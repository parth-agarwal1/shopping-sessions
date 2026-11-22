#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "parquet.hpp"

namespace fs = std::filesystem;
using View = std::string_view;
using Time = int64_t;
constexpr Time day_ms = 86400000, week = 7 * day_ms, gap = 1800000;
constexpr Time invalid = std::numeric_limits<Time>::min();
constexpr size_t bins = 128;
// Fewer ID partitions keep successive raw-row reads closer together.
constexpr size_t id_bins = 32;
int workers = 8;
size_t bucket(View s) { return std::hash<View>{}(s) % bins; }

template<class F> void parallel_workers(size_t n,int limit,F fn) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    std::exception_ptr failure;
    std::mutex failure_lock;
    for (size_t w = 0; w < std::min<size_t>(limit,n); ++w)
        pool.emplace_back([&,w]{
            try {
                for (size_t i=next++; i<n; i=next++) fn(i,w);
            } catch (...) {
                next.store(n);
                std::lock_guard lock(failure_lock);
                if (!failure) failure=std::current_exception();
            }
        });
    for (auto& thread:pool) thread.join();
    if (failure) std::rethrow_exception(failure);
}

template<class F> void parallel(size_t n,F fn) {
    parallel_workers(n,workers,[&](size_t i,size_t){ fn(i); });
}

// All views into input files remain valid until output has been produced.
struct Input {
    std::vector<std::pair<char*,size_t>> maps;
    ~Input() { for (auto [p,n]:maps) munmap(p,n); }
    View open(const fs::path& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        struct stat st{};
        if (fd < 0 || fstat(fd,&st)) throw std::runtime_error("cannot open " + path.string());
        size_t n = st.st_size;
        if (!n) { close(fd); return {}; }
        auto p = static_cast<char*>(mmap(nullptr,n,PROT_READ,MAP_PRIVATE,fd,0));
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        maps.emplace_back(p,n);
        return {p,n};
    }
};
View body(View s) {
    auto pos=s.find('\n');
    return pos==View::npos ? View{} : s.substr(pos+1);
}
template<class F> void lines(View s,F fn) {
    while (!s.empty()) {
        auto end=s.find('\n');
        if (end==View::npos) { fn(s); break; }
        fn(s.substr(0,end)); s.remove_prefix(end+1);
    }
}
bool decode(View& v) {
    if (v.size()>=2 && v.front()=='"' && v.back()=='"') {
        v.remove_prefix(1); v.remove_suffix(1);
    }
    return v.find('"')==View::npos;
}
template<size_t N> bool fields(View s,std::array<View,N>& dst) {
    for (size_t i=0;i<N;++i) {
        auto pos=s.find(',');
        if ((pos==View::npos)!=(i+1==N)) return false;
        dst[i]=s.substr(0,pos);
        if (!decode(dst[i])) return false;
        if (pos!=View::npos) s.remove_prefix(pos+1);
    }
    return true;
}
int digits(View s,size_t pos,int n) {
    int v=0;
    for (int i=0;i<n;++i) {
        unsigned d=unsigned(s[pos+i]-'0'); if (d>9) return -1;
        v=v*10+int(d);
    }
    return v;
}
Time decimal(View s,Time limit) {
    if (s.empty()) return invalid;
    Time v=0;
    // Up to 18 decimal digits fit in an i64 regardless of their value.
    if (s.size()<=18) {
        for (char c:s) {
            unsigned d=unsigned(c-'0');
            if (d>9) return invalid;
            v=v*10+d;
        }
        return v<=limit ? v : invalid;
    }
    for (char c:s) {
        unsigned d=unsigned(c-'0');
        if (d>9 || v>(limit-int(d))/10) return invalid;
        v=v*10+d;
    }
    return v;
}
Time date_epoch(View s) {
    int y=digits(s,0,4), m=digits(s,5,2), d=digits(s,8,2);
    if (y<0 || m<1 || m>12 || d<1) return invalid;
    constexpr int lengths[]={31,28,31,30,31,30,31,31,30,31,30,31};
    bool leap=y%4==0 && (y%100!=0 || y%400==0);
    if (d>lengths[m-1]+(m==2 && leap)) return invalid;
    // March-based Gregorian calendar; 719468 translates its origin to 1970-01-01.
    y-=(m<=2);
    int era=(y>=0?y:y-399)/400, year=y-era*400;
    int ordinal=(153*(m+(m>2?-3:9))+2)/5+d-1;
    return (era*146097+year*365+year/4-year/100+ordinal-719468)*day_ms;
}
Time stamp(View s) {
    if ((s.size()!=20 && s.size()!=25) || s[4]!='-')
        return decimal(s,std::numeric_limits<Time>::max());
    if (s[7]!='-' || s[10]!='T' || s[13]!=':' || s[16]!=':') return invalid;
    int h=digits(s,11,2), min=digits(s,14,2), sec=digits(s,17,2);
    if (h<0 || h>=24 || min<0 || min>=60 || sec<0 || sec>=60) return invalid;
    // Cache checked dates per thread. A hit requires the entire date prefix.
    struct Date { uint64_t front=0; uint16_t tail=0; Time epoch=invalid; };
    static thread_local std::array<Date,128> days;
    uint64_t front;
    uint16_t tail;
    std::memcpy(&front,s.data(),sizeof front);
    std::memcpy(&tail,s.data()+sizeof front,sizeof tail);
    auto& date=days[(tail+(tail>>5)+uint8_t(s[6])*17+uint8_t(s[3])*39)%days.size()];
    if (date.front!=front || date.tail!=tail) date={front,tail,date_epoch(s)};
    if (date.epoch==invalid) return invalid;
    int offset=0;
    if (s.size()==20) { if (s[19]!='Z') return invalid; }
    else {
        int oh=digits(s,20,2), om=digits(s,23,2);
        if ((s[19]!='+' && s[19]!='-') || s[22]!=':' || oh<0 || oh>=24 || om<0 || om>=60) return invalid;
        offset=(s[19]=='+'?1:-1)*(oh*60+om);
    }
    return date.epoch+
        ((h*60+min-offset)*60+sec)*1000LL;
}

struct People {
    // A tag rejects hash collisions without reading the mapped CSV again.
    struct Entry { View id; int index=0; uint32_t tag=0; };
    std::vector<Entry> ids = std::vector<Entry>(16);
    std::vector<int> parent, size;
    std::vector<View> name;
    static size_t probe(const std::vector<Entry>& table,View id,size_t hash) {
        size_t at=hash&(table.size()-1);
        uint32_t tag=hash>>32;
        while (!table[at].id.empty() && (table[at].tag!=tag || table[at].id!=id))
            at=(at+1)&(table.size()-1);
        return at;
    }
    void grow() {
        std::vector<Entry> table(ids.size()*2);
        for (auto entry:ids) if (!entry.id.empty()) {
            auto hash=std::hash<View>{}(entry.id);
            table[probe(table,entry.id,hash)]=entry;
        }
        ids.swap(table);
    }
    int add(View id) {
        if (parent.size()*4>=ids.size()*3) grow();
        auto hash=std::hash<View>{}(id);
        auto& entry=ids[probe(ids,id,hash)];
        if (entry.id.empty()) {
            int index=parent.size();
            entry={id,index,uint32_t(hash>>32)};
            parent.push_back(index);
            size.push_back(1);
            name.push_back(id);
        }
        return entry.index;
    }
    int root(int i) {
        while (i!=parent[i]) { parent[i]=parent[parent[i]]; i=parent[i]; }
        return i;
    }
    void join(View a,View b) {
        int x=root(add(a)), y=root(add(b));
        if (x==y) return;
        if (size[x]<size[y]) std::swap(x,y);
        parent[y]=x; size[x]+=size[y]; name[x]=std::min(name[x],name[y]);
    }
    void freeze() {
        for (auto& entry:ids) if (!entry.id.empty()) entry.index=root(entry.index);
        std::vector<int>().swap(parent);
        std::vector<int>().swap(size);
    }
    View person(View id) const {
        auto hash=std::hash<View>{}(id);
        const auto& entry=ids[probe(ids,id,hash)];
        return entry.id.empty() ? id : name[entry.index];
    }
};
struct Context {
    People people;
    std::unordered_map<View,int> products, campaigns;
    std::vector<View> categories{"unknown"}, campaign{"organic"}, channel{"organic"};
    Time start,end;
};
template<class F> void pairs(View v,F fn) {
    lines(body(v),[&](View line){
        std::array<View,2> a;
        if (fields(line,a) && !a[0].empty() && !a[1].empty()) fn(a[0],a[1]);
    });
}
std::vector<fs::path> files(const fs::path& dir) {
    std::vector<fs::path> out;
    for (auto& e: fs::directory_iterator(dir)) if (e.path().extension()==".csv") out.push_back(e.path());
    return out;
}
Time manifest_time(const std::string& json,const std::string& key) {
    std::smatch match;
    std::regex pattern("\""+key+"\"\\s*:\\s*\"([^\"]+)\"");
    if (!std::regex_search(json,match,pattern)) throw std::runtime_error("missing "+key);
    Time t=stamp(match[1].str());
    if (t==invalid) throw std::runtime_error("invalid manifest date");
    return t;
}
Context load(Input& input,const fs::path& data) {
    Context c;
    std::string manifest(input.open(data/"manifest.json"));
    c.start=manifest_time(manifest,"window_start"); c.end=manifest_time(manifest,"window_end");
    std::unordered_map<View,int> cats{{"unknown",0}};
    pairs(input.open(data/"products.csv"),[&](View p,View cat){
        auto [it,added]=cats.try_emplace(cat,c.categories.size());
        if (added) c.categories.push_back(cat);
        c.products.emplace(p,it->second);
    });
    pairs(input.open(data/"campaigns.csv"),[&](View id,View channel){
        c.campaigns.emplace(id,c.campaign.size()); c.campaign.push_back(id); c.channel.push_back(channel);
    });
    return c;
}
void load_people(Input& input,const fs::path& dir,People& people) {
    for (auto& path:files(dir)) pairs(input.open(path),[&](View a,View b){people.join(a,b);});
    people.freeze();
}

struct Event { View user; Time ts; int price,cat; };
struct Click { View user,id; Time ts; int campaign; };
using Events=std::vector<Event>;
using Clicks=std::vector<Click>;
template<class T> using Shards=std::array<std::vector<T>,bins>;

// Route by ID before reading payloads. Chunk boundaries always follow a physical LF.
std::vector<View> scan_chunks(Input& input,const fs::path& dir) {
    std::vector<View> chunks;
    for (auto& path:files(dir)) {
        View s=body(input.open(path));
        constexpr size_t target=2*1024*1024;
        while (s.size()>target) {
            auto cut=s.find('\n',target);
            if (cut==View::npos) break;
            chunks.push_back(s.substr(0,cut+1)); s.remove_prefix(cut+1);
        }
        if (!s.empty()) chunks.push_back(s);
    }
    return chunks;
}

std::array<std::vector<View>,id_bins> route(Input& input,const fs::path& dir) {
    auto chunks=scan_chunks(input,dir);
    std::array<std::vector<View>,id_bins> result;
    std::array<std::mutex,id_bins> locks;
    parallel(chunks.size(),[&](size_t i){
        std::array<std::vector<View>,id_bins> local;
        lines(chunks[i],[&](View s){
            View id=s.substr(0,s.find(','));
            if (decode(id) && !id.empty()) local[std::hash<View>{}(id)%id_bins].push_back(s);
        });
        for (size_t b=0;b<id_bins;++b) {
            std::lock_guard lock(locks[b]);
            result[b].insert(result[b].end(),local[b].begin(),local[b].end());
        }
    });
    return result;
}
struct Winner { View id; Time ingest; Event event; };
template<bool events> auto clean(Input& input,const fs::path& dir,const Context& c) {
    using Row=std::conditional_t<events,Event,Click>;
    auto raw=route(input,dir);
    Shards<Row> result;
    std::array<std::mutex,bins> locks;
    parallel(id_bins,[&](size_t b){
        Shards<Row> local;
        // Open addressing, with full byte equality on collisions. Invalid earliest
        // events still occupy their slot and can replace a valid later event.
        size_t capacity=1;
        while (capacity<raw[b].size()*1.5) capacity*=2;
        struct Slot { int index=-1; uint32_t tag=0; };
        std::vector<Slot> slots(capacity);
        std::vector<std::conditional_t<events,Winner,View>> winners;
        winners.reserve(raw[b].size());
        for (View line:raw[b]) {
            std::array<View,events?7:4> a;
            if (!fields(line,a)) continue;
            const auto hash=std::hash<View>{}(a[0]);
            const uint32_t tag=hash>>32;
            size_t pos=(hash/id_bins)&(capacity-1);
            while (slots[pos].index>=0) {
                if (slots[pos].tag==tag) {
                    View existing;
                    if constexpr (events) existing=winners[slots[pos].index].id;
                    else existing=winners[slots[pos].index];
                    if (existing==a[0]) break;
                }
                pos=(pos+1)&(capacity-1);
            }
            slots[pos].tag=tag;
            int& slot=slots[pos].index;
            if constexpr (events) {
                Time ingest=stamp(a[3]);
                if (slot>=0 && winners[slot].ingest<=ingest) continue;
                if (slot<0) { slot=winners.size(); winners.push_back({a[0],ingest,{}}); }
                auto& w=winners[slot]; w.ingest=ingest; w.event.user={};
                Time ts=stamp(a[2]);
                if (a[1].empty() || a[5].empty() || ts<c.start || ts>=c.end || ingest-ts>day_ms) continue;
                int price=0,cat=0;
                if (a[4]=="purchase") {
                    Time p=decimal(a[6],10000000);
                    if (p<=0) continue;
                    price=int(p);
                    auto it=c.products.find(a[5]); if (it!=c.products.end()) cat=it->second;
                } else if (a[4]!="cart" && a[4]!="view") continue;
                w.event={a[1],ts,price,cat};
            } else {
                if (slot>=0) continue;
                slot=winners.size(); winners.push_back(a[0]);
                Time ts=stamp(a[3]); auto it=c.campaigns.find(a[2]);
                if (ts<c.start-week || ts>=c.end || a[1].empty() || it==c.campaigns.end()) continue;
                View user=c.people.person(a[1]);
                local[bucket(user)].push_back({user,a[0],ts,it->second});
            }
        }
        if constexpr (events) for (auto& w:winners) if (!w.event.user.empty()) {
            w.event.user=c.people.person(w.event.user);
            local[bucket(w.event.user)].push_back(w.event);
        }
        std::vector<View>().swap(raw[b]);
        for (size_t k=0;k<bins;++k) {
            std::lock_guard lock(locks[k]);
            result[k].insert(result[k].end(),local[k].begin(),local[k].end());
        }
    });
    return result;
}

#include "packed.hpp"

struct Totals {
    int64_t sessions=0,purchasing=0,revenue=0,assists=0;
    void add(const Totals& o) {
        sessions+=o.sessions; purchasing+=o.purchasing; revenue+=o.revenue; assists+=o.assists;
    }
};
using CampaignMap=std::unordered_map<uint64_t,Totals>;
uint64_t campaign_day(int id,int32_t day) {
    return (uint64_t(id)<<32)|uint32_t(day);
}
int32_t utc_day(Time t) { return int32_t(t/day_ms-(t%day_ms<0)); }
struct CampaignDays {
    CampaignMap sparse;
    std::vector<Totals> dense;
    int32_t first_day=0;
    size_t campaigns=0;
    bool prepared=false;

    void prepare(const Context& ctx,bool use_dense) {
        if (prepared) return;
        prepared=true;
        if (!use_dense || ctx.end<=ctx.start) return;
        first_day=utc_day(ctx.start);
        const size_t span=int64_t(utc_day(ctx.end-1))-first_day+1;
        campaigns=ctx.campaign.size();
        if (campaigns<=65536/span) dense.resize(span*campaigns);
    }
    Totals& at(int campaign,int32_t day) {
        if (dense.empty()) return sparse[campaign_day(campaign,day)];
        return dense[size_t(day-first_day)*campaigns+campaign];
    }
    void merge(CampaignMap& out) const {
        for (const auto& [key,totals]:sparse) out[key].add(totals);
        for (size_t i=0;i<dense.size();++i) {
            const auto& totals=dense[i];
            if (totals.sessions || totals.assists)
                out[campaign_day(i%campaigns,first_day+int32_t(i/campaigns))].add(totals);
        }
    }
};
void quoted(std::ostream& stream,View v) { stream << '"' << v << '"'; }
struct UserParts {
    fs::path dir;
    size_t shard, rows=0;
    bool direct;
    std::ofstream stream;
    parquet::Table table{{"user_id",parquet::String},{"day",parquet::Date},
        {"sessions",parquet::Int64},{"events",parquet::Int64},{"revenue_cents",parquet::Int64},
        {"top_category",parquet::String,true}};
    UserParts(fs::path dir,size_t shard,bool direct):dir(std::move(dir)),shard(shard),direct(direct) {
        stream.exceptions(std::ios::badbit|std::ios::failbit);
    }
    void flush() {
        auto path=dir/"user_daily"/("part-"+std::to_string(shard)+"-"+std::to_string(rows)+".parquet");
        table.flush(path);
    }
    void finish() {
        if (table.size()) flush();
        if (stream.is_open()) stream.close();
    }
    void write(View user,int day,int64_t sessions,int64_t count,int64_t revenue,View top) {
        if (direct) {
            table.string(0,user);
            table.integer(1,day);
            table.integer(2,sessions);
            table.integer(3,count);
            table.integer(4,revenue);
            table.string(5,top,revenue!=0);
            table.finish_row();
            ++rows;
            if (table.size()==65536) flush();
            return;
        }
        if (rows%65536==0) {
            if (stream.is_open()) stream.close();
            stream.open(dir/("user_daily-"+std::to_string(shard)+"-"+std::to_string(rows)+".csv"));
        }
        ++rows; quoted(stream,user); stream<<','<<day<<','<<sessions<<','<<count<<','<<revenue<<',';
        if (revenue) quoted(stream,top);
        stream<<'\n';
    }
};
template<class E,class C> void aggregate(std::vector<E>& e,std::vector<C>& c,const Context& ctx,
        CampaignDays& totals,UserParts& out,Time base,bool verified,packed::RunStats& stats,
        packed::EventBuffers* blocks) {
    constexpr bool compact=std::is_same_v<E,packed::Evt>;
    auto time_of=[base](const auto& row) { return base+Time(row.ts); };
    std::vector<int64_t> by_category(ctx.categories.size());
    std::vector<int> touched;
    std::vector<int> active;
    std::vector<int> active_counts(ctx.campaign.size());
    std::vector<size_t> active_index(ctx.campaign.size());

    auto person_output=[&](View user,const auto& events,size_t i,size_t event_end,
                           const std::vector<C>& clicks,size_t left,size_t click_end) {
        size_t right=left;
        for (int campaign:active) active_counts[campaign]=0;
        active.clear();
        while (i<event_end) {
            int day=utc_day(time_of(events[i]));
            int64_t sessions=0,count=0,revenue=0;
            do {
                Time start=time_of(events[i]);
                while (right<click_end && time_of(clicks[right])<=start) {
                    const int campaign=clicks[right++].campaign;
                    if (active_counts[campaign]++==0) {
                        active_index[campaign]=active.size();
                        active.push_back(campaign);
                    }
                }
                while (left<right && time_of(clicks[left])<start-week) {
                    const int campaign=clicks[left++].campaign;
                    if (--active_counts[campaign]==0) {
                        const size_t at=active_index[campaign];
                        active[at]=active.back();
                        active_index[active[at]]=at;
                        active.pop_back();
                    }
                }
                int winner=left==right?0:clicks[right-1].campaign;
                for (int campaign:active) ++totals.at(campaign,day).assists;
                int64_t session_revenue=0;
                do {
                    const E& event=events[i++];
                    int price,cat;
                    if constexpr (compact) { price=event.price(); cat=event.cat(); }
                    else { price=event.price; cat=event.cat; }
                    if (price) {
                        auto& sum=by_category[cat];
                        if (!sum) touched.push_back(cat);
                        sum+=price; session_revenue+=price;
                    }
                    ++count;
                } while (i<event_end && events[i].ts-events[i-1].ts<=gap);
                auto& t=totals.at(winner,day);
                ++t.sessions; t.revenue+=session_revenue; t.purchasing+=(session_revenue>0);
                ++sessions; revenue+=session_revenue;
            } while (i<event_end && utc_day(time_of(events[i]))==day);
            View top; int64_t highest=0;
            for (int k:touched) {
                if (by_category[k]>highest || (by_category[k]==highest && ctx.categories[k]<top)) {
                    top=ctx.categories[k]; highest=by_category[k];
                }
                by_category[k]=0;
            }
            touched.clear(); out.write(user,day,sessions,count,revenue,top);
        }
    };

    std::vector<packed::Key> compact_names;
    if constexpr (compact) {
        if (verified) {
            std::vector<packed::EventSpan> spans;
            if (blocks) {
                spans.reserve(blocks->size());
                for (auto& block:*blocks) spans.push_back({block.data(),block.size()});
            } else if (!e.empty()) spans.push_back({e.data(),e.size()});
            auto groups=packed::verify_runs(spans,c);
            stats=groups.stats;
            size_t click_start=0;
            for (const auto& person:groups.people) {
                std::array<char,9> text;
                auto span=person.split ? packed::EventSpan{groups.regrouped.data(),groups.regrouped.size()}
                                       : spans[person.block];
                person_output(packed::name(person.name,text),span,person.first,person.first+person.count,
                              groups.grouped_clicks,click_start,person.click_end);
                click_start=person.click_end;
            }
            return;
        }
        compact_names=packed::group(e,c);
    } else {
        // Canonicalize backing addresses so even arbitrary IDs sort as one word.
        std::unordered_set<View> names;
        for (auto& row:e) row.user=*names.insert(row.user).first;
        for (auto& row:c) row.user=*names.insert(row.user).first;
    }
    auto key_of=[](const auto& row) {
        if constexpr (compact) return row.user;
        else return row.user.data();
    };
    auto before=std::less<decltype(key_of(std::declval<E&>()))>{};
    if constexpr (!compact) {
        std::sort(e.begin(),e.end(),[&](const E& a,const E& b){
            if (key_of(a)!=key_of(b)) return before(key_of(a),key_of(b));
            return a.ts<b.ts;
        });
        std::sort(c.begin(),c.end(),[&](const C& a,const C& b){
            if (key_of(a)!=key_of(b)) return before(key_of(a),key_of(b));
            return a.ts!=b.ts?a.ts<b.ts:a.id>b.id;
        });
    }
    size_t click_start=0;
    for (size_t start=0;start<e.size();) {
        const auto key=key_of(e[start]);
        size_t end=start+1;
        while (end<e.size() && key_of(e[end])==key) ++end;
        while (click_start<c.size() && before(key_of(c[click_start]),key)) ++click_start;
        size_t click_end=click_start;
        while (click_end<c.size() && key_of(c[click_end])==key) ++click_end;
        [[maybe_unused]] std::array<char,9> text;
        View user;
        if constexpr (compact) user=packed::name(compact_names[key],text);
        else user=e[start].user;
        person_output(user,e,start,end,c,click_start,click_end);
        start=end;
        click_start=click_end;
    }
}
int main(int argc,char** argv) try {
    const bool direct = argc==1 || (argc==4 && View(argv[3])=="--parquet");
    if (!direct && argc!=3) throw std::runtime_error("usage: shop [DATA_DIR OUTPUT_DIR [--parquet]]");
    if (auto n=getenv("NCPU")) workers=std::clamp(atoi(n),1,128);
    auto path_env=[](const char* variable,const char* default_path) {
        auto value=getenv(variable);
        return fs::path(value && *value ? value : default_path);
    };
    fs::path data=argc==1 ? path_env("DATA_DIR","/data") : fs::path(argv[1]);
    fs::path stage=argc==1 ? path_env("OUTPUT_DIR","/output") : fs::path(argv[2]);
    auto last=std::chrono::steady_clock::now();
    auto mark=[&](const char* label) {
        auto now=std::chrono::steady_clock::now();
        std::cerr<<label<<' '<<std::chrono::duration<double>(now-last).count()<<" s\n"; last=now;
    };
    std::array<CampaignDays,bins> partial;
    std::atomic<bool> any_users{false};
    if (direct) {
        fs::create_directories(stage/"user_daily");
        fs::create_directories(stage/"campaign_daily");
    }
    auto summarize=[&](auto& events,auto& clicks,Time base,const Context& ctx,bool verified,
                       std::array<packed::EventBuffers,bins>* segments=nullptr) {
        std::array<packed::RunStats,bins> stats;
        const bool dense_totals=!getenv("SHOP_SPARSE_TOTALS");
        std::array<std::unique_ptr<UserParts>,bins> writers;
        std::array<size_t,bins> order;
        std::iota(order.begin(),order.end(),0);
        // Start the largest shards first, and keep one writer and totals
        // table per worker so their buffers survive across shard boundaries.
        std::array<size_t,bins> weights;
        for (size_t b=0;b<bins;++b) {
            weights[b]=events[b].size()+2*clicks[b].size();
            if (segments) for (const auto& block:(*segments)[b]) weights[b]+=block.size();
        }
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){ return weights[a]>weights[b]; });
        parallel_workers(bins,workers,[&](size_t item,size_t worker){
            const size_t b=order[item];
            auto blocks=segments ? &(*segments)[b] : nullptr;
            if (events[b].empty() && (!blocks || blocks->empty())) {
                std::decay_t<decltype(clicks[b])>().swap(clicks[b]);
                return;
            }
            auto& writer=writers[worker];
            if (!writer) writer=std::make_unique<UserParts>(stage,worker,direct);
            auto& campaign_days=partial[worker];
            campaign_days.prepare(ctx,dense_totals);
            aggregate(events[b],clicks[b],ctx,campaign_days,*writer,base,verified,stats[b],blocks);
            if (writer->rows) any_users.store(true,std::memory_order_relaxed);
            if (blocks) packed::EventBuffers().swap(*blocks);
            std::decay_t<decltype(events[b])>().swap(events[b]);
            std::decay_t<decltype(clicks[b])>().swap(clicks[b]);
        });
        if (any_users.load(std::memory_order_relaxed))
            parallel(workers,[&](size_t i){ if (writers[i]) writers[i]->finish(); });
        if (verified && getenv("SHOP_PROFILE")) {
            packed::RunStats total;
            for (const auto& s:stats) {
                total.ordered+=s.ordered;
                total.sorted_in_place+=s.sorted_in_place;
                total.regrouped+=s.regrouped;
            }
            std::cerr<<"shopper rows: ordered "<<total.ordered<<", sorted in place "
                     <<total.sorted_in_place<<", regrouped "<<total.regrouped<<'\n';
        }
    };
    Input input; auto ctx=load(input,data); mark("dimensions");
    std::optional<packed::Clean> compact;
    if (!getenv("SHOP_GENERIC")) {
        // Release trial mappings before starting the generic path if packing fails.
        Input mapped;
        compact=packed::read(mapped,data,ctx);
        if (compact) {
            mark("packed links, events and clicks");
            summarize(compact->events,compact->clicks,compact->base,ctx,
                      compact->dense && !getenv("SHOP_REGROUP_ALL"),
                      compact->segmented ? &compact->segments : nullptr);
        }
    }
    if (!compact) {
        load_people(input,data/"links",ctx.people);
        auto events=clean<true>(input,data/"events",ctx); mark("general events");
        auto clicks=clean<false>(input,data/"clicks",ctx); mark("general clicks");
        summarize(events,clicks,0,ctx,false);
    }
    if (direct && !any_users.load()) {
        UserParts empty(stage,0,true);
        empty.flush();
    }
    CampaignMap totals;
    for (const auto& shard:partial) shard.merge(totals);
    if (direct) {
        parquet::Table table{{"campaign_id",parquet::String},{"channel",parquet::String},
            {"day",parquet::Date},{"sessions",parquet::Int64},{"purchasing_sessions",parquet::Int64},
            {"revenue_cents",parquet::Int64},{"assisted_sessions",parquet::Int64}};
        size_t part=0;
        auto flush=[&] {
            table.flush(stage/"campaign_daily"/("part-"+std::to_string(part++)+".parquet"));
        };
        for (auto& [key,t]:totals) {
            auto id=key>>32;
            table.string(0,ctx.campaign[id]); table.string(1,ctx.channel[id]);
            table.integer(2,int32_t(key)); table.integer(3,t.sessions);
            table.integer(4,t.purchasing); table.integer(5,t.revenue); table.integer(6,t.assists);
            table.finish_row();
            if (table.size()==65536) flush();
        }
        if (table.size() || !part) flush();
    } else if (!totals.empty()) {
        std::ofstream out;
        out.exceptions(std::ios::failbit|std::ios::badbit); out.open(stage/"campaign_daily-0.csv");
        for (auto& [key,t]:totals) {
            auto id=key>>32;
            quoted(out,ctx.campaign[id]); out<<','; quoted(out,ctx.channel[id]);
            out<<','<<int32_t(key)<<','<<t.sessions<<','<<t.purchasing<<','<<t.revenue<<','<<t.assists<<'\n';
        }
        out.close();
    }
    mark("sessions, aggregation and output");
    return 0;
} catch (const std::exception& e) {
    std::cerr<<e.what()<<'\n'; return 1;
}
