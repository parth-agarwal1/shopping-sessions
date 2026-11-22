#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

template<class F> void parallel(size_t n, F fn) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    std::exception_ptr failure;
    std::mutex failure_lock;
    for (size_t w = 0; w < std::min<size_t>(workers,n); ++w)
        pool.emplace_back([&]{
            try {
                for (size_t i=next++; i<n; i=next++) fn(i);
            } catch (...) {
                next.store(n);
                std::lock_guard lock(failure_lock);
                if (!failure) failure=std::current_exception();
            }
        });
    for (auto& thread:pool) thread.join();
    if (failure) std::rethrow_exception(failure);
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
    for (char c:s) {
        unsigned d=unsigned(c-'0');
        if (d>9 || v>(limit-int(d))/10) return invalid;
        v=v*10+d;
    }
    return v;
}
Time stamp(View s) {
    if ((s.size()!=20 && s.size()!=25) || s[4]!='-')
        return decimal(s,std::numeric_limits<Time>::max());
    if (s[7]!='-' || s[10]!='T' || s[13]!=':' || s[16]!=':') return invalid;
    int y=digits(s,0,4), m=digits(s,5,2), d=digits(s,8,2);
    int h=digits(s,11,2), min=digits(s,14,2), sec=digits(s,17,2);
    if (y<0 || m<1 || m>12 || d<1 || h<0 || h>=24 || min<0 || min>=60 || sec<0 || sec>=60) return invalid;
    constexpr int lengths[]={31,28,31,30,31,30,31,31,30,31,30,31};
    bool leap=y%4==0 && (y%100!=0 || y%400==0);
    if (d>lengths[m-1]+(m==2 && leap)) return invalid;
    // March-based Gregorian calendar; 719468 translates its origin to 1970-01-01.
    y-=(m<=2);
    int era=(y>=0?y:y-399)/400, year=y-era*400;
    int ordinal=(153*(m+(m>2?-3:9))+2)/5+d-1;
    Time epoch_day=era*146097+year*365+year/4-year/100+ordinal-719468;
    int offset=0;
    if (s.size()==20) { if (s[19]!='Z') return invalid; }
    else {
        int oh=digits(s,20,2), om=digits(s,23,2);
        if ((s[19]!='+' && s[19]!='-') || s[22]!=':' || oh<0 || oh>=24 || om<0 || om>=60) return invalid;
        offset=(s[19]=='+'?1:-1)*(oh*60+om);
    }
    return epoch_day*day_ms+
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
    for (auto& path:files(data/"links")) pairs(input.open(path),[&](View a,View b){c.people.join(a,b);});
    c.people.freeze();
    return c;
}

struct Event { View user; Time ts; int price,cat; };
struct Click { View user,id; Time ts; int campaign; };
using Events=std::vector<Event>;
using Clicks=std::vector<Click>;
template<class T> using Shards=std::array<std::vector<T>,bins>;

// Route by ID before reading payloads. Chunk boundaries always follow a physical LF.
std::array<std::vector<View>,id_bins> route(Input& input,const fs::path& dir) {
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

struct Totals {
    int64_t sessions=0,purchasing=0,revenue=0,assists=0;
    void add(const Totals& o) {
        sessions+=o.sessions; purchasing+=o.purchasing; revenue+=o.revenue; assists+=o.assists;
    }
};
using CampaignDays=std::unordered_map<uint64_t,Totals>;
uint64_t campaign_day(int id,int32_t day) {
    return (uint64_t(id)<<32)|uint32_t(day);
}
int32_t utc_day(Time t) { return int32_t(t/day_ms-(t%day_ms<0)); }
void quoted(std::ostream& stream,View v) { stream << '"' << v << '"'; }
struct UserParts {
    fs::path dir;
    size_t shard, rows=0;
    std::ofstream stream;
    UserParts(fs::path dir,size_t shard):dir(std::move(dir)),shard(shard) {
        stream.exceptions(std::ios::badbit|std::ios::failbit);
    }
    void write(View user,int day,int64_t sessions,int64_t count,int64_t revenue,View top) {
        if (rows%65536==0) {
            if (stream.is_open()) stream.close();
            stream.open(dir/("user_daily-"+std::to_string(shard)+"-"+std::to_string(rows)+".csv"));
        }
        ++rows; quoted(stream,user); stream<<','<<day<<','<<sessions<<','<<count<<','<<revenue<<',';
        if (revenue) quoted(stream,top);
        stream<<'\n';
    }
};
void aggregate(Events& e,Clicks& c,const Context& ctx,CampaignDays& totals,UserParts& out) {
    // Canonicalize the backing address as well as the value, including IDs
    // absent from the link graph. Sorting can then use a fixed-width key.
    std::unordered_set<View> names;
    for (Event& row:e) row.user=*names.insert(row.user).first;
    for (Click& row:c) row.user=*names.insert(row.user).first;
    auto before=std::less<const char*>{};
    std::sort(e.begin(),e.end(),[&](const Event& a,const Event& b){
        if (a.user.data()!=b.user.data()) return before(a.user.data(),b.user.data());
        return a.ts<b.ts;
    });
    // Descending click ID puts the smallest ID last in a timestamp group.
    std::sort(c.begin(),c.end(),[&](const Click& a,const Click& b){
        if (a.user.data()!=b.user.data()) return before(a.user.data(),b.user.data());
        return a.ts!=b.ts?a.ts<b.ts:a.id>b.id;
    });
    size_t i=0, cursor=0;
    std::vector<int64_t> by_category(ctx.categories.size());
    std::vector<int> touched;
    std::unordered_map<int,int> active;
    while (i<e.size()) {
        View user=e[i].user;
        while (cursor<c.size() && before(c[cursor].user.data(),user.data())) ++cursor;
        size_t right=cursor,left=cursor;
        active.clear();
        while (i<e.size() && e[i].user.data()==user.data()) {
            int day=utc_day(e[i].ts);
            int64_t sessions=0,count=0,revenue=0;
            do {
                Time start=e[i].ts;
                while (right<c.size() && c[right].user.data()==user.data() && c[right].ts<=start)
                    ++active[c[right++].campaign];
                while (left<right && c[left].ts<start-week) {
                    auto it=active.find(c[left++].campaign);
                    if (--it->second==0) active.erase(it);
                }
                int winner=left==right?0:c[right-1].campaign;
                for (auto [campaign,n]:active) ++totals[campaign_day(campaign,day)].assists;
                int64_t session_revenue=0;
                do {
                    const Event& event=e[i++];
                    if (event.price) {
                        auto& sum=by_category[event.cat];
                        if (!sum) touched.push_back(event.cat);
                        sum+=event.price; session_revenue+=event.price;
                    }
                    ++count;
                } while (i<e.size() && e[i].user.data()==user.data() && e[i].ts-e[i-1].ts<=gap);
                auto& t=totals[campaign_day(winner,day)];
                ++t.sessions; t.revenue+=session_revenue; t.purchasing+=(session_revenue>0);
                ++sessions; revenue+=session_revenue;
            } while (i<e.size() && e[i].user.data()==user.data() && utc_day(e[i].ts)==day);
            View top; int64_t highest=0;
            for (int k:touched) {
                if (by_category[k]>highest || (by_category[k]==highest && ctx.categories[k]<top)) {
                    top=ctx.categories[k]; highest=by_category[k];
                }
                by_category[k]=0;
            }
            touched.clear(); out.write(user,day,sessions,count,revenue,top);
        }
        cursor=right;
    }
}
int main(int argc,char** argv) try {
    if (argc!=3) throw std::runtime_error("usage: shop DATA_DIR STAGE_DIR");
    if (auto n=getenv("NCPU")) workers=std::clamp(atoi(n),1,128);
    fs::path data=argv[1],stage=argv[2];
    auto last=std::chrono::steady_clock::now();
    auto mark=[&](const char* label) {
        auto now=std::chrono::steady_clock::now();
        std::cerr<<label<<' '<<std::chrono::duration<double>(now-last).count()<<" s\n"; last=now;
    };
    Input input; auto ctx=load(input,data); mark("dimensions and links");
    auto events=clean<true>(input,data/"events",ctx); mark("events");
    auto clicks=clean<false>(input,data/"clicks",ctx); mark("clicks");
    std::array<CampaignDays,bins> partial;
    parallel(bins,[&](size_t b){
        UserParts out(stage,b);
        aggregate(events[b],clicks[b],ctx,partial[b],out);
        if (out.stream.is_open()) out.stream.close();
        Events().swap(events[b]); Clicks().swap(clicks[b]);
    });
    CampaignDays totals;
    for (const auto& shard:partial) for (auto& [key,t]:shard) totals[key].add(t);
    if (!totals.empty()) {
        std::ofstream out;
        out.exceptions(std::ios::failbit|std::ios::badbit); out.open(stage/"campaign_daily-0.csv");
        for (auto& [key,t]:totals) {
            auto id=key>>32;
            quoted(out,ctx.campaign[id]); out<<','; quoted(out,ctx.channel[id]);
            out<<','<<int32_t(key)<<','<<t.sessions<<','<<t.purchasing<<','<<t.revenue<<','<<t.assists<<'\n';
        }
        out.close();
    }
    mark("sessions and aggregation");
    return 0;
} catch (const std::exception& e) {
    std::cerr<<e.what()<<'\n'; return 1;
}
