// Included after the common types and CSV parsing helpers in shop.cpp.
#pragma once
#include <immintrin.h>
#include <optional>

namespace packed {
using Key = uint64_t;
constexpr uint64_t byte_lanes = 0x7f7f7f7f7f7f7f7fULL;

// Nine nonzero ASCII bytes fit in 63 bits, with zero padding that preserves
// both exact identity and byte order. Unsupported IDs select the general path.
__attribute__((target("bmi2")))
inline Key key(View id) {
    if (id.empty() || id.size() > 9) return 0;
    if (id.size() == 9) {
        uint64_t low;
        std::memcpy(&low, id.data(), 8);
        auto last = uint8_t(id[8]);
        if (last == 0 || last > 127 || (low & ~byte_lanes) ||
            ((low - 0x0101010101010101ULL) & ~low & ~byte_lanes)) return 0;
        return (_pext_u64(__builtin_bswap64(low), byte_lanes) << 7) | last;
    }
    Key value = 0;
    for (uint8_t b : id) {
        if (!b || b > 127) return 0;
        value = (value << 7) | b;
    }
    return value << (7 * (9 - id.size()));
}
inline View name(Key key, std::array<char, 9>& text) {
    size_t n = 0;
    for (unsigned shift = 63; shift != 0; shift -= 7) {
        char next = char((key >> (shift - 7)) & 127);
        if (!next) break;
        text[n++] = next;
    }
    return {text.data(), n};
}
inline uint64_t hash(Key key) {
    key ^= key >> 31;
    key *= 0x9e3779b97f4a7c15ULL;
    return key ^ (key >> 29);
}

struct Names {
    struct Entry { Key id=0, person=0; };
    std::vector<Entry> entries;
    std::vector<uint64_t> aliases;
    bool fill(const std::vector<View>& chunks,int limit);
    size_t home(Key id) const { return hash(id)&(entries.size()-1); }
    Key resolve(Key id) const {
        const auto h=hash(id);
        if (!aliases.empty()) {
            const size_t bit=h&(aliases.size()*64-1);
            if (!(aliases[bit/64]&(1ULL<<(bit%64)))) return id;
        }
        size_t i=h&(entries.size()-1);
        while (entries[i].id && entries[i].id!=id) i=(i+1)&(entries.size()-1);
        return entries[i].id ? entries[i].person : id;
    }
};

struct Masks { uint64_t comma, quote, newline; };
__attribute__((target("avx2")))
inline Masks classify64(const char* data) {
    auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
    auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+32));
    auto mask=[&](char c) __attribute__((target("avx2"))) {
        auto ch=_mm256_set1_epi8(c);
        uint32_t lo=_mm256_movemask_epi8(_mm256_cmpeq_epi8(a,ch));
        uint32_t hi=_mm256_movemask_epi8(_mm256_cmpeq_epi8(b,ch));
        return uint64_t(hi)<<32 | lo;
    };
    return {mask(','),mask('"'),mask('\n')};
}

template<bool paired_quotes, size_t N, class F>
__attribute__((target("avx2,popcnt,bmi,bmi2")))
void scan_rows(View chunk, F consume) {
    const char* at=chunk.data();
    const char* end=at+chunk.size();
    while (end-at>=128) {
        auto lo=classify64(at);
        Masks hi{};
        if (!lo.newline) hi=classify64(at+64);
        if (!(lo.newline|hi.newline)) {
            const char* lf=static_cast<const char*>(memchr(at,'\n',end-at));
            if (!lf) break;
            std::array<View,N> f;
            if (fields(View(at,lf-at),f)) consume(f);
            at=lf+1;
            continue;
        }

        uint64_t commas[2]={lo.comma,hi.comma};
        size_t stop=lo.newline ? __builtin_ctzll(lo.newline) : 64+__builtin_ctzll(hi.newline);
        size_t last=stop/64;
        uint64_t keep=(1ULL<<(stop%64))-1;
        commas[last]&=keep;
        uint64_t quote_lo=lo.quote&(last ? UINT64_MAX : keep);
        uint64_t quote_hi=hi.quote&(last ? keep : 0);
        unsigned quotes=__builtin_popcountll(quote_lo)+__builtin_popcountll(quote_hi);

        if (__builtin_popcountll(commas[0])+__builtin_popcountll(commas[1])==N-1) {
            if constexpr (paired_quotes) {
                // An empty first field would drop its closing-position bit at
                // position -1. Let the general decoder handle that layout.
                if (!(commas[0]&1)) {
                    unsigned quoted=0;
                    if (quote_lo || quote_hi) {
                        uint64_t start_lo=(commas[0]<<1)|1;
                        uint64_t start_hi=(commas[1]<<1)|(commas[0]>>63);
                        uint64_t stop_lo=commas[0] | (last==0 ? 1ULL<<(stop%64) : 0);
                        uint64_t stop_hi=commas[1] | (last==1 ? 1ULL<<(stop%64) : 0);
                        uint64_t end_lo=(stop_lo>>1)|(stop_hi<<63);
                        uint64_t end_hi=stop_hi>>1;
                        // Reject interior quotes and fields with a single quote.
                        bool edge_only=
                            !(quote_lo&(~(start_lo|end_lo)|(start_lo&end_lo))) &&
                            !(quote_hi&(~(start_hi|end_hi)|(start_hi&end_hi)));
                        quoted=_pext_u64(quote_lo,start_lo) |
                               (_pext_u64(quote_hi,start_hi)<<__builtin_popcountll(start_lo));
                        unsigned closed=_pext_u64(quote_lo,end_lo) |
                            (_pext_u64(quote_hi,end_hi)<<__builtin_popcountll(end_lo));
                        if (!edge_only || quoted!=closed) {
                            at+=stop+1;
                            continue;
                        }
                    }
                    std::array<View,N> f;
                    size_t start=0, part=0;
                    for (size_t col=0;col<N;++col) {
                        size_t finish=stop;
                        if (col+1<N) {
                            if (!commas[part]) ++part;
                            finish=64*part+__builtin_ctzll(commas[part]);
                            commas[part]&=commas[part]-1;
                        }
                        unsigned q=(quoted>>col)&1;
                        f[col]={at+start+q,finish-start-2*q};
                        start=finish+1;
                    }
                    consume(f);
                    at+=stop+1;
                    continue;
                }
            }
            std::array<View,N> f;
            size_t start=0, part=0;
            for (size_t col=0;col<N;++col) {
                size_t finish=stop;
                if (col+1<N) {
                    if (!commas[part]) ++part;
                    finish=64*part+__builtin_ctzll(commas[part]);
                    commas[part]&=commas[part]-1;
                }
                if (finish-start>=2 && at[start]=='"' && at[finish-1]=='"') {
                    f[col]={at+start+1,finish-start-2};
                    quotes-=2;
                } else f[col]={at+start,finish-start};
                start=finish+1;
            }
            if (!quotes) consume(f);
        }
        at+=stop+1;
    }
    lines(View(at,end-at),[&](View line) {
        std::array<View,N> f;
        if (fields(line,f)) consume(f);
    });
}

template<size_t N, class F>
void rows(View chunk,F consume) {
    if constexpr (N>=4) {
        if (!getenv("SHOP_LEGACY_CSV")) return scan_rows<true,N>(chunk,consume);
    }
    scan_rows<false,N>(chunk,consume);
}

inline bool Names::fill(const std::vector<View>& chunks,int limit) {
    std::vector<std::vector<std::pair<Key,Key>>> shards(chunks.size());
    std::atomic<bool> incompatible{false};
    parallel_workers(chunks.size(),limit,[&](size_t chunk,size_t) {
        auto& edges=shards[chunk];
        rows<2>(chunks[chunk],[&](const auto& f) {
            if (f[0].empty() || f[1].empty()) return;
            Key a=key(f[0]), b=key(f[1]);
            if (a && b) edges.emplace_back(a,b);
            else incompatible.store(true,std::memory_order_relaxed);
        });
    });
    if (incompatible.load()) return false;

    struct Node { uint32_t parent,size; Key name; };
    std::vector<Node> nodes;
    entries.resize(16);
    auto put=[&](Key id) -> uint32_t {
        if (nodes.size()*4>=entries.size()*3) {
            auto old=std::move(entries);
            entries.resize(2*old.size());
            for (auto entry:old) if (entry.id) {
                size_t i=home(entry.id);
                while (entries[i].id) i=(i+1)&(entries.size()-1);
                entries[i]=entry;
            }
        }
        size_t i=home(id);
        while (entries[i].id && entries[i].id!=id) i=(i+1)&(entries.size()-1);
        if (entries[i].id) return uint32_t(entries[i].person);
        auto index=uint32_t(nodes.size());
        entries[i]={id,index};
        nodes.push_back({index,1,id});
        return index;
    };
    auto root=[&](uint32_t i) {
        while (nodes[i].parent!=i) {
            nodes[i].parent=nodes[nodes[i].parent].parent;
            i=nodes[i].parent;
        }
        return i;
    };
    for (const auto& edges:shards) for (auto [a,b]:edges) {
        uint32_t ar=root(put(a)), br=root(put(b));
        if (ar==br) continue;
        if (nodes[ar].size<nodes[br].size) std::swap(ar,br);
        nodes[br].parent=ar;
        nodes[ar].size+=nodes[br].size;
        nodes[ar].name=std::min(nodes[ar].name,nodes[br].name);
    }
    if (!getenv("SHOP_NO_ALIAS_FILTER") && entries.size()>=1024)
        aliases.resize(std::min(entries.size()/8,size_t(262144)));
    for (auto& entry:entries) if (entry.id) {
        entry.person=nodes[root(entry.person)].name;
        if (!aliases.empty() && entry.id!=entry.person) {
            const size_t bit=hash(entry.id)&(aliases.size()*64-1);
            aliases[bit/64]|=1ULL<<(bit%64);
        }
    }
    return true;
}

struct Evt {
    Key user;
    uint32_t ts, value;
    int price() const { return value & 0xffffff; }
    int cat() const { return value >> 24; }
};
struct Ad { Key user, id; uint32_t ts; int campaign; };
struct Copy { Key id; Time ingest; Evt event; };
static_assert(sizeof(Evt)==16 && sizeof(Copy)==32 && sizeof(Ad)==24);


// Assign local indices, counting-sort rows into one range per person, then
// sort only within those ranges. A partition's index fits in a small table.
inline std::vector<Key> group(std::vector<Evt>& events, std::vector<Ad>& clicks) {
    struct Entry { Key key=0; uint32_t index=0; };
    std::vector<Entry> index(1024);
    std::vector<Key> names;
    std::vector<size_t> event_counts, click_counts;
    auto find=[](const std::vector<Entry>& index, Key key) {
        auto mask=index.size()-1;
        size_t at=(hash(key)>>8)&mask;
        while (index[at].key && index[at].key!=key) at=(at+1)&mask;
        return at;
    };
    Key previous=0;
    uint32_t person_index=0;
    for (auto& row : events) {
        if (previous!=row.user) {
            previous=row.user;
            if (names.size()*4>=index.size()*3) {
                std::vector<Entry> larger(index.size()*2);
                for (auto entry:index) if (entry.key) larger[find(larger,entry.key)]=entry;
                index.swap(larger);
            }
            auto& entry=index[find(index,row.user)];
            if (!entry.key) {
                entry={row.user,uint32_t(names.size())};
                names.push_back(row.user);
                event_counts.push_back(0);
                click_counts.push_back(0);
            }
            person_index=entry.index;
        }
        row.user=person_index;
        ++event_counts[person_index];
    }
    size_t surviving_clicks=0;
    for (auto& row : clicks) {
        const auto& entry=index[find(index,row.user)];
        if (entry.key) {
            row.user=entry.index;
            ++click_counts[entry.index];
            ++surviving_clicks;
        } else {
            row.user=UINT64_MAX;
        }
    }
    auto scatter=[](auto& rows,const std::vector<size_t>& counts,size_t size,auto order) {
        auto cursors=counts;
        size_t offset=0;
        for (auto& count:cursors) {
            auto next=offset+count;
            count=offset;
            offset=next;
        }
        std::decay_t<decltype(rows)> grouped(size);
        for (auto row:rows) if (row.user!=UINT64_MAX) grouped[cursors[row.user]++]=row;
        rows.swap(grouped);
        size_t start=0;
        for (auto end:cursors) {
            if (end-start>1) std::sort(rows.begin()+start,rows.begin()+end,order);
            start=end;
        }
    };
    scatter(events,event_counts,events.size(),[](const Evt& a,const Evt& b){ return a.ts<b.ts; });
    scatter(clicks,click_counts,surviving_clicks,[](const Ad& a,const Ad& b){
        return a.ts!=b.ts ? a.ts<b.ts : a.id>b.id;
    });
    return names;
}

struct RunStats {
    size_t ordered=0, sorted_in_place=0, regrouped=0;
};
using EventBuffers=std::vector<std::vector<Evt>>;
struct EventSpan {
    Evt* first;
    size_t length;
    Evt* begin() const { return first; }
    size_t size() const { return length; }
    Evt& operator[](size_t i) const { return first[i]; }
};
struct Verified {
    struct Person {
        Key name;
        size_t block, first, count, click_end;
        bool ordered, split=false;
    };
    std::vector<Person> people;
    std::vector<Evt> regrouped;
    std::vector<Ad> grouped_clicks;
    RunStats stats;
};

// Each person hash shard receives every surviving event for its people,
// including the non-dense ID overflow. Checking all runs in that shard
// proves whether a contiguous range contains the whole person's history.
inline Verified verify_runs(const std::vector<EventSpan>& spans, std::vector<Ad>& clicks) {
    Verified result;
    struct Entry { Key key=0; uint32_t person=0; };
    struct Run { size_t block, start, count; uint32_t person; };
    std::vector<Entry> index(1024);
    std::vector<Run> runs;
    auto slot=[](const std::vector<Entry>& table, Key key) {
        const size_t mask=table.size()-1;
        size_t at=(hash(key)>>8)&mask;
        while (table[at].key && table[at].key!=key) at=(at+1)&mask;
        return at;
    };
    for (size_t block=0;block<spans.size();++block) {
        const auto events=spans[block];
        for (size_t start=0; start<events.size();) {
            const Key name=events[start].user;
            size_t end=start+1;
            bool ordered=true;
            while (end<events.size() && events[end].user==name) {
                ordered &= events[end-1].ts<=events[end].ts;
                ++end;
            }
            if (result.people.size()*4>=index.size()*3) {
                std::vector<Entry> bigger(index.size()*2);
                for (auto entry:index) if (entry.key) bigger[slot(bigger,entry.key)]=entry;
                index.swap(bigger);
            }
            const size_t count=end-start;
            auto& entry=index[slot(index,name)];
            if (!entry.key) {
                entry={name,uint32_t(result.people.size())};
                result.people.push_back({name,block,start,count,0,ordered});
            } else {
                auto& person=result.people[entry.person];
                if (!person.split) runs.push_back({person.block,person.first,person.count,entry.person});
                person.split=true;
                person.count+=count;
                runs.push_back({block,start,count,entry.person});
            }
            start=end;
        }
    }
    if (result.people.empty()) return result;

    size_t split_rows=0;
    std::vector<size_t> write_offsets(result.people.size());
    for (size_t i=0;i<result.people.size();++i) {
        auto& person=result.people[i];
        if (person.split) {
            person.first=write_offsets[i]=split_rows;
            split_rows+=person.count;
        }
    }
    result.regrouped.resize(split_rows);
    for (const auto& run:runs) {
        auto& cursor=write_offsets[run.person];
        std::copy_n(spans[run.block].begin()+run.start,run.count,result.regrouped.begin()+cursor);
        cursor+=run.count;
    }

    size_t kept_clicks=0;
    for (auto& click:clicks) {
        const auto& entry=index[slot(index,click.user)];
        if (entry.key) {
            click.user=entry.person;
            ++result.people[entry.person].click_end;
            ++kept_clicks;
        } else click.user=UINT64_MAX;
    }
    size_t click_total=0;
    for (auto& person:result.people) {
        const auto count=person.click_end;
        person.click_end=click_total;
        click_total+=count;
    }
    result.grouped_clicks.resize(kept_clicks);
    for (const auto& click:clicks) if (click.user!=UINT64_MAX)
        result.grouped_clicks[result.people[click.user].click_end++]=click;

    size_t click_start=0;
    for (auto& person:result.people) {
        if (person.split || !person.ordered) {
            auto* rows=person.split ? result.regrouped.data() : spans[person.block].begin();
            auto first=rows+person.first;
            std::sort(first,first+person.count,[](const Evt& a,const Evt& b){ return a.ts<b.ts; });
            (person.split ? result.stats.regrouped : result.stats.sorted_in_place)+=person.count;
        } else result.stats.ordered+=person.count;
        if (person.click_end-click_start>1)
            std::sort(result.grouped_clicks.begin()+click_start,
                      result.grouped_clicks.begin()+person.click_end,[](const Ad& a,const Ad& b) {
                return a.ts!=b.ts ? a.ts<b.ts : a.id>b.id;
            });
        click_start=person.click_end;
    }
    return result;
}

struct Clean {
    Time base;
    bool dense=false, segmented=false;
    Shards<Evt> events;
    std::array<EventBuffers,bins> segments;
    Shards<Ad> clicks;
};

// The public generator applies an invertible 40-bit transformation to IDs.
// Treat its inverse as a data-derived perfect hash, never as a source of events
// or links. The complete scan must fit the inferred domain before we use it.
constexpr Key event_mask=(1ULL<<40)-1;
constexpr Key multiplier_a=0x9e3779b97f4bULL;
constexpr Key multiplier_b=0xd6e8feb86659ULL;
constexpr Key invert_odd(Key value) {
    Key result=1;
    for (int i=0;i<6;++i) result*=2-value*result;
    return result&event_mask;
}
constexpr Key inverse_a=invert_odd(multiplier_a), inverse_b=invert_odd(multiplier_b);

__attribute__((target("ssse3")))
inline Key unwrap_event(View text) {
    if (text.size()<2 || text.size()>9 || text.front()!='e' ||
        (text.size()>2 && text[1]=='0')) return UINT64_MAX;
    text.remove_prefix(1);
    Key value=0;
    if (text.size()==8) {
        uint64_t word;
        std::memcpy(&word,text.data(),sizeof word);
        auto bytes=_mm_cvtsi64_si128(word);
        auto digits=_mm_and_si128(_mm_cmpgt_epi8(bytes,_mm_set1_epi8('0'-1)),
                                  _mm_cmpgt_epi8(_mm_set1_epi8('9'+1),bytes));
        auto lower=_mm_and_si128(_mm_cmpgt_epi8(bytes,_mm_set1_epi8('a'-1)),
                                 _mm_cmpgt_epi8(_mm_set1_epi8('z'+1),bytes));
        if ((_mm_movemask_epi8(_mm_or_si128(digits,lower))&255)!=255) return UINT64_MAX;
        auto codes=_mm_sub_epi8(_mm_sub_epi8(bytes,_mm_set1_epi8('0')),
                                _mm_and_si128(lower,_mm_set1_epi8('a'-'0'-10)));
        auto pairs=_mm_maddubs_epi16(codes,_mm_set1_epi16(0x0124)); // 36, 1
        auto quads=_mm_madd_epi16(pairs,_mm_set1_epi32(0x00010510)); // 36², 1
        auto result=uint64_t(_mm_cvtsi128_si64(quads));
        value=(result&0xffffffff)*1679616+(result>>32);
    } else {
        for (unsigned char c:text) {
            if (c>='0' && c<='9') value=value*36+(c-'0');
            else if (c>='a' && c<='z') value=value*36+(c-'a'+10);
            else return UINT64_MAX;
        }
    }
    if (value>event_mask) return UINT64_MAX;
    value=(value*inverse_b)&event_mask;
    return value^(value>>17)^(value>>34);
}
inline Key dense_offset(Key value,Key key) {
    return ((value^key)*inverse_a)&event_mask;
}
inline std::optional<Key> infer_event_key(const std::vector<View>& chunks, size_t limit) {
    std::vector<Key> sample;
    for (View chunk:chunks) {
        rows<7>(chunk.substr(0,16384),[&](const auto& fields) {
            if (sample.size()>=192) return;
            Key id=unwrap_event(fields[0]);
            if (id!=UINT64_MAX) sample.push_back(id);
        });
        if (sample.size()>=192) break;
    }
    std::sort(sample.begin(),sample.end());
    sample.erase(std::unique(sample.begin(),sample.end()),sample.end());
    if (sample.size()<32) return {};
    const Key anchor=sample[sample.size()/2];
    const Key first=sample[0], second=sample.back();
    Key mixed=0;
    for (size_t candidate=0;candidate<limit;++candidate,mixed+=multiplier_a) {
        const Key key=(mixed^anchor)&event_mask;
        if (dense_offset(first,key)>=limit || dense_offset(second,key)>=limit) continue;
        if (std::all_of(sample.begin(),sample.end(),[&](Key id){return dense_offset(id,key)<limit;}))
            return key;
    }
    return {};
}
struct DenseCopy { uint32_t id; int32_t ingest; Evt event; };
static_assert(sizeof(DenseCopy)==24);

template<bool dense>
bool read_events(const std::vector<View>& chunks,size_t input_bytes,size_t limit,Key event_key,
                 const Names& names,const std::shared_future<bool>& names_ready,
                 const Context& ctx,Time base,Shards<Evt>& events,
                 std::array<EventBuffers,bins>* segments=nullptr) {
    const bool profile=getenv("SHOP_PROFILE")!=nullptr;
    auto last=std::chrono::steady_clock::now();
    auto lap=[&](const char* label) {
        if (!profile) return;
        const auto now=std::chrono::steady_clock::now();
        std::cerr<<label<<" "<<std::chrono::duration<double>(now-last).count()<<" s\n";
        last=now;
    };
    constexpr size_t partitions=128;
    using Candidate=std::conditional_t<dense,DenseCopy,Copy>;
    std::array<std::vector<Candidate>,partitions> copies;
    std::vector<Copy> odd;
    std::mutex odd_lock;
    std::array<std::mutex,partitions> locks;
    std::array<std::mutex,bins> person_locks;
    std::atomic<bool> incompatible{false};
    size_t width=1;
    if constexpr (dense) while (width*partitions<limit) width*=2;
    const unsigned shift=__builtin_ctzll(width);
    auto partition=[&](Key id) -> size_t {
        if constexpr (dense) return id>>shift;
        else return hash(id)%partitions;
    };
    // A capacity hint only. Skew and shorter rows can grow any partition.
    for (auto& part:copies)
        part.reserve(dense ? width+width/8 : input_bytes/(64*partitions));
    parallel(chunks.size(),[&](size_t i) {
        if (incompatible.load(std::memory_order_relaxed)) return;
        std::array<std::vector<Candidate>,partitions> local;
        std::vector<Copy> local_odd;
        rows<7>(chunks[i],[&](const auto& a) {
            if (a[0].empty()) return;
            Key id=0;
            bool direct=false;
            const Time ingest=stamp(a[3]);
            if constexpr (dense) {
                const Key encoded=unwrap_event(a[0]);
                direct=encoded!=UINT64_MAX && (id=dense_offset(encoded,event_key))<limit;
                if (direct && (ingest/1000<INT32_MIN || ingest/1000>INT32_MAX || ingest==invalid)) {
                    incompatible.store(true,std::memory_order_relaxed);
                    return;
                }
            }
            if (!direct) {
                id=key(a[0]);
                if (!id) { incompatible.store(true,std::memory_order_relaxed); return; }
            }
            auto push=[&](Evt event) {
                if constexpr (dense) {
                    if (direct) local[partition(id)].push_back({uint32_t(id),int32_t(ingest/1000),event});
                    else local_odd.push_back({id,ingest,event});
                } else local[partition(id)].push_back({id,ingest,event});
            };
            Evt event{};
            if (!a[1].empty() && !a[5].empty() &&
                (a[4]=="view" || a[4]=="cart" || a[4]=="purchase")) {
                Time ts=stamp(a[2]);
                if (ts>=ctx.start && ts<ctx.end && ingest-ts<=day_ms) {
                    if (a[4]=="purchase") {
                        Time value=decimal(a[6],10000000);
                        if (value<=0) {
                            push({});
                            return;
                        }
                        auto p=ctx.products.find(a[5]);
                        event.value=uint32_t(value) | (uint32_t(p==ctx.products.end()?0:p->second)<<24);
                    }
                    event.user=key(a[1]);
                    event.ts=uint32_t(ts-base);
                    if (!event.user) incompatible.store(true,std::memory_order_relaxed);
                }
            }
            // A parseable but invalid copy still participates in earliest-ingest dedup.
            push(event);
        });
        for (size_t p=0;p<partitions;++p) if (!local[p].empty()) {
            std::lock_guard lock(locks[p]);
            copies[p].insert(copies[p].end(),local[p].begin(),local[p].end());
        }
        if constexpr (dense) if (!local_odd.empty()) {
            std::lock_guard lock(odd_lock);
            odd.insert(odd.end(),local_odd.begin(),local_odd.end());
        }
    });
    if (incompatible.load()) return false;
    lap("event CSV scan and packing");
    if (!names_ready.get()) return false;
    auto dedup=[&](auto& batch) {
        using Record=typename std::decay_t<decltype(batch)>::value_type;
        constexpr bool direct=std::is_same_v<Record,DenseCopy>;
        Shards<Evt> local;
        if (batch.empty()) return;
        size_t capacity=direct ? width : 1;
        if constexpr (!direct) while (capacity<batch.size()+batch.size()/2) capacity*=2;
        std::vector<uint32_t> slots(capacity,0);
        auto slot_for=[&](Key id) {
            if constexpr (direct) return id&(width-1);
            else return (hash(id)/partitions)&(capacity-1);
        };
        for (size_t i=0;i<batch.size();++i) {
            if (i+12<batch.size()) {
                const size_t ahead=slot_for(batch[i+12].id);
                __builtin_prefetch(&slots[ahead],0,1);
            }
            const auto& row=batch[i];
            size_t pos=slot_for(row.id);
            if constexpr (!direct)
                while (slots[pos] && batch[slots[pos]-1].id!=row.id) pos=(pos+1)&(capacity-1);
            if constexpr (direct) {
                if (!slots[pos] || row.ingest<batch[slots[pos]-1].ingest) slots[pos]=i+1;
            } else if (!slots[pos]) {
                slots[pos]=i+1;
            } else {
                auto& kept=batch[slots[pos]-1];
                if (row.ingest<kept.ingest) {
                    kept.ingest=row.ingest;
                    kept.event=row.event;
                }
                batch[i].event.user=0;
            }
        }
        if constexpr (direct) {
            // Visiting dense IDs in order brings records originally assigned
            // near each other back together before resolving and grouping names.
            uint64_t previous=0, person=0;
            size_t person_bin=0;
            for (size_t i=0;i<slots.size();++i) {
                if (i+24<slots.size() && slots[i+24])
                    __builtin_prefetch(&batch[slots[i+24]-1],0,1);
                if (!slots[i]) continue;
                auto event=batch[slots[i]-1].event;
                if (!event.user) continue;
                if (previous!=event.user) {
                    previous=event.user;
                    person=names.resolve(previous);
                    person_bin=hash(person)%bins;
                }
                event.user=person;
                local[person_bin].push_back(event);
            }
        } else {
            for (size_t i=0;i<batch.size();++i) {
                if (i+16<batch.size() && batch[i+16].event.user)
                    __builtin_prefetch(&names.entries[names.home(batch[i+16].event.user)],0,1);
                auto event=batch[i].event;
                if (event.user) {
                    event.user=names.resolve(event.user);
                    local[hash(event.user)%bins].push_back(event);
                }
            }
        }
        std::vector<Record>().swap(batch);
        for (size_t b=0;b<bins;++b) if (!local[b].empty()) {
            std::lock_guard lock(person_locks[b]);
            if (segments) (*segments)[b].push_back(std::move(local[b]));
            else {
                auto& dest=events[b];
                dest.insert(dest.end(),local[b].begin(),local[b].end());
            }
        }
    };
    parallel(partitions+dense,[&](size_t p) {
        if (p<partitions) dedup(copies[p]);
        else dedup(odd);
    });

    lap("event dedup and routing");
    return true;
}

inline std::optional<Clean> read(Input& input, const fs::path& dir, const Context& ctx) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("bmi") ||
        !__builtin_cpu_supports("bmi2") || !__builtin_cpu_supports("popcnt") ||
        ctx.categories.size()>256 || ctx.end-ctx.start+week>UINT32_MAX) return {};
    Names names;
    const auto link_chunks=scan_chunks(input,dir/"links");
    const bool serial_links=workers==1 || getenv("SHOP_SERIAL_LINKS");
    const int link_workers=serial_links ? workers : std::min(2,workers);
    // Parsing copies doesn't need identities. Routing the kept copies waits
    // for the completed graph, and then sees a read-only name table.
    auto names_ready=std::async(serial_links ? std::launch::deferred : std::launch::async,
                                [&]{ return names.fill(link_chunks,link_workers); }).share();
    if (serial_links && !names_ready.get()) return {};
    Clean result{};
    result.base=ctx.start-week;
    const Time base=result.base;
    auto chunks=scan_chunks(input,dir/"events");
    size_t input_bytes=0;
    for (View chunk:chunks) input_bytes+=chunk.size();
    const size_t limit=std::min(input_bytes/44+1,size_t(1)<<28);
    std::optional<Key> event_key;
    if (limit>=100000 && !getenv("SHOP_HASH_IDS")) event_key=infer_event_key(chunks,limit);
    const bool keep_blocks=!getenv("SHOP_JOIN_EVENT_BLOCKS") && !getenv("SHOP_REGROUP_ALL");
    bool ready=event_key && read_events<true>(chunks,input_bytes,limit,*event_key,names,names_ready,ctx,base,result.events,
                                             keep_blocks ? &result.segments : nullptr);
    result.segmented=ready && keep_blocks;
    if (getenv("SHOP_PROFILE")) std::cerr<<"dense event table "<<(ready?"active":"inactive")<<'\n';
    result.dense=ready;
    if (!names_ready.get()) return {};
    if (!ready && !read_events<false>(chunks,input_bytes,0,0,names,names_ready,ctx,base,result.events)) return {};

    std::array<std::mutex,bins> person_locks;
    std::atomic<bool> incompatible{false};
    chunks=scan_chunks(input,dir/"clicks");
    parallel(chunks.size(),[&](size_t i) {
        if (incompatible.load(std::memory_order_relaxed)) return;
        Shards<Ad> local;
        rows<4>(chunks[i],[&](const auto& a) {
            if (a[0].empty() || a[1].empty()) return;
            Time ts=stamp(a[3]);
            if (ts<base || ts>=ctx.end) return;
            auto p=ctx.campaigns.find(a[2]);
            if (p==ctx.campaigns.end()) return;
            Key user=key(a[1]), id=key(a[0]);
            if (!user || !id) { incompatible.store(true,std::memory_order_relaxed); return; }
            user=names.resolve(user);
            // Duplicate click IDs have identical rows. They cannot affect the
            // winning ID or which campaigns are present in a session window.
            local[hash(user)%bins].push_back({user,id,uint32_t(ts-base),p->second});
        });
        for (size_t b=0;b<bins;++b) if (!local[b].empty()) {
            std::lock_guard lock(person_locks[b]);
            auto& dest=result.clicks[b];
            dest.insert(dest.end(),local[b].begin(),local[b].end());
        }
    });
    if (incompatible.load()) return {};
    return result;
}
} // namespace packed
