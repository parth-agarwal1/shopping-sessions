// Included after the common types and CSV parsing helpers in shop.cpp.
#pragma once
#include <immintrin.h>
#include <optional>

namespace packed {
using Key = uint64_t;
constexpr uint64_t byte_lanes = 0x7f7f7f7f7f7f7f7fULL;
constexpr Key view_tag = 1ULL << 63;
constexpr Key address_mask = (1ULL << 48) - 1;

inline Key mapped_key(View id) {
    auto address=reinterpret_cast<uintptr_t>(id.data());
    if ((address&~address_mask) || id.size()>32767) return 0;
    // Keep the length and mapped address in one word. The caller falls back
    // to ordinary string views if either exceeds its bit field.
    return view_tag | (Key(id.size())<<48) | address;
}
inline View mapped_name(Key id) {
    return {reinterpret_cast<const char*>(id&address_mask),size_t((id&~view_tag)>>48)};
}

// Nine nonzero ASCII bytes fit in 63 bits, with zero padding that preserves
// byte order. Other IDs retain their exact bytes in the mapped input.
__attribute__((target("bmi2")))
inline Key key(View id) {
    if (id.empty()) return 0;
    if (id.size()>9) return mapped_key(id);
    if (id.size() == 9) {
        uint64_t low;
        std::memcpy(&low, id.data(), 8);
        auto last = uint8_t(id[8]);
        if (last == 0 || last > 127 || (low & ~byte_lanes) ||
            ((low - 0x0101010101010101ULL) & ~low & ~byte_lanes)) return mapped_key(id);
        return (_pext_u64(__builtin_bswap64(low), byte_lanes) << 7) | last;
    }
    Key value = 0;
    for (uint8_t b : id) {
        if (!b || b > 127) return mapped_key(id);
        value = (value << 7) | b;
    }
    return value << (7 * (9 - id.size()));
}
inline View name(Key key, std::array<char, 9>& text) {
    if (key&view_tag) return mapped_name(key);
    size_t n = 0;
    for (unsigned shift = 63; shift != 0; shift -= 7) {
        char next = char((key >> (shift - 7)) & 127);
        if (!next) break;
        text[n++] = next;
    }
    return {text.data(), n};
}
inline bool same(Key a,Key b) {
    return a==b || ((a&b&view_tag) && mapped_name(a)==mapped_name(b));
}
inline bool less(Key a,Key b) {
    if (!((a|b)&view_tag)) return a<b;
    std::array<char,9> left,right;
    return name(a,left)<name(b,right);
}
inline uint64_t hash(Key key) {
    if (key&view_tag) return std::hash<View>{}(mapped_name(key));
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
    bool might_alias(uint64_t h) const {
        if (aliases.empty()) return true;
        const size_t bit=h&(aliases.size()*64-1);
        return aliases[bit/64]&(1ULL<<(bit%64));
    }
    Key resolve_alias(Key id,uint64_t h) const {
        size_t i=h&(entries.size()-1);
        while (entries[i].id && !same(entries[i].id,id)) i=(i+1)&(entries.size()-1);
        return entries[i].id ? entries[i].person : id;
    }
    Key resolve(Key id,uint64_t h) const { return might_alias(h) ? resolve_alias(id,h) : id; }
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
    struct Edge { Key left,right; uint64_t left_hash,right_hash; };
    std::vector<std::vector<Edge>> shards(chunks.size());
    std::atomic<bool> incompatible{false};
    parallel_workers(chunks.size(),limit,[&](size_t chunk,size_t) {
        auto& edges=shards[chunk];
        rows<2>(chunks[chunk],[&](const auto& f) {
            if (f[0].empty() || f[1].empty()) return;
            Key a=key(f[0]), b=key(f[1]);
            if (a && b) edges.push_back({a,b,hash(a),hash(b)});
            else incompatible.store(true,std::memory_order_relaxed);
        });
    });
    if (incompatible.load()) return false;

    struct Node { uint32_t parent,size; Key name; };
    size_t edge_count=0;
    for (const auto& shard:shards) edge_count+=shard.size();
    // Reserve from the edge count and grow if more IDs are discovered.
    const size_t estimate=edge_count+edge_count/2;
    std::vector<Node> nodes;
    nodes.reserve(estimate);
    size_t capacity=16;
    while (capacity/4*3<estimate) capacity*=2;
    entries.resize(capacity);
    auto put=[&](Key id,uint64_t h) -> uint32_t {
        if (nodes.size()*4>=entries.size()*3) {
            auto old=std::move(entries);
            entries.resize(2*old.size());
            for (auto entry:old) if (entry.id) {
                size_t i=home(entry.id);
                while (entries[i].id) i=(i+1)&(entries.size()-1);
                entries[i]=entry;
            }
        }
        size_t i=h&(entries.size()-1);
        while (entries[i].id && !same(entries[i].id,id)) i=(i+1)&(entries.size()-1);
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
    constexpr size_t ahead=12;
    for (const auto& edges:shards) for (size_t i=0;i<edges.size();++i) {
        if (i+ahead<edges.size()) {
            const auto& next=edges[i+ahead];
            const size_t mask=entries.size()-1;
            __builtin_prefetch(&entries[next.left_hash&mask],1,1);
            __builtin_prefetch(&entries[next.right_hash&mask],1,1);
        }
        const auto& edge=edges[i];
        uint32_t ar=root(put(edge.left,edge.left_hash)), br=root(put(edge.right,edge.right_hash));
        if (ar==br) continue;
        if (nodes[ar].size<nodes[br].size) std::swap(ar,br);
        nodes[br].parent=ar;
        nodes[ar].size+=nodes[br].size;
        if (less(nodes[br].name,nodes[ar].name)) nodes[ar].name=nodes[br].name;
    }
    if (!getenv("SHOP_NO_ALIAS_FILTER") && entries.size()>=1024)
        aliases.resize(std::min(entries.size()/8,size_t(262144)));
    for (auto& entry:entries) if (entry.id) {
        entry.person=nodes[root(entry.person)].name;
        if (!aliases.empty() && !same(entry.id,entry.person)) {
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
// Ingest seconds are relative to the window base.
struct Copy { Key id; uint32_t hash; int32_t ingest; Evt event; };
static_assert(sizeof(Evt)==16 && sizeof(Copy)==32 && sizeof(Ad)==24);

inline void sort_events(Evt* first,Evt* end,std::unique_ptr<Evt[]>& scratch,size_t& capacity) {
    const size_t count=end-first;
    if (count<1024) {
        std::sort(first,end,[](const Evt& a,const Evt& b){ return a.ts<b.ts; });
        return;
    }
    if (count>capacity) {
        scratch.reset(new Evt[count]);
        capacity=count;
    }
    constexpr unsigned bits=11, buckets=1<<bits, mask=buckets-1;
    std::array<std::array<uint32_t,buckets>,3> offsets{};
    for (auto* at=first;at!=end;++at) {
        ++offsets[0][at->ts&mask];
        ++offsets[1][(at->ts>>bits)&mask];
        ++offsets[2][at->ts>>(2*bits)];
    }
    Evt* source=first;
    Evt* target=scratch.get();
    for (unsigned pass=0;pass<offsets.size();++pass) {
        auto& next=offsets[pass];
        if (next[(source[0].ts>>(pass*bits))&mask]==count) continue;
        uint32_t sum=0;
        for (auto& value:next) {
            const auto size=value;
            value=sum;
            sum+=size;
        }
        for (size_t i=0;i<count;++i) {
            const auto row=source[i];
            target[next[(row.ts>>(pass*bits))&mask]++]=row;
        }
        std::swap(source,target);
    }
    if (source!=first) std::copy_n(source,count,first);
}


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
        while (index[at].key && !same(index[at].key,key)) at=(at+1)&mask;
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
    auto scatter=[](auto& rows,const std::vector<size_t>& counts,size_t size,auto sort_range) {
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
            if (end-start>1) sort_range(rows.data()+start,rows.data()+end);
            start=end;
        }
    };
    std::unique_ptr<Evt[]> scratch;
    size_t capacity=0;
    scatter(events,event_counts,events.size(),[&](Evt* first,Evt* end){
        sort_events(first,end,scratch,capacity);
    });
    scatter(clicks,click_counts,surviving_clicks,[](Ad* first,Ad* end){
        std::sort(first,end,[](const Ad& a,const Ad& b){
            return a.ts!=b.ts ? a.ts<b.ts : less(b.id,a.id);
        });
    });
    return names;
}

struct Clean {
    Time base;
    Shards<Evt> events;
    Shards<Ad> clicks;
};

struct ClickWork {
    const Names& names;
    const Context& ctx;
    const Time base;
    Shards<Ad>& output;
    const std::vector<View> chunks;
    std::array<std::mutex,bins> locks;
    std::atomic<bool> incompatible{false};
    std::vector<Shards<Ad>> routing;

    ClickWork(const Names& graph,const Context& data,Time first,Shards<Ad>& clicks,std::vector<View> input)
        : names(graph),ctx(data),base(first),output(clicks),chunks(std::move(input)),routing(workers) {}

    void parse(size_t i,size_t worker) {
        if (incompatible.load(std::memory_order_relaxed)) return;
        auto& local=routing[worker];
        rows<4>(chunks[i],[&](const auto& a) {
            if (a[0].empty() || a[1].empty()) return;
            Time ts=stamp(a[3]);
            if (ts<base || ts>=ctx.end) return;
            auto p=ctx.campaigns.find(a[2]);
            if (p==ctx.campaigns.end()) return;
            Key user=key(a[1]), id=key(a[0]);
            if (!user || !id) { incompatible.store(true,std::memory_order_relaxed); return; }
            auto h=hash(user);
            auto person=names.resolve(user,h);
            if (person!=user) h=hash(person);
            // Identical duplicate clicks leave both attribution and assists unchanged.
            local[h%bins].push_back({person,id,uint32_t(ts-base),p->second});
        });
        for (size_t b=0;b<bins;++b) if (!local[b].empty()) {
            std::lock_guard lock(locks[b]);
            auto& dest=output[b];
            dest.insert(dest.end(),local[b].begin(),local[b].end());
            local[b].clear();
        }
    }
};

inline bool read_records(const std::vector<View>& chunks,size_t input_bytes,
                         const Names& names,const std::shared_future<bool>& names_ready,
                         const Context& ctx,Time base,Shards<Evt>& events,ClickWork& clicks) {
    const bool profile=getenv("SHOP_PROFILE")!=nullptr;
    auto last=std::chrono::steady_clock::now();
    auto lap=[&](const char* label) {
        if (!profile) return;
        const auto now=std::chrono::steady_clock::now();
        std::cerr<<label<<" "<<std::chrono::duration<double>(now-last).count()<<" s\n";
        last=now;
    };
    constexpr size_t partitions=512;
    std::array<std::vector<Copy>,partitions> copies;
    std::array<std::mutex,partitions> locks;
    std::array<std::mutex,bins> person_locks;
    std::atomic<bool> incompatible{false};
    // A capacity hint only. Skew and shorter rows can grow any partition.
    for (auto& part:copies) part.reserve(input_bytes/(72*partitions));
    std::vector<std::array<std::vector<Copy>,partitions>> staging(workers);
    parallel_workers(chunks.size(),workers,[&](size_t i,size_t worker) {
        if (incompatible.load(std::memory_order_relaxed)) return;
        auto& local=staging[worker];
        rows<7>(chunks[i],[&](const auto& a) {
            if (a[0].empty()) return;
            Key id=key(a[0]);
            if (!id) {
                incompatible.store(true,std::memory_order_relaxed);
                return;
            }
            const Time ingest=stamp(a[3]);
            const Time seconds=ingest/1000-base/1000;
            if (ingest==invalid || ingest%1000 || seconds<INT32_MIN || seconds>INT32_MAX) {
                incompatible.store(true,std::memory_order_relaxed);
                return;
            }
            const auto h=hash(id);
            auto& partition=local[h%partitions];
            const auto tag=uint32_t(h/partitions);
            Evt event{};
            if (!a[1].empty() && !a[5].empty() &&
                (a[4]=="view" || a[4]=="cart" || a[4]=="purchase")) {
                Time ts=stamp(a[2]);
                if (ts>=ctx.start && ts<ctx.end && ingest-ts<=day_ms) {
                    if (a[4]=="purchase") {
                        Time value=decimal(a[6],10000000);
                        if (value<=0) {
                            partition.push_back({id,tag,int32_t(seconds),{}});
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
            partition.push_back({id,tag,int32_t(seconds),event});
        });
        for (size_t p=0;p<partitions;++p) if (!local[p].empty()) {
            std::lock_guard lock(locks[p]);
            copies[p].insert(copies[p].end(),local[p].begin(),local[p].end());
            local[p].clear();
        }
    });
    if (incompatible.load()) return false;
    lap("event CSV scan and packing");
    if (!names_ready.get()) return false;
    std::vector<Shards<Evt>> routing(workers);
    auto dedup=[&](size_t partition,size_t worker) {
        auto& batch=copies[partition];
        if (batch.empty()) return;
        if (batch.size()>UINT32_MAX) {
            incompatible.store(true,std::memory_order_relaxed);
            return;
        }
        auto& local=routing[worker];
        size_t capacity=1;
        while (capacity<batch.size()+batch.size()/2) capacity*=2;
        // Reserve just enough bits for a one-based row index. The remaining
        // bits hold a hash tag that filters probes before the exact ID check.
        std::vector<uint32_t> slots(capacity,0);
        const uint32_t index_mask=UINT32_MAX>>__builtin_clz(uint32_t(batch.size()));
        const uint32_t tag_mask=~index_mask;
        const size_t mask=capacity-1;
        for (size_t i=0;i<batch.size();++i) {
            if (i+12<batch.size())
                __builtin_prefetch(&slots[batch[i+12].hash&mask],0,1);
            const auto& row=batch[i];
            size_t pos=row.hash&mask;
            auto slot=slots[pos];
            while (slot && (((slot^row.hash)&tag_mask) ||
                   !same(batch[(slot&index_mask)-1].id,row.id))) {
                pos=(pos+1)&mask;
                slot=slots[pos];
            }
            if (!slot) {
                slots[pos]=(row.hash&tag_mask)|uint32_t(i+1);
            } else {
                auto& kept=batch[(slot&index_mask)-1];
                if (row.ingest<kept.ingest) {
                    kept.ingest=row.ingest;
                    kept.event=row.event;
                }
                batch[i].event.user=0;
            }
        }
        // Resolve shopper IDs in batches so the name table loads can
        // overlap. Long ID bytes for the next batch are also prefetched.
        constexpr size_t block_size=32;
        for (size_t offset=0;offset<batch.size();offset+=block_size) {
            const auto* rows=batch.data()+offset;
            const auto count=std::min(block_size,batch.size()-offset);
            std::array<uint64_t,block_size> hashes;
            std::array<bool,block_size> aliases;
            const size_t next=std::min(block_size,batch.size()-offset-count);
            for (size_t i=0;i<next;++i) {
                const auto id=rows[count+i].event.user;
                if (id&view_tag) __builtin_prefetch(mapped_name(id).data(),0,1);
            }
            for (size_t i=0;i<count;++i) {
                const auto id=rows[i].event.user;
                if (!id) continue;
                const auto h=hash(id);
                hashes[i]=h;
                const bool maybe=names.might_alias(h);
                aliases[i]=maybe;
                if (maybe) __builtin_prefetch(&names.entries[h&(names.entries.size()-1)],0,1);
            }
            for (size_t i=0;i<count;++i) {
                auto row=rows[i].event;
                if (!row.user) continue;
                auto h=hashes[i];
                if (aliases[i]) {
                    const auto person=names.resolve_alias(row.user,h);
                    if (person!=row.user) h=hash(person);
                    row.user=person;
                }
                local[h%bins].push_back(row);
            }
        }
        std::vector<Copy>().swap(batch);
        for (size_t b=0;b<bins;++b) if (!local[b].empty()) {
            std::lock_guard lock(person_locks[b]);
            auto& dest=events[b];
            dest.insert(dest.end(),local[b].begin(),local[b].end());
            local[b].clear();
        }
    };
    struct Job { size_t index; bool click; };
    std::vector<Job> jobs;
    jobs.reserve(partitions+clicks.chunks.size());
    size_t partition=0, click_chunk=0;
    // Both jobs read the completed graph and write to separate tables.
    while (partition<partitions || click_chunk<clicks.chunks.size()) {
        for (int i=0;i<2 && partition<partitions;++i,++partition) {
            if (!copies[partition].empty()) jobs.push_back({partition,false});
        }
        if (click_chunk<clicks.chunks.size()) jobs.push_back({click_chunk++,true});
    }
    parallel_workers(jobs.size(),workers,[&](size_t i,size_t worker) {
        const auto job=jobs[i];
        if (job.click) clicks.parse(job.index,worker);
        else dedup(job.index,worker);
    });
    lap("event dedup and click routing");
    return !incompatible.load() && !clicks.incompatible.load();
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
    ClickWork clicks(names,ctx,base,result.clicks,scan_chunks(input,dir/"clicks"));
    if (!read_records(chunks,input_bytes,names,names_ready,ctx,base,result.events,clicks)) return {};
    return result;
}
} // namespace packed
