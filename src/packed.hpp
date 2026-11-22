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
    bool fill(const std::vector<View>& chunks);
    size_t home(Key id) const { return hash(id)&(entries.size()-1); }
    Key resolve(Key id) const {
        size_t i=home(id);
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

template<size_t N, class F>
__attribute__((target("avx2,popcnt,bmi,bmi2")))
void rows(View chunk, F consume) {
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
        unsigned quotes=last ? __builtin_popcountll(lo.quote)+__builtin_popcountll(hi.quote&keep)
                             : __builtin_popcountll(lo.quote&keep);

        if (__builtin_popcountll(commas[0])+__builtin_popcountll(commas[1])==N-1) {
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

inline bool Names::fill(const std::vector<View>& chunks) {
    std::vector<std::vector<std::pair<Key,Key>>> shards(chunks.size());
    std::atomic<bool> incompatible{false};
    parallel(chunks.size(),[&](size_t chunk) {
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
    for (auto& entry:entries) if (entry.id) entry.person=nodes[root(entry.person)].name;
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
    for (auto& row : events) {
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
        row.user=entry.index;
        ++event_counts[entry.index];
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

struct Clean {
    Time base;
    Shards<Evt> events;
    Shards<Ad> clicks;
};

inline std::optional<Clean> read(Input& input, const fs::path& dir, const Context& ctx) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("bmi") ||
        !__builtin_cpu_supports("bmi2") || !__builtin_cpu_supports("popcnt") ||
        ctx.categories.size()>256 || ctx.end-ctx.start+week>UINT32_MAX) return {};
    Names names;
    if (!names.fill(scan_chunks(input,dir/"links"))) return {};
    Clean result{};
    result.base=ctx.start-week;
    const Time base=result.base;

    constexpr size_t partitions=64;
    std::array<std::vector<Copy>,partitions> copies;
    std::array<std::mutex,partitions> locks;
    std::array<std::mutex,bins> person_locks;
    std::atomic<bool> incompatible{false};
    auto chunks=scan_chunks(input,dir/"events");
    size_t input_bytes=0;
    for (View chunk:chunks) input_bytes+=chunk.size();
    // A capacity hint only. Skew and shorter rows can grow any partition.
    for (auto& part:copies) part.reserve(input_bytes/(64*partitions));
    parallel(chunks.size(),[&](size_t i) {
        if (incompatible.load(std::memory_order_relaxed)) return;
        std::array<std::vector<Copy>,partitions> local;
        rows<7>(chunks[i],[&](const auto& a) {
            if (a[0].empty()) return;
            Key id=key(a[0]);
            if (!id) { incompatible.store(true,std::memory_order_relaxed); return; }
            Time ingest=stamp(a[3]);
            Evt event{};
            if (!a[1].empty() && !a[5].empty() &&
                (a[4]=="view" || a[4]=="cart" || a[4]=="purchase")) {
                Time ts=stamp(a[2]);
                if (ts>=ctx.start && ts<ctx.end && ingest-ts<=day_ms) {
                    if (a[4]=="purchase") {
                        Time value=decimal(a[6],10000000);
                        if (value<=0) {
                            local[hash(id)%partitions].push_back({id,ingest,{}});
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
            local[hash(id)%partitions].push_back({id,ingest,event});
        });
        for (size_t p=0;p<partitions;++p) if (!local[p].empty()) {
            std::lock_guard lock(locks[p]);
            copies[p].insert(copies[p].end(),local[p].begin(),local[p].end());
        }
    });
    if (incompatible.load()) return {};
    parallel(partitions,[&](size_t p) {
        auto& batch=copies[p];
        Shards<Evt> local;
        size_t capacity=1;
        while (capacity<batch.size()+batch.size()/2) capacity*=2;
        std::vector<uint32_t> slots(capacity,0);
        for (size_t i=0;i<batch.size();++i) {
            if (i+12<batch.size()) {
                const size_t ahead=(hash(batch[i+12].id)/partitions)&(capacity-1);
                __builtin_prefetch(&slots[ahead],0,1);
            }
            const auto& row=batch[i];
            size_t pos=(hash(row.id)/partitions)&(capacity-1);
            while (slots[pos] && batch[slots[pos]-1].id!=row.id) pos=(pos+1)&(capacity-1);
            if (!slots[pos]) {
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
        for (size_t i=0;i<batch.size();++i) {
            if (i+16<batch.size() && batch[i+16].event.user)
                __builtin_prefetch(&names.entries[names.home(batch[i+16].event.user)],0,1);
            auto row=batch[i];
            if (!row.event.user) continue;

            row.event.user=names.resolve(row.event.user);
            local[hash(row.event.user)%bins].push_back(row.event);
        }
        std::vector<Copy>().swap(batch);
        for (size_t b=0;b<bins;++b) if (!local[b].empty()) {
            std::lock_guard lock(person_locks[b]);
            auto& dest=result.events[b];
            dest.insert(dest.end(),local[b].begin(),local[b].end());
        }
    });

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
