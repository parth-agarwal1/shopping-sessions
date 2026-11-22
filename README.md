# Shopping Sessions

Exact Shopping Sessions pipeline for the Builders Buildathon. The image runs one statically linked C++ binary as uid 1000. The binary reads `DATA_DIR` (default `/data`), writes both Parquet tables under `OUTPUT_DIR` (default `/output`), and uses `NCPU` foreground workers (default 8). Identity links can be built on a small background team while event rows are parsed.

On supported x86-64 CPUs, an AVX2 scanner parses CSV into 32-byte event candidates. Short ASCII IDs have a reversible encoding that preserves byte order. Union-find applies the supplied links, deduplication retains the earliest ingest even if its event is invalid, and the remaining records are grouped by shopper before sorting by time. A native Parquet writer emits the two tables directly. Inputs outside the compact representation use the general engine. `SHOP_GENERIC=1` selects it explicitly.

When compatible event IDs are detected, the engine finds an inverse mapping from the public generator's ID encoding using a sample of the mounted input. The mapping is bijective. Every row is still parsed and deduplicated with the same ingest-time comparison, and IDs outside the dense range go through exact key lookup. If there is no suitable inverse mapping, the packed hash engine handles the entire input. Set `SHOP_HASH_IDS=1` to force that path for comparisons.

After deduplication, the engine checks canonical shoppers within each shopper partition. A complete run in timestamp order is sessionized in place. Unordered single runs are sorted in place, and only shoppers with multiple runs are copied into a new grouping buffer. Set `SHOP_REGROUP_ALL=1` to compare against grouping every event. `SHOP_PROFILE=1` reports coverage and stage timings.

The CSV scanner validates whole-field quotes from the row's separator and quote masks before extracting fields. Rows with an empty first field, long lines and chunk tails use the general decoder as applicable. `SHOP_LEGACY_CSV=1` switches off quote-mask pairing for A/B checks.

The dense path hands off whole survivor buffers to the shopper stage. Workers consume verified runs directly, reuse their campaign counters and Parquet buffers, and start the largest shopper shards first. Campaign totals use a bounded array when the day and campaign range fits, with a sparse map for larger ranges. Use a fresh output directory for each run.

```sh
make native
make check-native
make fixtures-native             # requires Polars and the extracted starter fixtures
scripts/native_run.sh dev out
make check-dev ANSWER=dev-answer
```

`make fixtures` builds the image and invokes the contest's Docker wrapper. The rules, checker, and fixtures are from the organizers. Generated datasets and answers are kept outside Git.

Performance design references include [Express Lane](https://github.com/jason-scheffel/express-lane), [jklein07's packed-record pipeline](https://github.com/jklein07/shopping-sessions), and [Fused Parallel CSV Processing](https://wwwbayer.informatik.tu-muenchen.de/~ellmann/papers/csveee.pdf). No competitor source is included. The writer follows [Apache Parquet's format definition](https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift) and [encoding specification](https://parquet.apache.org/docs/file-format/data-pages/encodings/). See [LICENSE](LICENSE).
