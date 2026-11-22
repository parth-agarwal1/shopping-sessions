# Shopping Sessions

Exact Shopping Sessions pipeline for the Builders Buildathon. The image runs one statically linked C++ binary as uid 1000. The binary reads `DATA_DIR` (default `/data`), writes both Parquet tables under `OUTPUT_DIR` (default `/output`), and uses `NCPU` foreground workers (default 8). Identity links can be built on a small background team while event rows are parsed.

On supported x86-64 CPUs, an AVX2 scanner parses CSV into 32-byte event candidates. IDs are opaque variable-width strings. Short ASCII IDs are stored byte-for-byte in a compact encoding, and other IDs are represented with views into the memory-mapped files. Hash probes verify equality against the exact bytes. Union-find chooses the byte-wise smallest linked ID, and event deduplication retains the earliest ingest even when its event is invalid. Inputs outside the compact representation use the general engine. `SHOP_GENERIC=1` selects it explicitly.

The CSV scanner validates whole-field quotes from the row's separator and quote masks before extracting fields. Rows with an empty first field, long lines and chunk tails use the general decoder as applicable. `SHOP_LEGACY_CSV=1` switches off quote-mask pairing for A/B checks.

Event deduplication and click parsing share workers once the identity graph is ready. Workers then group surviving events by shopper and sort by timestamp. Large event groups use a radix sort. Workers start the largest shopper shards first and reuse their campaign counters and Parquet buffers. Campaign totals use a bounded array when the day and campaign range fits, with a sparse map for larger ranges. `SHOP_PROFILE=1` reports stage timings. Use a fresh output directory for each run.

```sh
make native
make check-native
make fixtures-native             # requires Polars and the extracted starter fixtures
scripts/native_run.sh dev out
make check-dev ANSWER=dev-answer
```

`make fixtures` builds the image and invokes the contest's Docker wrapper. Use the generator and golden answers for scorer version `shopping-sessions-5`. The rules, checker, and fixtures are from the organizers. Generated datasets and answers are kept outside Git.

Performance design references include [Express Lane](https://github.com/jason-scheffel/express-lane), [jklein07's packed-record pipeline](https://github.com/jklein07/shopping-sessions), and [Fused Parallel CSV Processing](https://wwwbayer.informatik.tu-muenchen.de/~ellmann/papers/csveee.pdf). No competitor source is included. The writer follows [Apache Parquet's format definition](https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift) and [encoding specification](https://parquet.apache.org/docs/file-format/data-pages/encodings/). See [LICENSE](LICENSE).
