# Shopping Sessions

Exact Shopping Sessions pipeline for the Builders Buildathon. The image runs one statically linked C++ binary as uid 1000. The binary reads `DATA_DIR` (default `/data`), writes both Parquet tables under `OUTPUT_DIR` (default `/output`), and uses up to `NCPU` workers (default 8).

On supported x86-64 CPUs, an AVX2 scanner parses CSV into 32-byte event candidates. Short ASCII IDs have a reversible encoding that preserves byte order. Union-find applies the supplied links, deduplication retains the earliest ingest even if its event is invalid, and the remaining records are grouped by shopper before sorting by time. A native Parquet writer emits the two tables directly. Inputs outside the compact representation use the general engine. `SHOP_GENERIC=1` selects it explicitly.

```sh
make native
make check-native
make fixtures-native             # requires Polars and the extracted starter fixtures
scripts/native_run.sh dev out
make check-dev ANSWER=dev-answer
```

`make fixtures` builds the image and invokes the contest's Docker wrapper. The rules, checker, and fixtures are from the organizers. Generated datasets and answers are kept outside Git.

Performance design references include [Express Lane](https://github.com/jason-scheffel/express-lane), [jklein07's packed-record pipeline](https://github.com/jklein07/shopping-sessions), and [Fused Parallel CSV Processing](https://wwwbayer.informatik.tu-muenchen.de/~ellmann/papers/csveee.pdf). No competitor source is included. The writer follows [Apache Parquet's format definition](https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift) and [encoding specification](https://parquet.apache.org/docs/file-format/data-pages/encodings/). See [LICENSE](LICENSE).
