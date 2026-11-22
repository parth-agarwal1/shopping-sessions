# Shopping Sessions handoff

Read rules.md first. It is authoritative. Use the Polars starter's independent checker. The runtime uses native C++ for parsing, union-find, deduplication, sessions and attribution, then Polars for typed Parquet output. C++ permits local testing without a Rust toolchain. Rust 1.85+ is required for the official generator only.

Preserve LF-only records, byte-exact IDs, malformed-row rejection before global earliest-ingest deduplication, link-only minimum IDs, strict session gaps and inclusive attribution boundaries. Keep the entire session on its UTC start day. Count assists once per distinct campaign per session. Keep all counters in 64 bits.

Compile locally with make native. For the scorer image run make fixtures. Build the official generator with cargo build --release --manifest-path generator/Cargo.toml. Run PATH="$PWD/generator/target/release:$PATH" make dev, then make run-dev and make check-dev ANSWER=dev-answer. The fixture runner and dev checker require Polars on the host. Keep local challenge assets outside version control; copy the official starter-polars/fixtures into this workspace when reproducing tests on a fresh checkout.

Time fresh containers from startup to exit. Use 8 pinned logical CPUs, 24 GB including filesystem cache, no swap, read-only root and input, and uid 1000. Generate 20M/5M and 60M/15M event/click datasets separately with recorded seeds. Record commit, seed, host, cache condition, elapsed time, memory.peak and scratch plus output size. Target 20M below 60 seconds and all three 60M runs below 180 seconds. Report missing measurements explicitly. Compare uncompressed and Snappy output. Optimize only after a profile, change one variable at a time, and re-run exact comparisons. Keep the last verified commit. Never publish datasets, benchmark artifacts or hidden submissions as part of a routine edit.

## Current verification

Run `make check-native` for the dependency-free checks. All 36 official fixtures matched both native tables before Parquet conversion. A separate brute-force reference matched 20 shuffled seeds with one and eight workers. After each algorithm change rerun these checks, then check the actual image and Parquet schemas with the official runner.

Local native-only smoke benchmark on an i5-13450HX, CPUs 0–7, fresh processes with cached input: 20M events + 5M clicks took 3.88 s after canonicalizing person string addresses before sorting (5.57 s before). Three 60M + 15M runs took 11.84, 11.92 and 11.66 s, with peak process RSS up to 7.74 GiB. The simplified input used one event shard, one click shard, one product/campaign, no links, numeric event times and a one-day distribution. These runs exclude container start and Parquet encoding and cannot predict official times. Counts and revenue reconciled on the 60M input.

Docker, host Polars, Cargo and the generated official dev input were unavailable here. Official image fixtures, the 1M dev comparison, compression comparison, cold-container scale benchmarks and a hidden score are pending. The workspace has no configured GitHub remote. Keep the first practice attempt for a commit that passes the official checks.
