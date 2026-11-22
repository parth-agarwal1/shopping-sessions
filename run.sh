#!/bin/sh
set -eu
export HOME="${SCRATCH_DIR:=/scratch}" TMPDIR="$SCRATCH_DIR" POLARS_MAX_THREADS="${NCPU:-8}"
stage=$(mktemp -d "$SCRATCH_DIR/shop.XXXXXXXX")
trap 'rm -rf "$stage"' EXIT
/app/shop "${DATA_DIR:-/data}" "$stage"
python3 /app/write.py "$stage" "${OUTPUT_DIR:-/output}"
