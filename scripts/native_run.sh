#!/bin/sh
set -eu
root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
DATA_DIR="$1" OUTPUT_DIR="$2" exec "$root/build-local/shop"
