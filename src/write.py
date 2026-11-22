"""Convert bounded native CSV batches into schema-stable Parquet parts."""
import os
import sys
from pathlib import Path

import polars as pl

schemas = {
    "user_daily": ("user_id day sessions events revenue_cents top_category", {"user_id", "top_category"}),
    "campaign_daily": ("campaign_id channel day sessions purchasing_sessions revenue_cents assisted_sessions", {"campaign_id", "channel"}),
}
stage, output = map(Path, sys.argv[1:])
for table, (columns, strings) in schemas.items():
    dest = output / table
    dest.mkdir(parents=True, exist_ok=True)
    schema = {k: pl.String if k in strings else pl.Int32 if k == "day" else pl.Int64
              for k in columns.split()}
    parts = sorted(stage.glob(table + "-*.csv"))
    for i, path in enumerate(parts or [None]):
        frame = pl.read_csv(path, has_header=False, schema=schema) if path else pl.DataFrame(schema=schema)
        frame.with_columns(pl.col("day").cast(pl.Date)).write_parquet(
            dest / f"part-{i}.parquet", compression=os.getenv("SHOP_COMPRESSION", "uncompressed"))
        if path:
            path.unlink()
