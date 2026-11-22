"""Check pre-Parquet native rows without external Python packages."""
import csv
import os
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys
import tempfile


def check(data, expected, threads):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["build-local/shop", data, tmp], check=True,
                       env=dict(os.environ, NCPU=str(threads)), stderr=subprocess.PIPE)
        for table in ("campaign_daily", "user_daily"):
            with (expected / (table + ".csv")).open(newline="") as f:
                reader = csv.reader(f)
                day = next(reader).index("day")
                want = sorted(reader)
            actual = []
            for part in Path(tmp).glob(table + "-*.csv"):
                with part.open(newline="") as f:
                    for row in csv.reader(f):
                        row[day] = (date(1970, 1, 1) + timedelta(days=int(row[day]))).isoformat()
                        actual.append(row)
            if sorted(actual) != want:
                raise AssertionError((data, table, actual[:5], want[:5]))


if __name__ == "__main__":
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "starter-polars/fixtures")
    cases = sorted(p for p in base.iterdir() if p.is_dir())
    assert cases
    for case in cases:
        data = case / "input"
        check(data if data.is_dir() else case / "data", case / "expected", 8)
    print(f"{len(cases)} native fixture comparisons passed")
