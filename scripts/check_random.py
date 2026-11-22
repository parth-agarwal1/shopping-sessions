"""Independent tiny-data oracle for linked people, deduplication and attribution."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import tempfile

from check_native import check

DAY = 86400000
BASE = 1790812800000


def iso(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writerow(header.split())
        writer.writerows(rows)


def trial(root, seed):
    rand = random.Random(seed)
    groups = [[" ghost across several linked IDs", "a", "A", "a\r"],
              ["b0000000000000000000000000000000a", "b"],
              ["c"], ["é00000000000000000000000000000000", "α"]]
    names = {alias: min(group) for group in groups for alias in group}
    rows, clicks, links = [], [], []
    for group in groups:
        links.extend(zip(group, group[1:]))
    for i in range(240):
        ts = BASE + rand.randrange(5 * 24 * 2) * 1800000 + rand.choice([0, 0, 1])
        earliest = ts // 1000 * 1000 + rand.choice([0, DAY, DAY + 1000])
        user = rand.choice(list(names))
        product = rand.choice(["p1", "p2", "other"])
        kind = rand.choice(["view", "cart", "purchase", "purchase", "bad"])
        event_id = f"e{i:032x}" if i % 3 == 0 else str(i)
        row = [event_id, user, str(ts), iso(earliest), kind, product, str(rand.randint(1, 3))]
        rows.append(row)
        if i % 4 == 0:
            rows.append([event_id, "c", str(ts), iso(earliest + 3000), "purchase", "p2", "9"])
    for i in range(140):
        t = BASE + rand.randrange(-8 * 48, 5 * 48) * 1800000
        click_id = f"k{i:032x}" if i % 3 == 0 else f"k{i}"
        row = [click_id, rand.choice(list(names)), rand.choice(["c1", "c2", "bad"]), str(t)]
        clicks.append(row)
        if i % 5 == 0:
            clicks.append(row)
    (root / "manifest.json").write_text(json.dumps({"window_start": iso(BASE), "window_end": iso(BASE+5*DAY)}))
    write(root/"products.csv", "product_id category", [("p1","z"),("p2","a")])
    write(root/"campaigns.csv", "campaign_id channel", [("c1","x"),("c2","y")])
    for name, header, data in [
        ("events", "event_id user_id ts ingest_ts event_type product_id price_cents", rows),
        ("clicks", "click_id user_id campaign_id ts", clicks), ("links", "user_id_a user_id_b", links)
    ]:
        (root/name).mkdir(exist_ok=True)
        rand.shuffle(data)
        for shard in range(3):
            write(root/name/f"{shard}.csv", header, data[shard::3])
    by_person = defaultdict(list)
    seen = set()
    for key, user, raw_ts, ingest, kind, product, price in sorted(rows, key=lambda r:r[3]):
        if key in seen:
            continue
        seen.add(key)
        ts = int(raw_ts)
        if kind == "bad" or ingest > iso(ts + DAY):
            continue
        by_person[names[user]].append((ts, kind, product, int(price)))
    users = []
    campaigns = defaultdict(lambda: [0, 0, 0, 0])
    for person, events in by_person.items():
        sessions = []
        for row in sorted(events):
            if not sessions or row[0] - sessions[-1][-1][0] > 1800000:
                sessions.append([])
            sessions[-1].append(row)
        daily = defaultdict(list)
        for session in sessions:
            start = session[0][0]
            day = iso(start)[:10]
            daily[day].append(session)
            eligible = [(int(t), k, camp) for k, u, camp, t in clicks
                        if names[u] == person and camp != "bad" and start-7*DAY <= int(t) <= start]
            winner = min(eligible, key=lambda x: (-x[0], x[1]))[2] if eligible else "organic"
            cost = sum(p for _, kind, _, p in session if kind == "purchase")
            sums = campaigns[winner, day]
            sums[0] += 1
            sums[1] += cost > 0
            sums[2] += cost
            for camp in {r[2] for r in eligible}:
                campaigns[camp, day][3] += 1
        for day, sessions in daily.items():
            events = [e for s in sessions for e in s]
            sums = Counter()
            for _, kind, product, p in events:
                if kind == "purchase":
                    sums[{"p1":"z", "p2":"a"}.get(product, "unknown")] += p
            top = min(sums, key=lambda cat: (-sums[cat], cat)) if sums else ""
            users.append([person, day, len(sessions), len(events), sum(sums.values()), top])
    write(root/"user_daily.csv", "user_id day sessions events revenue_cents top_category", users)
    write(root/"campaign_daily.csv", "campaign_id channel day sessions purchasing_sessions revenue_cents assisted_sessions",
          [[c, {"c1":"x","c2":"y"}.get(c,c), day, *sums] for (c, day), sums in campaigns.items()])
    for threads in (1, 8):
        check(root, root, threads)


def check_cell_thresholds(root):
    """Cross the promotion threshold with an exact gap and a gap one ms longer."""
    for part in ("events", "links", "clicks"):
        (root / part).mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "window_start": iso(BASE), "window_end": iso(BASE + 2 * DAY)
    }))
    write(root / "products.csv", "product_id category", [("p", "tea")])
    write(root / "campaigns.csv", "campaign_id channel", [])
    shards = [[], []]
    users = []
    for count in (127, 128, 129):
        user = f"u{count}"
        users.append([user, iso(BASE)[:10], 2, count, count * 3, "tea"])
        timestamps = [BASE] * (count - 2) + [BASE + 1800000, BASE + 3600001]
        for i, timestamp in enumerate(timestamps):
            event = f"{user}-event-{i}"
            ingest = iso(timestamp)
            shards[i % 2].append(f'{event},{user},{timestamp},{ingest},purchase,p,3')
            shards[(i + 1) % 2].append(
                f'"{event}",other,{timestamp},{iso(timestamp + 1000)},purchase,p,99'
            )
    shards[0].append(f'"invalid-earliest",other,bad,{iso(BASE)},view,p,')
    shards[1].append(f'invalid-earliest,u127,{BASE},{iso(BASE + 1000)},purchase,p,99')
    header = "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents"
    for i, rows in enumerate(shards):
        (root / "events" / f"{i}.csv").write_text("\n".join([header, *rows, ""]))
    write(root / "user_daily.csv", "user_id day sessions events revenue_cents top_category", users)
    write(root / "campaign_daily.csv",
          "campaign_id channel day sessions purchasing_sessions revenue_cents assisted_sessions",
          [["organic", "organic", iso(BASE)[:10], 6, 6, 1152, 0]])
    for threads in (1, 8):
        check(root, root, threads)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check_cell_thresholds(Path(tmp) / "thresholds")
        for seed in range(20):
            trial(Path(tmp), seed)
    print("20 shuffled seeds matched at both 1 and 8 workers")
