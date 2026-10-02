"""Print scrape progress: games done, reviews collected, rough time remaining.

Usage: python -m src.collect.status
"""
from __future__ import annotations

import csv
import json

from .common import RAW, ROOT

apps = list(csv.DictReader(open(ROOT / "docs" / "app_ids.csv", encoding="utf-8")))
rev = RAW / "reviews"
done = {int(p.name.split(".")[0]): json.loads(p.read_text()) for p in rev.glob("*.done.json")}
partial = [json.loads(p.read_text()) for p in rev.glob("*.state.json")]
n_done = sum(m["n"] for m in done.values())
n_part = sum(s["n"] for s in partial)
# Remaining estimate: search counts are English-only, so scale by the observed all-language ratio.
ratio = [m["n"] / int(a["search_review_count"]) for a in apps if (m := done.get(int(a["appid"]))) and int(a["search_review_count"]) >= 100]
scale = sum(ratio) / len(ratio) if ratio else 2.0
left = [a for a in apps if int(a["appid"]) not in done]
left_reviews = sum(int(a["search_review_count"]) for a in left) * scale - n_part
left_requests = left_reviews / 100 + len(left)
print(f"games done      : {len(done)}/{len(apps)}  (in progress: {len(partial)})")
print(f"reviews on disk : {n_done + n_part:,}")
print(f"all-lang / search-count ratio: {scale:.2f}")
print(f"est. remaining  : ~{left_reviews:,.0f} reviews, ~{left_requests * 1.55 / 3600:.1f} h (1.25 s/request + block waits)")
for k in ("appdetails", "tags", "news", "steamspy"):
    print(f"{k:<16}: {len(list((RAW / k).glob('*.json'))) if (RAW / k).exists() else 0}/{len(apps)}")
prog = rev / "_progress.json"
if prog.exists():
    print("last update     :", json.loads(prog.read_text())["updated_utc"], "UTC")
