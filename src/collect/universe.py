"""Build the game universe from Steam store search.

Search = Indie tag (492) AND one roguelike-family tag, games only (category1=998).
Raw result pages are saved untouched to data/raw/search/; parsed candidates go to
data/raw/universe/candidates.csv and the filtered list to docs/app_ids.csv.

Usage: python -m src.collect.universe [--min-reviews 10]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from datetime import datetime, timezone

from .common import RAW, ROOT, Throttled, atomic_write_json, get_logger

SEARCH_URL = "https://store.steampowered.com/search/results/"
INDIE = 492
ROGUE_TAGS = {1716: "Roguelike", 3959: "Roguelite", 42804: "Action Roguelike",
              1091588: "Roguelike Deckbuilder", 454187: "Traditional Roguelike"}
YEARS = range(2022, 2026)
PAGE = 100

log = get_logger("universe")

ROW_RE = re.compile(r'<a href="[^"]*/app/(\d+)/[^"]*"(.*?)</a>', re.S)
TITLE_RE = re.compile(r'<span class="title">(.*?)</span>', re.S)
RELEASED_RE = re.compile(r'search_released[^>]*>(.*?)</div>', re.S)
TOOLTIP_RE = re.compile(r'data-tooltip-html="([^"]*)"')
REVIEWS_RE = re.compile(r'(\d+)% of the ([\d,]+) user reviews')
TAGIDS_RE = re.compile(r'data-ds-tagids="\[([^\]]*)\]"')
PRICE_RE = re.compile(r'data-price-final="(\d+)"')
DISCOUNT_RE = re.compile(r'data-discount="(\d+)"')


def parse_year(s: str) -> int | None:
    m = re.search(r"(20\d\d)", s)
    return int(m.group(1)) if m else None


def parse_date(s: str) -> str:
    for fmt in ("%b %d, %Y", "%d %b, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def parse_page(results_html: str) -> list[dict]:
    rows = []
    for appid, body in ROW_RE.findall(results_html):
        title = TITLE_RE.search(body)
        rel = RELEASED_RE.search(body)
        rel_s = " ".join(html.unescape(rel.group(1)).split()) if rel else ""
        tip = TOOLTIP_RE.search(body)
        pct = n_rev = None
        if tip:
            m = REVIEWS_RE.search(html.unescape(tip.group(1)))
            if m:
                pct, n_rev = int(m.group(1)), int(m.group(2).replace(",", ""))
        tags = TAGIDS_RE.search(body)
        price = PRICE_RE.search(body)
        disc = DISCOUNT_RE.search(body)
        rows.append({
            "appid": int(appid),
            "name": html.unescape(title.group(1)).strip() if title else "",
            "released_raw": rel_s,
            "release_date": parse_date(rel_s),
            "release_year": parse_year(rel_s),
            "search_pct_positive": pct,
            "search_review_count": n_rev,
            "search_tagids": tags.group(1) if tags else "",
            "price_final_usd_cents": int(price.group(1)) if price else None,
            "discount_pct": int(disc.group(1)) if disc else None,
        })
    return rows


def crawl(http: Throttled, tagid: int) -> list[dict]:
    out_dir = RAW / "search" / f"{tagid}_released_desc"
    rows, start, total = [], 0, None
    while total is None or start < total:
        f = out_dir / f"{start:06d}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
        else:
            r = http.get(SEARCH_URL, params={"tags": f"{INDIE},{tagid}", "category1": 998, "start": start,
                                             "count": PAGE, "infinite": 1, "json": 1, "cc": "us", "l": "english",
                                             "ndl": 1, "sort_by": "Released_DESC"})
            if r is None or r.status_code != 200:
                log.error("search failed tag=%s start=%s", tagid, start)
                break
            d = r.json()
            atomic_write_json(f, d)
        total = d.get("total_count", 0)
        page_rows = parse_page(d.get("results_html", ""))
        if not page_rows:
            log.warning("empty page tag=%s start=%s total=%s", tagid, start, total)
            break
        for row in page_rows:
            row["matched_tag"] = ROGUE_TAGS[tagid]
        rows += page_rows
        start += PAGE
        log.info("tag %s: %d/%d", ROGUE_TAGS[tagid], min(start, total), total)
        # Sorted newest first: once a whole page is dated before the window, older pages are irrelevant.
        years = [r["release_year"] for r in page_rows if r["release_year"]]
        if years and max(years) < min(YEARS):
            log.info("tag %s: passed %d, stopping", ROGUE_TAGS[tagid], min(YEARS))
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reviews", type=int, default=10)
    args = ap.parse_args()

    http = Throttled(1.5, log, block_wait=90)
    merged: dict[int, dict] = {}
    for tagid in ROGUE_TAGS:
        for row in crawl(http, tagid):
            if row["appid"] in merged:
                merged[row["appid"]]["matched_tags"].add(row["matched_tag"])
            else:
                row["matched_tags"] = {row.pop("matched_tag")}
                merged[row["appid"]] = row
            row.pop("matched_tag", None)

    cands = sorted(merged.values(), key=lambda r: r["appid"])
    for r in cands:
        r["matched_tags"] = ";".join(sorted(r["matched_tags"]))
    fields = ["appid", "name", "released_raw", "release_date", "release_year", "search_pct_positive",
              "search_review_count", "search_tagids", "price_final_usd_cents", "discount_pct", "matched_tags"]
    (RAW / "universe").mkdir(parents=True, exist_ok=True)
    with open(RAW / "universe" / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cands)

    sel = [r for r in cands if r["release_year"] in YEARS and (r["search_review_count"] or 0) >= args.min_reviews]
    sel.sort(key=lambda r: r["search_review_count"])
    (ROOT / "docs").mkdir(exist_ok=True)
    with open(ROOT / "docs" / "app_ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sel)

    in_years = [r for r in cands if r["release_year"] in YEARS]
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates_total": len(cands),
        "candidates_released_2022_2025": len(in_years),
        "min_reviews": args.min_reviews,
        "selected": len(sel),
        "selected_review_sum_search": sum(r["search_review_count"] for r in sel),
        "requests": http.n_requests,
    }
    atomic_write_json(RAW / "universe" / "summary.json", summary)
    log.info("summary %s", summary)


if __name__ == "__main__":
    main()
