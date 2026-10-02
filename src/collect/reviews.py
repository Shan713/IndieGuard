"""Throttled, checkpointed, resumable /appreviews scraper.

Per game, reviews are paged with filter=recent and the cursor. Every page is appended to
data/raw/reviews/<appid>.part.jsonl and the cursor is checkpointed to <appid>.state.json, so a
crash or Ctrl+C resumes mid-game. A finished game becomes <appid>.jsonl.gz plus <appid>.done.json.
A page can be written twice if the process dies between append and checkpoint: dedupe on
recommendationid downstream.

Privacy: the author's persona name, profile URL and avatar are dropped; steamid is replaced by a
salted SHA256 (author.steamid_hash) so per-reviewer dedupe and bot checks still work.

Before each game it also fetches appdetails + store tags (src.collect.metadata.fetch_store) with the
same throttle, because Steam rate-limits the whole store host as one bucket.

Usage: python -m src.collect.reviews [--delay 1.25] [--limit N]
Stop gracefully by creating data/raw/reviews/STOP.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

from .common import RAW, ROOT, Throttled, atomic_write_json, get_logger, keep_awake
from .metadata import fetch_store

URL = "https://store.steampowered.com/appreviews/{appid}"
OUT = RAW / "reviews"
SALT_FILE = ROOT / "data" / "raw" / ".steamid_salt"
DROP_AUTHOR = {"personaname", "profile_url", "avatar", "persona_status", "steamid"}

log = get_logger("reviews")


def load_salt() -> bytes:
    if not SALT_FILE.exists():
        SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SALT_FILE.write_bytes(os.urandom(16).hex().encode())
    return SALT_FILE.read_bytes()


def sanitise(review: dict, appid: int, salt: bytes) -> dict:
    a = review.get("author", {})
    sid = a.get("steamid", "")
    a = {k: v for k, v in a.items() if k not in DROP_AUTHOR}
    a["steamid_hash"] = hashlib.sha256(salt + sid.encode()).hexdigest()[:20] if sid else ""
    review["author"] = a
    review["appid"] = appid
    return review


def scrape_game(http: Throttled, appid: int, salt: bytes, max_reviews: int = 0) -> dict | None:
    done_f = OUT / f"{appid}.done.json"
    if done_f.exists():
        return None
    part_f, state_f = OUT / f"{appid}.part.jsonl", OUT / f"{appid}.state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {
        "appid": appid, "cursor": "*", "pages": 0, "n": 0, "query_summary": None,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if state["pages"]:
        log.info("resume %s at page %d (%d reviews)", appid, state["pages"], state["n"])

    empty_retries = 0
    stop_reason = "end"
    with open(part_f, "a", encoding="utf-8") as out:
        while True:
            if (OUT / "STOP").exists():
                return {"stopped": True}
            if max_reviews and state["n"] >= max_reviews:
                stop_reason = "capped"  # newest max_reviews only (filter=recent pages newest first)
                break
            r = http.get(URL.format(appid=appid), params={
                "json": 1, "num_per_page": 100, "cursor": state["cursor"], "filter": "recent",
                "language": "all", "review_type": "all", "purchase_type": "all", "filter_offtopic_activity": 0})
            if r is None:
                # Leave the checkpoint in place: the game stays in progress and is resumed on restart.
                log.error("app %s: giving up for now at page %d", appid, state["pages"])
                return {"failed": True}
            try:
                d = r.json()
            except ValueError:
                d = {}
            if r.status_code != 200 or d.get("success") != 1:
                empty_retries += 1
                log.warning("app %s bad page (HTTP %s, success=%s), retry %d", appid, r.status_code, d.get("success"), empty_retries)
                if empty_retries > 3:
                    stop_reason = f"bad_response_{r.status_code}"
                    break
                time.sleep(15)
                continue
            if state["cursor"] == "*" and state["query_summary"] is None:
                state["query_summary"] = d.get("query_summary")
            expected = (state["query_summary"] or {}).get("total_reviews", 0)
            reviews = d.get("reviews", [])
            new_cursor = d.get("cursor")
            if not reviews:
                # Steam occasionally returns an empty page early; retry before accepting the end.
                if state["n"] < 0.95 * expected and empty_retries < 3:
                    empty_retries += 1
                    log.warning("app %s empty page at %d/%d, retry %d", appid, state["n"], expected, empty_retries)
                    time.sleep(10)
                    continue
                break
            empty_retries = 0
            for rv in reviews:
                out.write(json.dumps(sanitise(rv, appid, salt), ensure_ascii=False) + "\n")
            out.flush()
            state["n"] += len(reviews)
            state["pages"] += 1
            if not new_cursor or new_cursor == state["cursor"]:
                state["cursor"] = new_cursor
                atomic_write_json(state_f, state)
                stop_reason = "cursor_repeat"
                break
            state["cursor"] = new_cursor
            atomic_write_json(state_f, state)
            if len(reviews) < 100 and state["n"] >= expected:
                break  # short final page and Steam's total reached: skip the extra empty request
            if state["pages"] % 100 == 0:
                log.info("app %s: %d/%d reviews (%d pages)", appid, state["n"], expected, state["pages"])

    gz_f = OUT / f"{appid}.jsonl.gz"
    if part_f.exists():
        with open(part_f, "rb") as src, gzip.open(gz_f, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
    else:
        gzip.open(gz_f, "wb").close()
    meta = {**{k: state[k] for k in ("appid", "pages", "n", "query_summary", "started_utc")},
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stop_reason": stop_reason,
            "params": "filter=recent&language=all&review_type=all&purchase_type=all&filter_offtopic_activity=0"}
    atomic_write_json(done_f, meta)
    part_f.unlink(missing_ok=True)
    state_f.unlink(missing_ok=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.25)
    ap.add_argument("--limit", type=int, default=0, help="only the first N games (testing)")
    ap.add_argument("--ids", default=str(ROOT / "docs" / "app_ids.csv"))
    ap.add_argument("--max-per-game", type=int, default=int(os.environ.get("MAX_REVIEWS_PER_GAME", 0)),
                    help="stop each game after this many (newest) reviews; 0 = no cap")
    args = ap.parse_args()

    keep_awake()
    OUT.mkdir(parents=True, exist_ok=True)
    salt = load_salt()
    with open(args.ids, encoding="utf-8") as f:
        apps = [(int(r["appid"]), r["name"], int(r["search_review_count"] or 0)) for r in csv.DictReader(f)]
    if args.limit:
        apps = apps[: args.limit]
    # In-progress games first so a resume finishes them, then the listed order (ascending review count).
    apps.sort(key=lambda a: not (OUT / f"{a[0]}.state.json").exists())
    # Steam allows ~150 requests then blocks for <1 min regardless of pace (1.25-1.84s tested),
    # so run at the brief's minimum spacing and wait briefly: slowing down only lowers throughput.
    http = Throttled(args.delay, log, block_wait=45)
    t0, done_n, done_reviews, failed = time.time(), 0, 0, 0
    total = len(apps)
    for i, (appid, name, est) in enumerate(apps, 1):
        if (OUT / "STOP").exists():
            log.info("STOP file found, exiting cleanly")
            break
        fetch_store(http, appid)  # appdetails + tags through the same store-host throttle
        meta = scrape_game(http, appid, salt, args.max_per_game)
        if meta and meta.get("stopped"):
            log.info("STOP file found, exiting cleanly")
            break
        if meta and meta.get("failed"):
            failed += 1
            continue
        if meta:
            done_n += 1
            done_reviews += meta["n"]
            exp = (meta["query_summary"] or {}).get("total_reviews")
            log.info("[%d/%d] done %s %r: %d reviews (steam total %s, %s) | session %d games, %d reviews, %d req, %d x429, %.0f rev/h",
                     i, total, appid, name, meta["n"], exp, meta["stop_reason"], done_n, done_reviews,
                     http.n_requests, http.n_429, done_reviews / max(time.time() - t0, 1) * 3600)
        atomic_write_json(OUT / "_progress.json", {
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "position": i, "of": total, "session_games": done_n, "session_reviews": done_reviews,
            "requests": http.n_requests, "http_429": http.n_429})
    log.info("finished: %d games, %d reviews this session, %d games left in progress", done_n, done_reviews, failed)
    if failed:
        sys.exit(1)  # the supervisor restarts us, and in-progress games are resumed first


if __name__ == "__main__":
    main()
