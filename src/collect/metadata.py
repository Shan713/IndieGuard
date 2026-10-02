"""Per-game metadata: Steam appdetails, store-page user tags (with vote counts), Steam news, SteamSpy.

Raw outputs, one file per game, skipped if already present (resumable):
  data/raw/appdetails/<appid>.json   store API response, untouched
  data/raw/tags/<appid>.json         [{"tagid", "name", "count", ...}] parsed from the store page
  data/raw/news/<appid>.json         ISteamNews/GetNewsForApp/v2 response, untouched
  data/raw/steamspy/<appid>.json     SteamSpy appdetails (skipped when steamspy.com is unreachable)

appdetails + tags are fetched by the review scraper (fetch_store) so all store-host requests share one
throttle; this script's main() collects news + SteamSpy, and the store parts only with --with-store.

Usage: python -m src.collect.metadata [--ids docs/app_ids.csv] [--with-store]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone

import requests

from .common import RAW, ROOT, Throttled, atomic_write_json, get_logger, keep_awake

APPDETAILS = "https://store.steampowered.com/api/appdetails"
APPPAGE = "https://store.steampowered.com/app/{appid}/"
NEWS = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
STEAMSPY = "https://steamspy.com/api.php"
TAGS_RE = re.compile(r"InitAppTagModal\(\s*\d+,\s*(\[.*?\])\s*,", re.S)

log = get_logger("metadata")


def steamspy_reachable() -> bool:
    try:
        requests.get(STEAMSPY, params={"request": "appdetails", "appid": 730}, timeout=20).raise_for_status()
        return True
    except requests.RequestException as e:
        log.warning("SteamSpy unreachable, skipping it this run: %s", e)
        return False


def fetch_store(http: Throttled, appid: int) -> None:
    """appdetails + store-page tags. Shares the caller's throttle: Steam rate-limits the store host as one bucket."""
    f = RAW / "appdetails" / f"{appid}.json"
    if not f.exists():
        r = http.get(APPDETAILS, params={"appids": appid, "cc": "us", "l": "english"})
        if r is not None and r.status_code == 200:
            atomic_write_json(f, {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "response": r.json()})
    f = RAW / "tags" / f"{appid}.json"
    if not f.exists():
        r = http.get(APPPAGE.format(appid=appid), params={"l": "english", "cc": "us"})
        if r is not None and r.status_code == 200:
            m = TAGS_RE.search(r.text)
            atomic_write_json(f, {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "final_url": r.url, "tags": json.loads(m.group(1)) if m else None})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=str(ROOT / "docs" / "app_ids.csv"))
    ap.add_argument("--with-store", action="store_true",
                    help="also fetch appdetails + tags (normally done inside the review scraper)")
    args = ap.parse_args()
    keep_awake()

    with open(args.ids, encoding="utf-8") as f:
        apps = [int(r["appid"]) for r in csv.DictReader(f)]
    # Largest games first, so the games that matter most get metadata soonest.
    apps.reverse()

    store_api = Throttled(1.6, log, block_wait=300)
    news_api = Throttled(0.5, log, block_wait=120)
    spy = Throttled(1.1, log, block_wait=60) if steamspy_reachable() else None
    fetched_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for i, appid in enumerate(apps, 1):
        if args.with_store:
            fetch_store(store_api, appid)

        f = RAW / "news" / f"{appid}.json"
        if not f.exists():
            r = news_api.get(NEWS, params={"appid": appid, "count": 1000, "maxlength": 600, "format": "json"})
            if r is not None and r.status_code == 200:
                atomic_write_json(f, {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "response": r.json()})

        if spy:
            f = RAW / "steamspy" / f"{appid}.json"
            if not f.exists():
                r = spy.get(STEAMSPY, params={"request": "appdetails", "appid": appid})
                if r is not None and r.status_code == 200:
                    atomic_write_json(f, {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                          "response": r.json()})

        if i % 50 == 0 or i == len(apps):
            log.info("%d/%d games | requests: store %d, news %d | 429s: %d/%d",
                     i, len(apps), store_api.n_requests, news_api.n_requests, store_api.n_429, news_api.n_429)
    atomic_write_json(RAW / "metadata_run.json", {"started_utc": fetched_utc,
                                                  "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                  "games": len(apps), "steamspy": bool(spy)})
    log.info("metadata done")


if __name__ == "__main__":
    main()
