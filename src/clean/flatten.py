"""Stage 1: raw JSON -> typed interim tables (no rows dropped, no values changed).

  data/interim/reviews.parquet   one row per scraped review, author fields flattened
  data/interim/games.parquet     one row per game in docs/app_ids.csv: search + appdetails + tags + SteamSpy + review totals
  data/interim/news.parquet      one row per ISteamNews item

Only games whose review scrape finished (<appid>.done.json) are included in reviews.parquet, so this
can be run on a partial scrape. Reviews are streamed game by game to keep memory low.

Usage: python -m src.clean.flatten
"""
from __future__ import annotations

import csv
import gzip
import html
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq

from src.collect.common import RAW, ROOT, get_logger

INTERIM = ROOT / "data" / "interim"
log = get_logger("flatten")

REVIEW_SCHEMA = pa.schema([
    ("recommendationid", pa.int64()), ("appid", pa.int32()), ("language", pa.string()), ("review", pa.string()),
    ("timestamp_created", pa.int64()), ("timestamp_updated", pa.int64()), ("voted_up", pa.bool_()),
    ("votes_up", pa.int64()), ("votes_funny", pa.int64()), ("weighted_vote_score", pa.float64()),
    ("comment_count", pa.int64()), ("steam_purchase", pa.bool_()), ("received_for_free", pa.bool_()),
    ("refunded", pa.bool_()), ("written_during_early_access", pa.bool_()), ("primarily_steam_deck", pa.bool_()),
    ("n_reactions", pa.int32()), ("has_dev_response", pa.bool_()), ("timestamp_dev_responded", pa.int64()),
    ("author_steamid_hash", pa.string()), ("author_num_games_owned", pa.int64()), ("author_num_reviews", pa.int64()),
    ("author_playtime_forever", pa.int64()), ("author_playtime_last_two_weeks", pa.int64()),
    ("author_playtime_at_review", pa.int64()), ("author_last_played", pa.int64()),
    ("app_release_date", pa.int64()),  # original launch incl. Early Access; store date moves to the 1.0 date
])


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def review_row(r: dict) -> dict:
    a = r.get("author") or {}
    return {
        "recommendationid": _int(r.get("recommendationid")),
        "appid": _int(r.get("appid")),
        "language": r.get("language"),
        "review": r.get("review"),
        "timestamp_created": _int(r.get("timestamp_created")),
        "timestamp_updated": _int(r.get("timestamp_updated")),
        "voted_up": r.get("voted_up"),
        "votes_up": _int(r.get("votes_up")),
        "votes_funny": _int(r.get("votes_funny")),
        "weighted_vote_score": _float(r.get("weighted_vote_score")),
        "comment_count": _int(r.get("comment_count")),
        "steam_purchase": r.get("steam_purchase"),
        "received_for_free": r.get("received_for_free"),
        "refunded": r.get("refunded"),
        "written_during_early_access": r.get("written_during_early_access"),
        "primarily_steam_deck": r.get("primarily_steam_deck"),
        "n_reactions": sum(_int(x.get("count")) or 0 for x in (r.get("reactions") or [])),
        "has_dev_response": bool(r.get("developer_response")),
        "timestamp_dev_responded": _int(r.get("timestamp_dev_responded")),
        "author_steamid_hash": a.get("steamid_hash"),
        "author_num_games_owned": _int(a.get("num_games_owned")),
        "author_num_reviews": _int(a.get("num_reviews")),
        "author_playtime_forever": _int(a.get("playtime_forever")),
        "author_playtime_last_two_weeks": _int(a.get("playtime_last_two_weeks")),
        "author_playtime_at_review": _int(a.get("playtime_at_review")),
        "author_last_played": _int(a.get("last_played")),
        "app_release_date": _int(r.get("app_release_date")),
    }


def flatten_reviews() -> int:
    out = INTERIM / "reviews.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    done = sorted((RAW / "reviews").glob("*.done.json"))
    writer, n = None, 0
    try:
        for i, d in enumerate(done, 1):
            appid = int(d.name.split(".")[0])
            with gzip.open(RAW / "reviews" / f"{appid}.jsonl.gz", "rt", encoding="utf-8") as f:
                rows = [review_row(json.loads(line)) for line in f if line.strip()]
            if not rows:
                continue
            if writer is None:
                writer = pq.ParquetWriter(tmp, REVIEW_SCHEMA, compression="zstd")
            writer.write_table(pa.Table.from_pylist(rows, schema=REVIEW_SCHEMA))
            n += len(rows)
            if i % 200 == 0:
                log.info("reviews: %d/%d games, %d rows", i, len(done), n)
    finally:
        if writer is not None:
            writer.close()
    if n:
        tmp.replace(out)
    log.info("reviews.parquet: %d rows from %d finished games", n, len(done))
    return n


def _load(folder: str, appid: int) -> dict | None:
    p = RAW / folder / f"{appid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _parse_owners(s: str | None) -> tuple[int | None, int | None]:
    nums = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", s or "") if x.replace(",", "")]
    return (nums[0], nums[1]) if len(nums) == 2 else (None, None)


def _n_languages(s: str | None) -> int | None:
    if not s:
        return None
    text = re.sub(r"<[^>]+>", "", html.unescape(s.split("<br>")[0]))  # drop the "*languages with full audio" footnote
    return len([x for x in text.split(",") if x.strip()])


def game_row(base: dict) -> dict:
    appid = int(base["appid"])
    row = {"appid": appid, "name": base["name"], "search_released_raw": base["released_raw"],
           "search_release_date": base["release_date"] or None,
           "search_review_count": _int(base["search_review_count"]),
           "search_pct_positive": _int(base["search_pct_positive"]),
           "search_tagids": base["search_tagids"], "matched_tags": base["matched_tags"]}

    ad = _load("appdetails", appid)
    body = (ad or {}).get("response", {}).get(str(appid), {})
    d = body.get("data") or {}
    po = d.get("price_overview") or {}
    rd = d.get("release_date") or {}
    row.update({
        "appdetails_ok": bool(body.get("success")) and bool(d),
        "appdetails_fetched_utc": (ad or {}).get("fetched_utc"),
        "type": d.get("type"),
        "is_free": d.get("is_free"),
        "price_currency": po.get("currency"),
        "price_initial_cents": po.get("initial"),
        "price_final_cents": po.get("final"),
        "discount_percent": po.get("discount_percent"),
        "store_release_date_raw": rd.get("date"),
        "coming_soon": rd.get("coming_soon"),
        "developers": ";".join(d.get("developers") or []),
        "publishers": ";".join(p for p in (d.get("publishers") or []) if p),
        "genres": ";".join(g["description"] for g in d.get("genres") or []),
        "categories": ";".join(c["description"] for c in d.get("categories") or []),
        "n_dlc": len(d.get("dlc") or []),
        "platform_windows": (d.get("platforms") or {}).get("windows"),
        "platform_mac": (d.get("platforms") or {}).get("mac"),
        "platform_linux": (d.get("platforms") or {}).get("linux"),
        "metacritic_score": (d.get("metacritic") or {}).get("score"),
        "recommendations_total": (d.get("recommendations") or {}).get("total"),
        "required_age": _int(d.get("required_age")),
        "controller_support": d.get("controller_support"),
        "n_supported_languages": _n_languages(d.get("supported_languages")),
        "n_achievements": (d.get("achievements") or {}).get("total"),
        "short_description": d.get("short_description"),
    })

    tg = _load("tags", appid)
    tags = (tg or {}).get("tags") or []
    row["tags"] = ";".join(t["name"] for t in tags)
    row["tags_votes_json"] = json.dumps({t["name"]: t.get("count") for t in tags}, ensure_ascii=False) if tags else None

    sp = (_load("steamspy", appid) or {}).get("response") or {}
    lo, hi = _parse_owners(sp.get("owners"))
    row.update({
        "spy_owners_raw": sp.get("owners"), "spy_owners_low": lo, "spy_owners_high": hi,
        "spy_ccu": _int(sp.get("ccu")), "spy_positive": _int(sp.get("positive")), "spy_negative": _int(sp.get("negative")),
        "spy_tags_votes_json": json.dumps(sp["tags"], ensure_ascii=False) if isinstance(sp.get("tags"), dict) else None,
    })

    done = RAW / "reviews" / f"{appid}.done.json"
    meta = json.loads(done.read_text()) if done.exists() else {}
    qs = meta.get("query_summary") or {}
    row.update({
        "reviews_scraped": meta.get("n"), "review_scrape_stop": meta.get("stop_reason"),
        "steam_total_reviews": qs.get("total_reviews"), "steam_total_positive": qs.get("total_positive"),
        "steam_total_negative": qs.get("total_negative"), "steam_review_score": qs.get("review_score"),
        "steam_review_score_desc": qs.get("review_score_desc"),
    })
    return row


def news_rows(appid: int) -> list[dict]:
    nw = _load("news", appid)
    items = ((nw or {}).get("response") or {}).get("appnews", {}).get("newsitems", [])
    return [{"appid": appid, "gid": str(it.get("gid")), "title": it.get("title"), "date": _int(it.get("date")),
             "feedname": it.get("feedname"), "feedlabel": it.get("feedlabel"), "feed_type": _int(it.get("feed_type")),
             "author": it.get("author"), "news_tags": ";".join(it.get("tags") or []),
             "is_external_url": it.get("is_external_url"), "contents": it.get("contents"), "url": it.get("url")}
            for it in items]


def main() -> None:
    import pandas as pd

    INTERIM.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "docs" / "app_ids.csv", encoding="utf-8") as f:
        base = list(csv.DictReader(f))

    games = pd.DataFrame([game_row(b) for b in base])
    games.to_parquet(INTERIM / "games.parquet", index=False)
    log.info("games.parquet: %d rows (%d with appdetails, %d with finished review scrape)",
             len(games), games["appdetails_ok"].sum(), games["reviews_scraped"].notna().sum())

    news = pd.DataFrame([r for b in base for r in news_rows(int(b["appid"]))])
    news.to_parquet(INTERIM / "news.parquet", index=False)
    log.info("news.parquet: %d rows", len(news))

    flatten_reviews()


if __name__ == "__main__":
    main()
