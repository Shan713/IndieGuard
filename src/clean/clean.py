"""Stage 2: interim tables -> cleaned tables, with every rule logged to docs/cleaning_log.md.

Scope: cleaning only. Columns are the scraped fields plus parsed dates, unit fixes and the quality flags
behind each rule; derived analysis features and merging tables are left to feature engineering (M2).

Rows are only dropped when they are unusable (duplicates, no text, impossible values, out-of-scope games).
Everything that is merely suspicious is kept and flagged (is_*/low_*/*_outlier columns) so each analysis
can choose; `text_mining_ok` marks English reviews with analysable text.

  data/processed/reviews_clean/part-*.parquet  (split so each file stays under GitHub's 100 MB limit;
                                               read the folder: pd.read_parquet('data/processed/reviews_clean'))
  data/processed/games_clean.parquet
  data/processed/events.parquet           developer announcements classified as patch or sale events

Usage: python -m src.clean.clean
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.collect.common import ROOT, get_logger

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
log = get_logger("clean")

YEARS = (2022, 2025)
REVIEW_PART_ROWS = 350_000     # ~80 MB per parquet part, under GitHub's 100 MB file limit
PLAYTIME_CAP_Q = 0.995           # winsorise playtime at this quantile (Steam counts idle time)
COPYPASTA_MIN_LEN, COPYPASTA_MIN_COUNT = 50, 10
# Patch = developer post tagged 'patchnotes' by Steam, or titled like a shipped patch, minus teasers/marketing.
PATCH_TITLE_RE = re.compile(
    r"\b(?:patch|hotfix|hot-fix|bug ?fix(?:es)?|fixes|changelog|update|v?\d+\.\d+(?:\.\d+)*[a-z]?|20\d{6})\b", re.I)
NOT_PATCH_RE = re.compile(
    r"\b(?:demo|sale|discount|% ?off|deal|bundle|nominat\w*|award\w*|vote|playtest\w*|next ?fest|festival|"
    r"soon|coming|upcoming|next (?:week|month)|roadmap|preview|dev ?(?:log|blog|diary|journal|update)|sneak peek|trading cards|"
    r"wishlist|giveaway|contest|survey)\b", re.I)
SALE_TITLE_RE = re.compile(r"\b(?:sale|discount|deal|\d{1,2} ?% ?off|-\d{1,2} ?%)", re.I)


class Log:
    """Collects one entry per rule: rows before/after and the reason, rendered into docs/cleaning_log.md."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def drop(self, table: str, rule: str, before: int, after: int, why: str) -> None:
        self.rows.append({"table": table, "rule": rule, "action": "drop", "before": before, "after": after,
                          "affected": before - after, "why": why})
        log.info("%s | %s: %d -> %d (-%d)", table, rule, before, after, before - after)

    def flag(self, table: str, rule: str, n_rows: int, n_flagged: int, why: str) -> None:
        self.rows.append({"table": table, "rule": rule, "action": "flag", "before": n_rows, "after": n_rows,
                          "affected": n_flagged, "why": why})
        log.info("%s | %s: flagged %d of %d", table, rule, n_flagged, n_rows)

    def fix(self, table: str, rule: str, n_rows: int, n_changed: int, why: str) -> None:
        self.rows.append({"table": table, "rule": rule, "action": "fix", "before": n_rows, "after": n_rows,
                          "affected": n_changed, "why": why})
        log.info("%s | %s: changed %d of %d", table, rule, n_changed, n_rows)


# --------------------------------------------------------------------------- games

def parse_store_date(s) -> pd.Timestamp:
    if not isinstance(s, str) or not s.strip():
        return pd.NaT
    for fmt in ("%b %d, %Y", "%d %b, %Y", "%B %d, %Y", "%b %Y", "%B %Y", "%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s.strip(), fmt))
        except ValueError:
            pass
    return pd.NaT


def clean_games(g: pd.DataFrame, launch: pd.Series, L: Log) -> pd.DataFrame:
    """launch: appid -> original launch timestamp (reviews' app_release_date, includes Early Access)."""
    T = "games"
    n0 = len(g)
    g = g.drop_duplicates("appid")
    L.drop(T, "Duplicate appid", n0, len(g), "Same game matched by several roguelike tags; one row per game.")

    n0 = len(g)
    g = g[g["appdetails_ok"] & (g["type"] == "game")].copy()
    L.drop(T, "No store details / not a game", n0, len(g),
           "Delisted or non-game apps have no price, genre or release data.")

    store = g["store_release_date_raw"].map(parse_store_date)
    search = pd.to_datetime(g["search_release_date"], errors="coerce")
    g["release_date"] = store.fillna(search)
    L.fix(T, "Release date from store API (search date as fallback)", len(g), int(store.isna().sum()),
          "appdetails is the authoritative date; search date only used where the store date is unparseable.")
    n0 = len(g)
    yr = g["release_date"].dt.year
    g = g[yr.between(*YEARS)].copy()
    L.drop(T, f"Release outside {YEARS[0]}-{YEARS[1]}", n0, len(g),
           "Scope of the study; store date can differ from the search listing used to build the universe.")

    # Steam moves the store release date to the 1.0 date when a game leaves Early Access. The reviews'
    # app_release_date keeps the original launch, which is what launch-timing analyses need.
    g["launch_date"] = pd.to_datetime(g["appid"].map(launch), unit="s").fillna(g["release_date"])
    g["launch_date"] = g[["launch_date", "release_date"]].min(axis=1)
    L.fix(T, "launch_date = original launch (Early Access start) where earlier than store date", len(g),
          int(((g["release_date"] - g["launch_date"]).dt.days > 7).sum()),
          "Store date is the 1.0 date for games that left Early Access; launch_date is the true launch.")
    g["launch_before_window"] = g["launch_date"].dt.year < YEARS[0]
    L.flag(T, f"Early Access launch before {YEARS[0]} (1.0 inside window)", len(g),
           int(g["launch_before_window"].sum()), "Kept per the store-date scope rule; can be excluded in sensitivity checks.")

    g["price_usd"] = np.where(g["is_free"].fillna(False), 0.0, g["price_initial_cents"] / 100)
    g["price_final_usd_snapshot"] = np.where(g["is_free"].fillna(False), 0.0, g["price_final_cents"] / 100)
    g["price_missing"] = g["price_usd"].isna()
    L.flag(T, "No price (not purchasable in US store at snapshot)", len(g), int(g["price_missing"].sum()),
           "Kept: reviews are valid; price-based analyses must exclude or impute these.")

    def _split(s):
        return [t.strip() for t in s.split(";") if t.strip()] if isinstance(s, str) else []

    list_cols = ["tags", "genres", "categories", "developers", "publishers"]
    before = g[list_cols].fillna("").copy()
    for col in list_cols:
        g[col] = g[col].map(lambda s: ";".join(_split(s)))
    L.fix(T, "Strip whitespace in tag/genre/developer lists", len(g), int((before != g[list_cols]).any(axis=1).sum()),
          "Store tag names sometimes carry trailing spaces ('Dystopian ').")

    g["spy_missing"] = g["spy_owners_raw"].isna()
    L.flag(T, "No SteamSpy record", len(g), int(g["spy_missing"].sum()), "Owner estimates unavailable; kept.")

    g["review_scrape_complete"] = g["reviews_scraped"] >= 0.99 * g["steam_total_reviews"]
    L.flag(T, "Review scrape < 99% of Steam total", len(g),
           int((g["reviews_scraped"].notna() & ~g["review_scrape_complete"]).sum()),
           "Steam totals include hidden/deleted reviews the API no longer returns; per-game coverage is reported.")
    g["reviews_capped"] = g["review_scrape_stop"].eq("capped")
    L.flag(T, "Reviews capped (newest 15,000 only)", len(g), int(g["reviews_capped"].sum()),
           "Largest hits capped to finish the scrape on time and stop them dominating review-level models. "
           "Their launch period is missing: exclude from launch/refund-window timing and patch before/after "
           "analyses, and use the steam_total_* columns (full Steam totals) for their rating.")
    L.flag(T, "Review scrape not finished", len(g), int(g["reviews_scraped"].isna().sum()),
           "Game still in progress when cleaning ran; its reviews are absent from reviews_clean.")
    return g


# --------------------------------------------------------------------------- events

def build_events(news: pd.DataFrame, games: pd.DataFrame, L: Log) -> pd.DataFrame:
    """Developer announcements classified as 'patch' or 'sale' events (one row per event)."""
    T = "news"
    n0 = len(news)
    news = news.drop_duplicates(["appid", "gid"])
    L.drop(T, "Duplicate news item", n0, len(news), "Same gid listed twice.")
    news = news[news["appid"].isin(games["appid"])].copy()
    news["date"] = pd.to_datetime(news["date"], unit="s", utc=True)
    title = news["title"].fillna("").astype(object)  # Python re: Unicode-aware \b and \w
    dev = news["feedname"].eq("steam_community_announcements")
    L.flag(T, "Developer announcement (not press coverage)", len(news), int(dev.sum()),
           "Only the developer's own Steam announcements are used as event dates.")
    tagged = news["news_tags"].fillna("").str.contains("patchnotes", regex=False)
    titled = title.str.contains(PATCH_TITLE_RE) & ~title.str.contains(NOT_PATCH_RE)
    news["is_patch"] = dev & (tagged | titled)
    L.flag(T, "Patch event", len(news), int(news["is_patch"].sum()),
           "Steam 'patchnotes' tag, or title like patch/hotfix/update/version number, excluding teasers "
           "('coming soon', roadmap), demos, sales, awards and dev blogs.")
    news["is_sale"] = dev & ~news["is_patch"] & title.str.contains(SALE_TITLE_RE)
    L.flag(T, "Sale/discount event", len(news), int(news["is_sale"].sum()),
           "Developer announcement titled sale/discount/deal/% off; used for price-effect analysis.")
    ev = news[news["is_patch"] | news["is_sale"]].copy()
    ev["event_type"] = np.where(ev["is_patch"], "patch", "sale")
    ev = ev.sort_values(["appid", "date"])
    return ev[["appid", "gid", "date", "event_type", "title", "news_tags", "url"]].reset_index(drop=True)


# --------------------------------------------------------------------------- reviews

def norm_text(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def clean_reviews(r: pd.DataFrame, games: pd.DataFrame, L: Log) -> pd.DataFrame:
    T = "reviews"
    n0 = len(r)
    r = r[r["appid"].isin(games["appid"])]
    L.drop(T, "Game not in cleaned game table", n0, len(r), "Follows the game-level scope rules above.")

    n0 = len(r)
    r = r.sort_values("timestamp_updated").drop_duplicates("recommendationid", keep="last")
    L.drop(T, "Duplicate recommendationid", n0, len(r),
           "A page can be saved twice after a crash/resume, or shift when a new review is posted mid-scrape; "
           "the latest edit is kept.")

    n0 = len(r)
    text = r["review"].fillna("").str.strip()
    r = r[text.ne("")].copy()
    L.drop(T, "Empty review text", n0, len(r), "No text to analyse; Steam allows submitting a blank review.")

    created = pd.to_datetime(r["timestamp_created"], unit="s", utc=True)
    scrape_end = pd.Timestamp(datetime.now(timezone.utc))
    n0 = len(r)
    ok = created.between(pd.Timestamp("2015-01-01", tz="UTC"), scrape_end)
    r, created = r[ok].copy(), created[ok]
    L.drop(T, "Impossible creation timestamp", n0, len(r), "Before 2015 or in the future.")
    r["created"] = created
    r["updated"] = pd.to_datetime(r["timestamp_updated"], unit="s", utc=True)

    # Steam occasionally reports unsigned-overflow counters (e.g. 4294967295) and negative playtimes.
    bad_funny = r["votes_funny"] >= 2**31
    r.loc[bad_funny, "votes_funny"] = np.nan
    L.fix(T, "votes_funny overflow -> missing", len(r), int(bad_funny.sum()), "Known Steam API counter bug.")
    pt_cols = ["author_playtime_at_review", "author_playtime_forever", "author_playtime_last_two_weeks"]
    neg = (r[pt_cols] < 0).any(axis=1)
    r[pt_cols] = r[pt_cols].where(r[pt_cols] >= 0)
    L.fix(T, "Negative playtime -> missing", len(r), int(neg.sum()), "Impossible values.")
    bad_wvs = ~r["weighted_vote_score"].between(0, 1)
    r.loc[bad_wvs, "weighted_vote_score"] = np.nan
    L.fix(T, "weighted_vote_score outside [0,1] -> missing", len(r), int(bad_wvs.sum()), "Score is a probability.")

    cap = r["author_playtime_at_review"].quantile(PLAYTIME_CAP_Q)
    r["author_playtime_at_review_capped"] = r["author_playtime_at_review"].clip(upper=cap)
    r["playtime_outlier"] = r["author_playtime_at_review"] > cap
    L.flag(T, f"Playtime at review above {PLAYTIME_CAP_Q:.1%} quantile ({cap / 60:,.0f} h), capped copy added",
           len(r), int(r["playtime_outlier"].sum()),
           "Steam counts idle/AFK time; raw minutes kept, author_playtime_at_review_capped is the winsorised copy.")
    r["no_playtime"] = r["author_playtime_at_review"].fillna(0).eq(0)
    L.flag(T, "Zero or missing playtime at review", len(r), int(r["no_playtime"].sum()),
           "Pre-release keys, family sharing or playtime not recorded.")

    gi = games.set_index("appid")
    launch = r["appid"].map(gi["launch_date"]).dt.tz_localize("UTC")
    r["is_prerelease"] = r["created"] < launch
    L.flag(T, "Written before launch (incl. Early Access start)", len(r), int(r["is_prerelease"].sum()),
           "Playtests/early keys; kept but can be excluded from launch-timing analyses.")

    # Text metrics run on object dtype so Python's Unicode-aware regex is used: pandas' Arrow-backed
    # strings use RE2, where \w is ASCII-only and would mark all Chinese/Russian/Japanese text as symbols.
    text = r["review"].str.strip().astype(object)
    letters = text.str.count(r"[^\W\d_]")
    cjk = text.str.count(r"[぀-ヿ㐀-鿿豈-﫿가-힯]")
    words = text.str.split().str.len()
    n_chars = text.str.len()
    is_cjk = cjk > letters / 2
    r["low_content"] = np.where(is_cjk, letters < 4, (letters < 3) | (words < 2))
    L.flag(T, "Low content (<3 letters or <2 words; CJK: <4 characters)", len(r), int(r["low_content"].sum()),
           "e.g. 'good', '10/10', '好玩', emoji only: valid votes, little text signal.")
    symbols = text.str.count(r"[^\w\s]")
    r["is_ascii_art"] = (n_chars >= 50) & (symbols / n_chars > 0.5)
    L.flag(T, "ASCII art / symbol spam (>50% symbols, 50+ chars)", len(r), int(r["is_ascii_art"].sum()),
           "Meme reviews with no analysable text.")
    norm = norm_text(text)
    counts = norm.map(norm.value_counts())
    r["is_copypasta"] = (norm.str.len() >= COPYPASTA_MIN_LEN) & (counts >= COPYPASTA_MIN_COUNT)
    L.flag(T, f"Copypasta (same {COPYPASTA_MIN_LEN}+ char text >= {COPYPASTA_MIN_COUNT} times)", len(r),
           int(r["is_copypasta"].sum()), "Copied meme/template text or coordinated posting.")
    dup_author = r.duplicated(["author_steamid_hash", "appid"], keep=False) & r["author_steamid_hash"].ne("")
    L.flag(T, "Same author, same game, >1 review (info only)", len(r), int(dup_author.sum()),
           "Steam allows one review per account per game; non-zero values would point to scrape problems.")

    r["text_mining_ok"] = r["language"].eq("english") & ~r["low_content"] & ~r["is_ascii_art"] & ~r["is_copypasta"]
    L.flag(T, "Text-mining subset (English, not low-content/ASCII/copypasta)", len(r), int(r["text_mining_ok"].sum()),
           "Steam's language field is the reviewer's chosen language; topic models are run on English only.")
    L.flag(T, "Received for free", len(r), int(r["received_for_free"].sum()),
           "Kept; free copies may bias sentiment, so analyses can control for it.")

    r["appid"] = r["appid"].astype("int64")  # same key dtype as games/events for joins
    r = r.drop(columns=["timestamp_created", "timestamp_updated", "app_release_date"])
    return r.sort_values(["appid", "created"]).reset_index(drop=True)


def write_parts(df: pd.DataFrame, folder) -> None:
    """Write df as folder/part-000.parquet, part-001.parquet, ... in row order (one logical table)."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for i, start in enumerate(range(0, len(df), REVIEW_PART_ROWS)):
        df.iloc[start:start + REVIEW_PART_ROWS].to_parquet(folder / f"part-{i:03d}.parquet", index=False)
    (PROCESSED / "reviews_clean.parquet").unlink(missing_ok=True)  # pre-split single-file layout


# --------------------------------------------------------------------------- report

def write_log(L: Log, games: pd.DataFrame, reviews: pd.DataFrame, events: pd.DataFrame) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# Cleaning log", "",
             f"Generated {now} UTC by `python -m src.clean.clean` from `data/interim/` (see `src/clean/flatten.py`).",
             "Raw data is never modified. Rows are dropped only when unusable; suspicious rows are flagged.", "",
             "| Table | Rule | Action | Rows before | Rows after | Affected | Justification |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for x in L.rows:
        lines.append(f"| {x['table']} | {x['rule']} | {x['action']} | {x['before']:,} | {x['after']:,} | "
                     f"{x['affected']:,} | {x['why']} |")
    lines += ["", "## Output", "",
              f"- `data/processed/games_clean.parquet`: {len(games):,} games",
              f"- `data/processed/reviews_clean/` (parquet parts): {len(reviews):,} reviews "
              f"({reviews['voted_up'].eq(False).mean():.1%} negative, {reviews['language'].eq('english').mean():.1%} English, "
              f"{reviews['text_mining_ok'].sum():,} in the text-mining subset)",
              f"- `data/processed/events.parquet`: {(events['event_type'] == 'patch').sum():,} patch and "
              f"{(events['event_type'] == 'sale').sum():,} sale announcements across {events['appid'].nunique():,} games",
              ""]
    (ROOT / "docs" / "cleaning_log.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    L = Log()
    raw_reviews = pd.read_parquet(INTERIM / "reviews.parquet")
    launch = raw_reviews.groupby("appid")["app_release_date"].median()
    games = clean_games(pd.read_parquet(INTERIM / "games.parquet"), launch, L)
    events = build_events(pd.read_parquet(INTERIM / "news.parquet"), games, L)
    reviews = clean_reviews(raw_reviews, games, L)
    del raw_reviews

    games.to_parquet(PROCESSED / "games_clean.parquet", index=False)
    events.to_parquet(PROCESSED / "events.parquet", index=False)
    write_parts(reviews, PROCESSED / "reviews_clean")
    write_log(L, games, reviews, events)
    with open(PROCESSED / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump(L.rows, f, indent=1)
    log.info("done: %d games, %d reviews, %d events", len(games), len(reviews), len(events))


if __name__ == "__main__":
    main()
