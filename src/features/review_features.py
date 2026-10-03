"""Review-level feature table (one row per review) for "will this review be negative?".

  data/processed/features/review_features/part-00k.parquet    key + split + target + features

Leaves out the 5 capped games (their launch period is missing, so timing features would be wrong).
Leaves out columns that only exist after the review is written, so they cannot leak the label:
votes_up, votes_funny, weighted_vote_score, comment_count, n_reactions, has_dev_response, refunded,
author_playtime_forever / last_two_weeks / last_played (measured at collection time), and every game-level
outcome (steam_total_*). See docs/feature_engineering.md.

Streams reviews_clean part by part so the 1.6M reviews never sit in memory at once.
"""
from __future__ import annotations

import glob
import shutil

import numpy as np
import pandas as pd

from .common import OUT, PROCESSED, get_logger, load_games

log = get_logger("features")

REFUND_WINDOW_MIN = 120                      # Steam refunds games played under 2 hours
PLAYTIME_CUTS = [120, 600, 3000]             # minutes: bucket 0 <2h, 1 2-10h, 2 10-50h, 3 50h+
READ = ["recommendationid", "appid", "language", "review", "voted_up", "steam_purchase", "received_for_free",
        "written_during_early_access", "primarily_steam_deck", "author_num_games_owned", "author_num_reviews",
        "author_playtime_at_review", "author_playtime_at_review_capped", "created"]


def last_event_before(reviews: pd.DataFrame, events: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Latest `kind` event at or before each review (same game), and how many such events came before it."""
    ev = events[events["event_type"] == kind][["appid", "date"]].sort_values("date").copy()
    ev["n_before"] = ev.groupby("appid").cumcount() + 1
    ev = ev.rename(columns={"date": "last_date"})
    left = reviews[["recommendationid", "appid", "created"]].sort_values("created")
    out = pd.merge_asof(left, ev, left_on="created", right_on="last_date", by="appid", direction="backward")
    out["days_since"] = (out["created"] - out["last_date"]).dt.total_seconds() / 86400
    out["n_before"] = out["n_before"].fillna(0)
    return out[["recommendationid", "days_since", "n_before"]]


def build_part(raw: pd.DataFrame, games: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    r = raw[~raw["appid"].isin(games.loc[games["reviews_capped"] == 1, "appid"])].copy()
    r = r.merge(games, on="appid", how="inner", validate="many_to_one")

    out = pd.DataFrame({"recommendationid": r["recommendationid"].to_numpy(), "appid": r["appid"].to_numpy(),
                        "split": r["split"].to_numpy(), "cv_fold": r["cv_fold"].astype("int8").to_numpy(),
                        "target_is_negative": (~r["voted_up"]).astype("int8").to_numpy()})

    minutes = r["author_playtime_at_review"]
    out["playtime_at_review_min_capped"] = r["author_playtime_at_review_capped"].astype("float32").to_numpy()
    out["playtime_bucket"] = np.where(minutes.isna(), -1, np.digitize(minutes.fillna(0), PLAYTIME_CUTS)).astype("int8")
    out["in_refund_window"] = (minutes < REFUND_WINDOW_MIN).fillna(False).astype("int8").to_numpy()

    launch = r["launch_date"].dt.tz_localize("UTC")
    days = (r["created"] - launch).dt.total_seconds() / 86400
    out["days_since_launch"] = days.astype("float32").to_numpy()
    out["is_prerelease"] = (days < 0).astype("int8").to_numpy()
    out["launch_week"] = ((days >= 0) & (days < 7)).astype("int8").to_numpy()

    for kind, plural in (("patch", "patches"), ("sale", "sales")):
        e = last_event_before(r, events, kind).set_index("recommendationid").reindex(r["recommendationid"])
        out[f"days_since_last_{kind}"] = e["days_since"].astype("float32").to_numpy()
        out[f"n_{plural}_before"] = e["n_before"].astype("int16").to_numpy()
    out["has_prior_patch"] = out["days_since_last_patch"].notna().astype("int8")

    for c in ("steam_purchase", "received_for_free", "written_during_early_access", "primarily_steam_deck"):
        out[c] = r[c].astype("int8").to_numpy()
    out["author_num_games_owned_log1p"] = np.log1p(r["author_num_games_owned"]).astype("float32").to_numpy()
    out["author_num_reviews_log1p"] = np.log1p(r["author_num_reviews"]).astype("float32").to_numpy()

    text = r["review"].astype(object)
    out["review_n_chars"] = text.str.len().clip(upper=32000).astype("int16").to_numpy()
    out["review_n_words"] = text.str.split().str.len().clip(upper=32000).astype("int16").to_numpy()
    out["language"] = r["language"].to_numpy()
    out["is_english"] = (r["language"] == "english").astype("int8").to_numpy()

    out["price_usd"] = r["price_usd"].astype("float32").to_numpy()
    out["price_tier"] = r["price_tier"].astype("int8").to_numpy()
    out["is_free"] = r["is_free"].astype("int8").to_numpy()
    return out


def build() -> int:
    games = load_games()[["appid", "launch_date"]].merge(
        pd.read_parquet(OUT / "game_features.parquet", columns=["appid", "split", "cv_fold", "reviews_capped",
                                                                "price_usd", "price_tier", "is_free"]), on="appid")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    folder = OUT / "review_features"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    total = 0
    parts = sorted(glob.glob(str(PROCESSED / "reviews_clean" / "*.parquet")))
    for i, path in enumerate(parts):
        raw = pd.read_parquet(path, columns=READ)
        feats = build_part(raw, games, events)
        feats.to_parquet(folder / f"part-{i:03d}.parquet", index=False)
        total += len(feats)
        log.info("review_features part %d/%d: %d reviews (of %d read)", i + 1, len(parts), len(feats), len(raw))
    log.info("review_features: %d reviews written (capped games left out)", total)
    return total


CATALOG = [
    ("review_features", "target_is_negative", "target", "voted_up", "1 if the review does not recommend the game (the label)"),
    ("review_features", "playtime_at_review_min_capped", "at_review", "author_playtime_at_review_capped", "Minutes played when the review was written, winsorised at the 99.5% quantile"),
    ("review_features", "playtime_bucket", "at_review", "author_playtime_at_review", "0 under 2 h, 1 2-10 h, 2 10-50 h, 3 50 h+; -1 unknown (12 reviews)"),
    ("review_features", "in_refund_window", "at_review", "author_playtime_at_review", "Played under 2 hours (inside Steam's refund policy); unknown playtime counts as 0"),
    ("review_features", "days_since_launch", "at_review", "created, games.launch_date", "Days from the original launch (Early Access start if any) to the review; negative = before launch"),
    ("review_features", "is_prerelease", "at_review", "days_since_launch", "Written before launch"),
    ("review_features", "launch_week", "at_review", "days_since_launch", "Written in the first 7 days after launch"),
    ("review_features", "days_since_last_patch", "at_review", "events", "Days since the developer's latest patch announcement at the time of the review; missing before the first patch"),
    ("review_features", "n_patches_before", "at_review", "events", "Patch announcements published before the review"),
    ("review_features", "has_prior_patch", "at_review", "events", "At least one patch announcement before the review"),
    ("review_features", "days_since_last_sale", "at_review", "events", "Days since the latest sale announcement; missing before the first one"),
    ("review_features", "n_sales_before", "at_review", "events", "Sale announcements published before the review"),
    ("review_features", "steam_purchase, received_for_free, written_during_early_access, primarily_steam_deck", "at_review", "reviews_clean", "Flags as scraped (0/1)"),
    ("review_features", "author_num_games_owned_log1p, author_num_reviews_log1p", "at_review", "reviews_clean", "log(1 + reviewer's library size / number of reviews)"),
    ("review_features", "review_n_chars, review_n_words", "at_review", "review", "Length of the review text (words are unreliable for Chinese/Japanese)"),
    ("review_features", "language, is_english", "at_review", "language", "Reviewer's chosen language and an English flag"),
    ("review_features", "price_usd, price_tier, is_free", "at_review", "games", "The game's regular price, tier and free flag"),
    ("review_features", "recommendationid, appid, split, cv_fold", "key", "game_split.csv", "Review key, game, shared split and cv fold (-1 = test)"),
]


if __name__ == "__main__":
    build()
