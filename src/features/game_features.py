"""Game-level feature table (one row per game) for the success-tier / game-level models and association rules.

  data/processed/features/game_features.parquet   keys + split + launch-time and post-launch features + tag PCs
  data/processed/features/game_outcomes.parquet   Steam's outcome columns (prefix `outcome_`): targets only, never inputs

Feature groups (also in docs/feature_engineering/feature_catalog.csv):
  launch_time  known when a developer decides on the launch: price, content, platforms, tags
  post_launch  only exists after launch (patch cadence): usable for a post-launch monitor, not for a launch-risk scorer
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import FIGS, OUT, PROCESSED, get_logger, load_games

log = get_logger("features")

PRICE_CUTS = [5, 10, 20]             # tier 1: < $5, 2: $5-10, 3: $10-20, 4: $20+ (tier 0 = free)
CONTROLLER_LEVEL = {"full": 2, "partial": 1}


def price_tier(price_usd: pd.Series, is_free: pd.Series) -> pd.Series:
    paid = np.digitize(price_usd.fillna(0), PRICE_CUTS) + 1
    return pd.Series(np.where(is_free | (price_usd.fillna(0) == 0), 0, paid), index=price_usd.index).astype("int8")


def patches_after_launch(games: pd.DataFrame, events: pd.DataFrame, days: int, kind: str = "patch") -> pd.Series:
    """Number of `kind` events within `days` days after each game's launch_date."""
    launch = games.set_index("appid")["launch_date"].dt.tz_localize("UTC")
    ev = events[events["event_type"] == kind].copy()
    ev["delta_days"] = (ev["date"] - ev["appid"].map(launch)).dt.total_seconds() / 86400
    inside = ev[(ev["delta_days"] >= 0) & (ev["delta_days"] < days)]
    return inside.groupby("appid").size().reindex(games["appid"]).fillna(0).astype("int16").set_axis(games.index)


def build() -> pd.DataFrame:
    games = load_games()
    events = pd.read_parquet(PROCESSED / "events.parquet")
    tags = pd.read_parquet(OUT / "tag_components.parquet").drop(columns="split")

    f = pd.DataFrame({"appid": games["appid"], "split": games["split"], "cv_fold": games["cv_fold"].astype("int8"),
                      "reviews_capped": games["reviews_capped"].astype("int8")})
    f["price_usd"] = games["price_usd"].astype("float32")
    f["price_tier"] = price_tier(games["price_usd"], games["is_free"])
    f["is_free"] = games["is_free"].astype("int8")
    f["launch_year"] = games["launch_date"].dt.year.astype("int16")
    f["launch_quarter"] = games["launch_date"].dt.quarter.astype("int8")
    f["launched_in_early_access"] = ((games["release_date"] - games["launch_date"]).dt.days > 7).astype("int8")
    f["launch_before_window"] = games["launch_before_window"].astype("int8")
    f["n_dlc"] = games["n_dlc"].astype("int16")
    f["n_supported_languages"] = games["n_supported_languages"].astype("int16")
    f["n_achievements"] = games["n_achievements"].fillna(0).astype("int16")
    f["has_achievements"] = (games["n_achievements"].fillna(0) > 0).astype("int8")
    f["platform_mac"] = games["platform_mac"].astype("int8")
    f["platform_linux"] = games["platform_linux"].astype("int8")
    f["controller_support_level"] = games["controller_support"].map(CONTROLLER_LEVEL).fillna(0).astype("int8")
    developers = games["developers"].fillna("").str.split(";").apply(lambda x: {t.strip() for t in x if t.strip()})
    publishers = games["publishers"].fillna("").str.split(";").apply(lambda x: {t.strip() for t in x if t.strip()})
    f["self_published"] = [int(bool(d & p)) for d, p in zip(developers, publishers)]
    f["patches_first_30d"] = patches_after_launch(games, events, 30)
    f["patches_first_90d"] = patches_after_launch(games, events, 90)
    f["sales_first_90d"] = patches_after_launch(games, events, 90, kind="sale")
    f = f.merge(tags, on="appid", how="left", validate="one_to_one")

    neg = games["steam_total_negative"]
    total = games["steam_total_reviews"]
    outcomes = pd.DataFrame({
        "appid": games["appid"], "split": games["split"], "cv_fold": games["cv_fold"].astype("int8"),
        "outcome_total_reviews": total.astype("int32"), "outcome_total_positive": games["steam_total_positive"].astype("int32"),
        "outcome_total_negative": neg.astype("int32"), "outcome_neg_ratio": (neg / total).astype("float32"),
        "outcome_pct_positive": (1 - neg / total).astype("float32"), "outcome_review_score": games["steam_review_score"].astype("int8"),
        "outcome_review_score_desc": games["steam_review_score_desc"]})

    f.to_parquet(OUT / "game_features.parquet", index=False)
    outcomes.to_parquet(OUT / "game_outcomes.parquet", index=False)
    log.info("game_features.parquet: %d games x %d columns (%d tag PCs); game_outcomes.parquet: %d x %d",
             len(f), f.shape[1], len(tags.columns) - 1, len(outcomes), outcomes.shape[1])
    return f


if __name__ == "__main__":
    build()


CATALOG = [
    ("game_features", "price_usd", "launch_time", "games.price_usd", "Regular (undiscounted) US price in USD; 0 for free games"),
    ("game_features", "price_tier", "launch_time", "price_usd, is_free", "0 free, 1 under $5, 2 $5-10, 3 $10-20, 4 $20+ (fixed cut-offs)"),
    ("game_features", "is_free", "launch_time", "games.is_free", "Free to play"),
    ("game_features", "launch_year", "launch_time", "games.launch_date", "Year of the original launch (Early Access start if any)"),
    ("game_features", "launch_quarter", "launch_time", "games.launch_date", "Quarter of the launch (1 to 4)"),
    ("game_features", "launched_in_early_access", "launch_time", "release_date, launch_date", "Store release (1.0) date is more than 7 days after the original launch"),
    ("game_features", "launch_before_window", "launch_time", "games.launch_before_window", "Early Access launch before 2022 though 1.0 is in 2022 to 2025"),
    ("game_features", "n_dlc", "launch_time", "games.n_dlc", "Number of DLCs on the store page (snapshot date, may include post-launch DLC)"),
    ("game_features", "n_supported_languages", "launch_time", "games.n_supported_languages", "Interface languages"),
    ("game_features", "n_achievements", "launch_time", "games.n_achievements", "Achievements; missing (none listed) set to 0"),
    ("game_features", "has_achievements", "launch_time", "games.n_achievements", "At least one achievement"),
    ("game_features", "platform_mac", "launch_time", "games.platform_mac", "Mac supported (Windows is true for all games and omitted)"),
    ("game_features", "platform_linux", "launch_time", "games.platform_linux", "Linux supported"),
    ("game_features", "controller_support_level", "launch_time", "games.controller_support", "0 none listed, 1 partial, 2 full"),
    ("game_features", "self_published", "launch_time", "developers, publishers", "Developer and publisher share a name"),
    ("game_features", "patches_first_30d", "post_launch", "events", "Developer patch announcements in the first 30 days after launch_date"),
    ("game_features", "patches_first_90d", "post_launch", "events", "Developer patch announcements in the first 90 days after launch_date"),
    ("game_features", "sales_first_90d", "post_launch", "events", "Developer sale announcements in the first 90 days after launch_date"),
    ("game_features", "pc_01 .. pc_68", "launch_time", "tags, categories", "Principal components of the multi-hot tag/category block (src/features/tags.py)"),
    ("game_features", "appid, split, cv_fold, reviews_capped", "key", "game_split.csv", "Key, shared train/test split, cv fold (-1 = test) and the capped-games flag"),
    ("game_outcomes", "outcome_total_reviews .. outcome_review_score_desc", "outcome", "games.steam_total_*", "Steam's own totals and rating band: targets and descriptions, NEVER predictors"),
]
