"""Build the small tables the dashboard reads, so the app does not need the 650 MB of review data.

  python -m src.dashboard.build_data

Writes data/dashboard/{games.csv, monthly.csv, refund_heatmap.csv, refund_summary.csv}. Everything else the dashboard shows comes
from the committed tables in reports/results/. Rerun after the data or the targets change ("final data refresh").
"""
from __future__ import annotations

import pandas as pd

from .data import DASH, PROC, FEAT

PRICE_LABELS = ["free", "<$5", "$5-10", "$10-20", "$20+"]
PLAYTIME_LABELS = ["<2 h", "2-10 h", "10-50 h", "50 h+"]


def main() -> None:
    DASH.mkdir(parents=True, exist_ok=True)

    g = pd.read_parquet(PROC / "games_clean.parquet", columns=[
        "appid", "name", "release_date", "price_usd", "is_free", "n_dlc", "n_achievements", "n_supported_languages", "tags",
        "steam_total_reviews", "steam_total_positive", "steam_total_negative", "reviews_capped"])
    t = pd.read_parquet(FEAT / "game_targets.parquet", columns=["appid", "split", "tier"])
    gf = pd.read_parquet(FEAT / "game_features.parquet", columns=["appid", "price_tier", "launched_in_early_access"])
    d = g.merge(t, on="appid", how="left").merge(gf, on="appid", how="left")
    d["release_year"] = d.release_date.dt.year
    d["neg_share"] = d.steam_total_negative / d.steam_total_reviews.where(d.steam_total_reviews > 0)
    d["price_band"] = d.price_tier.map(dict(enumerate(PRICE_LABELS)))
    d["name"] = d.name.astype(str)
    d[["appid", "name", "release_year", "price_usd", "price_band", "is_free", "n_dlc", "n_achievements", "n_supported_languages",
       "launched_in_early_access", "tags", "steam_total_reviews", "neg_share", "tier", "split", "reviews_capped"]].to_csv(DASH / "games.csv", index=False)

    r = pd.read_parquet(PROC / "reviews_clean", columns=["created", "voted_up"])
    r["month"] = r.created.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    m = r.groupby("month").agg(reviews=("voted_up", "size"), neg_share=("voted_up", lambda s: 1 - s.mean())).reset_index()
    m[m.month >= "2022-01-01"].to_csv(DASH / "monthly.csv", index=False)

    rf = pd.read_parquet(FEAT / "review_features", columns=["target_is_negative", "playtime_bucket", "in_refund_window", "price_tier"])
    h = rf[rf.playtime_bucket >= 0].groupby(["playtime_bucket", "price_tier"]).target_is_negative.agg(["mean", "size"]).reset_index()
    h["playtime"] = h.playtime_bucket.map(dict(enumerate(PLAYTIME_LABELS)))
    h["price_band"] = h.price_tier.map(dict(enumerate(PRICE_LABELS)))
    h.rename(columns={"mean": "neg_share", "size": "reviews"})[["playtime", "price_band", "neg_share", "reviews"]].to_csv(DASH / "refund_heatmap.csv", index=False)

    inside, outside = rf[rf.in_refund_window == 1], rf[rf.in_refund_window == 0]
    pd.DataFrame([{"reviews": len(rf), "neg_share_inside_window": inside.target_is_negative.mean(),
                   "neg_share_outside_window": outside.target_is_negative.mean(),
                   "inside_window_share_of_reviews": rf.in_refund_window.mean(),
                   "inside_window_share_of_negatives": rf[rf.target_is_negative == 1].in_refund_window.mean()}]).to_csv(DASH / "refund_summary.csv", index=False)
    print({p.name: p.stat().st_size for p in sorted(DASH.glob("*.csv"))})


if __name__ == "__main__":
    main()
