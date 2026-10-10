"""Patch-impact analysis with launch timing and game-level uncertainty.

Run from the repository root:
    python -m src.analysis.patch_impact

Descriptive observational analysis, not a causal estimate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RES = ROOT / "reports" / "results"
FIG = ROOT / "figures" / "findings"

WINDOW_DAYS = 30
MIN_REVIEWS = 20
BOOTSTRAP_REPS = 10_000
RANDOM_SEED = 42


def deduplicate_patches(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate announcements for a game, timestamp, and title."""
    patches = events.loc[events["event_type"].eq("patch")].copy()
    patches["date"] = pd.to_datetime(patches["date"], utc=True, errors="coerce")
    patches = patches.dropna(subset=["appid", "date", "gid"])
    patches = patches.sort_values(["appid", "date", "gid"])
    original_count = len(patches)
    patches = patches.drop_duplicates(["appid", "date", "title"], keep="first")
    print(f"Repeated patch announcements collapsed: {original_count - len(patches):,}")
    return patches.reset_index(drop=True)


def launch_group(days: float) -> str:
    if pd.isna(days):
        return "unknown"
    if days <= 30:
        return "<=30 days (including pre-launch)"
    if days <= 180:
        return "31-180 days"
    return ">180 days"


def bootstrap_mean_ci(values: pd.Series) -> tuple[float, float]:
    """Percentile bootstrap confidence interval, resampling games."""
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    means = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        means[i] = rng.choice(array, size=len(array), replace=True).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def create_plots(results: pd.DataFrame) -> None:
    if results.empty:
        print("No eligible patch comparisons; plots skipped.")
        return
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(results["positive_pct_change"].dropna(), bins=40)
    ax.axvline(0, linestyle="--", color="black", label="No change")
    ax.axvline(results["positive_pct_change"].median(), linestyle=":", color="red",
               label=f"Median = {results['positive_pct_change'].median():.2f} pp")
    ax.set_xlabel("Change in positive-review share (percentage points)")
    ax.set_ylabel("Patch comparisons")
    ax.set_title("Review positivity change around patch announcements")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "patch_impact_change_distribution.png", dpi=200)
    plt.close(fig)

    launch = results.groupby("launch_group").agg(
        comparisons=("positive_pct_change", "size"),
        mean_change=("positive_pct_change", "mean"),
    )
    order = ["<=30 days (including pre-launch)", "31-180 days", ">180 days", "unknown"]
    launch = launch.reindex([label for label in order if label in launch.index])
    fig, ax = plt.subplots(figsize=(9, 5))
    if not launch.empty:
        ax.bar(launch.index, launch["mean_change"])
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Time from recorded launch to patch")
        ax.set_ylabel("Mean change in positive share (percentage points)")
        ax.set_title("Patch-window changes by time since launch")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG / "patch_impact_by_launch.png", dpi=200)
    plt.close(fig)

    strata = results.groupby("before_positive_stratum").agg(
        comparisons=("positive_pct_change", "size"),
        mean_change=("positive_pct_change", "mean"),
    )
    order = ["<70%", "70-90%", ">90%"]
    strata = strata.reindex([label for label in order if label in strata.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    if not strata.empty:
        ax.bar(strata.index, strata["mean_change"])
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Positive-review share before patch")
        ax.set_ylabel("Mean change (percentage points)")
        ax.set_title("Change stratified by pre-patch positivity")
    fig.tight_layout()
    fig.savefig(FIG / "patch_impact_by_before_positivity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    print("Loading datasets...")
    reviews = pd.read_parquet(
        DATA / "reviews_clean",
        columns=["appid", "created", "voted_up"],
    )
    events = pd.read_parquet(DATA / "events.parquet")
    games = pd.read_parquet(
        DATA / "games_clean.parquet",
        columns=["appid", "launch_date", "reviews_capped"],
    )

    capped_ids = set(games.loc[games["reviews_capped"].fillna(False), "appid"])
    reviews = reviews.loc[~reviews["appid"].isin(capped_ids)].copy()
    reviews["created"] = pd.to_datetime(reviews["created"], utc=True, errors="coerce")
    reviews = reviews.dropna(subset=["appid", "created", "voted_up"])
    reviews["voted_up"] = reviews["voted_up"].astype(bool)
    reviews = reviews.sort_values(["appid", "created"])
    reviews_by_game = {
        appid: group.reset_index(drop=True)
        for appid, group in reviews.groupby("appid", sort=False)
    }

    patches = deduplicate_patches(events)
    patches = patches.sort_values(["appid", "date", "gid"]).reset_index(drop=True)
    launch_dates = games.set_index("appid")["launch_date"]
    launch_dates = pd.to_datetime(launch_dates, utc=True, errors="coerce")

    records = []
    last_window_end: dict[int, pd.Timestamp] = {}
    overlap_skipped = insufficient = no_reviews = 0

    for patch in patches.itertuples(index=False):
        game_reviews = reviews_by_game.get(patch.appid)
        if game_reviews is None:
            no_reviews += 1
            continue

        patch_date = patch.date
        before_start = patch_date - pd.Timedelta(days=WINDOW_DAYS)
        after_end = patch_date + pd.Timedelta(days=WINDOW_DAYS)
        previous_end = last_window_end.get(patch.appid)
        if previous_end is not None and before_start < previous_end:
            overlap_skipped += 1
            continue

        before = game_reviews.loc[
            (game_reviews["created"] >= before_start)
            & (game_reviews["created"] < patch_date)
        ]
        after = game_reviews.loc[
            (game_reviews["created"] >= patch_date)
            & (game_reviews["created"] < after_end)
        ]
        if len(before) < MIN_REVIEWS or len(after) < MIN_REVIEWS:
            insufficient += 1
            continue

        last_window_end[patch.appid] = after_end
        before_positive = float(before["voted_up"].mean() * 100)
        after_positive = float(after["voted_up"].mean() * 100)
        before_negative = 100.0 - before_positive
        after_negative = 100.0 - after_positive
        launch_date = launch_dates.get(patch.appid, pd.NaT)
        days_since_launch = (
            (patch_date - launch_date).total_seconds() / 86400
            if pd.notna(launch_date) else np.nan
        )
        if before_positive < 70:
            stratum = "<70%"
        elif before_positive <= 90:
            stratum = "70-90%"
        else:
            stratum = ">90%"

        records.append({
            "appid": patch.appid,
            "patch_id": patch.gid,
            "patch_date": patch_date,
            "patch_title": patch.title,
            "launch_date": launch_date,
            "days_since_launch": days_since_launch,
            "launch_group": launch_group(days_since_launch),
            "before_reviews": len(before),
            "after_reviews": len(after),
            "before_positive_pct": before_positive,
            "after_positive_pct": after_positive,
            "positive_pct_change": after_positive - before_positive,
            "before_negative_pct": before_negative,
            "after_negative_pct": after_negative,
            "negative_pct_change": after_negative - before_negative,
            "review_volume_change": len(after) - len(before),
            "review_volume_pct_change": (len(after) - len(before)) / len(before) * 100,
            "before_positive_stratum": stratum,
        })

    RES.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results_path = RES / "patch_impact_results.csv"
    results.to_csv(results_path, index=False)

    if results.empty:
        print("No patch comparisons met the review-count requirement.")
        return

    # Average within each game before calculating the overall estimate.
    game_level = results.groupby("appid", as_index=False).agg(
        patch_comparisons=("positive_pct_change", "size"),
        mean_positive_pct_change=("positive_pct_change", "mean"),
        mean_negative_pct_change=("negative_pct_change", "mean"),
        mean_review_volume_change=("review_volume_change", "mean"),
        mean_review_volume_pct_change=("review_volume_pct_change", "mean"),
    )
    ci_low, ci_high = bootstrap_mean_ci(game_level["mean_positive_pct_change"])
    summary = pd.DataFrame([{
        "games": len(game_level),
        "eligible_patch_events": len(results),
        "mean_game_level_positive_pct_change": game_level["mean_positive_pct_change"].mean(),
        "median_game_level_positive_pct_change": game_level["mean_positive_pct_change"].median(),
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "random_seed": RANDOM_SEED,
        "mean_game_level_negative_pct_change": game_level["mean_negative_pct_change"].mean(),
        "mean_game_level_review_volume_change": game_level["mean_review_volume_change"].mean(),
        "mean_game_level_review_volume_pct_change": game_level["mean_review_volume_pct_change"].mean(),
    }])
    game_level.to_csv(RES / "patch_impact_game_level.csv", index=False)
    summary.to_csv(RES / "patch_impact_summary.csv", index=False)

    launch_summary = results.groupby("launch_group", as_index=False).agg(
        patch_comparisons=("positive_pct_change", "size"),
        games=("appid", "nunique"),
        mean_positive_pct_change=("positive_pct_change", "mean"),
        median_positive_pct_change=("positive_pct_change", "median"),
        mean_review_volume_change=("review_volume_change", "mean"),
        mean_review_volume_pct_change=("review_volume_pct_change", "mean"),
    )
    launch_summary.to_csv(RES / "patch_impact_by_launch.csv", index=False)

    strata_summary = results.groupby("before_positive_stratum", as_index=False).agg(
        patch_comparisons=("positive_pct_change", "size"),
        games=("appid", "nunique"),
        mean_before_positive_pct=("before_positive_pct", "mean"),
        mean_positive_pct_change=("positive_pct_change", "mean"),
        median_positive_pct_change=("positive_pct_change", "median"),
        mean_review_volume_change=("review_volume_change", "mean"),
    )
    strata_summary.to_csv(RES / "patch_impact_by_before_positivity.csv", index=False)

    create_plots(results)
    print(f"Reviews loaded (required columns only): {len(reviews):,}")
    print(f"Capped games excluded: {len(capped_ids):,}")
    print(f"Deduplicated patch announcements: {len(patches):,}")
    print(f"Accepted non-overlapping comparisons: {len(results):,}")
    print(f"Games represented: {len(game_level):,}")
    print(f"Overlapping comparisons skipped: {overlap_skipped:,}")
    print(f"Insufficient-review comparisons skipped: {insufficient:,}")
    print(f"Patch announcements without review data: {no_reviews:,}")
    print("\nGame-level summary:")
    print(summary.to_string(index=False))
    print("\nBy time since launch:")
    print(launch_summary.to_string(index=False))
    print("\nBy pre-patch positivity:")
    print(strata_summary.to_string(index=False))
    print("\nResults:", results_path.relative_to(ROOT))
    print("Tables:", RES.relative_to(ROOT))
    print("Figures:", FIG.relative_to(ROOT))
    print("\nInterpretation: these are descriptive associations, not causal patch effects.")
    print("Negative-share change is the inverse of positive-share change for binary reviews.")
    print("Refund-window share is unavailable because playtime is not read from reviews_clean.")


if __name__ == "__main__":
    main()
