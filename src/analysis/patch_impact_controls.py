
"""Sensitivity analysis: patch windows vs. matched no-patch controls.

Run from the repository root:
    python -m src.analysis.patch_impact_controls

This is a descriptive sensitivity analysis, not a causal estimate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RES = ROOT / "reports" / "results"

WINDOW_DAYS = 30
MIN_REVIEWS = 20
CANDIDATE_STEP_DAYS = 7
BOOTSTRAP_REPS = 10_000
RANDOM_SEED = 42

# A control must satisfy all three matching tolerances.
MAX_LAUNCH_DAY_GAP = 90
MAX_CALENDAR_DAY_GAP = 90
MAX_PRE_POSITIVITY_GAP_PP = 10.0


def launch_group(days: float) -> str:
    if pd.isna(days):
        return "unknown"
    if days <= 30:
        return "<=30 days (including pre-launch)"
    if days <= 180:
        return "31-180 days"
    return ">180 days"


def load_data():
    reviews = pd.read_parquet(
        DATA / "reviews_clean",
        columns=["appid", "created", "voted_up"],
    )
    events = pd.read_parquet(DATA / "events.parquet")
    games = pd.read_parquet(
        DATA / "games_clean.parquet",
        columns=["appid", "launch_date", "reviews_capped"],
    )

    capped_ids = set(
        games.loc[games["reviews_capped"].fillna(False), "appid"]
    )
    reviews = reviews.loc[~reviews["appid"].isin(capped_ids)].copy()
    reviews["created"] = pd.to_datetime(
        reviews["created"], utc=True, errors="coerce"
    )
    reviews = reviews.dropna(subset=["appid", "created", "voted_up"])
    reviews["voted_up"] = reviews["voted_up"].astype(bool)
    reviews = reviews.sort_values(["appid", "created"])

    reviews_by_game = {
        appid: group.reset_index(drop=True)
        for appid, group in reviews.groupby("appid", sort=False)
    }

    patches = events.loc[events["event_type"].eq("patch")].copy()
    patches["date"] = pd.to_datetime(
        patches["date"], utc=True, errors="coerce"
    )
    patches = patches.dropna(subset=["appid", "date", "gid"])
    patches = patches.sort_values(["appid", "date", "gid"])
    patches = patches.drop_duplicates(
        ["appid", "date", "title"], keep="first"
    )
    patches = patches.sort_values(["appid", "date"]).reset_index(drop=True)

    launch_dates = games.set_index("appid")["launch_date"]
    launch_dates = pd.to_datetime(launch_dates, utc=True, errors="coerce")

    return reviews_by_game, patches, launch_dates


def measure_window(
    game_reviews: pd.DataFrame, anchor: pd.Timestamp
) -> dict | None:
    start = anchor - pd.Timedelta(days=WINDOW_DAYS)
    end = anchor + pd.Timedelta(days=WINDOW_DAYS)

    before = game_reviews.loc[
        (game_reviews["created"] >= start)
        & (game_reviews["created"] < anchor)
    ]
    after = game_reviews.loc[
        (game_reviews["created"] >= anchor)
        & (game_reviews["created"] < end)
    ]

    if len(before) < MIN_REVIEWS or len(after) < MIN_REVIEWS:
        return None

    before_pct = float(before["voted_up"].mean() * 100)
    after_pct = float(after["voted_up"].mean() * 100)

    return {
        "before_reviews": len(before),
        "after_reviews": len(after),
        "before_positive_pct": before_pct,
        "after_positive_pct": after_pct,
        "positive_pct_change": after_pct - before_pct,
    }


def overlaps(start, end, intervals) -> bool:
    return any(
        start < old_end and end > old_start
        for old_start, old_end in intervals
    )


def build_record(
    appid, anchor, measured, launch_date, kind, title=None
):
    days = (
        (anchor - launch_date).total_seconds() / 86400
        if pd.notna(launch_date)
        else np.nan
    )
    return {
        **measured,
        "appid": appid,
        "anchor_date": anchor,
        "anchor_type": kind,
        "days_since_launch": days,
        "launch_group": launch_group(days),
        "patch_title": title,
    }


def bootstrap_ci(values: pd.Series) -> tuple[float, float]:
    """Bootstrap the mean across games, treating games as independent units."""
    values = values.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(RANDOM_SEED)
    means = np.empty(BOOTSTRAP_REPS)

    for i in range(BOOTSTRAP_REPS):
        means[i] = rng.choice(
            values, size=len(values), replace=True
        ).mean()

    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    print("Loading review, event, and game data...")
    reviews_by_game, patches, launch_dates = load_data()

    patch_dates = {
        appid: pd.DatetimeIndex(group["date"].sort_values())
        for appid, group in patches.groupby("appid")
    }

    # Build eligible patch windows without overlapping patch comparisons.
    patch_records = []
    patch_intervals = {}
    last_patch_end = {}

    for patch in patches.itertuples(index=False):
        appid = patch.appid
        game_reviews = reviews_by_game.get(appid)
        if game_reviews is None:
            continue

        anchor = patch.date
        start = anchor - pd.Timedelta(days=WINDOW_DAYS)
        end = anchor + pd.Timedelta(days=WINDOW_DAYS)

        if appid in last_patch_end and start < last_patch_end[appid]:
            continue

        measured = measure_window(game_reviews, anchor)
        if measured is None:
            continue

        launch_date = launch_dates.get(appid, pd.NaT)
        patch_records.append(
            build_record(
                appid, anchor, measured, launch_date, "patch", patch.title
            )
        )
        patch_intervals.setdefault(appid, []).append((start, end))
        last_patch_end[appid] = end

    patches_df = pd.DataFrame(patch_records)
    print(f"Eligible patch windows: {len(patches_df):,}")

    if patches_df.empty:
        print("No eligible patch windows.")
        return

    controls = []
    rng = np.random.default_rng(RANDOM_SEED)

    for appid, target_rows in patches_df.groupby("appid", sort=True):
        game_reviews = reviews_by_game[appid]
        dates = game_reviews["created"]

        candidates = pd.date_range(
            dates.min().ceil("D"),
            dates.max().floor("D"),
            freq=f"{CANDIDATE_STEP_DAYS}D",
            tz="UTC",
        )
        if candidates.empty:
            continue

        known_dates = patch_dates.get(appid, pd.DatetimeIndex([]))
        candidate_records = []

        for anchor in candidates:
            start = anchor - pd.Timedelta(days=WINDOW_DAYS)
            end = anchor + pd.Timedelta(days=WINDOW_DAYS)

            # Exclude controls near any recorded patch announcement.
            if len(known_dates) and (
                abs(known_dates - anchor)
                <= pd.Timedelta(days=WINDOW_DAYS)
            ).any():
                continue

            if overlaps(start, end, patch_intervals.get(appid, [])):
                continue

            measured = measure_window(game_reviews, anchor)
            if measured is None:
                continue

            launch_date = launch_dates.get(appid, pd.NaT)
            candidate_records.append(
                build_record(
                    appid, anchor, measured, launch_date, "no_patch_control"
                )
            )

        # Match on calendar date, launch timing, and pre-window positivity.
        # Do not reuse controls or select overlapping control windows.
        selected_intervals = []
        used_anchors = set()
        targets = target_rows.sort_values("anchor_date")

        for target in targets.itertuples(index=False):
            eligible = []

            for candidate in candidate_records:
                anchor = candidate["anchor_date"]
                start = anchor - pd.Timedelta(days=WINDOW_DAYS)
                end = anchor + pd.Timedelta(days=WINDOW_DAYS)

                if anchor in used_anchors:
                    continue
                if overlaps(start, end, selected_intervals):
                    continue

                calendar_gap = abs(
                    (anchor - target.anchor_date).total_seconds() / 86400
                )
                if calendar_gap > MAX_CALENDAR_DAY_GAP:
                    continue

                target_days = target.days_since_launch
                candidate_days = candidate["days_since_launch"]

                # Unknown launch dates match only other unknown launch dates.
                if pd.isna(target_days) or pd.isna(candidate_days):
                    if not (
                        pd.isna(target_days) and pd.isna(candidate_days)
                    ):
                        continue
                    launch_gap = 0.0
                else:
                    launch_gap = abs(target_days - candidate_days)

                positivity_gap = abs(
                    target.before_positive_pct
                    - candidate["before_positive_pct"]
                )

                if launch_gap > MAX_LAUNCH_DAY_GAP:
                    continue
                if positivity_gap > MAX_PRE_POSITIVITY_GAP_PP:
                    continue

                score = (
                    calendar_gap / MAX_CALENDAR_DAY_GAP
                    + launch_gap / MAX_LAUNCH_DAY_GAP
                    + positivity_gap / MAX_PRE_POSITIVITY_GAP_PP
                )
                eligible.append(
                    (
                        score,
                        candidate,
                        calendar_gap,
                        launch_gap,
                        positivity_gap,
                    )
                )

            if not eligible:
                continue

            eligible.sort(key=lambda item: item[0])
            best_score = eligible[0][0]
            best_candidates = [
                item for item in eligible
                if np.isclose(item[0], best_score)
            ]
            chosen_item = best_candidates[
                int(rng.integers(0, len(best_candidates)))
            ]
            _, chosen, calendar_gap, launch_gap, positivity_gap = chosen_item

            anchor = chosen["anchor_date"]
            start = anchor - pd.Timedelta(days=WINDOW_DAYS)
            end = anchor + pd.Timedelta(days=WINDOW_DAYS)

            used_anchors.add(anchor)
            selected_intervals.append((start, end))

            selected = chosen.copy()
            selected["matched_patch_anchor_date"] = target.anchor_date
            selected["calendar_gap_days"] = calendar_gap
            selected["launch_day_gap"] = launch_gap
            selected["pre_positivity_gap_pp"] = positivity_gap
            controls.append(selected)

            candidate_records = [
                c for c in candidate_records if c["anchor_date"] != anchor
            ]

    controls_df = pd.DataFrame(controls)
    RES.mkdir(parents=True, exist_ok=True)

    patches_df.to_csv(
        RES / "patch_impact_control_patch_windows.csv", index=False
    )
    controls_df.to_csv(
        RES / "patch_impact_control_windows.csv", index=False
    )

    print(f"Selected no-patch controls: {len(controls_df):,}")
    print(f"Games with patch windows: {patches_df['appid'].nunique():,}")
    print(
        "Games with controls: "
        f"{controls_df['appid'].nunique() if not controls_df.empty else 0:,}"
    )

    if controls_df.empty:
        print("No eligible controls satisfy the matching tolerances.")
        return

    # Pair only patch windows that actually received controls.
    matched_patch_keys = controls_df[
        ["appid", "matched_patch_anchor_date"]
    ].drop_duplicates()

    matched_patches = patches_df.merge(
        matched_patch_keys,
        left_on=["appid", "anchor_date"],
        right_on=["appid", "matched_patch_anchor_date"],
        how="inner",
        validate="one_to_one",
    )

    if matched_patches.empty:
        print("No patch windows matched to controls.")
        return

    # Average matched windows within each game. The bootstrap resamples games,
    # not individual patch or control windows.
    patch_game = matched_patches.groupby("appid")[
        "positive_pct_change"
    ].mean()
    control_game = controls_df.groupby("appid")[
        "positive_pct_change"
    ].mean()

    common = patch_game.index.intersection(control_game.index)

    paired = pd.DataFrame({
        "patch_mean_change_pp": patch_game.loc[common],
        "control_mean_change_pp": control_game.loc[common],
    })
    paired["patch_minus_control_pp"] = (
        paired["patch_mean_change_pp"] - paired["control_mean_change_pp"]
    )

    ci_low, ci_high = bootstrap_ci(paired["patch_minus_control_pp"])

    summary = pd.DataFrame([{
        "matched_games": len(paired),
        "eligible_patch_windows": len(patches_df),
        "matched_patch_windows": len(matched_patches),
        "selected_control_windows": len(controls_df),
        "games_with_patch_windows": patches_df["appid"].nunique(),
        "games_with_controls": controls_df["appid"].nunique(),
        "mean_patch_change_pp": paired["patch_mean_change_pp"].mean(),
        "mean_control_change_pp": paired["control_mean_change_pp"].mean(),
        "mean_patch_minus_control_pp":
            paired["patch_minus_control_pp"].mean(),
        "median_patch_minus_control_pp":
            paired["patch_minus_control_pp"].median(),
        "paired_bootstrap_95_ci_low_pp": ci_low,
        "paired_bootstrap_95_ci_high_pp": ci_high,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "random_seed": RANDOM_SEED,
        "max_launch_day_gap": MAX_LAUNCH_DAY_GAP,
        "max_calendar_day_gap": MAX_CALENDAR_DAY_GAP,
        "max_pre_positivity_gap_pp": MAX_PRE_POSITIVITY_GAP_PP,
        "mean_matched_calendar_gap_days":
            controls_df["calendar_gap_days"].mean(),
        "interpretation":
            "descriptive sensitivity analysis, not a causal estimate",
    }])

    paired.to_csv(
        RES / "patch_impact_control_paired_games.csv", index_label="appid"
    )
    summary.to_csv(
        RES / "patch_impact_control_summary.csv", index=False
    )

    print("\nControl comparison summary:")
    print(summary.to_string(index=False))
    print("\nControl launch groups:")
    print(controls_df["launch_group"].value_counts().to_string())
    print("\nResults saved under reports/results/")
    print(
        "\nControls are matched observational windows. Calendar effects, "
        "unrecorded updates, and residual confounding may remain."
    )


if __name__ == "__main__":
    main()
