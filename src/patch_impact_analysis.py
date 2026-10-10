
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Resolve paths from the repository root.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "patch_impact"

WINDOW_DAYS = 30
MIN_REVIEWS = 20


def create_plots(results_df, output_dir):
    """Generate visualizations for patch-impact results."""

    if results_df.empty:
        print("No eligible results available for plotting.")
        return

    sns.set_theme(style="whitegrid")

    # Plot 1: Distribution of positivity changes.
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=results_df,
        x="positive_pct_change",
        bins=50,
        kde=True,
    )
    plt.axvline(
        0,
        linestyle="--",
        color="red",
        label="No change",
    )
    plt.axvline(
        results_df["positive_pct_change"].median(),
        linestyle=":",
        color="black",
        label="Median",
    )
    plt.xlabel("Change in positive reviews (percentage points)")
    plt.ylabel("Number of patch events")
    plt.title("Change in Review Positivity After Patches")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "positivity_change_distribution.png",
        dpi=300,
    )
    plt.close()

    # Plot 2: Positivity before versus after each patch.
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=results_df,
        x="before_positive_pct",
        y="after_positive_pct",
        alpha=0.25,
        s=18,
    )
    plt.plot(
        [0, 100],
        [0, 100],
        "--",
        color="red",
        label="No change",
    )
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("Positive reviews before patch (%)")
    plt.ylabel("Positive reviews after patch (%)")
    plt.title("Review Positivity: Before vs. After Patches")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "positivity_before_after.png",
        dpi=300,
    )
    plt.close()

    print("Plots saved:")
    print(" - positivity_change_distribution.png")
    print(" - positivity_before_after.png")


def main():
    print("Loading datasets...")

    reviews = pd.read_parquet(DATA / "reviews_clean")
    events = pd.read_parquet(DATA / "events.parquet")
    games = pd.read_parquet(DATA / "games_clean.parquet")

    # Exclude games with capped review histories.
    capped_ids = set(
        games.loc[
            games["reviews_capped"].fillna(False),
            "appid",
        ]
    )
    reviews = reviews.loc[
        ~reviews["appid"].isin(capped_ids)
    ].copy()

    patches = events.loc[
        events["event_type"] == "patch"
    ].copy()

    # Normalize timestamps to UTC.
    reviews["created"] = pd.to_datetime(
        reviews["created"], utc=True
    )
    patches["date"] = pd.to_datetime(
        patches["date"], utc=True
    )

    # Remove records with missing required fields.
    reviews = reviews.dropna(
        subset=["appid", "created", "voted_up"]
    )
    reviews["voted_up"] = reviews["voted_up"].astype(bool)

    # Index reviews by game to avoid repeatedly filtering the full dataset.
    reviews = reviews.sort_values(["appid", "created"])
    reviews_by_game = {
        appid: group.reset_index(drop=True)
        for appid, group in reviews.groupby("appid", sort=False)
    }

    print(f"Reviews after exclusions: {len(reviews):,}")
    print(f"Patch announcements: {len(patches):,}")
    print(f"Capped games excluded: {len(capped_ids):,}")

    results = []

    for patch in patches.itertuples(index=False):
        game_reviews = reviews_by_game.get(patch.appid)

        if game_reviews is None:
            continue

        patch_date = patch.date
        before_start = patch_date - pd.Timedelta(
            days=WINDOW_DAYS
        )
        after_end = patch_date + pd.Timedelta(
            days=WINDOW_DAYS
        )

        before = game_reviews.loc[
            (game_reviews["created"] >= before_start)
            & (game_reviews["created"] < patch_date)
        ]

        after = game_reviews.loc[
            (game_reviews["created"] >= patch_date)
            & (game_reviews["created"] < after_end)
        ]

        # Require sufficient reviews in both windows.
        if (
            len(before) < MIN_REVIEWS
            or len(after) < MIN_REVIEWS
        ):
            continue

        before_positive = before["voted_up"].mean() * 100
        after_positive = after["voted_up"].mean() * 100

        results.append(
            {
                "appid": patch.appid,
                "patch_id": patch.gid,
                "patch_date": patch_date,
                "patch_title": patch.title,
                "before_reviews": len(before),
                "after_reviews": len(after),
                "before_positive_pct": before_positive,
                "after_positive_pct": after_positive,
                "positive_pct_change": (
                    after_positive - before_positive
                ),
                "review_volume_change": (
                    len(after) - len(before)
                ),
                "review_volume_pct_change": (
                    (len(after) - len(before))
                    / len(before)
                    * 100
                ),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)

    # Save patch-level results.
    patch_file = OUTPUT / "patch_impact_results.csv"
    results_df.to_csv(patch_file, index=False)

    # Summarize eligible patch events.
    if not results_df.empty:
        changes = results_df["positive_pct_change"]

        summary = pd.DataFrame(
            [
                {
                    "eligible_patch_events": len(results_df),
                    "games_covered": results_df["appid"].nunique(),
                    "mean_positive_pct_change": changes.mean(),
                    "median_positive_pct_change": changes.median(),
                    "patches_with_improved_positivity": int(
                        (changes > 0).sum()
                    ),
                    "patches_with_declined_positivity": int(
                        (changes < 0).sum()
                    ),
                    "patches_with_no_change": int(
                        (changes == 0).sum()
                    ),
                    "mean_review_volume_change": (
                        results_df["review_volume_change"].mean()
                    ),
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "eligible_patch_events",
                "games_covered",
                "mean_positive_pct_change",
                "median_positive_pct_change",
                "patches_with_improved_positivity",
                "patches_with_declined_positivity",
                "patches_with_no_change",
                "mean_review_volume_change",
            ]
        )

    summary_file = OUTPUT / "patch_impact_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"\nEligible patch events: {len(results_df):,}")
    print(f"Results saved to: {patch_file.relative_to(ROOT)}")
    print(f"Summary saved to: {summary_file.relative_to(ROOT)}")

    if not summary.empty:
        print("\nSummary:")
        print(summary.to_string(index=False))

    # Generate plots from the saved analysis results.
    create_plots(results_df, OUTPUT)

    print(
        "\nInterpretation: these results describe associations "
        "before and after patch announcements. They do not prove "
        "that patches caused the observed changes."
    )


if __name__ == "__main__":
    main()
