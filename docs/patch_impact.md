# Patch Impact Analysis

## Objective

This analysis examines changes in Steam review sentiment and review volume around patch announcements. It is descriptive and observational; it does not establish that patches cause changes in player sentiment.

## Method

Patch announcements are processed chronologically within each game. Repeated announcements sharing the same game, timestamp, and title are deduplicated. Games with capped review histories are excluded.

Each eligible patch is compared using a 30-day window before and after its announcement. Both windows must contain at least 20 reviews. Overlapping accepted patch-comparison windows within a game are skipped.

Positive-review share is calculated from `voted_up`. Negative-review share is its complement. Review-volume change is the difference between the number of reviews in the two windows; percentage change is also reported.

Uncertainty for the main estimate is calculated by bootstrapping games rather than treating individual patch comparisons as independent.

## Main findings

The analysis produced **1,685 eligible patch comparisons across 617 games**.

The mean of the per-game average change in positive-review share was **-2.06 percentage points**, with a 95% game-level bootstrap confidence interval of **[-2.75, -1.37] percentage points**.

The corresponding mean change in negative-review share was +2.06 percentage points. The mean per-game review-volume change was approximately +85 reviews, with a mean percentage change of approximately +165%. Review-volume changes are descriptive and can vary substantially across games.

## Confounding checks

### Time since launch

| Time group | Comparisons | Mean positivity change |
|---|---:|---:|
| 30 days or less, including pre-launch | 617 | -1.90 pp |
| 31–180 days | 317 | -0.76 pp |
| More than 180 days | 751 | -0.36 pp |

The larger decline around launch suggests that launch-related review surges and Early Access activity may influence the overall association.

### Pre-patch positivity

| Pre-patch positive share | Comparisons | Mean positivity change |
|---|---:|---:|
| Below 70% | 105 | +9.45 pp |
| 70–90% | 757 | -0.05 pp |
| Above 90% | 823 | -3.21 pp |

The opposing changes across pre-patch positivity groups are consistent with regression to the mean being a potential explanation. These stratified results do not isolate a causal patch effect.

## No-patch control sensitivity analysis

The supplementary script `src/analysis/patch_impact_controls.py` compares eligible patch windows with observed windows containing no recorded patch announcement within 30 days of the control anchor.

Control windows use the same 30-day before-and-after windows and require at least 20 reviews on each side. Controls are selected within the same game, subject to three matching tolerances:

- **Calendar time:** control anchor within 90 days of the matched patch anchor.
- **Time since launch:** no more than 90 days apart.
- **Pre-window positivity:** no more than 10 percentage points apart.

Control windows must not overlap eligible patch windows or previously selected control windows.

The corrected analysis compares only patch windows that actually received controls. It averages matched patch-window and control-window changes within each game, then bootstraps games rather than individual windows.

The latest run produced:

| Metric | Result |
|---|---:|
| Eligible patch windows | 1,685 |
| Matched patch windows | 282 |
| Selected control windows | 282 |
| Games with eligible patch windows | 617 |
| Games with selected controls | 164 |
| Mean patch-window change | -0.536 pp |
| Mean control-window change | -0.806 pp |
| Mean paired patch-minus-control difference | +0.270 pp |
| Median paired difference | +0.353 pp |
| Paired game-level bootstrap 95% CI | [-0.824, +1.357] pp |
| Maximum calendar gap | 90 days |
| Mean absolute calendar gap | 69.61 days |

The estimated difference is small relative to its uncertainty, and the confidence interval includes zero. This sensitivity analysis does not establish a clear difference between patch and control windows in the matched subset.

Only 164 of 617 games had selected controls. The 282 controls were concentrated in games more than 180 days after launch (210 controls); 65 were 31–180 days after launch, and 7 were within 30 days of launch or before launch. Consequently, the matched subset is not evenly representative of all launch periods.

Calendar-time matching reduces differences in observation timing but cannot eliminate calendar effects, unrecorded updates, or other confounding. The analysis remains descriptive, not causal.

## Under-2-hour playtime share

The processed review-feature dataset contains **1,573,436 reviews**. Of these, **1,573,424 have known playtime** and 12 have unknown playtime.

Among reviews with known playtime, **228,899 reviews have less than 120 minutes of playtime**, representing **14.55%**.

This is a descriptive playtime metric, not an actual refund rate. Playtime below two hours does not establish that a player requested or received a refund. The metric is calculated from `data/processed/features/review_features` and is reported separately from the patch-window analysis.

The under-2-hour playtime share is not currently included as a comparison between patch and control windows.

## Outputs

The main script `src/analysis/patch_impact.py` generates:

- `reports/results/patch_impact_results.csv` — patch-level comparisons.
- `reports/results/patch_impact_game_level.csv` — one row per game.
- `reports/results/patch_impact_summary.csv` — overall game-level estimate and bootstrap confidence interval.
- `reports/results/patch_impact_by_launch.csv` — results grouped by time since launch.
- `reports/results/patch_impact_by_before_positivity.csv` — results stratified by pre-patch positivity.
- `figures/findings/patch_impact_change_distribution.png` — distribution of changes.
- `figures/findings/patch_impact_by_launch.png` — changes by time since launch.
- `figures/findings/patch_impact_by_before_positivity.png` — changes by pre-patch positivity.

The control-analysis script `src/analysis/patch_impact_controls.py` generates:

- `reports/results/patch_impact_control_patch_windows.csv` — eligible patch windows used by the control analysis.
- `reports/results/patch_impact_control_windows.csv` — selected no-patch control windows, including matching diagnostics.
- `reports/results/patch_impact_control_paired_games.csv` — game-level paired comparison.
- `reports/results/patch_impact_control_summary.csv` — control-comparison summary.

The review-feature dataset used for the under-2-hour playtime calculation is `data/processed/features/review_features`.

The review-two notebook is `notebooks/review2/patch_impact.ipynb`, and the analysis documentation is `docs/patch_impact.md`.

## Limitations

These are before-and-after associations, not causal estimates. Launch timing and regression to the mean may explain part of the observed decline. The bootstrap confidence intervals reflect variation across games in this dataset; they do not remove confounding.

The control analysis is limited to games with suitable observed control windows. Matching uses calendar time, time since launch, and pre-window positivity, but cannot eliminate unrecorded updates or other confounding. Only 164 of 617 games had selected controls, and the controls were concentrated more than 180 days after launch. The control estimate applies to this matched subset.

Review-volume changes are descriptive and do not establish whether a patch increased or decreased engagement. The under-2-hour playtime share is not a measure of actual refunds and is not included in the patch-window comparison.

The results should be interpreted as descriptive evidence about sentiment changes around recorded patch announcements, not as evidence that patches cause those changes.