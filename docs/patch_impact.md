# Patch Impact Analysis

## Objective

This analysis examines changes in Steam review sentiment and review volume around patch announcements. It is descriptive and observational; it does not establish that patches cause changes in player sentiment.

## Method

Patch announcements are processed chronologically within each game. Repeated announcements sharing the same game, timestamp, and title are deduplicated. Games with capped review histories are excluded.

Each eligible patch is compared using a 30-day window before and after its announcement. Both windows must contain at least 20 reviews. Overlapping accepted comparison windows within a game are skipped.

Positive-review share is calculated from `voted_up`. Negative-review share is its complement. Review-volume change is the difference between the number of reviews in the two windows; percentage change is also reported.

## Main findings

The analysis produced 1,685 eligible patch comparisons across 617 games.

The mean of the per-game average change in positive-review share was **-2.06 percentage points**, with a 95% game-level bootstrap confidence interval of **[-2.75, -1.37] percentage points**. The bootstrap resamples games rather than treating individual patch comparisons as independent.

## Confounding checks

### Time since launch

| Time group | Comparisons | Mean positivity change |
|---|---:|---:|
| 30 days or less, including pre-launch | 617 | -1.90 pp |
| 31-180 days | 317 | -0.76 pp |
| More than 180 days | 751 | -0.36 pp |

The larger decline around launch suggests that launch-related review surges and Early Access activity may influence the overall association.

### Pre-patch positivity

| Pre-patch positive share | Comparisons | Mean positivity change |
|---|---:|---:|
| Below 70% | 105 | +9.45 pp |
| 70-90% | 757 | -0.05 pp |
| Above 90% | 823 | -3.21 pp |

The opposing changes across pre-patch positivity groups are consistent with regression to the mean being a potential explanation. These stratified results do not isolate a causal patch effect.

## Outputs

The script `src/analysis/patch_impact.py` generates:

- `reports/results/patch_impact_results.csv` - patch-level comparisons.
- `reports/results/patch_impact_game_level.csv` - one row per game.
- `reports/results/patch_impact_summary.csv` - overall game-level estimate and bootstrap confidence interval.
- `reports/results/patch_impact_by_launch.csv` - results grouped by time since launch.
- `reports/results/patch_impact_by_before_positivity.csv` - results stratified by pre-patch positivity.
- `figures/findings/patch_impact_change_distribution.png` - distribution of changes.
- `figures/findings/patch_impact_by_launch.png` - changes by time since launch.
- `figures/findings/patch_impact_by_before_positivity.png` - changes by pre-patch positivity.

## Limitations

These are before-and-after associations, not causal estimates. Launch timing and regression to the mean may explain part of the observed decline. The bootstrap confidence interval reflects variation across games in this dataset; it does not remove confounding.

Review-volume changes are descriptive and do not establish whether a patch increased or decreased engagement. Refund-window share is not calculated because the analysis reads only `appid`, `created`, and `voted_up` from the review dataset.

The current analysis does not include a no-patch control group. A controlled comparison would be needed to strengthen causal interpretation.
