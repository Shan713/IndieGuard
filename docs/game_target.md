# Game-level target: success tier

The brief asks the game-level model to predict "which success tier the game will land in". This file defines the
tiers. They are **proposed by M1 and owned by M4**; change them only by pull request, and before anyone reports results.

Code: `src/features/targets.py`. Table: `data/processed/features/game_targets.parquet`.

```python
import pandas as pd
t = pd.read_parquet("data/processed/features/game_targets.parquet")
t = t[t.eligible == 1]                      # 1,490 games with a tier
train, test = t[t.split == "train"], t[t.split == "test"]
```

## Definition

Three ordered tiers, from Steam's all-language review totals at collection time (1 October 2026):

| Tier | Code | Rule | Games (train / test) | Share |
| --- | ---: | --- | --- | ---: |
| Struggling | 0 | 30% or more of the game's reviews are negative | 165 / 51 | 14.5% |
| Solid | 1 | in between | 605 / 147 | 50.5% |
| Strong | 2 | 10% or less negative (90%+ positive) | 421 / 101 | 35.0% |

- `high_risk` is the binary version: Struggling versus the rest (216 games, the same as the EDA notebook).
- **Eligible games:** at least 20 reviews, **1,490 of 1,861** (1,191 train, 299 test). The 371 games below 20 reviews
  get no tier, because their ratios are too noisy to label.
- **Boundaries use whole-number arithmetic** (`10 * negative >= 3 * total`). A float comparison such as
  `positive_share < 0.70` misclassifies games sitting exactly on a cut-off (five games here).
- The cut-offs are round numbers near Steam's own rating bands and were fixed in advance; nothing is fitted from the data.
- The capped games (Megabonk, Cult of the Lamb, Hades II, Balatro, Vampire Survivors) are included, because their
  Steam totals are complete.

## What the tiers mean, and do not mean

- **Reception only.** Reach (review count, owners) is not part of the tiers: it depends on how long a game has been
  out, and a "hit" tier (90%+ positive and 1,000+ reviews) would hold only 83 games.
- **Current reception, not launch-day reception.** Totals are as of 1 October 2026.
- **The Struggling share varies by release year** (20.6% in 2022, 13.7% in 2023, 17.2% in 2024, 9.8% in 2025). Part of that
  may be exposure time, since older games had longer to collect negative reviews, but the pattern is not a clean trend.
  `launch_year` is a feature; be careful when interpreting its effect.

## Cautions for modelling

- Only **51 Struggling games are in the test set**, so test results are noisy. Always report cross-validation
  (mean ± spread over the 5 folds in `cv_fold`) next to the test score, and prefer macro-F1 or balanced accuracy to accuracy.
- **Never use the outcome columns as features**: everything in `game_outcomes`, plus `steam_*`, `spy_*`,
  `recommendations_total` and `metacritic_score` from `games_clean`. They encode the label.
- A model for a launch decision uses launch-time features only; see `docs/modelling_protocol.md`.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Binary high-risk only (the EDA's) | Throws away the middle; it is already available as `high_risk` |
| Reception × reach ("Flop / Middle / Hit") | Only 83 hits, and reach is confounded by time since release |
| Quantile tiers (terciles) | Cut-offs would depend on the data; the fixed ones are easier to explain to a studio |

To change the rule, edit the constants in `src/features/targets.py`, run `python -m src.features --skip-text`, and
update this page and the class-balance table (`docs/feature_engineering/game_target_balance.csv`).
