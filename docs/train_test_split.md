# Train / test split

Generated 2026-10-03T03:25:02+00:00 UTC by `python -m src.splits.make_split` (test seed 250, fold seed 71). File: `data/processed/splits/game_split.csv`.

**Every model in the team uses this split**, so results are directly comparable.

## Method

- **Unit: game.** All reviews of a game are on the same side, so no game-specific signal leaks into the test set.
- **Test: 20% of games**, held out. Use it **once**, for the final comparison. Never tune on it.
- **5 cross-validation folds** inside train (`cv_fold` 0-4), also by game, for tuning and model selection.
- **Stratified** by release year × review-volume quartile × rating tertile (Steam totals), so both sides have the same mix of small/large and liked/disliked games.
- **The 5 capped games are always in train** (`reviews_capped`): the test set contains only fully scraped games. For review-level models, drop them from train too if you use launch/patch timing features.
- **Review-level balance.** The top 20 games hold ~53% of all reviews, so a plain stratified split can put very different review volumes and negative rates in test. Among 500 stratified candidate splits, the one whose test set is closest to 20% of reviews with the same negative-review rate as train was kept (folds chosen the same way). This was decided before any model was trained.

## How to use

```python
import pandas as pd
split = pd.read_csv('data/processed/splits/game_split.csv')
reviews = pd.read_parquet('data/processed/reviews_clean').merge(split, on='appid')
train, test = reviews[reviews.split == 'train'], reviews[reviews.split == 'test']

# tuning: GroupKFold-style CV with the shared folds
for k in range(5):
    fit_part, val_part = train[train.cv_fold != k], train[train.cv_fold == k]
```

Anything learned from data (scalers, encoders, PCA/SVD, TF-IDF, bucket edges from quantiles) must be fitted on the training part only and then applied to the test part.

## Balance checks

| Part | Games | Reviews | Negative reviews | Text-mining reviews | Median price (USD) | Median game rating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,489 | 1,332,046 | 10.3% | 476,177 | 5.99 | 86.2% |
| test | 372 | 316,341 | 10.6% | 121,600 | 5.99 | 85.7% |
| train (excl. capped) | 1,484 | 1,257,095 | 10.6% | 442,361 | 5.99 | 86.2% |

Cross-validation folds (train only):

| Part | Games | Reviews | Negative reviews | Text-mining reviews | Median price (USD) | Median game rating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fold 0 | 298 | 273,193 | 10.3% | 100,578 | 5.99 | 85.7% |
| fold 1 | 298 | 221,781 | 10.9% | 71,489 | 5.99 | 85.8% |
| fold 2 | 298 | 288,576 | 6.9% | 109,293 | 6.99 | 85.3% |
| fold 3 | 298 | 249,716 | 11.3% | 98,310 | 5.99 | 86.3% |
| fold 4 | 297 | 298,780 | 12.3% | 96,507 | 5.99 | 87.5% |

Release year share (% of games in each part):

| Year | train | test |
| --- | ---: | ---: |
| 2022 | 15.4 | 15.1 |
| 2023 | 20.0 | 19.9 |
| 2024 | 29.9 | 30.1 |
| 2025 | 34.7 | 34.9 |

**Why the folds still differ at review level.** Three games dominate their folds: Dave the Diver (162k reviews, 3.5% negative) is ~56% of fold 2, Buckshot Roulette ~49% of fold 0 and Escape From Duckov ~41% of fold 4. A game cannot be split across folds without reintroducing leakage, and putting the three giants in three different folds is the best possible. So for review-level models:

- report CV results as **mean ± standard deviation across the 5 folds**, not a single fold;
- report **per-game (macro) metrics** next to pooled ones, so a few huge games do not decide the result;
- optionally weight reviews so each game contributes at most a fixed amount (e.g. `sample_weight` = min(1, 5000 / reviews_of_that_game)), and say so in the report.

The held-out **test** set is balanced at review level (see table above), so the final comparison is fair.
