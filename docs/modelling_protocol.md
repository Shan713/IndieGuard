# Shared modelling protocol

Everyone builds models for the same two tasks and compares them. The comparison is only fair if everyone uses the
same data, split, features and metrics, so this page fixes them. **Proposed by M1, owned by M4.** Change it by pull
request, and agree before anyone reports results.

## 1. The two tasks

| | Review level | Game level |
| --- | --- | --- |
| Question | Will this review be negative? | Which success tier will the game land in? |
| Label | `target_is_negative` in `review_features` | `tier_code` in `game_targets` (0 Struggling, 1 Solid, 2 Strong); also `high_risk` |
| Table | `data/processed/features/review_features` (1,573,436 reviews) | `game_features` joined to `game_targets` (1,490 games with a tier) |
| Rows to use | all (the 5 capped games are already left out) | `eligible == 1` |
| Models (brief) | Logistic Regression, Random Forest, XGBoost | the same |
| Main metric | **PR-AUC** (only 10.6% of reviews are negative) | **macro-F1** and balanced accuracy |
| Also report | ROC-AUC, per-game average (section 4) | per-tier recall, one-vs-rest AUC |

```python
import pandas as pd
rf = pd.read_parquet("data/processed/features/review_features")
gf = pd.read_parquet("data/processed/features/game_features.parquet")
gt = pd.read_parquet("data/processed/features/game_targets.parquet")
games = gf.merge(gt[["appid", "tier_code", "eligible"]], on="appid").query("eligible == 1")
train, test = rf[rf.split == "train"], rf[rf.split == "test"]            # same for games
```

## 2. Rules for everyone

1. **Use the shared split** (`split`, `cv_fold` in the tables, from `data/processed/splits/game_split.csv`). Do not make another.
2. **The test set is used once**, for the final comparison. Choose models, features and settings with cross-validation on the training part only.
3. **Cross-validate with the given folds**: for fold k, fit on train rows with `cv_fold != k` and validate on `cv_fold == k`. Report the **mean and spread (standard deviation) over the 5 folds**.
4. **Fit anything learned from data on the training part only**: scalers, encoders, imputers, bucket edges. The tag and text components are already fitted on training games (see `docs/feature_engineering.md`).
5. **Never use outcome or post-review columns as features** (list in section 3).
6. **Report the same table** (section 5) so results line up.
7. **Fix random seeds and share your code** in `src/models/`.

## 3. Which features each model may use

The feature catalog (`docs/feature_engineering/feature_catalog.csv`) gives every feature a group.

| Model | Features allowed |
| --- | --- |
| Game-level **launch-risk scorer** (what a studio can know before launch) | `launch_time` features of `game_features` only: price, launch, content, platforms, tag components `pc_*` |
| Game-level **post-launch monitor** | the above plus `post_launch`: `patches_first_30d`, `patches_first_90d`, `sales_first_90d` |
| **Review-level** model | `review_features` columns (groups `launch_time` and `at_review`), optionally joined with `game_features` (`launch_time`) and the text components `t_*` |

**Not features, ever:** `appid`, `recommendationid`, `split`, `cv_fold`, `reviews_capped`, the label columns, and any
`outcome_*`, `steam_*`, `spy_*`, `recommendations_total`, `metacritic_score`, `votes_*`, `comment_count`,
`weighted_vote_score`, `refunded`, `has_dev_response`. `language` is a string: encode it (for example one-hot the top
languages). Missing values: tree models can use them as they are; impute for linear models (see `docs/feature_engineering.md`, section 5).

## 4. Metrics and how to report them

- **Review level.** PR-AUC and ROC-AUC on the pooled reviews, **and the per-game average** (ROC-AUC computed within each
  test game with at least 200 reviews, then the median). A few huge games hold much of the data, so a pooled score alone can
  mislead. Optionally weight reviews so that no game counts for more than 5,000 of them, and say so.
- **Game level.** Macro-F1, balanced accuracy, recall for each tier, and one-vs-rest AUC for Struggling and Strong. With 51
  Struggling test games, add a bootstrap interval (resample test games, 1,000 times).
- **Class imbalance.** Use class weights or sample weights. Choose any decision threshold on the cross-validation folds, never on test.
- **Interpretation.** SHAP or permutation importance for the best model, as the brief asks, and say what it means for a studio.

## 5. Table every member reports

| Model | Task | Feature set | CV (mean ± sd, 5 folds) | Test | Notes |
| --- | --- | --- | --- | --- | --- |
| (name) | review / game | launch-risk / monitor / review | main metric | main metric | settings, seed |

## 6. Reference baselines to beat

Quick, untuned models run with exactly the rules above, so a real model should beat them.

| Task | Baseline | Result |
| --- | --- | --- |
| Review | always predict the majority class | PR-AUC = base rate, **0.106** |
| Review | gradient boosting, 300k training reviews, all features except the text components | test PR-AUC **0.275**, ROC-AUC **0.725**; CV PR-AUC 0.263 ± 0.032; within-game ROC-AUC (median) 0.732 |
| Game, 3 tiers | always predict "Solid" | macro-F1 **0.220**, balanced accuracy 0.333 |
| Game, 3 tiers | random forest on the 83 launch-time features | test macro-F1 **0.412**, balanced accuracy 0.417; CV macro-F1 0.415 ± 0.025; recall Struggling 0.12, Solid 0.62, Strong 0.51; one-vs-rest AUC Struggling 0.68, Strong 0.64 |

**Expect modest scores, especially at game level.** Launch-time features give some signal, but a model finds only
about one Struggling game in eight. That is a finding, not a failure: present it honestly, and focus on which features
matter and on calibrated risk rather than on accuracy.

## 7. Known caveats (details in `docs/feature_engineering.md`, section 7)

- The tag and text components were fitted on all training games, so they leak slightly into each cross-validation fold.
  For a strict check, refit them inside each fold.
- `author_num_games_owned` and `author_num_reviews` were captured at collection time; drop them for a strict version.
- The text components are not committed: run `python -m src.features.text` once (about a minute).
- Review-level results are dominated by a few very large games; see the per-game average above.
- Reception tiers reflect Steam totals on 1 October 2026, and older games had longer to collect negative reviews.
