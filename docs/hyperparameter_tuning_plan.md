# Hyperparameter tuning plan (reviewer comment from Review 1)

**Comment received:** the hyperparameters were fixed. We must tune them by grid search for every model we reported.

## What is tuned today

| Model | Review level | Game level | Text (English reviews) |
| --- | --- | --- | --- |
| Logistic Regression | small grid, 25% of games | small grid | C in {0.3, 1, 3}; best was the edge value 3.0 |
| Random Forest | small grid, 25% of games | small grid | not run |
| LightGBM | small grid, 25% of games | small grid | fixed (31 leaves, 100 min child) |
| XGBoost | **default settings** | **default settings** | not run |
| Dummy | nothing to tune | nothing to tune | nothing to tune |

## Rules for every search (so results stay comparable)

1. Select on the **5 shared cross-validation folds inside train** only. The test set is not touched until the grid is finished.
2. Metric: **PR-AUC** (review level and text), **macro-F1** (game level). Ties go to the simpler setting.
3. Anything learned from data (imputer, scaler, vectoriser) is fitted inside each fold.
4. Boosting models may use early stopping on the validation fold; record the chosen number of rounds.
5. Seed 42. Record every grid point, not only the winner.
6. Then: refit on the full train, set the threshold (review) or class scales (game) from out-of-fold predictions as before, and score the test set **once**.
7. Review level uses all training rows. Random Forest may search on 50% of the training games (whole games, so folds stay grouped) and must say so.

## Files each search writes

- `reports/results/tuning_<task>_<model>.csv` with one row per grid point: `params`, `cv_mean`, `cv_sd`, `fit_seconds`, `rank`. Tasks: `review`, `game`, `reviewtext`.
- The usual `reports/results/<task>_<model>.csv` row, updated with `tuned (grid)` and the chosen parameters.
- One line in the issue: default CV score, tuned CV score, and the test score.

## Grids

| Model | Review level | Game level |
| --- | --- | --- |
| Logistic Regression | C in {0.001, 0.01, 0.1, 1, 10}; class_weight in {none, balanced} (10) | C in {0.001, 0.01, 0.1, 1, 10, 100}; class_weight balanced; l1 and l2 (12) |
| Random Forest | min_samples_leaf in {5, 20, 50, 200}; max_features in {sqrt, 0.3, 0.5}; 300 trees (12) | min_samples_leaf in {1, 2, 3, 5, 10}; max_features in {sqrt, 0.3, 0.5}; n_estimators in {300, 600}; max_depth in {none, 8} (60) |
| LightGBM | num_leaves in {15, 31, 63, 127}; min_child_samples in {50, 200, 500}; learning_rate in {0.03, 0.1}; early stopping (24) | num_leaves in {4, 8, 16}; min_child_samples in {5, 10, 20}; learning_rate in {0.03, 0.1}; reg_lambda in {0, 1, 5} (54) |
| XGBoost | max_depth in {4, 6, 8}; min_child_weight in {1, 10, 50}; learning_rate in {0.03, 0.1}; subsample = colsample in {0.7, 0.9}; early stopping (36) | max_depth in {2, 3, 4, 6}; min_child_weight in {1, 5, 10}; learning_rate in {0.03, 0.1}; subsample in {0.7, 1.0}; gamma in {0, 1} (96) |

Text (English reviews, 436,715 train / 120,487 test):

| Model | Grid |
| --- | --- |
| Logistic Regression on TF-IDF + metadata | C in {0.3, 1, 3, 10, 30}; ngram_range in {(1,1), (1,2)}; max_features in {20k, 40k, 80k} (30). Search on 50% of the training games, then refit. |
| LightGBM on metadata, and on metadata + 78 text components | the review-level LightGBM grid above (24 each) |

## Size of the job

Measured on a 24-core machine, one fit on the full review-level train takes about 20 s for LightGBM and 23 s for XGBoost; a 100-tree Random Forest on 200k rows takes about 19 s and grows faster than linearly. With 5 folds per grid point:

| Search | Approx. time (24 cores) |
| --- | --- |
| Review XGBoost (36 points) | 1.2 h |
| Review Random Forest (12 points, 50% of games) | 2 h |
| Review LightGBM (24 points) | 40 min |
| Review Logistic Regression (10 points) | 25 min |
| Text Logistic Regression (30 points, 50% of games) | 3 h |
| Text LightGBM (2 x 24 points) | 1.5 h |
| All game-level searches (small data) | under 1 h in total |

On a laptop with 6 to 8 cores, multiply by 2 to 4. That is why the work is split.

## Split of the work (60:40)

| Owner | Work packages | Effort |
| --- | --- | --- |
| Srihitha | review Random Forest, review XGBoost, text Logistic Regression, game Random Forest, game XGBoost, game Logistic Regression | 10 of 17 (59%) |
| Shantharam | review Logistic Regression, review LightGBM, game LightGBM, text LightGBM, and the integration (results table, `compare.py`, docs, notebook, slides) | 7 of 17 (41%) |

Effort units: review RF 3, review XGBoost 3, text LR 2, review LightGBM 2, text LightGBM 1.5, integration 1.5, game XGBoost 1, game LightGBM 1, review LR 1, game RF 0.5, game LR 0.5.

## Done means

- Every reported model has a `tuning_*.csv` and a tuned row in its results file.
- `python -m src.models.compare` is rerun and `docs/model_comparison.md` shows default vs tuned.
- The notebook, slides and `docs/modelling_protocol.md` say the models were tuned by grid search on the shared folds.
