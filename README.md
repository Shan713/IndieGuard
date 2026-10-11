# IndieGuard: Indie Game Launch Analytics

Predicting negative-review risk and identifying winning feature combinations on Steam. Business Analytics capstone (Units 1 to 3), team of 5.

**Business objective:** help a small indie studio or publisher decide which feature combinations to back, what to fix before the refund window closes, and when to patch.

**Project board:** <https://github.com/users/Shan713/projects/1> (private: if you cannot open it, ask Shantharam to add you). Every task has one owner, a due date and the evidence that proves it is done.

## Where we are

| Stage | State |
| --- | --- |
| Data collection | Done: 1,861 indie roguelike/roguelite games (2022 to 2025), 1,652,999 reviews, all collected by us from Steam and SteamSpy |
| Cleaning | Done: `docs/cleaning_log.md` |
| Train/test split | Done: by game, with 5 cross-validation folds |
| Features and dimensionality reduction | Done: `docs/feature_engineering.md` |
| Game-level target | Defined, awaiting team agreement: `docs/game_target.md` |
| EDA | Done: `notebooks/02_EDA_Visualizations.ipynb` |
| Models | Done for Review 1: Dummy baseline, Logistic Regression, Random Forest, LightGBM and XGBoost on both tasks, plus text-based review models: `docs/model_comparison.md` |
| Business findings | First version done: `docs/business_findings.md` |
| Problem statement | Done: `docs/problem_statement.md` |
| Review 1 notebook | Done: `notebooks/review1/IndieGuard_Review1.ipynb` (runs the analysis and models live, about 6 minutes) |
| **Next** | **Review 1 slides and rehearsal, contribution summaries, and the team's agreement on the game-level tiers. Then Review 2: association rules, topic modelling, patch-impact analysis and the dashboard** |

## Getting started

### Requirements

- Python 3.12
- Git
- Internet access to install Python dependencies

Run commands from the repository root.

### Setup

Clone the repository and enter its directory:

```powershell
git clone https://github.com/Shan713/IndieGuard.git
cd IndieGuard
```

Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Build features and validate

Run the complete feature-generation pipeline and validation script:

```powershell
.\scripts\run_all.ps1
```

The script builds the feature tables and runs the validation checks. It stops if feature generation or validation fails. A successful run should finish with all validation checks passing.

### Validate existing artifacts without rebuilding

To run validation only:

```powershell
python -m src.features.validate
```

The cleaned data and existing feature tables are stored in `data/processed/`. Raw scraped data is **not** included in the repository (it is frozen on M1's machine; counts and SHA256 hashes are in `docs/data_manifest.md`). Raw data is not required for this feature pipeline.

## The data you will use

```python
import pandas as pd
rf = pd.read_parquet("data/processed/features/review_features")        # 1,573,436 reviews, one row each
gf = pd.read_parquet("data/processed/features/game_features.parquet")  # 1,861 games, launch-time + post-launch features
gt = pd.read_parquet("data/processed/features/game_targets.parquet")   # game labels (tier_code 0/1/2, high_risk)
txt = pd.read_parquet("data/processed/features/text_svd")              # text components (after running the feature pipeline), join on recommendationid
```

| Task | Label | Table | Rows |
| --- | --- | --- | ---: |
| Review level: will this review be negative? | `target_is_negative` | `review_features` | 1,573,436 |
| Game level: which success tier? | `tier_code` (0 Struggling, 1 Solid, 2 Strong) | `game_features` + `game_targets`, keep `eligible == 1` | 1,490 |

Every table has `split` (`train` or `test`) and `cv_fold` (0 to 4 for train, -1 for test). **They come from one shared split by game; use them as they are.** More: `docs/train_test_split.md`, `docs/feature_engineering.md`, `docs/data_documentation.md` (data dictionary).

## Building a model: the steps

1. **Read `docs/modelling_protocol.md` (10 minutes).** It fixes the split, the features each model may use, the metrics and the table everyone reports. Results can only be compared if we all follow it.
2. **Pick a card on the board** (To Do), make sure it is assigned to you, and move it to *In Progress* the day you start.
3. **Branch from `main`:** `git checkout -b feature/<issue-number>-short-name`.
4. **Put your work in** `src/models/` (code) and `notebooks/review1/` (notebooks). Fix your random seeds. Do not commit files over 50 MB.
5. **Tune with cross-validation on the training part only**, using the given folds: for fold `k`, fit on `cv_fold != k` and validate on `cv_fold == k`. Report the mean and spread over the 5 folds.
6. **Touch the test set once**, at the end, for the final comparison.
7. **Save your results** as `reports/results/<task>_<model>.csv` with the columns of the table in the protocol (model, task, feature set, CV mean, CV sd, test score, notes).
8. **Open a pull request** that says `Closes #<issue>` and attach the evidence (notebook, plot, results file). **A teammate reviews and merges.** Then the card moves to *Completed*.

### The rules that matter most

- **Fit anything learned from data on training rows only** (scalers, encoders, imputers, bucket edges).
- **Never use outcome or after-the-fact columns as features.** Not allowed: `outcome_*`, `steam_*`, `spy_*`, `metacritic_score`, `recommendations_total`, `votes_*`, `comment_count`, `weighted_vote_score`, `refunded`, `has_dev_response`, and keys (`appid`, `recommendationid`, `split`, `cv_fold`, `reviews_capped`). The full list is in the protocol.
- **A launch-risk model uses only `launch_time` features.** Patch counts after launch (`patches_first_30d`, `patches_first_90d`, `sales_first_90d`) are for a post-launch monitor only. Each feature's group is in `docs/feature_engineering/feature_catalog.csv`.
- **Metrics:** PR-AUC for reviews (only 10.6% are negative), macro-F1 and balanced accuracy for games. Not accuracy.
- **Do not make another split.** Do not edit raw or cleaned data; propose changes by pull request.

### What to expect

Reference scores to beat are in the protocol. At game level, expect modest results: a random forest on launch-time features reaches macro-F1 0.41 (guessing the majority class gives 0.22) and finds only about 1 in 8 struggling games, and only 51 struggling games are in the test set. That is a finding, not a failure. Report it honestly and focus on which features matter (SHAP) and what it means for a studio.

## Repository layout

```text
data/processed/        cleaned tables, split, features (committed)   data/raw, data/interim: local only
src/collect/           scrapers and the raw-data manifest             src/clean/    cleaning pipeline
src/splits/            the shared train/test split                    src/features/ features, tag PCA, text SVD, game targets, validation
src/models/            your model code (create it)                    notebooks/    EDA now; models in notebooks/review1/
docs/                  documentation: start with the list below       figures/, reports/    charts and results
```

## Documentation

| File | What it explains |
| --- | --- |
| `docs/modelling_protocol.md` | **Rules for model development, metrics, reference baselines** |
| `docs/model_comparison.md` | Results of all five models on both tasks, plus the text-based review models, with how to read them |
| `docs/business_findings.md` | First business findings for a studio: refund window, complaint themes, what goes with struggling games |
| `docs/problem_statement.md` | The business problem, objective, questions, scope, success measures and hypotheses |
| `docs/game_target.md` | The three success tiers and why |
| `docs/feature_engineering.md` | Every feature table, how it was built, limitations |
| `docs/train_test_split.md` | The shared split and fold balance |
| `docs/data_documentation.md` | Sources, collection method, data dictionary, limitations, privacy |
| `docs/cleaning_log.md` | Each cleaning rule with before and after counts |
| `docs/data_manifest.md` | Raw data row counts and hashes |

## Team and ownership

M1 (Shantharam): data lead and board admin. M2 (Gayas): cleaning and features. M3 (Srihitha): EDA and dashboard. M4: modelling and association rules. M5: text mining and storytelling. The board is the source of truth for who owns what; M4 and M5 are not assigned yet.

## Known limitations (read before trusting a number)

- The tag and text components were fitted on all training games, so they leak very slightly into each cross-validation fold.
- Review-level results are dominated by a few very large games: always report the per-game average too.
- Five very large games (Megabonk, Cult of the Lamb, Hades II, Balatro, Vampire Survivors) have only their newest ~15,000 reviews. They are left out of the review-level tables and kept at game level.
- About 1% of reviews labelled English are in another language; a little remains in the text features.
- Details: `docs/data_documentation.md` (section 8) and `docs/feature_engineering.md` (section 7).

## Working agreement

1. Every piece of work is an issue with one owner, a due date and evidence.
2. Branch from `main` as `feature/<issue-number>-short-name`; open a pull request that says `Closes #<issue>`.
3. A teammate reviews and merges. Update your card the same day the work happens; the board history is used to verify individual contribution.
4. Raw data is never edited and never committed. Large files stay out of git.
