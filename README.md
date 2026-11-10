# IndieGuard: Indie Game Launch Analytics

Predicting negative-review risk and identifying winning feature combinations on Steam.
Business Analytics capstone (23CSE452, Units 1 to 3), **Team 1**.

| | |
| --- | --- |
| Final report | [`Final_Project_Report.pdf`](Final_Project_Report.pdf) |
| Final presentation | [`Final_Project_Presentation.pptx`](Final_Project_Presentation.pptx) |
| Live dashboard | <https://indieguard.streamlit.app> |
| Project board | <https://github.com/users/Shan713/projects/1> |

## Team

| Member | Register number | Main contribution |
| --- | --- | --- |
| Thotakura Srihitha | CB.SC.U4CSE23450 | Exploratory analysis (29 figures, association tests); dashboard wireframe and downloads; grid-search tuning |
| Gayas Sheik | CB.SC.U4CSE23118 | Feature engineering and tag PCA; patch impact with matched controls; launch-risk scorer; reproducibility |
| Shantharam P | CB.SC.U4CSE23146 | Data collection and cleaning; split and targets; grid-search tuning; dashboard and deployment |
| Pradeep Sasidharan | CB.SC.U4CSE23232 | Model development (Logistic Regression, Random Forest, XGBoost; SHAP); model comparison; association rules |
| Valikala Tejaswini | CB.SC.U4CSE23752 | Text mining and topic modelling; combined business insights |

Guide: Rohini S. Dates, pull requests and board cards for each member are in Section 8.2 and Appendix D of the report.

## The business problem

A small indie studio has little room for a bad launch. Steam refunds a game bought in the last 14 days if it was played for under two
hours, and early impressions shape a game's rating. Refunds are not public, so we study **negative reviews written in that first
two-hour window as a proxy for launch risk**. We ask what drives negative reviews, which feature combinations go with a strong
reception, what players complain about, and whether patches help.

## Data

All data was collected by our own Python scrapers from public Steam endpoints and SteamSpy on 1 and 2 October 2026. No ready-made
dataset was used.

- **1,861** indie roguelike and roguelite games with a store release date in 2022 to 2025 and at least 10 reviews
- **1,652,999** reviews in 31 languages (1,648,387 after cleaning; 10.4% negative)
- **45,183** news items, of which 29,918 are patch or sale announcements
- Store details, top-20 user tags, and SteamSpy owner estimates for every game

Raw responses are kept off GitHub (frozen with a SHA-256 hash per file: [`docs/data_manifest.md`](docs/data_manifest.md)); the cleaned
and feature tables are in [`data/processed/`](data/processed), samples in [`data/sample/`](data/sample). Reviewer IDs are replaced by a
salted hash. Sources, collection method and data dictionary: [`docs/data_documentation.md`](docs/data_documentation.md); every cleaning
rule with before and after counts: [`docs/cleaning_log.md`](docs/cleaning_log.md).

## Methods

| Unit | What we did |
| --- | --- |
| Unit 1 | Exploratory analysis and visualisation of reviews and games (refund window, price, tags, store features) |
| Unit 2 | Review-level (negative or not) and game-level (Struggling / Solid / Strong tier) models: a Dummy baseline, Logistic Regression, Random Forest, LightGBM and XGBoost, tuned by grid search on 5 shared folds; data split by game (80/20), test set scored once; SHAP |
| Unit 3, method 1 | Text mining: TF-IDF + Logistic Regression to detect negative reviews; NMF topic model (8 complaint themes) compared with LDA |
| Unit 3, method 2 | Association rule mining: FP-Growth over tags, price band and Early Access, with false-discovery control and a held-out check |
| Extra | Patch impact against matched games that did not patch |
| Application | Streamlit dashboard: market explorer, tag rules and complaint themes, patch impact, and a launch-risk scorer that explains its verdict and suggests changes |

## Key results

- Reviews written in the **first two hours are 21.2% negative, against 8.8% later**, and make up 29% of all negative reviews. Inside the
  window, negativity rises with price, from 11% under $5 to 32% at $20 or more.
- The **text detector** finds negative reviews well: PR-AUC **0.84** and ROC-AUC **0.97** on reviews of unseen games (guessing: 0.12).
- **Store data alone predicts the game tier only modestly**: tuned XGBoost reaches macro-F1 **0.437** against 0.22 for guessing.
- **95 tag combinations** pass false-discovery control for the Strong tier and about two-thirds of the testable ones hold on unseen
  games; focused 2D, pixel, retro and procedural scopes reach Strong about 60% of the time (35% overall). No combination marks a struggling launch.
- **Complaints change with playtime**: early negative reviews lean to multiplayer issues and crashes, later ones to boss design and difficulty.
- **Patches show no effect of their own**: +0.27 points against matched games without a patch (95% interval -0.82 to +1.36).

All results are associations in observational data.

## Recommendations for a studio (hypotheses to test)

1. Treat the first hour as a standalone product: blind-test it, check controls, enable crash reporting.
2. Match the price to what the first hour delivers.
3. Test a focused 2D, pixel, retro or procedural scope before committing to multiplayer or 3D.
4. Test whether a complete store page (achievements, controller support) helps.
5. Patch specific problems, not to lift sentiment in general.

## Repository layout

```text
Final_Project_Report.pdf      final report             Final_Project_Presentation.pptx   final presentation
data/processed/               cleaned tables, the shared split, feature tables      data/sample/   small samples
notebooks/                    Units 1 to 3 (see below)
src/collect/                  scrapers and the raw-data manifest      src/clean/      cleaning pipeline
src/splits/                   the shared train/test split             src/features/   features, tag PCA, text SVD, targets, checks
src/models/                   models, grid search, model comparison   src/analysis/   text mining, association rules, patch impact
src/dashboard/                the Streamlit app (pointer: dashboard/) models/         the saved launch-risk model
figures/                      all charts (pointer: visualizations/)    reports/results/   result tables
docs/                         documentation (data, cleaning, split, features, tuning, dashboard)
```

## Notebooks

| Notebook | Content |
| --- | --- |
| [`notebooks/02_EDA_Visualizations.ipynb`](notebooks/02_EDA_Visualizations.ipynb) | Unit 1: exploratory analysis and visualisation |
| [`notebooks/review1/IndieGuard_Review1.ipynb`](notebooks/review1/IndieGuard_Review1.ipynb) | Unit 2: data, split, baselines and the five model families (with a tuning addendum) |
| [`notebooks/03_Text_Mining_and_Topic_Modelling.ipynb`](notebooks/03_Text_Mining_and_Topic_Modelling.ipynb) | Unit 3, method 1: text mining and topic modelling |
| [`notebooks/review2/association_rules.ipynb`](notebooks/review2/association_rules.ipynb) | Unit 3, method 2: association rules |
| [`notebooks/review2/patch_impact.ipynb`](notebooks/review2/patch_impact.ipynb) | Extra: patch impact with matched controls |

## Reproduce

Python 3.12. From the repository root:

```bash
pip install -r requirements.txt

# data (needs internet; about 12 hours for the reviews; raw files stay local)
python -m src.collect.universe --min-reviews 10
python -m src.collect.metadata
python -m src.collect.reviews
python -m src.collect.manifest

# cleaning, split, features and the automated checks
python -m src.clean
python -m src.splits.make_split
python -m src.features
python -m src.features.validate

# models, tuning and comparison
python -m src.models.review_models
python -m src.models.game_models
python -m src.models.review_text_models
python -m src.models.grid_search --task game --model xgboost   # one search per task and model: docs/hyperparameter_tuning_plan.md
python -m src.models.compare

# Unit 3 and patch impact
python -m src.analysis.text_mining
python -m src.analysis.text_evaluation
python -m src.analysis.association_rules
python -m src.analysis.patch_impact
python -m src.analysis.patch_impact_controls

# dashboard
python -m src.dashboard.build_data
streamlit run src/dashboard/app.py
```

The cleaned data and features are committed, so everything from the split onward runs without the raw data. The dashboard is
deployed on Streamlit Community Cloud from `main` (main file `src/dashboard/app.py`, dependencies `src/dashboard/requirements.txt`);
see [`docs/dashboard.md`](docs/dashboard.md).

## Limitations

- Associations, not causes: the data is observational.
- One niche and one snapshot (October 2026); results may not carry over to other genres.
- Revenue and refunds are not public, so early negative reviews are a proxy for launch risk.
- About 60% of reviews are not in English; the text models use English only.
- Small game-level data (1,191 training games): game-tier prediction is modest, and tuning overfitted for two models.
- Five very large games are capped at their newest ~15,000 reviews and left out of the review-level models.

## Project management

Work was tracked on a public GitHub Projects board (<https://github.com/users/Shan713/projects/1>) with the stages Backlog, To Do,
In Progress, Review/Testing and Completed. Each task had one owner, a due date and evidence, and was delivered through a pull request
reviewed by a teammate. Generative-AI use and how it was verified are disclosed in Section 8.3 and Appendix D of the report.
