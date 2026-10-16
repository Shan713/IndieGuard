# Review 2 submission index

Where each required item is in the repository. Repository: https://github.com/Shan713/IndieGuard; board: https://github.com/users/Shan713/projects/1.

| Required item | Where |
| --- | --- |
| Dataset description | `docs/data_documentation.md` |
| Sample dataset | `data/sample/games_sample.csv`, `data/sample/reviews_sample.csv` |
| Final dataset | `data/processed/` (`games_clean.parquet`, `reviews_clean/`, `events.parquet`, `features/`) |
| Data dictionary | `docs/data_documentation.md` section 5; feature catalogue `docs/feature_engineering/feature_catalog.csv` |
| Validation summary | `docs/validation_summary.md`, `docs/cleaning_log.md` |
| Notebooks and source code | `notebooks/review1/IndieGuard_Review1.ipynb` (Units 1 and 2), `notebooks/03_Text_Mining_and_Topic_Modelling.ipynb` (Unit 3, text mining), `notebooks/review2/association_rules.ipynb` (Unit 3, association rules), `notebooks/review2/patch_impact.ipynb`, source in `src/` |
| Evaluation results and comparative analysis | `reports/results/` (for example `comparison_*.csv`, `tuning_*.csv`, `text_evaluation_metrics.csv`, `association_rule_configuration_comparison.csv`); `docs/model_comparison.md`, `docs/text_mining_and_evaluation.md` |
| Execution instructions | `README.md` (setup), `scripts/run_all.ps1` (features and validation), `docs/hyperparameter_tuning_plan.md`, `docs/dashboard.md` |
| Working dashboard | `streamlit run src/dashboard/app.py`; deployment steps in `docs/dashboard.md` |
| Final report | `docs/final_report.docx`, `docs/final_report.md` |
| Presentation slides | `slides/review2.pptx` |
| Project board, task history and contribution summary | board link above; `docs/final_report.md` section 15; slide 16 of the deck |

## The two selected Unit 3 methods

1. **Text mining**: topic modelling (NMF against LDA) and sentiment classification of negative reviews (TF-IDF Logistic Regression against metadata and compressed-text models), evaluated on held-out games with precision, recall, F1 and a confusion matrix.
2. **Association rule mining**: FP-Growth rules over tags, price band and Early Access, evaluated with support, confidence and lift on training games and checked on held-out test games, with six threshold and algorithm configurations compared.

Patch-impact analysis and the launch-risk scorer are additional analyses.
