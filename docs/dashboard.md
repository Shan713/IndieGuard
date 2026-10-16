# IndieGuard dashboard (card #43)

A Streamlit app with five pages. It reads small committed tables, not the 650 MB of review data.

| Page | What it shows | Reads |
| --- | --- | --- |
| Market explorer | 1,861 games: negative share, tier mix by price band, tags with the lowest and highest negative share, reviews over time; filters for year, price band, minimum reviews, tags | `data/dashboard/games.csv`, `monthly.csv` |
| Tag rules and complaints | Association rules (filter by stability, lift, tag), eight complaint themes by cohort, the refund-window heatmap | `reports/results/association_rules_strong.csv`, `topic_prevalence.csv`, `topic_terms.csv`, `data/dashboard/refund_*.csv` |
| Patch impact | Positive-share change around patches, by time since launch and pre-patch level, and the matched-control result | `reports/results/patch_impact_*.csv` |
| Launch-risk scorer | Describe a game (price, launch date, DLC, languages, achievements, platforms, controller support, tags, categories) or load an existing one; shows the predicted tier and scores | `models/launch_risk/`, `data/processed/features/tag_pca_model.npz` |
| About | The data, split, findings and limits in a few lines | none |

## Run locally

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

## Final data refresh

The app tables are rebuilt from the cleaned data with one command. Rerun it after the data or the targets change, then commit `data/dashboard/`.

```bash
python -m src.dashboard.build_data
```

## Deploy on Streamlit Community Cloud

1. Sign in at share.streamlit.io with the GitHub account that owns the repository and choose **Create app**.
2. Repository `Shan713/IndieGuard`, branch `main`, main file path `src/dashboard/app.py`.
3. Under **Advanced settings** select Python 3.12. The app installs `src/dashboard/requirements.txt` (slim: Streamlit, pandas, numpy, pyarrow, XGBoost, Altair).
4. Deploy. The first start takes a few minutes. Put the public URL in the README.

The repository is public and the app does not need secrets.

## How the scorer works

`src/dashboard/featurize.py` turns a game's raw metadata and tags into the 83 launch-time features, applying the tag PCA that was fitted on training games only. `src/dashboard/risk.py` scores them with the exported XGBoost model, then multiplies by the class scales chosen on out-of-fold predictions. The scores are class-scaled and are **not calibrated probabilities**. Cross-validation macro-F1 is about 0.43 against 0.22 for guessing, and Struggling games are hard to find (`docs/launch_risk_scorer.md`).

## Limits

Associations, not causes. Store attributes are snapshots from 1 to 2 October 2026. Five very large games are capped and show only their newest reviews. The scorer was fitted on training games, so a game from the training split looks better than a new one would.
