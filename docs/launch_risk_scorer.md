# Launch-Risk Scorer

## Purpose
Predicts a game's review-reception tier from launch-time features:
- Struggling
- Solid
- Strong

This is a three-class prediction of review reception, not sales, revenue,
profitability, or guaranteed future performance.

## Requirements
Activate the project environment and run commands from the repository root.

The model uses 83 numeric launch-time features:
- 15 price, launch, content, platform, and publishing features
- 68 tag-component features (`pc_01` to `pc_68`)

The raw-game featurizer converts metadata and semicolon-separated tags and
categories into this schema. It applies the saved training-fitted tag PCA
model from `data/processed/features/tag_pca_model.npz`. The expected feature
order is stored in `models/launch_risk/feature_columns.json`.

## Raw-game inference
Provide these fields to `featurize_game` or `predict_raw_game`:
- `price_usd`, `is_free`
- `launch_date`, `release_date`, `launch_before_window`
- `n_dlc`, `n_supported_languages`, `n_achievements`
- `platform_mac`, `platform_linux`, `controller_support`
- `developers`, `publishers`
- `tags`, `categories`

Example:

    from src.dashboard.risk import predict_raw_game

    game = {
        "price_usd": 9.99,
        "is_free": False,
        "launch_date": "2020-01-06",
        "release_date": "2022-05-02",
        "launch_before_window": True,
        "n_dlc": 0,
        "n_supported_languages": 3,
        "n_achievements": 21,
        "platform_mac": False,
        "platform_linux": False,
        "controller_support": "full",
        "developers": "Example Studio",
        "publishers": "Example Studio",
        "tags": "Action Roguelike;Indie;Adventure",
        "categories": "Single-player;Steam Achievements",
    }

    result = predict_raw_game(game)
    print(result["tier"], result["scores"])

## Prepared-feature inference
For already processed features, use:

    from src.dashboard.risk import predict_launch_risk

Pass a dictionary with all 83 required features or a DataFrame with those
feature columns. A DataFrame returns one prediction per row.

Each result contains `tier`, `tier_code`, and `scores`, a set of normalized
class scores for Struggling, Solid, and Strong. The scores sum to 1, but they
are **not calibrated probabilities** and should not be interpreted as
confidence estimates.

## Model evaluation
The exporter uses five-fold cross-validation on the training split and selects
class scales from out-of-fold predictions. The reported out-of-fold macro-F1
after scaling is approximately **0.43**. Struggling-class recall is weak; the
exact training-only out-of-fold value is recorded in `models/launch_risk/metadata.json`.

These are training-only cross-validation metrics, not held-out test results.
The export workflow does not evaluate the test split.

## Limitations
Predictions estimate review-reception tiers only. They do not forecast sales,
revenue, profitability, or guaranteed future performance. The model may miss
games that ultimately fall into the Struggling tier, so treat results as
indicative rather than definitive.
