# Launch-Risk Scorer

## Purpose
Predicts a game's review-reception tier from launch-time features:
- Struggling
- Solid
- Strong

This is a three-class prediction of review reception, not sales, revenue,
profitability, or guaranteed future performance.

## Requirements
Activate the `indieguard-repro` environment and run commands from the
repository root.

The scorer requires all 83 numeric launch-time features:
- 15 price, launch, content, platform, and publishing features
- 68 precomputed tag-component features (`pc_01` to `pc_68`)

Inputs must already be processed into the expected feature schema. The
scorer does not convert raw Steam store information or raw tags into these
features. Missing, non-numeric, or non-finite values are rejected.

The feature order is defined in `models/launch_risk/feature_columns.json`.

## Export the model
Run:

    python -m src.models.export_game_xgboost

The exporter performs five-fold cross-validation on the training split,
selects class scales using out-of-fold predictions, and fits the final
model on the full training split. It does not evaluate the test split.

The model, feature list, and metadata are saved under `models/launch_risk/`.

## Inference
Import the prediction function:

    from src.dashboard.risk import predict_launch_risk

Pass a dictionary containing all 83 required features, or a pandas DataFrame
with those feature columns:

    result = predict_launch_risk(your_prepared_feature_dictionary)
    print(result)

For a DataFrame, one prediction is returned per row. Each result includes
the predicted tier, its numeric code, and adjusted class probabilities
that sum to 1.

For existing processed games, the launch-time columns can be obtained from
`data/processed/features/game_features.parquet`. A new game's raw metadata
and tags must first be processed into the same feature schema before scoring.

## Limitations
Predictions are estimates of review-reception tiers. They are not guarantees
and should not be interpreted as sales, revenue, or profitability forecasts.
