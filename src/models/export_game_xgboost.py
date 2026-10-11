"""Export the launch-risk XGBoost model without evaluating the test split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.models.game_xgboost import (
    BASE_PARAMS,
    ROOT,
    TARGET,
    _feature_cols,
    class_weights,
    cross_validate,
    load_game_data,
    tune_class_scales,
)

OUTPUT_DIR = ROOT / "models" / "launch_risk"


def main() -> None:
    # Load eligible games and select only launch-time numeric features.
    df = load_game_data()
    feature_cols = _feature_cols(df, mode="all")

    train = df.query("split == 'train'").copy()
    if train.empty:
        raise RuntimeError("No training rows found.")

    print(f"Training rows: {len(train)}")
    print(f"Launch-time features: {len(feature_cols)}")
    print("Running 5-fold cross-validation on training data only...")

    # CV uses cv_fold within the training split; no test data is evaluated.
    cv = cross_validate(df, feature_cols, BASE_PARAMS, verbose=True)
    scales, oof_f1 = tune_class_scales(cv["oof_y"], cv["oof_prob"])
    n_rounds = cv["n_rounds"]

    if n_rounds < 1:
        raise RuntimeError(f"Invalid selected boosting rounds: {n_rounds}")

    print(f"Selected boosting rounds: {n_rounds}")
    print(f"Training-only OOF macro-F1 after scaling: {oof_f1:.4f}")
    print(f"Class scales: {scales.tolist()}")
    print("Fitting final model on the full training split...")

    dtrain = xgb.DMatrix(
        train[feature_cols],
        label=train[TARGET].values,
        weight=class_weights(train[TARGET]),
        feature_names=feature_cols,
    )
    model = xgb.train(
        BASE_PARAMS,
        dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "game_xgboost.json"
    metadata_path = OUTPUT_DIR / "metadata.json"
    features_path = OUTPUT_DIR / "feature_columns.json"

    model.save_model(str(model_path))

    metadata = {
        "model_name": "IndieGuard launch-risk scorer",
        "task": "three-class review-reception tier prediction",
        "target_column": TARGET,
        "tier_mapping": {
            "0": "Struggling",
            "1": "Solid",
            "2": "Strong",
        },
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "class_scales": [float(value) for value in scales],
        "boosting_rounds": int(n_rounds),
        "base_parameters": {
            key: (int(value) if isinstance(value, (np.integer,)) else
                  float(value) if isinstance(value, (np.floating,)) else value)
            for key, value in BASE_PARAMS.items()
        },
        "seed": 42,
        "training_rows": int(len(train)),
        "cross_validation_folds": 5,
        "oof_macro_f1_after_scaling": float(oof_f1),
        "training_protocol": (
            "Class scales selected from out-of-fold predictions on the training split. "
            "Final model fitted on the full training split. Test split not evaluated "
            "during export."
        ),
        "interpretation_warning": (
            "Predicts review-reception tier, not sales, revenue, or profitability."
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    features_path.write_text(
        json.dumps(feature_cols, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nExport complete:")
    print(f"Model:    {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Features: {features_path}")


if __name__ == "__main__":
    main()
