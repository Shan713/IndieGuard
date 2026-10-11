"""Inference utilities for the IndieGuard launch-risk scorer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "launch_risk"
MODEL_PATH = MODEL_DIR / "game_xgboost.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

TIER_NAMES = {0: "Struggling", 1: "Solid", 2: "Strong"}


def load_scorer() -> tuple[xgb.Booster, dict[str, Any]]:
    """Load the trained booster and its feature/threshold metadata."""
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Launch-risk model artifacts are missing. "
            "Run the model export script first."
        )

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    features = metadata.get("feature_columns")
    if not isinstance(features, list) or not features:
        raise ValueError("Model metadata has no valid feature_columns list.")
    if model.num_features() != len(features):
        raise ValueError("Model feature count does not match metadata.")

    return model, metadata


def predict_launch_risk(
    data: pd.DataFrame | dict[str, Any],
) -> list[dict[str, Any]]:
    """Predict review-reception tiers from prepared launch-time features.

    Scores are class-scaled and normalized for ranking. They are not calibrated
    probabilities and do not represent sales, revenue, or profitability.
    """
    model, metadata = load_scorer()
    features = metadata["feature_columns"]
    scales = np.asarray(metadata.get("class_scales", [1.0, 1.0, 1.0]), dtype=float)

    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Metadata must contain three positive finite class scales.")

    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required launch-time features: {missing}")

    frame = frame.loc[:, features].apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or non-finite values.")

    scores = model.predict(xgb.DMatrix(frame, feature_names=features))
    scores = scores * scales
    scores = scores / scores.sum(axis=1, keepdims=True)
    predicted = scores.argmax(axis=1)

    return [
        {
            "tier_code": int(tier),
            "tier": TIER_NAMES[int(tier)],
            "scores": {
                TIER_NAMES[class_id]: float(scores[index, class_id])
                for class_id in range(3)
            },
        }
        for index, tier in enumerate(predicted)
    ]


def predict_raw_game(game: dict[str, Any]) -> dict[str, Any]:
    """Featurize one raw game's metadata and tags, then predict its tier."""
    from src.dashboard.featurize import featurize_game

    return predict_launch_risk(featurize_game(game))[0]


if __name__ == "__main__":
    print("Launch-risk inference module is ready.")
    print(f"Expected model: {MODEL_PATH}")
