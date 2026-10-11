"""Convert raw game metadata and tags into launch-risk model features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PCA_PATH = ROOT / "data" / "processed" / "features" / "tag_pca_model.npz"
FEATURES_PATH = ROOT / "models" / "launch_risk" / "feature_columns.json"

PRICE_CUTS = [5, 10, 20]
CONTROLLER_LEVEL = {"full": 2, "partial": 1}


def _items(value: Any) -> list[str]:
    """Parse semicolon-separated tags/categories."""
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _flag(value: Any, field: str) -> int:
    """Validate boolean-like inputs without treating non-empty strings as True."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    raise ValueError(f"{field} must be a boolean or 0/1.")


def featurize_game(game: Mapping[str, Any]) -> dict[str, float | int]:
    """Transform one raw game's metadata and tags into the scorer's 83 features.

    Required fields: price_usd, is_free, launch_date, release_date,
    launch_before_window, n_dlc, n_supported_languages, n_achievements,
    platform_mac, platform_linux, controller_support, developers, publishers,
    tags, and categories.
    """
    required = {
        "price_usd", "is_free", "launch_date", "release_date",
        "launch_before_window", "n_dlc", "n_supported_languages",
        "n_achievements", "platform_mac", "platform_linux",
        "controller_support", "developers", "publishers", "tags", "categories",
    }
    missing = sorted(required - game.keys())
    if missing:
        raise ValueError(f"Missing raw game fields: {missing}")

    price = float(game["price_usd"])
    if not np.isfinite(price) or price < 0:
        raise ValueError("price_usd must be finite and non-negative.")

    is_free = _flag(game["is_free"], "is_free")
    launch = pd.to_datetime(game["launch_date"], errors="coerce")
    release = pd.to_datetime(game["release_date"], errors="coerce")
    if pd.isna(launch):
        raise ValueError("launch_date must be a valid date.")

    achievements = game["n_achievements"]
    achievements = 0 if pd.isna(achievements) else int(achievements)
    if achievements < 0:
        raise ValueError("n_achievements must be non-negative.")

    controller = game["controller_support"]
    controller = controller.strip().lower() if isinstance(controller, str) else ""

    developers = set(_items(game["developers"]))
    publishers = set(_items(game["publishers"]))

    result: dict[str, float | int] = {
        "price_usd": price,
        "price_tier": 0 if is_free or price == 0 else int(np.digitize(price, PRICE_CUTS) + 1),
        "is_free": is_free,
        "launch_year": int(launch.year),
        "launch_quarter": int(launch.quarter),
        "launched_in_early_access": int(
            not pd.isna(release) and (release - launch).days > 7
        ),
        "launch_before_window": _flag(game["launch_before_window"], "launch_before_window"),
        "n_dlc": int(game["n_dlc"]),
        "n_supported_languages": int(game["n_supported_languages"]),
        "n_achievements": achievements,
        "has_achievements": int(achievements > 0),
        "platform_mac": _flag(game["platform_mac"], "platform_mac"),
        "platform_linux": _flag(game["platform_linux"], "platform_linux"),
        "controller_support_level": CONTROLLER_LEVEL.get(controller, 0),
        "self_published": int(bool(developers & publishers)),
    }

    for field in ("n_dlc", "n_supported_languages"):
        if result[field] < 0:
            raise ValueError(f"{field} must be non-negative.")

    with np.load(PCA_PATH, allow_pickle=False) as pca:
        columns = pca["columns"].tolist()
        mean = pca["mean"].astype(np.float32)
        components = pca["components"].astype(np.float32)

    if len(columns) != len(mean) or components.shape != (68, len(columns)):
        raise ValueError("Saved tag PCA artifact has an unexpected shape.")

    vocabulary = {name: index for index, name in enumerate(columns)}
    multi_hot = np.zeros(len(columns), dtype=np.float32)

    for kind, field in (("tag", "tags"), ("category", "categories")):
        for item in set(_items(game[field])):
            index = vocabulary.get(f"{kind}__{item}")
            if index is not None:
                multi_hot[index] = 1.0

    pcs = (multi_hot - mean) @ components.T
    for index, value in enumerate(pcs, start=1):
        result[f"pc_{index:02d}"] = float(value)

    feature_columns = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    if set(result) != set(feature_columns):
        missing_features = sorted(set(feature_columns) - set(result))
        extra_features = sorted(set(result) - set(feature_columns))
        raise ValueError(
            f"Feature schema mismatch. Missing: {missing_features}; extra: {extra_features}"
        )

    return {name: result[name] for name in feature_columns}
