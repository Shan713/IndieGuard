"""Shared paths, loaders and helpers for the feature pipeline (the stage after src/clean).

Everything learned from data (vocabularies, PCA/SVD, TF-IDF) is fitted on TRAINING games only
(`split == "train"` in data/processed/splits/game_split.csv) and then applied to all games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.collect.common import ROOT, get_logger

PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "features"
INTERIM = ROOT / "data" / "interim"
DOCS = ROOT / "docs" / "feature_engineering"
FIGS = ROOT / "docs" / "figures"
SEED = 42

# Review-level tables leave out these games (their launch period is missing, see docs/data_documentation.md)
CAPPED_NOTE = "reviews_capped"


def load_games() -> pd.DataFrame:
    """games_clean joined with the shared train/test split (one row per game, split and cv_fold attached)."""
    games = pd.read_parquet(PROCESSED / "games_clean.parquet")
    split = pd.read_csv(PROCESSED / "splits" / "game_split.csv")
    return games.merge(split, on="appid", how="left", validate="one_to_one").pipe(_check_split)


def _check_split(g: pd.DataFrame) -> pd.DataFrame:
    if g["split"].isna().any():
        raise ValueError(f"{int(g['split'].isna().sum())} games have no train/test assignment in game_split.csv")
    return g


def elbow(cum: np.ndarray) -> int:
    """Number of components at the elbow of a cumulative-variance curve.

    The elbow is the point farthest above the straight line joining the first and last point of the
    (normalised) curve, so the rule is deterministic and the same for every block.
    """
    k = np.arange(1, len(cum) + 1)
    x = (k - 1) / (len(cum) - 1)
    y = (cum - cum[0]) / (cum[-1] - cum[0])
    return int(np.argmax(y - x) + 1)


def split_items(value) -> list[str]:
    """';'-separated list column -> list of stripped, non-empty items."""
    return [t.strip() for t in value.split(";") if t.strip()] if isinstance(value, str) else []
