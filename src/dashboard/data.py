"""Cached loaders for the dashboard. Small committed tables only (data/dashboard and reports/results)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FEAT = PROC / "features"
DASH = ROOT / "data" / "dashboard"
RES = ROOT / "reports" / "results"


def _cache(fn):
    try:
        import streamlit as st
        return st.cache_data(show_spinner=False)(fn)
    except Exception:                       # building the data without streamlit installed
        return fn


@_cache
def games() -> pd.DataFrame:
    d = pd.read_csv(DASH / "games.csv")
    d["tag_list"] = d.tags.fillna("").str.split(";")
    return d


@_cache
def monthly() -> pd.DataFrame:
    return pd.read_csv(DASH / "monthly.csv", parse_dates=["month"])


@_cache
def refund_heatmap() -> pd.DataFrame:
    return pd.read_csv(DASH / "refund_heatmap.csv")


@_cache
def refund_summary() -> pd.Series:
    return pd.read_csv(DASH / "refund_summary.csv").iloc[0]


@_cache
def rules() -> pd.DataFrame:
    return pd.read_csv(RES / "association_rules_strong.csv")


@_cache
def topic_prevalence() -> pd.DataFrame:
    return pd.read_csv(RES / "topic_prevalence.csv")


@_cache
def topic_terms() -> pd.DataFrame:
    return pd.read_csv(RES / "topic_terms.csv")


@_cache
def patch_tables() -> dict[str, pd.DataFrame]:
    return {"summary": pd.read_csv(RES / "patch_impact_summary.csv"), "launch": pd.read_csv(RES / "patch_impact_by_launch.csv"),
            "before": pd.read_csv(RES / "patch_impact_by_before_positivity.csv"), "control": pd.read_csv(RES / "patch_impact_control_summary.csv")}


@_cache
def raw_games() -> pd.DataFrame:
    return pd.read_parquet(PROC / "games_clean.parquet")


@_cache
def tag_vocabulary() -> tuple[list[str], list[str]]:
    import numpy as np
    cols = np.load(FEAT / "tag_pca_model.npz", allow_pickle=False)["columns"].tolist()
    return ([c.split("__", 1)[1] for c in cols if c.startswith("tag__")], [c.split("__", 1)[1] for c in cols if c.startswith("category__")])
