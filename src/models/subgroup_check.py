"""Subgroup error check (Section 4 of the final report): does a model work equally well across groups?

  python -m src.models.subgroup_check            # both checks (the text part refits the TF-IDF model, ~10 min)
  python -m src.models.subgroup_check --game     # game-level scorer only (seconds)

Game level: the exported launch-risk scorer (models/launch_risk) on the 299 held-out test games, split by price band,
review volume, launch year and Early Access. Review text: the tuned TF-IDF Logistic Regression refitted exactly as in
src.models.review_text_detail, scored once on the 120,487 held-out English reviews, split by refund window, game price,
review length and free copies. Nothing is tuned here; groups are only used to slice the test predictions.
Writes reports/results/game_subgroup_check.csv and reports/results/reviewtext_subgroup_check.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "reports" / "results"
FEAT = ROOT / "data" / "processed" / "features"


def game_check() -> pd.DataFrame:
    from src.dashboard.risk import load_scorer
    model, meta = load_scorer()
    feats = meta["feature_columns"]
    d = (pd.read_parquet(FEAT / "game_features.parquet")
         .merge(pd.read_parquet(FEAT / "game_targets.parquet")[["appid", "eligible", "tier_code"]], on="appid")
         .merge(pd.read_parquet(FEAT / "game_outcomes.parquet")[["appid", "outcome_total_reviews"]], on="appid"))
    d = d[(d.split == "test") & d.eligible.astype(bool)].copy()
    p = model.predict(xgb.DMatrix(d[feats].astype(float), feature_names=feats)) * np.asarray(meta["class_scales"])
    d["pred"], d["y"] = p.argmax(1), d.tier_code.astype(int)
    groups = {
        "price": pd.cut(d.price_usd, [-1, 4.99, 14.99, 1e9], labels=["under $5", "$5 to $14.99", "$15 or more"]),
        "reviews": pd.cut(d.outcome_total_reviews, [0, 99, 499, 1e12], labels=["20 to 99", "100 to 499", "500 or more"]),
        "year": d.launch_year.astype(int).map(lambda y: "2022-2023" if y <= 2023 else "2024-2025"),
        "early access": d.launched_in_early_access.map(lambda v: "Early Access" if v else "Full release"),
    }
    rows = [["all", "all test games", len(d), f1_score(d.y, d.pred, average="macro"),
             recall_score(d.y, d.pred, labels=[0], average="macro"), int((d.y == 0).sum())]]
    for g, s in groups.items():
        for lv in s.dropna().unique():
            sub = d[s == lv]
            rows.append([g, str(lv), len(sub), f1_score(sub.y, sub.pred, average="macro", labels=[0, 1, 2]),
                         recall_score(sub.y, sub.pred, labels=[0], average="macro", zero_division=0) if (sub.y == 0).any() else np.nan,
                         int((sub.y == 0).sum())])
    out = pd.DataFrame(rows, columns=["group", "level", "n", "macro_f1", "recall_struggling", "n_struggling"])
    out.to_csv(RES / "game_subgroup_check.csv", index=False)
    return out


def text_check() -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .review_text_models import TARGET, _fit_score, load_subset, tfidf_matrix
    thr = float(re.search(r"threshold ([\d.]+)", pd.read_csv(RES / "reviewtext_tfidf_lr.csv").iloc[0]["notes"]).group(1))
    best = json.loads(pd.read_csv(RES / "tuning_reviewtext_logreg.csv").sort_values("rank").iloc[0]["params"])
    d, meta, _, vec0 = load_subset()
    keep = {k: v for k, v in vec0.get_params().items() if k in ("min_df", "max_df", "sublinear_tf", "token_pattern", "lowercase", "norm", "use_idf", "smooth_idf")}
    tr, te = np.where(d["split"] == "train")[0], np.where(d["split"] == "test")[0]
    vec = TfidfVectorizer(ngram_range=tuple(best["ngram_range"]), max_features=int(best["max_features"]), dtype=np.float32, **keep).fit(d["review"].iloc[tr])
    prob = _fit_score(tfidf_matrix(d, vec, meta), d[TARGET].values, tr, te, float(best["C"]))
    t = d.iloc[te].copy()
    t["prob"], t["pred"] = prob, (prob >= thr).astype(int)
    groups = {
        "window": t.in_refund_window.astype(bool).map({True: "inside 2 h", False: "after 2 h"}),
        "price": pd.cut(t.price_usd.fillna(0), [-1, 0, 4.99, 14.99, 1e9], labels=["free", "under $5", "$5 to $14.99", "$15 or more"]),
        "length": pd.cut(t.review_n_words, [0, 9, 49, 1e9], labels=["under 10 words", "10 to 49 words", "50+ words"]),
        "free copy": t.received_for_free.astype(bool).map({True: "free copy", False: "bought"}),
    }
    rows = []
    for g, s in groups.items():
        for lv in s.dropna().unique():
            sub = t[s == lv]
            rows.append([g, str(lv), len(sub), sub[TARGET].mean(), average_precision_score(sub[TARGET], sub.prob),
                         precision_score(sub[TARGET], sub.pred, zero_division=0), recall_score(sub[TARGET], sub.pred)])
    out = pd.DataFrame(rows, columns=["group", "level", "n", "base_rate", "pr_auc", "precision", "recall"])
    out.to_csv(RES / "reviewtext_subgroup_check.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", action="store_true", help="only the game-level check")
    a = ap.parse_args()
    print(game_check().round(3).to_string())
    if not a.game:
        print(text_check().round(3).to_string())
