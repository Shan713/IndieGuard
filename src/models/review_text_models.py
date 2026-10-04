"""Review-level models that read the review text, on English reviews.

The earlier review-level models use metadata only (length, playtime, language, price, patch timing ...). This script
adds the text and measures what it buys, on the English reviews that have text features (557,202 reviews):

  reviewtext_dummy        Dummy (prior)                                   the baseline to beat on this subset
  reviewtext_meta         LightGBM on the 33 metadata features            like-for-like reference (same subset, same rows)
  reviewtext_meta_svd     LightGBM on metadata + 78 text SVD components   what the compressed text features add
  reviewtext_tfidf_lr     Logistic Regression on the full TF-IDF matrix   what the whole text adds (42k terms + metadata)

Same folds, threshold rule and test rule as the other review models (docs/modelling_protocol.md); the test set is scored once.
Needs the text features first:  python -m src.features.text
Usage:  python -m src.models.review_text_models [--cv]
Reading a review's own text to predict whether it is negative is detection of a negative review, not forecasting risk.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import review_models as rm
from .review_xgboost import (N_FOLDS, RESULTS, SEED, TARGET, _feature_cols, best_f1_threshold, check_group_integrity,
                             load_review_data, per_game_roc_auc)

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("review_text_models")
LGBM = {"num_leaves": 31, "min_child_samples": 100}
C_GRID = (0.3, 1.0, 3.0)
TUNE_FRAC = 0.25

LABELS = {"dummy": "Dummy (prior)", "meta": "LightGBM, metadata only", "meta_svd": "LightGBM, metadata + text components",
          "tfidf_lr": "Logistic Regression, full TF-IDF text + metadata"}
FEATURE_SETS = {"dummy": "English reviews with text features (no features)",
                "meta": "metadata only (33 cols), English reviews with text features",
                "meta_svd": "metadata (33) + 78 text SVD components, English reviews",
                "tfidf_lr": "full TF-IDF (42k terms) + scaled metadata, English reviews"}


def load_subset() -> tuple[pd.DataFrame, list[str], list[str], object]:
    """English reviews that have text features, with their text, metadata and the fitted TF-IDF vectoriser."""
    model_path = ROOT / "data" / "interim" / "text_svd_model.joblib"
    text_dir = ROOT / "data" / "processed" / "features" / "text_svd"
    if not model_path.exists() or not text_dir.exists():
        raise SystemExit("Text features are missing. Build them first: python -m src.features.text")
    df = load_review_data()
    check_group_integrity(df)
    meta = _feature_cols(df)
    txt = pd.read_parquet(text_dir).drop(columns="appid")
    tcols = [c for c in txt.columns if c.startswith("t_")]
    d = df.merge(txt, on="recommendationid", how="inner")
    reviews = pd.concat([pd.read_parquet(p, columns=["recommendationid", "review"])
                         for p in sorted((ROOT / "data" / "processed" / "reviews_clean").glob("*.parquet"))])
    d = d.merge(reviews, on="recommendationid", how="left").reset_index(drop=True)
    log.info("%d English reviews with text (%d train, %d test), base rate %.3f", len(d), (d["split"] == "train").sum(),
             (d["split"] == "test").sum(), d[TARGET].mean())
    return d, meta, tcols, joblib.load(model_path)["vectorizer"]


def tfidf_matrix(d: pd.DataFrame, vec, meta: list[str]) -> sp.csr_matrix:
    """TF-IDF of the text plus scaled metadata (imputer and scaler fitted on training rows only)."""
    chunks = [vec.transform(d["review"].iloc[i:i + 50_000]) for i in range(0, len(d), 50_000)]
    pre = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    pre.fit(d.loc[d["split"] == "train", meta])
    scaled = sp.csr_matrix(pre.transform(d[meta]).astype("float32"))
    return sp.hstack([sp.vstack(chunks), scaled]).tocsr()


def _fit_score(X, y, tr_idx, te_idx, C) -> np.ndarray:
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=300).fit(X[tr_idx], y[tr_idx])
    return lr.predict_proba(X[te_idx])[:, 1]


def lr_cross_validate(d: pd.DataFrame, X: sp.csr_matrix, C: float, rows: np.ndarray | None = None) -> dict:
    """5-fold CV of the text Logistic Regression on the train rows (optionally a subset of them), folds in parallel."""
    train = np.where((d["split"] == "train").values)[0] if rows is None else rows
    folds = d["cv_fold"].values
    y = d[TARGET].values
    splits = [(train[folds[train] != k], train[folds[train] == k]) for k in range(N_FOLDS)]
    probs = Parallel(n_jobs=N_FOLDS)(delayed(_fit_score)(X, y, tr, va, C) for tr, va in splits)
    oof = np.zeros(len(train))
    pos = {i: p for p, i in enumerate(train)}
    folds_m = []
    for (tr, va), p in zip(splits, probs):
        oof[[pos[i] for i in va]] = p
        folds_m.append({"pr_auc": average_precision_score(y[va], p), "roc_auc": roc_auc_score(y[va], p),
                        "per_game": per_game_roc_auc(d[TARGET].iloc[va], p, d["appid"].iloc[va]), "prevalence": float(y[va].mean())})
    fm = pd.DataFrame(folds_m)
    summary = {"pr_auc": (fm["pr_auc"].mean(), fm["pr_auc"].std()), "roc_auc": (fm["roc_auc"].mean(), fm["roc_auc"].std()),
               "per_game": fm["per_game"].mean(), "prevalence": fm["prevalence"].mean()}
    return {"summary": summary, "oof_prob": oof, "oof_y": y[train], "n_rounds": None}


def lr_test(d: pd.DataFrame, X, C: float, threshold: float) -> dict:
    tr, te = np.where(d["split"] == "train")[0], np.where(d["split"] == "test")[0]
    y = d[TARGET].values
    prob = _fit_score(X, y, tr, te, C)
    pred = (prob >= threshold).astype(int)
    pr = average_precision_score(y[te], prob)
    return {"pr_auc": pr, "lift": pr / y[te].mean(), "roc_auc": roc_auc_score(y[te], prob),
            "per_game": per_game_roc_auc(d[TARGET].iloc[te], prob, d["appid"].iloc[te]), "threshold": threshold,
            "precision": precision_score(y[te], pred, zero_division=0), "recall": recall_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0), "n_rounds": None}


def run_lr(d, vec, meta, cv_only):
    X = tfidf_matrix(d, vec, meta)
    train = np.where((d["split"] == "train").values)[0]
    games = d["appid"].iloc[train].unique()
    keep = set(np.random.RandomState(SEED).choice(games, int(TUNE_FRAC * len(games)), replace=False))
    sub = train[d["appid"].iloc[train].isin(keep).values]
    best_c, best = None, -1.0
    for c in C_GRID:
        score = lr_cross_validate(d, X, c, rows=sub)["summary"]["pr_auc"][0]
        log.info("  tfidf_lr C=%s  CV PR-AUC %.4f (25%% of games)", c, score)
        if score > best:
            best_c, best = c, score
    cv = lr_cross_validate(d, X, best_c)
    test = None
    if not cv_only:
        threshold, _ = best_f1_threshold(cv["oof_y"], cv["oof_prob"])
        test = lr_test(d, X, best_c, threshold)
    return cv, test, {"C": best_c}


def run_lgbm_like(d, feats, model, params, cv_only):
    """CV, then (unless cv_only) the OOF threshold and the single test evaluation, for a zoo model."""
    cv = rm.cross_validate(d, feats, model, params)
    test = None
    if not cv_only:
        threshold, _ = best_f1_threshold(cv["oof_y"], cv["oof_prob"])
        test = rm.evaluate_test(d, feats, model, params, cv["n_rounds"], threshold)
    return cv, test, params


def save(key: str, cv: dict, test: dict | None, params: dict, tuned: bool) -> None:
    s = cv["summary"]
    notes = [f"base rate {s['prevalence']:.3f}", f"ROC-AUC CV {s['roc_auc'][0]:.3f}+/-{s['roc_auc'][1]:.3f}",
             f"per-game median ROC-AUC CV {s['per_game']:.3f}"]
    if test:
        notes.append(f"test ROC-AUC {test['roc_auc']:.3f}; test per-game ROC-AUC {test['per_game']:.3f}; PR-AUC lift x{test['lift']:.2f}")
        notes.append(f"threshold {test['threshold']:.3f} (max-F1 on OOF): precision {test['precision']:.3f}, recall {test['recall']:.3f}, F1 {test['f1']:.3f}")
        if test["n_rounds"]:
            notes.append(f"{test['n_rounds']} rounds, refit on full train")
    notes += [f"seed {SEED}", ("tuned " if tuned else "default ") + json.dumps(params)]
    row = {"model": LABELS[key], "task": "review (English, text)", "feature_set": FEATURE_SETS[key],
           "cv_mean": f"{s['pr_auc'][0]:.4f} +/- {s['pr_auc'][1]:.4f}", "test": f"{test['pr_auc']:.4f}" if test else "",
           "notes": "; ".join(notes)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULTS / f"reviewtext_{key}.csv", index=False)
    log.info("saved reports/results/reviewtext_%s.csv", key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Review-level models that read the text")
    ap.add_argument("--cv", action="store_true", help="CV only; the test set is not touched")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    d, meta, tcols, vec = load_subset()
    for key in LABELS:
        log.info("=== %s", LABELS[key])
        if key == "dummy":
            cv, test, params = run_lgbm_like(d, meta, "dummy", {}, args.cv)
        elif key == "meta":
            cv, test, params = run_lgbm_like(d, meta, "lgbm", LGBM, args.cv)
        elif key == "meta_svd":
            cv, test, params = run_lgbm_like(d, meta + tcols, "lgbm", LGBM, args.cv)
        else:
            cv, test, params = run_lr(d, vec, meta, args.cv)
        s = cv["summary"]["pr_auc"]
        log.info("CV PR-AUC %.4f +/- %.4f%s", s[0], s[1], f" | TEST PR-AUC {test['pr_auc']:.4f} ROC-AUC {test['roc_auc']:.4f} per-game {test['per_game']:.4f}" if test else "")
        save(key, cv, test, params, tuned=(key == "tfidf_lr"))


if __name__ == "__main__":
    main()
