"""Review-level model comparison: Dummy, Logistic Regression, Random Forest, LightGBM (XGBoost is in review_xgboost.py).

Same data, features (33 columns), folds, threshold rule and test rule as the XGBoost run, because the loading, the
leak checks and the metric helpers are imported from src.models.review_xgboost. Writes reports/results/review_<model>.csv
in the protocol table format.

Usage:
    python -m src.models.review_models                         # all four models: tune on CV, then test once
    python -m src.models.review_models --models logreg,lgbm   # a subset
    python -m src.models.review_models --no-tune              # default settings only
    python -m src.models.review_models --cv                   # CV only, test set untouched
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .review_xgboost import (N_FOLDS, RESULTS, SEED, TARGET, _feature_cols, best_f1_threshold, check_group_integrity,
                             load_review_data, per_game_roc_auc)
from .zoo import DEFAULTS, GRIDS, LABELS, Model

log = logging.getLogger("review_models")
TUNE_FRAC = 0.25      # share of training GAMES used to compare grid points (whole games, so folds stay grouped)


def cross_validate(df: pd.DataFrame, feats: list[str], name: str, params: dict) -> dict:
    """5-fold CV on the train split using the shared cv_fold column."""
    train = df.query("split == 'train'").reset_index(drop=True)
    oof = np.zeros(len(train))
    folds, rounds = [], []
    for fold in range(N_FOLDS):
        va = (train["cv_fold"] == fold).values
        tr, vl = train[~va], train[va]
        m = Model(name, "review", params).fit(tr[feats], tr[TARGET], vl[feats], vl[TARGET])
        oof[va] = m.score(vl[feats])
        y = vl[TARGET].values
        folds.append({"pr_auc": average_precision_score(y, oof[va]), "roc_auc": roc_auc_score(y, oof[va]),
                      "per_game": per_game_roc_auc(vl[TARGET], oof[va], vl["appid"]), "prevalence": float(y.mean())})
        rounds.append(m.rounds_)
    fm = pd.DataFrame(folds)
    summary = {"pr_auc": (fm["pr_auc"].mean(), fm["pr_auc"].std()), "roc_auc": (fm["roc_auc"].mean(), fm["roc_auc"].std()),
               "per_game": fm["per_game"].mean(), "prevalence": fm["prevalence"].mean()}
    n_rounds = int(np.ceil(np.median(rounds))) if rounds[0] is not None else None
    return {"summary": summary, "oof_prob": oof, "oof_y": train[TARGET].values, "n_rounds": n_rounds}


def tune(df: pd.DataFrame, feats: list[str], name: str) -> dict:
    """Best grid point by CV PR-AUC on a sample of training games."""
    train_games = df.loc[df["split"] == "train", "appid"].unique()
    keep = np.random.RandomState(SEED).choice(train_games, int(TUNE_FRAC * len(train_games)), replace=False)
    sub = df[df["appid"].isin(keep) | (df["split"] == "test")]
    best, best_score = None, -1.0
    for params in GRIDS["review"][name]:
        score = cross_validate(sub, feats, name, params)["summary"]["pr_auc"][0]
        log.info("  %s %s  CV PR-AUC %.4f (25%% of games)", name, params, score)
        if score > best_score:
            best, best_score = params, score
    return best


def evaluate_test(df, feats, name, params, n_rounds, threshold) -> dict:
    """Refit on the whole train split and score the test reviews once."""
    train, test = df.query("split == 'train'"), df.query("split == 'test'")
    m = Model(name, "review", params).fit(train[feats], train[TARGET], n_rounds=n_rounds)
    prob, y = m.score(test[feats]), test[TARGET].values
    pred = (prob >= threshold).astype(int)
    pr = average_precision_score(y, prob)
    return {"pr_auc": pr, "lift": pr / y.mean(), "roc_auc": roc_auc_score(y, prob),
            "per_game": per_game_roc_auc(test[TARGET], prob, test["appid"]), "threshold": threshold,
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "n_rounds": n_rounds}


def save_results(name: str, cv: dict, test: dict | None, params: dict, n_features: int, tuned: bool) -> None:
    s = cv["summary"]
    notes = [f"base rate {s['prevalence']:.3f}", f"ROC-AUC CV {s['roc_auc'][0]:.3f}+/-{s['roc_auc'][1]:.3f}",
             f"per-game median ROC-AUC CV {s['per_game']:.3f}"]
    if test:
        notes.append(f"test ROC-AUC {test['roc_auc']:.3f}; test per-game ROC-AUC {test['per_game']:.3f}; PR-AUC lift x{test['lift']:.2f}")
        notes.append(f"threshold {test['threshold']:.3f} (max-F1 on OOF): precision {test['precision']:.3f}, "
                     f"recall {test['recall']:.3f}, F1 {test['f1']:.3f}")
        if test["n_rounds"]:
            notes.append(f"{test['n_rounds']} rounds, refit on full train")
    notes += [f"seed {SEED}", ("tuned " if tuned else "default ") + json.dumps(params)]
    row = {"model": LABELS[name], "task": "review", "feature_set": f"review_features + one-hot language ({n_features} cols)",
           "cv_mean": f"{s['pr_auc'][0]:.4f} +/- {s['pr_auc'][1]:.4f}", "test": f"{test['pr_auc']:.4f}" if test else "",
           "notes": "; ".join(notes)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULTS / f"review_{name}.csv", index=False)
    log.info("saved reports/results/review_%s.csv", name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Review-level model comparison")
    ap.add_argument("--models", default="dummy,logreg,rf,lgbm")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--cv", action="store_true", help="CV only; the test set is not touched")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    df = load_review_data()
    check_group_integrity(df)
    feats = _feature_cols(df)
    log.info("%d training reviews, %d features", int((df["split"] == "train").sum()), len(feats))

    for name in args.models.split(","):
        log.info("=== %s", LABELS[name])
        grid = GRIDS["review"][name]
        tuned = not args.no_tune and len(grid) > 1
        params = tune(df, feats, name) if tuned else DEFAULTS["review"][name]
        cv = cross_validate(df, feats, name, params)
        s = cv["summary"]["pr_auc"]
        log.info("CV PR-AUC %.4f +/- %.4f  params %s", s[0], s[1], params)
        test = None
        if not args.cv:
            threshold, _ = best_f1_threshold(cv["oof_y"], cv["oof_prob"])
            test = evaluate_test(df, feats, name, params, cv["n_rounds"], threshold)
            log.info("TEST PR-AUC %.4f  ROC-AUC %.4f  per-game %.4f  | at threshold %.3f: precision %.3f recall %.3f F1 %.3f",
                     test["pr_auc"], test["roc_auc"], test["per_game"], threshold, test["precision"], test["recall"], test["f1"])
        save_results(name, cv, test, params, len(feats), tuned)


if __name__ == "__main__":
    main()
