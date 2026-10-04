"""Game-level model comparison: Dummy, Logistic Regression, Random Forest, LightGBM (XGBoost is in game_xgboost.py).

Same data, launch-time features, folds, metrics, class-scale rule and test rule as the XGBoost run, because
everything shared is imported from src.models.game_xgboost. Writes reports/results/game_<model>.csv in the
protocol table format.

Usage:
    python -m src.models.game_models                          # all four models: tune on CV, then test once
    python -m src.models.game_models --models logreg,rf      # a subset
    python -m src.models.game_models --no-tune               # default settings only
    python -m src.models.game_models --cv                    # CV only, test set untouched
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from .game_xgboost import (RESULTS, SEED, TARGET, N_FOLDS, _feature_cols, bootstrap_macro_f1, compute_metrics,
                           load_game_data, predict_labels, tune_class_scales)
from .zoo import DEFAULTS, GRIDS, LABELS, Model

log = logging.getLogger("game_models")


def cross_validate(df: pd.DataFrame, feats: list[str], name: str, params: dict) -> dict:
    """5-fold CV on the train split using the shared cv_fold column."""
    train = df.query("split == 'train'").reset_index(drop=True)
    oof = np.zeros((len(train), 3))
    folds, rounds = [], []
    for fold in range(N_FOLDS):
        va = (train["cv_fold"] == fold).values
        tr, vl = train[~va], train[va]
        m = Model(name, "game", params).fit(tr[feats], tr[TARGET], vl[feats], vl[TARGET])
        oof[va] = m.score(vl[feats])
        folds.append(compute_metrics(vl[TARGET].values, oof[va]))
        rounds.append(m.rounds_)
    fm = pd.DataFrame(folds)
    summary = {"macro_f1": (fm["macro_f1"].mean(), fm["macro_f1"].std()),
               "bal_acc": (fm["balanced_accuracy"].mean(), fm["balanced_accuracy"].std()),
               "pr_auc_Struggling": (fm["pr_auc_Struggling"].mean(), fm["pr_auc_Struggling"].std())}
    n_rounds = int(np.ceil(np.median(rounds))) if rounds[0] is not None else None
    return {"summary": summary, "oof_prob": oof, "oof_y": train[TARGET].values, "n_rounds": n_rounds}


def tune(df: pd.DataFrame, feats: list[str], name: str) -> tuple[dict, dict]:
    """Pick the grid point with the best CV macro-F1 (training part only)."""
    best_params, best_cv = None, None
    for params in GRIDS["game"][name]:
        cv = cross_validate(df, feats, name, params)
        score = cv["summary"]["macro_f1"][0]
        log.info("  %s %s  CV macro-F1 %.4f", name, params, score)
        if best_cv is None or score > best_cv["summary"]["macro_f1"][0]:
            best_params, best_cv = params, cv
    return best_params, best_cv


def evaluate_test(df, feats, name, params, n_rounds, scale) -> dict:
    """Refit on the whole train split and score the test games once."""
    train, test = df.query("split == 'train'"), df.query("split == 'test'")
    m = Model(name, "game", params).fit(train[feats], train[TARGET], n_rounds=n_rounds)
    prob, y = m.score(test[feats]), test[TARGET].values
    lo, hi = bootstrap_macro_f1(y, predict_labels(prob))
    return {"metrics": compute_metrics(y, prob), "tuned": compute_metrics(y, prob, scale),
            "lo": lo, "hi": hi, "scale": scale, "n_rounds": n_rounds}


def save_results(name: str, cv: dict, test: dict | None, params: dict, n_features: int, tuned: bool) -> None:
    s = cv["summary"]
    notes = [f"macro-F1; bal-acc CV {s['bal_acc'][0]:.3f}+/-{s['bal_acc'][1]:.3f}",
             f"CV PR-AUC Struggling {s['pr_auc_Struggling'][0]:.3f}"]
    if test:
        m, t = test["metrics"], test["tuned"]
        notes.append(f"test bal-acc {m['balanced_accuracy']:.3f}; "
                     f"recall Str/Sol/Stg {m['recall_Struggling']:.2f}/{m['recall_Solid']:.2f}/{m['recall_Strong']:.2f}; "
                     f"OvR AUC Str {m['ovr_auc_Struggling']:.3f} Stg {m['ovr_auc_Strong']:.3f}; "
                     f"PR-AUC Str {m['pr_auc_Struggling']:.3f} (prevalence {m['prevalence_Struggling']:.3f}); "
                     f"boot 95% [{test['lo']:.3f}, {test['hi']:.3f}]")
        notes.append(f"tuned class scales {np.round(test['scale'], 2).tolist()}: "
                     f"test macro-F1 {t['macro_f1']:.3f}, Struggling recall {t['recall_Struggling']:.2f}")
        if test["n_rounds"]:
            notes.append(f"{test['n_rounds']} rounds, refit on full train")
    notes += [f"seed {SEED}", ("tuned " if tuned else "default ") + json.dumps(params)]
    row = {"model": LABELS[name], "task": "game",
           "feature_set": f"launch-risk (launch_time features only, {n_features} cols, mode=all)",
           "cv_mean": f"{s['macro_f1'][0]:.4f} +/- {s['macro_f1'][1]:.4f}",
           "test": f"{test['metrics']['macro_f1']:.4f}" if test else "", "notes": "; ".join(notes)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULTS / f"game_{name}.csv", index=False)
    log.info("saved reports/results/game_%s.csv", name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Game-level model comparison")
    ap.add_argument("--models", default="dummy,logreg,rf,lgbm")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--cv", action="store_true", help="CV only; the test set is not touched")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    df = load_game_data()
    feats = _feature_cols(df)
    log.info("%d eligible games, %d launch-time features", len(df), len(feats))

    for name in args.models.split(","):
        log.info("=== %s", LABELS[name])
        if args.no_tune or len(GRIDS["game"][name]) == 1:
            params = DEFAULTS["game"][name]
            cv = cross_validate(df, feats, name, params)
        else:
            params, cv = tune(df, feats, name)
        s = cv["summary"]["macro_f1"]
        log.info("CV macro-F1 %.4f +/- %.4f  params %s", s[0], s[1], params)
        test = None
        if not args.cv:
            scale, _ = tune_class_scales(cv["oof_y"], cv["oof_prob"])
            test = evaluate_test(df, feats, name, params, cv["n_rounds"], scale)
            m = test["metrics"]
            log.info("TEST macro-F1 %.4f [%.3f, %.3f]  bal-acc %.3f  recall Str/Sol/Stg %.2f/%.2f/%.2f",
                     m["macro_f1"], test["lo"], test["hi"], m["balanced_accuracy"],
                     m["recall_Struggling"], m["recall_Solid"], m["recall_Strong"])
        save_results(name, cv, test, params, len(feats), not args.no_tune and len(GRIDS["game"][name]) > 1)


if __name__ == "__main__":
    main()
