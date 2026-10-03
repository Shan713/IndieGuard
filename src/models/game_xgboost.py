"""Game-level XGBoost: predict the success tier (Struggling / Solid / Strong).
 
Task:       3-class classification (tier_code 0/1/2)
Data:       game_features + game_targets, eligible == 1
Metric:     macro-F1 (primary), balanced accuracy, per-tier recall, one-vs-rest AUC,
            PR-AUC for the Struggling tier (the "negative-review risk" view)
Protocol:   docs/modelling_protocol.md - launch-time features only, 5-fold CV, test once.
 
Usage:
    python -m src.models.game_xgboost                    # CV + test + SHAP
    python -m src.models.game_xgboost --cv               # CV only (no test evaluation)
    python -m src.models.game_xgboost --tune 20          # random search on CV, then test
    python -m src.models.game_xgboost --features named   # without the pc_* columns
"""
from __future__ import annotations
 
import argparse
import json
import logging
import random
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
 
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "figures" / "models"
 
SEED = 42
N_FOLDS = 5
TIERS = {0: "Struggling", 1: "Solid", 2: "Strong"}
N_CLASSES = len(TIERS)
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 25
 
BASE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": SEED,
    "nthread": -1,
}
 
SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
    "gamma": [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.1, 1.0, 5.0],
    "reg_lambda": [1.0, 2.0, 5.0, 10.0],
}
 
# -- logging ------------------------------------------------------------------
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
 
log = logging.getLogger("game_xgboost")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
 
 
# -- feature selection --------------------------------------------------------
KEY_COLS = {"appid", "split", "cv_fold", "reviews_capped"}
TARGET = "tier_code"
LABEL_COLS = {"tier_code", "tier", "high_risk", "eligible"}
FORBIDDEN = {
    "outcome_total_reviews", "outcome_total_positive", "outcome_total_negative",
    "outcome_neg_ratio", "outcome_pct_positive", "outcome_review_score",
    "outcome_review_score_desc", "steam_total_positive", "steam_total_negative",
    "steam_total_reviews", "steam_review_score", "steam_review_score_desc",
    "spy_owners", "spy_owners_variance", "metacritic_score",
    "recommendations_total", "votes_up", "votes_funny",
    "weighted_vote_score", "comment_count", "refunded", "has_dev_response",
    "target_is_negative",
}
# Safety net: any future column with these prefixes is an outcome, never a feature.
FORBIDDEN_PREFIXES = ("outcome_", "steam_", "spy_")
POST_LAUNCH = {"patches_first_30d", "patches_first_90d", "sales_first_90d"}
 
 
def _feature_cols(df: pd.DataFrame, mode: str = "all") -> list[str]:
    """Launch-time numeric feature columns.
 
    mode: "all" = everything allowed, "named" = without pc_* columns,
          "pc" = only the pc_* columns.
    """
    drop = KEY_COLS | LABEL_COLS | FORBIDDEN | POST_LAUNCH
    cols = [c for c in df.columns if c not in drop and not c.startswith(FORBIDDEN_PREFIXES)]
 
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        log.warning("Dropping non-numeric columns: %s", non_numeric)
        cols = [c for c in cols if c not in non_numeric]
 
    if mode == "named":
        cols = [c for c in cols if not c.startswith("pc_")]
    elif mode == "pc":
        cols = [c for c in cols if c.startswith("pc_")]
    return cols
 
 
# -- data loading -------------------------------------------------------------
def load_game_data() -> pd.DataFrame:
    """Load game_features merged with game_targets, keep eligible games only."""
    log.info("Loading game features and targets ...")
    gf = pd.read_parquet(ROOT / "data" / "processed" / "features" / "game_features.parquet")
    gt = pd.read_parquet(ROOT / "data" / "processed" / "features" / "game_targets.parquet")
    games = gf.merge(gt[["appid", "tier_code", "eligible"]], on="appid").query("eligible == 1").copy()
    log.info("  loaded %d eligible games, %d columns", len(games), games.shape[1])
    return games
 
 
def save_feature_list(cols: list[str], tag: str) -> None:
    """Write the exact feature names used, so the list can be audited for leakage."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"game_xgboost{tag}_features.txt").write_text("\n".join(cols) + "\n")
 
 
def class_weights(y: pd.Series) -> np.ndarray:
    """Per-sample weights inversely proportional to class frequency (balanced)."""
    counts = y.value_counts()
    n, k = len(y), y.nunique()
    w_map = {cls: n / (k * cnt) for cls, cnt in counts.items()}
    return y.map(w_map).values.astype("float32")
 
 
# -- metrics ------------------------------------------------------------------
def predict_labels(y_prob: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Class with the highest (optionally rescaled) probability."""
    return (y_prob * (1.0 if scale is None else scale)).argmax(axis=1)
 
 
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, scale: np.ndarray | None = None) -> dict:
    """All protocol metrics for the game-level task."""
    y_pred = predict_labels(y_prob, scale)
    rec = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
 
    ovr_auc = {}
    for cls, name in [(0, "Struggling"), (2, "Strong")]:
        y_bin = (y_true == cls).astype(int)
        ok = 0 < y_bin.sum() < len(y_bin)
        ovr_auc[name] = roc_auc_score(y_bin, y_prob[:, cls]) if ok else float("nan")
 
    y_str = (y_true == 0).astype(int)
    pr_auc_str = average_precision_score(y_str, y_prob[:, 0]) if y_str.sum() > 0 else float("nan")
 
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_Struggling": rec[0],
        "recall_Solid": rec[1],
        "recall_Strong": rec[2],
        "ovr_auc_Struggling": ovr_auc["Struggling"],
        "ovr_auc_Strong": ovr_auc["Strong"],
        "pr_auc_Struggling": pr_auc_str,
        "prevalence_Struggling": float(y_str.mean()),
    }
 
 
def bootstrap_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                       n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 95% CI for macro-F1 on test predictions."""
    rng = np.random.RandomState(SEED)
    n = len(y_true)
    f1s = [f1_score(y_true[i], y_pred[i], average="macro")
           for i in (rng.choice(n, size=n, replace=True) for _ in range(n_boot))]
    alpha = (1 - ci) / 2
    return float(np.percentile(f1s, 100 * alpha)), float(np.percentile(f1s, 100 * (1 - alpha)))
 
 
def tune_class_scales(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, float]:
    """Pick per-class probability multipliers that maximise macro-F1 on out-of-fold predictions.
 
    Class weights alone often leave the rare Struggling tier under-predicted. This is a small
    grid on training-only predictions, so the test set stays untouched.
    """
    best_scale = np.ones(N_CLASSES)
    best_f1 = f1_score(y_true, predict_labels(y_prob), average="macro")
    for s0 in np.arange(1.0, 4.01, 0.25):
        for s2 in np.arange(0.5, 2.01, 0.25):
            scale = np.array([s0, 1.0, s2])
            f = f1_score(y_true, predict_labels(y_prob, scale), average="macro")
            if f > best_f1 + 1e-9:
                best_f1, best_scale = f, scale
    return best_scale, best_f1
 
 
# -- cross-validation ---------------------------------------------------------
def cross_validate(df: pd.DataFrame, feature_cols: list[str], params: dict,
                   num_boost_round: int = NUM_BOOST_ROUND,
                   early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
                   verbose: bool = True) -> dict:
    """5-fold CV on the train split (cv_fold column). Returns fold metrics, summary and OOF probs."""
    train = df.query("split == 'train'").copy()
    if verbose:
        log.info("CV on %d training games, %d features", len(train), len(feature_cols))
 
    oof = np.zeros((len(train), N_CLASSES))
    fold_metrics, rounds = [], []
    for fold in range(N_FOLDS):
        val_mask = (train["cv_fold"] == fold).values
        tr, va = train[~val_mask], train[val_mask]
        y_va = va[TARGET].values
 
        dtrain = xgb.DMatrix(tr[feature_cols], label=tr[TARGET].values,
                             weight=class_weights(tr[TARGET]))
        dval = xgb.DMatrix(va[feature_cols], label=y_va)
 
        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                        evals=[(dval, "val")],
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
 
        # Use the best iteration explicitly (do not rely on library defaults).
        y_prob = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        oof[val_mask] = y_prob
        rounds.append(bst.best_iteration + 1)
 
        m = compute_metrics(y_va, y_prob)
        fold_metrics.append({"fold": fold, **m, "best_iteration": bst.best_iteration})
        if verbose:
            log.info("  fold %d  macro-F1 %.4f  bal-acc %.4f  rec(Str/Sol/Stg) %.2f/%.2f/%.2f  (iter %d)",
                     fold, m["macro_f1"], m["balanced_accuracy"], m["recall_Struggling"],
                     m["recall_Solid"], m["recall_Strong"], bst.best_iteration)
 
    fm = pd.DataFrame(fold_metrics)
    summary = {
        "cv_macro_f1_mean": fm["macro_f1"].mean(),
        "cv_macro_f1_std": fm["macro_f1"].std(),
        "cv_bal_acc_mean": fm["balanced_accuracy"].mean(),
        "cv_bal_acc_std": fm["balanced_accuracy"].std(),
        "cv_ovr_auc_Struggling_mean": fm["ovr_auc_Struggling"].mean(),
        "cv_ovr_auc_Strong_mean": fm["ovr_auc_Strong"].mean(),
        "cv_pr_auc_Struggling_mean": fm["pr_auc_Struggling"].mean(),
    }
    if verbose:
        log.info("CV summary  macro-F1 %.4f +/- %.4f  bal-acc %.4f +/- %.4f",
                 summary["cv_macro_f1_mean"], summary["cv_macro_f1_std"],
                 summary["cv_bal_acc_mean"], summary["cv_bal_acc_std"])
    return {"folds": fold_metrics, "summary": summary, "oof_prob": oof,
            "oof_y": train[TARGET].values, "n_rounds": int(np.ceil(np.median(rounds)))}
 
 
# -- hyper-parameter tuning ---------------------------------------------------
def tune(df: pd.DataFrame, feature_cols: list[str], n_iter: int) -> tuple[dict, dict]:
    """Random search scored on CV macro-F1. The default params are always candidate 0."""
    rnd = random.Random(SEED)
    best_params, best_cv = BASE_PARAMS, cross_validate(df, feature_cols, BASE_PARAMS, verbose=False)
    log.info("tune  default params  CV macro-F1 %.4f", best_cv["summary"]["cv_macro_f1_mean"])
 
    for i in range(1, n_iter + 1):
        cand = {**BASE_PARAMS, **{k: rnd.choice(v) for k, v in SEARCH_SPACE.items()}}
        cv = cross_validate(df, feature_cols, cand, verbose=False)
        score = cv["summary"]["cv_macro_f1_mean"]
        better = score > best_cv["summary"]["cv_macro_f1_mean"]
        log.info("tune  %2d/%d  CV macro-F1 %.4f %s", i, n_iter, score, "<- best" if better else "")
        if better:
            best_params, best_cv = cand, cv
    return best_params, best_cv
 
 
# -- test evaluation ----------------------------------------------------------
def evaluate_test(df: pd.DataFrame, feature_cols: list[str], params: dict,
                  n_rounds: int, scale: np.ndarray) -> dict:
    """Train on the FULL train split with the CV-chosen number of rounds, evaluate once on test."""
    train = df.query("split == 'train'").copy()
    test = df.query("split == 'test'").copy()
    log.info("Test evaluation: train %d -> test %d  (%d rounds)", len(train), len(test), n_rounds)
 
    dtrain = xgb.DMatrix(train[feature_cols], label=train[TARGET].values,
                         weight=class_weights(train[TARGET]))
    dtest = xgb.DMatrix(test[feature_cols])
    model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
 
    y_te = test[TARGET].values
    y_prob = model.predict(dtest)
    m = compute_metrics(y_te, y_prob)
    tuned = compute_metrics(y_te, y_prob, scale)
    lo, hi = bootstrap_macro_f1(y_te, predict_labels(y_prob))
 
    log.info("TEST  macro-F1 %.4f [95%% CI %.4f, %.4f]  bal-acc %.4f", m["macro_f1"], lo, hi,
             m["balanced_accuracy"])
    log.info("      recall: Struggling %.2f, Solid %.2f, Strong %.2f",
             m["recall_Struggling"], m["recall_Solid"], m["recall_Strong"])
    log.info("      OvR AUC: Struggling %.3f, Strong %.3f | PR-AUC Struggling %.3f (prevalence %.3f)",
             m["ovr_auc_Struggling"], m["ovr_auc_Strong"], m["pr_auc_Struggling"],
             m["prevalence_Struggling"])
    log.info("      with tuned class scales %s: macro-F1 %.4f, Struggling recall %.2f",
             np.round(scale, 2).tolist(), tuned["macro_f1"], tuned["recall_Struggling"])
 
    return {"model": model, "y_true": y_te, "y_prob": y_prob, "metrics": m, "tuned": tuned,
            "boot_lo": lo, "boot_hi": hi, "n_rounds": n_rounds, "scale": scale}
 
 
# -- confusion matrix ---------------------------------------------------------
def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, scale: np.ndarray, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [TIERS[i] for i in range(N_CLASSES)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, sc) in zip(axes, [("argmax", None), ("tuned class scales", scale)]):
        cm = confusion_matrix(y_true, predict_labels(y_prob, sc), labels=[0, 1, 2])
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES), labels)
        ax.set_yticks(range(N_CLASSES), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Game tiers - {title} (row-normalised)")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    fig.savefig(FIGURES / f"game_xgb_confusion{tag}.png", dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrices -> figures/models/game_xgb_confusion%s.png", tag)
 
 
# -- SHAP interpretation ------------------------------------------------------
def shap_analysis(model: xgb.Booster, df_test: pd.DataFrame, feature_cols: list[str],
                  tag: str, top_n: int = 20) -> pd.DataFrame:
    """SHAP feature importance using XGBoost native tree SHAP."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
 
    X = df_test[feature_cols]
    contribs = model.predict(xgb.DMatrix(X), pred_contribs=True)
    shap_values = contribs[:, :, :-1]  # (n_samples, n_classes, n_features); drop bias
 
    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=1).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(RESULTS / f"game_xgb_shap_importance{tag}.csv", index=False)
 
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance.head(top_n)
    ax.barh(range(len(top)), top["mean_abs_shap"].values, color="#2ecc71", alpha=0.85)
    ax.set_yticks(range(len(top)), top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Game-Level XGBoost (Launch-Risk) - Top Feature Importances (SHAP)")
    plt.tight_layout()
    fig.savefig(FIGURES / f"game_xgb_shap_bar{tag}.png", dpi=150)
    plt.close(fig)
 
    # Beeswarm for the Struggling class - the most actionable for developers.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[:, 0, :], X, feature_names=feature_cols,
                      max_display=top_n, show=False, plot_type="dot")
    plt.title("SHAP for Struggling Tier (class 0)")
    plt.tight_layout()
    plt.savefig(FIGURES / f"game_xgb_shap_struggling_beeswarm{tag}.png", dpi=150)
    plt.close()
    log.info("Saved SHAP plots + importance table (tag '%s')", tag)
    return importance
 
 
# -- save protocol-format results ---------------------------------------------
def save_results(cv: dict, test: dict | None, params: dict, n_features: int,
                 mode: str, tag: str, tuned: bool) -> None:
    """Write reports/results/game_xgboost<tag>.csv in the shared protocol table format."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    s = cv["summary"]
 
    notes = [f"macro-F1; bal-acc CV {s['cv_bal_acc_mean']:.3f}+/-{s['cv_bal_acc_std']:.3f}",
             f"CV PR-AUC Struggling {s['cv_pr_auc_Struggling_mean']:.3f}"]
    if test:
        m, t = test["metrics"], test["tuned"]
        notes.append(
            f"test bal-acc {m['balanced_accuracy']:.3f}; "
            f"recall Str/Sol/Stg {m['recall_Struggling']:.2f}/{m['recall_Solid']:.2f}/{m['recall_Strong']:.2f}; "
            f"OvR AUC Str {m['ovr_auc_Struggling']:.3f} Stg {m['ovr_auc_Strong']:.3f}; "
            f"PR-AUC Str {m['pr_auc_Struggling']:.3f} (prevalence {m['prevalence_Struggling']:.3f}); "
            f"boot 95% [{test['boot_lo']:.3f}, {test['boot_hi']:.3f}]")
        notes.append(f"tuned class scales {np.round(test['scale'], 2).tolist()}: "
                     f"test macro-F1 {t['macro_f1']:.3f}, Struggling recall {t['recall_Struggling']:.2f}")
        notes.append(f"{test['n_rounds']} rounds, refit on full train")
    notes.append(f"seed {SEED}")
    notes.append(("tuned " if tuned else "default ") +
                 json.dumps({k: v for k, v in params.items() if k != "seed"}))
 
    row = {
        "model": "XGBoost",
        "task": "game",
        "feature_set": f"launch-risk (launch_time features only, {n_features} cols, mode={mode})",
        "cv_mean": f"{s['cv_macro_f1_mean']:.4f} +/- {s['cv_macro_f1_std']:.4f}",
        "test": f"{test['metrics']['macro_f1']:.4f}" if test else "",
        "notes": "; ".join(notes),
    }
    pd.DataFrame([row]).to_csv(RESULTS / f"game_xgboost{tag}.csv", index=False)
    if tuned:
        (RESULTS / f"game_xgboost{tag}_best_params.json").write_text(
            json.dumps({k: v for k, v in params.items()}, indent=2))
    log.info("Saved results -> reports/results/game_xgboost%s.csv", tag)
 
 
# -- main ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Game-level XGBoost model")
    parser.add_argument("--cv", action="store_true", help="Run CV only, skip test evaluation")
    parser.add_argument("--tune", type=int, default=0, metavar="N",
                        help="Random-search N parameter sets on CV before the final run")
    parser.add_argument("--features", choices=["all", "named", "pc"], default="all",
                        help="all = every launch-time column; named = without pc_*; pc = only pc_*")
    args = parser.parse_args()
    tag = "" if args.features == "all" else f"_{args.features}"
 
    df = load_game_data()
    feature_cols = _feature_cols(df, args.features)
    log.info("Using %d launch-time features (mode=%s)", len(feature_cols), args.features)
    save_feature_list(feature_cols, tag)
 
    if args.tune > 0:
        params, cv = tune(df, feature_cols, args.tune)
    else:
        params, cv = BASE_PARAMS, cross_validate(df, feature_cols, BASE_PARAMS)
 
    test = None
    if not args.cv:
        scale, oof_f1 = tune_class_scales(cv["oof_y"], cv["oof_prob"])
        log.info("Class scales from out-of-fold predictions: %s (OOF macro-F1 %.4f)",
                 np.round(scale, 2).tolist(), oof_f1)
 
        test = evaluate_test(df, feature_cols, params, cv["n_rounds"], scale)
        plot_confusion(test["y_true"], test["y_prob"], scale, tag)
        importance = shap_analysis(test["model"], df.query("split == 'test'").copy(),
                                   feature_cols, tag)
        log.info("Top-5 features by SHAP:\n%s", importance.head().to_string(index=False))
 
    save_results(cv, test, params, len(feature_cols), args.features, tag, args.tune > 0)
    log.info("Done.")
 
 
if __name__ == "__main__":
    main()