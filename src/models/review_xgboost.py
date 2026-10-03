"""Review-level XGBoost: predict whether a Steam review is negative.
 
Task:       Binary classification (target_is_negative 0/1)
Data:       data/processed/features/review_features
Metric:     PR-AUC (primary), ROC-AUC, per-game median ROC-AUC, precision/recall/F1 at a
            threshold chosen on out-of-fold predictions
Protocol:   docs/modelling_protocol.md - 5-fold CV on cv_fold, test once at the end.
 
Note: features such as review length and the reviewer's playtime exist only AFTER a review is
written. This model explains which kinds of reviews turn negative; it is not a launch-time
risk scorer (that is the game-level model).
 
Usage:
    python -m src.models.review_xgboost                  # CV + test + SHAP
    python -m src.models.review_xgboost --cv             # CV only (no test evaluation)
    python -m src.models.review_xgboost --tune 10        # random search on a sample of games
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
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
 
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "figures" / "models"
 
SEED = 42
N_FOLDS = 5
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30
 
BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": SEED,
    "nthread": -1,
    "tree_method": "hist",
}
 
SEARCH_SPACE = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.08],
    "min_child_weight": [5, 10, 20, 50],
    "subsample": [0.6, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "gamma": [0.0, 1.0, 3.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}
 
# -- logging ------------------------------------------------------------------
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
 
log = logging.getLogger("review_xgboost")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
 
 
# -- feature selection --------------------------------------------------------
KEY_COLS = {"recommendationid", "appid", "split", "cv_fold"}
TARGET = "target_is_negative"
FORBIDDEN = {
    "outcome_total_reviews", "outcome_total_positive", "outcome_total_negative",
    "outcome_neg_ratio", "outcome_pct_positive", "outcome_review_score",
    "outcome_review_score_desc", "steam_total_positive", "steam_total_negative",
    "steam_total_reviews", "steam_review_score", "steam_review_score_desc",
    "spy_owners", "spy_owners_variance", "metacritic_score",
    "recommendations_total", "votes_up", "votes_funny",
    "weighted_vote_score", "comment_count", "refunded", "has_dev_response",
    "reviews_capped", "tier_code", "tier", "high_risk", "eligible",
}
# Safety net: any future column with these prefixes is an outcome, never a feature.
FORBIDDEN_PREFIXES = ("outcome_", "steam_", "spy_")
STRING_COLS = {"language"}  # one-hot encoded below
 
 
def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns, excluding keys, target, forbidden outcomes and strings."""
    drop = KEY_COLS | {TARGET} | FORBIDDEN | STRING_COLS
    cols = [c for c in df.columns if c not in drop and not c.startswith(FORBIDDEN_PREFIXES)]
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        log.warning("Dropping non-numeric columns: %s", non_numeric)
        cols = [c for c in cols if c not in non_numeric]
    return cols
 
 
def save_feature_list(cols: list[str]) -> None:
    """Write the exact feature names used, so the list can be audited for leakage."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "review_xgboost_features.txt").write_text("\n".join(cols) + "\n")
 
 
# -- data loading -------------------------------------------------------------
def check_group_integrity(df: pd.DataFrame) -> None:
    """Games must not leak across train/test or across CV folds."""
    train_ids = set(df.loc[df["split"] == "train", "appid"])
    test_ids = set(df.loc[df["split"] == "test", "appid"])
    shared = train_ids & test_ids
    if shared:
        raise ValueError(f"{len(shared)} games appear in BOTH train and test - split is not grouped by game.")
 
    folds_per_game = df.loc[df["split"] == "train"].groupby("appid")["cv_fold"].nunique()
    n_bad = int((folds_per_game > 1).sum())
    if n_bad:
        log.warning("%d games span more than one CV fold: CV is NOT grouped by game, so CV scores "
                    "are optimistic (game-level columns let the model recognise the game). "
                    "Rebuild cv_fold with GroupKFold on appid.", n_bad)
    else:
        log.info("Group check OK: no game crosses train/test or CV folds.")
 
 
def load_review_data() -> pd.DataFrame:
    """Load review_features and one-hot encode the top languages (chosen on train only)."""
    log.info("Loading review features ...")
    rf = pd.read_parquet(ROOT / "data" / "processed" / "features" / "review_features")
    log.info("  loaded %d reviews, %d columns", len(rf), rf.shape[1])
 
    top_langs = rf.loc[rf["split"] == "train", "language"].value_counts().nlargest(10).index.tolist()
    for lang in top_langs:
        rf[f"lang_{lang}"] = (rf["language"] == lang).astype("int8")
    rf["lang_other"] = (~rf["language"].isin(top_langs)).astype("int8")
    return rf
 
 
# -- metrics ------------------------------------------------------------------
def per_game_roc_auc(y_true: pd.Series, y_score: np.ndarray, appids: pd.Series,
                     min_reviews: int = 200) -> float:
    """Median ROC-AUC within each game that has >= min_reviews reviews and both classes."""
    tmp = pd.DataFrame({"y": y_true.values, "s": y_score, "g": appids.values})
    aucs = [roc_auc_score(g["y"], g["s"]) for _, g in tmp.groupby("g")
            if len(g) >= min_reviews and g["y"].nunique() == 2]
    return float(np.median(aucs)) if aucs else float("nan")
 
 
def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Probability threshold that maximises F1 (used on out-of-fold predictions only)."""
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])
 
 
# -- cross-validation ---------------------------------------------------------
def cross_validate(df: pd.DataFrame, feature_cols: list[str], params: dict,
                   num_boost_round: int = NUM_BOOST_ROUND,
                   early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
                   verbose: bool = True) -> dict:
    """5-fold CV on the train split (cv_fold column). Returns metrics, summary and OOF probs."""
    train = df.query("split == 'train'").copy()
    if verbose:
        log.info("CV on %d training reviews, %d features", len(train), len(feature_cols))
 
    oof = np.zeros(len(train))
    fold_metrics, rounds = [], []
    for fold in range(N_FOLDS):
        val_mask = (train["cv_fold"] == fold).values
        tr, va = train[~val_mask], train[val_mask]
        y_tr, y_va = tr[TARGET], va[TARGET]
 
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        fold_params = {**params, "scale_pos_weight": spw}
        dtrain = xgb.DMatrix(tr[feature_cols], label=y_tr)
        dval = xgb.DMatrix(va[feature_cols], label=y_va)
 
        bst = xgb.train(fold_params, dtrain, num_boost_round=num_boost_round,
                        evals=[(dval, "val")],
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
 
        # Use the best iteration explicitly (do not rely on library defaults).
        y_prob = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        oof[val_mask] = y_prob
        rounds.append(bst.best_iteration + 1)
 
        pr_auc = average_precision_score(y_va, y_prob)
        roc = roc_auc_score(y_va, y_prob)
        pg_auc = per_game_roc_auc(y_va, y_prob, va["appid"])
        fold_metrics.append({"fold": fold, "pr_auc": pr_auc, "roc_auc": roc,
                             "per_game_roc_auc": pg_auc, "prevalence": float(y_va.mean()),
                             "best_iteration": bst.best_iteration})
        if verbose:
            log.info("  fold %d  PR-AUC %.4f (base rate %.3f)  ROC-AUC %.4f  per-game %.4f  (iter %d)",
                     fold, pr_auc, y_va.mean(), roc, pg_auc, bst.best_iteration)
 
    fm = pd.DataFrame(fold_metrics)
    summary = {
        "cv_pr_auc_mean": fm["pr_auc"].mean(),
        "cv_pr_auc_std": fm["pr_auc"].std(),
        "cv_roc_auc_mean": fm["roc_auc"].mean(),
        "cv_roc_auc_std": fm["roc_auc"].std(),
        "cv_per_game_roc_auc_mean": fm["per_game_roc_auc"].mean(),
        "cv_prevalence_mean": fm["prevalence"].mean(),
    }
    if verbose:
        log.info("CV summary  PR-AUC %.4f +/- %.4f (base rate %.3f)  ROC-AUC %.4f +/- %.4f",
                 summary["cv_pr_auc_mean"], summary["cv_pr_auc_std"], summary["cv_prevalence_mean"],
                 summary["cv_roc_auc_mean"], summary["cv_roc_auc_std"])
    return {"folds": fold_metrics, "summary": summary, "oof_prob": oof,
            "oof_y": train[TARGET].values, "n_rounds": int(np.ceil(np.median(rounds)))}
 
 
# -- hyper-parameter tuning ---------------------------------------------------
def tune(df: pd.DataFrame, feature_cols: list[str], n_iter: int, frac: float) -> dict:
    """Random search scored on CV PR-AUC, run on a random SAMPLE OF GAMES to keep it fast.
 
    Whole games are sampled (never single reviews) so the group structure of cv_fold holds.
    """
    train = df[df["split"] == "train"]
    games = train["appid"].drop_duplicates()
    keep = games.sample(frac=frac, random_state=SEED)
    sub = train[train["appid"].isin(keep)]
    log.info("Tuning on %d of %d games (%d reviews)", len(keep), len(games), len(sub))
 
    rnd = random.Random(SEED)
    best_params, best_score = BASE_PARAMS, None
    for i in range(0, n_iter + 1):
        cand = BASE_PARAMS if i == 0 else {**BASE_PARAMS, **{k: rnd.choice(v) for k, v in SEARCH_SPACE.items()}}
        cv = cross_validate(sub, feature_cols, cand, verbose=False)
        score = cv["summary"]["cv_pr_auc_mean"]
        better = best_score is None or score > best_score
        log.info("tune  %2d/%d  CV PR-AUC %.4f %s", i, n_iter, score,
                 "(default)" if i == 0 else ("<- best" if better else ""))
        if better:
            best_params, best_score = cand, score
    return best_params
 
 
# -- test evaluation ----------------------------------------------------------
def evaluate_test(df: pd.DataFrame, feature_cols: list[str], params: dict,
                  n_rounds: int, threshold: float) -> dict:
    """Train on the FULL train split with the CV-chosen number of rounds, evaluate once on test."""
    train = df.query("split == 'train'").copy()
    test = df.query("split == 'test'").copy()
    log.info("Test evaluation: train %d -> test %d  (%d rounds)", len(train), len(test), n_rounds)
 
    y_tr, y_te = train[TARGET], test[TARGET].values
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    dtrain = xgb.DMatrix(train[feature_cols], label=y_tr)
    model = xgb.train({**params, "scale_pos_weight": spw}, dtrain,
                      num_boost_round=n_rounds, verbose_eval=False)
 
    y_prob = model.predict(xgb.DMatrix(test[feature_cols]))
    y_pred = (y_prob >= threshold).astype(int)
    prevalence = float(y_te.mean())
    pr_auc = average_precision_score(y_te, y_prob)
 
    m = {
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / prevalence,
        "prevalence": prevalence,
        "roc_auc": roc_auc_score(y_te, y_prob),
        "per_game_roc_auc": per_game_roc_auc(test[TARGET], y_prob, test["appid"]),
        "threshold": threshold,
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
    }
    log.info("TEST  PR-AUC %.4f (base rate %.3f, lift x%.2f)  ROC-AUC %.4f  per-game ROC-AUC %.4f",
             m["pr_auc"], prevalence, m["pr_auc_lift"], m["roc_auc"], m["per_game_roc_auc"])
    log.info("      at threshold %.3f: precision %.3f  recall %.3f  F1 %.3f",
             threshold, m["precision"], m["recall"], m["f1"])
    return {"model": model, "y_true": y_te, "y_prob": y_prob, "metrics": m, "n_rounds": n_rounds}
 
 
# -- curves and confusion matrix ----------------------------------------------
def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
 
    p, r, _ = precision_recall_curve(y_true, y_prob)
    axes[0].plot(r, p, color="#3498db")
    axes[0].axhline(y_true.mean(), ls="--", color="grey", label=f"base rate {y_true.mean():.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision",
                title=f"PR curve (AP {average_precision_score(y_true, y_prob):.3f})")
    axes[0].legend()
 
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="#3498db")
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate",
                title=f"ROC curve (AUC {roc_auc_score(y_true, y_prob):.3f})")
 
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    axes[2].imshow(norm, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xticks([0, 1], ["Not negative", "Negative"])
    axes[2].set_yticks([0, 1], ["Not negative", "Negative"])
    axes[2].set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {threshold:.2f})")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.0%}", ha="center", va="center",
                         color="white" if norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    fig.savefig(FIGURES / "review_xgb_curves.png", dpi=150)
    plt.close(fig)
    log.info("Saved PR/ROC curves + confusion matrix -> figures/models/review_xgb_curves.png")
 
 
# -- SHAP interpretation ------------------------------------------------------
def shap_analysis(model: xgb.Booster, df_test: pd.DataFrame, feature_cols: list[str],
                  top_n: int = 20, sample_n: int = 10_000) -> pd.DataFrame:
    """SHAP feature importance on a sample of test reviews (for speed)."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
 
    sample = df_test.sample(sample_n, random_state=SEED) if len(df_test) > sample_n else df_test
    X = sample[feature_cols]
    shap_values = model.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]  # drop bias
 
    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(RESULTS / "review_xgb_shap_importance.csv", index=False)
 
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance.head(top_n)
    ax.barh(range(len(top)), top["mean_abs_shap"].values, color="#3498db", alpha=0.85)
    ax.set_yticks(range(len(top)), top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Review-Level XGBoost - Top Feature Importances (SHAP)")
    plt.tight_layout()
    fig.savefig(FIGURES / "review_xgb_shap_bar.png", dpi=150)
    plt.close(fig)
 
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols,
                      max_display=top_n, show=False, plot_type="dot")
    plt.title("Review-Level XGBoost - SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(FIGURES / "review_xgb_shap_beeswarm.png", dpi=150)
    plt.close()
    log.info("Saved SHAP plots + importance table")
    return importance
 
 
# -- save protocol-format results ---------------------------------------------
def save_results(cv: dict, test: dict | None, params: dict, n_features: int, tuned: bool) -> None:
    """Write reports/results/review_xgboost.csv in the shared protocol table format."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    s = cv["summary"]
 
    notes = [f"base rate {s['cv_prevalence_mean']:.3f}",
             f"ROC-AUC CV {s['cv_roc_auc_mean']:.3f}+/-{s['cv_roc_auc_std']:.3f}",
             f"per-game median ROC-AUC CV {s['cv_per_game_roc_auc_mean']:.3f}"]
    if test:
        m = test["metrics"]
        notes.append(f"test ROC-AUC {m['roc_auc']:.3f}; test per-game ROC-AUC {m['per_game_roc_auc']:.3f}; "
                     f"PR-AUC lift x{m['pr_auc_lift']:.2f}")
        notes.append(f"threshold {m['threshold']:.3f} (max-F1 on OOF): precision {m['precision']:.3f}, "
                     f"recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
        notes.append(f"{test['n_rounds']} rounds, refit on full train")
    notes.append(f"seed {SEED}")
    notes.append(("tuned " if tuned else "default ") +
                 json.dumps({k: v for k, v in params.items() if k != "seed"}))
 
    row = {
        "model": "XGBoost",
        "task": "review",
        "feature_set": f"review_features + one-hot language ({n_features} cols)",
        "cv_mean": f"{s['cv_pr_auc_mean']:.4f} +/- {s['cv_pr_auc_std']:.4f}",
        "test": f"{test['metrics']['pr_auc']:.4f}" if test else "",
        "notes": "; ".join(notes),
    }
    pd.DataFrame([row]).to_csv(RESULTS / "review_xgboost.csv", index=False)
    if tuned:
        (RESULTS / "review_xgboost_best_params.json").write_text(json.dumps(params, indent=2))
    log.info("Saved results -> reports/results/review_xgboost.csv")
 
 
# -- main ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Review-level XGBoost model")
    parser.add_argument("--cv", action="store_true", help="Run CV only, skip test evaluation")
    parser.add_argument("--tune", type=int, default=0, metavar="N",
                        help="Random-search N parameter sets on a sample of games first")
    parser.add_argument("--tune-frac", type=float, default=0.25,
                        help="Fraction of training GAMES used during tuning (default 0.25)")
    args = parser.parse_args()
 
    df = load_review_data()
    check_group_integrity(df)
    feature_cols = _feature_cols(df)
    log.info("Using %d features", len(feature_cols))
    save_feature_list(feature_cols)
 
    params = tune(df, feature_cols, args.tune, args.tune_frac) if args.tune > 0 else BASE_PARAMS
    cv = cross_validate(df, feature_cols, params)
 
    test = None
    if not args.cv:
        threshold, oof_f1 = best_f1_threshold(cv["oof_y"], cv["oof_prob"])
        log.info("Threshold from out-of-fold predictions: %.3f (OOF F1 %.3f)", threshold, oof_f1)
 
        test = evaluate_test(df, feature_cols, params, cv["n_rounds"], threshold)
        plot_curves(test["y_true"], test["y_prob"], threshold)
        importance = shap_analysis(test["model"], df.query("split == 'test'").copy(), feature_cols)
        log.info("Top-5 features by SHAP:\n%s", importance.head().to_string(index=False))
 
    save_results(cv, test, params, len(feature_cols), args.tune > 0)
    log.info("Done.")
 
 
if __name__ == "__main__":
    main()
 