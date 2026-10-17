"""Grid search for the reported models (reviewer comment from Review 1: the hyperparameters were fixed).

  python -m src.models.grid_search --task review --model lgbm
  python -m src.models.grid_search --task review --model xgboost
  python -m src.models.grid_search --task review --model rf --games-frac 0.5
  python -m src.models.grid_search --task review --model logreg
  python -m src.models.grid_search --task reviewtext --model lgbm       # metadata, and metadata + text components
  python -m src.models.grid_search --task reviewtext --model logreg --games-frac 0.5
  python -m src.models.grid_search --task game --model rf        # also logreg, lgbm, xgboost (macro-F1, small data)

Rules (docs/hyperparameter_tuning_plan.md): select on the 5 shared CV folds inside train only, metric PR-AUC (macro-F1 for games), seed 42,
every grid point recorded, then refit on the full train, set the threshold from out-of-fold predictions and score the test
set once. The default setting is evaluated as grid point 0, so the table shows default against tuned.

Writes reports/results/tuning_<task>_<model>.csv (one row per grid point, resumable) and the usual results row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import game_models as gm
from . import game_xgboost as gx
from . import review_models as rm
from . import review_text_models as rtm
from . import review_xgboost as rx
from .zoo import DEFAULTS, LABELS

log = logging.getLogger("grid_search")
RESULTS = rx.RESULTS
SEED = rx.SEED

LGBM_GRID = {"num_leaves": [15, 31, 63, 127], "min_child_samples": [50, 200, 500], "learning_rate": [0.03, 0.1]}
GRIDS: dict[tuple[str, str], dict] = {
    ("review", "logreg"): {"C": [0.001, 0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    ("review", "rf"): {"min_samples_leaf": [5, 20, 50, 200], "max_features": ["sqrt", 0.3, 0.5], "n_estimators": [300]},
    ("review", "lgbm"): LGBM_GRID,
    ("review", "xgboost"): {"max_depth": [4, 6, 8], "min_child_weight": [1, 10, 50], "learning_rate": [0.03, 0.1], "sub_col": [0.7, 0.9]},
    ("game", "logreg"): {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l1", "l2"]},
    ("game", "rf"): {"min_samples_leaf": [1, 2, 3, 5, 10], "max_features": ["sqrt", 0.3, 0.5], "n_estimators": [300, 600], "max_depth": [None, 8]},
    ("game", "lgbm"): {"num_leaves": [4, 8, 16], "min_child_samples": [5, 10, 20], "learning_rate": [0.03, 0.1], "reg_lambda": [0.0, 1.0, 5.0]},
    ("game", "xgboost"): {"max_depth": [2, 3, 4, 6], "min_child_weight": [1, 5, 10], "learning_rate": [0.03, 0.1], "subsample": [0.7, 1.0], "gamma": [0.0, 1.0]},
    ("reviewtext", "lgbm"): LGBM_GRID,
    ("reviewtext", "logreg"): {"C": [0.3, 1.0, 3.0, 10.0, 30.0], "ngram_range": [(1, 1), (1, 2)], "max_features": [20_000, 50_000, 100_000]},
}


def expand(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def jdump(d: dict) -> str:
    return json.dumps(d, sort_keys=True, default=str)


def load_done(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    return {r["params"]: r.to_dict() for _, r in t.iterrows()}


def finish_table(rows: list[dict], path) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    t["rank"] = t["cv_mean"].rank(ascending=False, method="min").astype(int)
    t.sort_values("rank").to_csv(path, index=False)
    return t


def search(points: list[dict], run_point, path, label: str) -> tuple[dict, pd.DataFrame]:
    """Evaluate every point (resuming from `path`), return the best parameters and the table."""
    done, rows = load_done(path), []
    for i, p in enumerate(points):
        key = jdump(p)
        if key in done:
            rows.append(done[key]); log.info("%s [%d/%d] cached  cv %.4f", label, i + 1, len(points), done[key]["cv_mean"]); continue
        t0 = time.time()
        res = run_point(p)
        row = {"params": key, "is_default": i == 0, "cv_mean": res["cv_mean"], "cv_sd": res["cv_sd"], "cv_roc_auc": res.get("roc"),
               "rounds": res.get("rounds"), "fit_seconds": round(time.time() - t0, 1)}
        rows.append(row)
        finish_table(rows, path)
        log.info("%s [%d/%d] cv %.4f +/- %.4f  %.0fs  %s", label, i + 1, len(points), row["cv_mean"], row["cv_sd"], row["fit_seconds"], key)
    t = finish_table(rows, path)
    best = t.sort_values(["cv_mean", "is_default"], ascending=[False, False]).iloc[0]
    log.info("%s best %s  cv %.4f (default %.4f)", label, best["params"], best["cv_mean"], t[t.is_default].cv_mean.iloc[0])
    return json.loads(best["params"]), t


def sample_games(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    games = df.loc[df["split"] == "train", "appid"].unique()
    keep = np.random.RandomState(SEED).choice(games, int(frac * len(games)), replace=False)
    return df[df["appid"].isin(keep) | (df["split"] == "test")]


# ---- zoo models (Logistic Regression, Random Forest, LightGBM) on review features or text-subset features
def run_zoo(task: str, name: str, frac: float) -> None:
    grid = GRIDS[(task, name)]
    if task == "review":
        df = rx.load_review_data(); rx.check_group_integrity(df)
        feature_sets = {None: rx._feature_cols(df)}
    else:
        df, meta, tcols, _ = rtm.load_subset()
        feature_sets = {"meta": meta, "meta_svd": meta + tcols}
    sub = sample_games(df, frac)
    for fs_key, feats in feature_sets.items():
        tag = f"{task}_{name}" + (f"_{fs_key}" if fs_key else "")
        path = RESULTS / f"tuning_{tag}.csv"
        default = dict(DEFAULTS["review"][name]) if task == "review" or name != "lgbm" else {"num_leaves": 31, "min_child_samples": 100}
        points = [default] + [p for p in expand(grid) if p != default]

        def run_point(p):
            cv = rm.cross_validate(sub, feats, name, p)["summary"]
            return {"cv_mean": cv["pr_auc"][0], "cv_sd": cv["pr_auc"][1], "roc": cv["roc_auc"][0]}

        best, _ = search(points, run_point, path, tag)
        cv = rm.cross_validate(df, feats, name, best)                        # full train: OOF for the threshold
        thr, _ = rx.best_f1_threshold(cv["oof_y"], cv["oof_prob"])
        test = rm.evaluate_test(df, feats, name, best, cv["n_rounds"], thr)  # the single test evaluation
        log.info("%s TEST PR-AUC %.4f ROC-AUC %.4f", tag, test["pr_auc"], test["roc_auc"])
        if task == "review":
            rm.save_results(name, cv, test, best, len(feats), True)
        else:
            rtm.save(fs_key, cv, test, best, tuned=True)


# ---- game level: macro-F1 on the shared folds; the class scales come from out-of-fold predictions
GAME_DEFAULTS = {"logreg": {"C": 0.1, "penalty": "l2"}, "rf": {"min_samples_leaf": 3, "max_features": "sqrt", "n_estimators": 300, "max_depth": None},
                 "lgbm": {"num_leaves": 8, "min_child_samples": 10, "learning_rate": 0.05, "reg_lambda": 1.0}}


def run_game(name: str) -> None:
    df = gx.load_game_data()
    feats = gx._feature_cols(df)
    tag = f"game_{name}"
    path = RESULTS / f"tuning_{tag}.csv"
    if name == "xgboost":
        base = {k: gx.BASE_PARAMS[k] for k in ("max_depth", "min_child_weight", "learning_rate", "subsample", "gamma")}
        points = [base] + [p for p in expand(GRIDS[("game", name)]) if p != base]

        def run_point(p):
            cv = gx.cross_validate(df, feats, {**gx.BASE_PARAMS, **p}, verbose=False)
            s = cv["summary"]
            return {"cv_mean": s["cv_macro_f1_mean"], "cv_sd": s["cv_macro_f1_std"], "rounds": cv["n_rounds"]}
    else:
        default = GAME_DEFAULTS[name]
        points = [default] + [p for p in expand(GRIDS[("game", name)]) if p != default]

        def run_point(p):
            s = gm.cross_validate(df, feats, name, p)["summary"]["macro_f1"]
            return {"cv_mean": s[0], "cv_sd": s[1]}

    best, _ = search(points, run_point, path, tag)
    if name == "xgboost":
        params = {**gx.BASE_PARAMS, **best}
        cv = gx.cross_validate(df, feats, params, verbose=False)
        scale, _ = gx.tune_class_scales(cv["oof_y"], cv["oof_prob"])
        test = gx.evaluate_test(df, feats, params, cv["n_rounds"], scale)
        gx.save_results(cv, test, params, len(feats), "all", "", True)
    else:
        cv = gm.cross_validate(df, feats, name, best)
        scale, _ = gx.tune_class_scales(cv["oof_y"], cv["oof_prob"])
        test = gm.evaluate_test(df, feats, name, best, cv["n_rounds"], scale)
        gm.save_results(name, cv, test, best, len(feats), True)
    log.info("%s TEST macro-F1 %.4f", tag, test["metrics"]["macro_f1"])


# ---- XGBoost on review features
def run_xgb(frac: float) -> None:
    df = rx.load_review_data(); rx.check_group_integrity(df)
    feats = rx._feature_cols(df)
    sub = sample_games(df, frac)

    def to_params(p):
        q = {k: v for k, v in p.items() if k != "sub_col"}
        if "sub_col" in p:
            q.update(subsample=p["sub_col"], colsample_bytree=p["sub_col"])
        return {**rx.BASE_PARAMS, **q}

    base = {k: rx.BASE_PARAMS[k] for k in ("max_depth", "min_child_weight", "learning_rate")}
    base["sub_col"] = 0.8                       # the defaults use subsample = colsample_bytree = 0.8
    points = [base] + [p for p in expand(GRIDS[("review", "xgboost")])]

    def run_point(p):
        cv = rx.cross_validate(sub, feats, to_params(p), verbose=False)
        s = cv["summary"]
        return {"cv_mean": s["cv_pr_auc_mean"], "cv_sd": s["cv_pr_auc_std"], "roc": s["cv_roc_auc_mean"], "rounds": cv["n_rounds"]}

    best, _ = search(points, run_point, RESULTS / "tuning_review_xgboost.csv", "review_xgboost")
    params = to_params(best)
    cv = rx.cross_validate(df, feats, params, verbose=False)
    thr, _ = rx.best_f1_threshold(cv["oof_y"], cv["oof_prob"])
    test = rx.evaluate_test(df, feats, params, cv["n_rounds"], thr)
    log.info("review_xgboost TEST PR-AUC %.4f", test["metrics"]["pr_auc"])
    rx.save_results(cv, test, params, len(feats), True)


# ---- text Logistic Regression on TF-IDF + metadata
def run_text_lr(frac: float) -> None:
    d, meta, _, vec0 = rtm.load_subset()
    train_idx = np.where((d["split"] == "train").values)[0]
    games = d["appid"].iloc[train_idx].unique()
    keep = set(np.random.RandomState(SEED).choice(games, int(frac * len(games)), replace=False)) if frac < 1 else set(games)
    sub = train_idx[d["appid"].iloc[train_idx].isin(keep).values]
    base_kw = {k: v for k, v in vec0.get_params().items() if k in ("min_df", "max_df", "sublinear_tf", "token_pattern", "lowercase", "norm", "use_idf", "smooth_idf")}
    path = RESULTS / "tuning_reviewtext_logreg.csv"
    done, rows = load_done(path), []
    default = {"C": 3.0, "ngram_range": (1, 2), "max_features": 50_000}
    variants = [(default["ngram_range"], default["max_features"])] + [
        (n, m) for n in GRIDS[("reviewtext", "logreg")]["ngram_range"] for m in GRIDS[("reviewtext", "logreg")]["max_features"]
        if (n, m) != (default["ngram_range"], default["max_features"])]
    for n, m in variants:
        todo = [C for C in GRIDS[("reviewtext", "logreg")]["C"] if jdump({"C": C, "max_features": m, "ngram_range": list(n)}) not in done]
        if default["ngram_range"] == n and default["max_features"] == m and default["C"] not in GRIDS[("reviewtext", "logreg")]["C"]:
            todo.append(default["C"])
        if not todo:
            rows += [r for k, r in done.items() if json.loads(k)["max_features"] == m and tuple(json.loads(k)["ngram_range"]) == n]; continue
        t0 = time.time()
        vec = TfidfVectorizer(ngram_range=n, max_features=m, dtype=np.float32, **base_kw)
        vec.fit(d["review"].iloc[sub])                        # vocabulary and idf from the search games' training reviews only
        X = rtm.tfidf_matrix(d, vec, meta)
        log.info("variant ngram %s max_features %d: %d terms, built in %.0fs", n, m, len(vec.vocabulary_), time.time() - t0)
        for C in todo:
            t1 = time.time()
            s = rtm.lr_cross_validate(d, X, C, rows=sub)["summary"]
            p = {"C": C, "max_features": m, "ngram_range": list(n)}
            rows.append({"params": jdump(p), "is_default": (C == default["C"] and n == default["ngram_range"] and m == default["max_features"]),
                         "cv_mean": s["pr_auc"][0], "cv_sd": s["pr_auc"][1], "cv_roc_auc": s["roc_auc"][0], "rounds": None,
                         "fit_seconds": round(time.time() - t1, 1)})
            finish_table(rows, path)
            log.info("text LR ngram %s max_features %d C %s  cv PR-AUC %.4f", n, m, C, s["pr_auc"][0])
    t = finish_table(rows, path)
    best = t.sort_values(["cv_mean", "is_default"], ascending=[False, False]).iloc[0]
    bp = json.loads(best["params"]); n, m, C = tuple(bp["ngram_range"]), bp["max_features"], bp["C"]
    log.info("text LR best %s cv %.4f", best["params"], best["cv_mean"])
    vec = TfidfVectorizer(ngram_range=n, max_features=m, dtype=np.float32, **base_kw).fit(d["review"].iloc[train_idx])
    X = rtm.tfidf_matrix(d, vec, meta)
    cv = rtm.lr_cross_validate(d, X, C)                       # full train: OOF for the threshold
    thr, _ = rx.best_f1_threshold(cv["oof_y"], cv["oof_prob"])
    test = rtm.lr_test(d, X, C, thr)
    log.info("text LR TEST PR-AUC %.4f ROC-AUC %.4f", test["pr_auc"], test["roc_auc"])
    rtm.save("tfidf_lr", cv, test, {"C": C, "ngram_range": list(n), "max_features": m}, tuned=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid search for a reported model")
    ap.add_argument("--task", choices=["review", "reviewtext", "game"], required=True)
    ap.add_argument("--model", choices=["logreg", "rf", "lgbm", "xgboost"], required=True)
    ap.add_argument("--games-frac", type=float, default=1.0, help="share of training games used to compare grid points")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if a.task == "game":
        run_game(a.model)
    elif a.model == "xgboost":
        run_xgb(a.games_frac)
    elif a.task == "reviewtext" and a.model == "logreg":
        run_text_lr(a.games_frac)
    else:
        run_zoo(a.task, a.model, a.games_frac)


if __name__ == "__main__":
    main()
