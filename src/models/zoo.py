"""Estimators for the model comparison: Dummy baseline, Logistic Regression, Random Forest, LightGBM.

One uniform wrapper (`Model`) so the review-level and game-level runners treat every model the same way:
  fit(X, y, X_val, y_val, n_rounds)   LightGBM stops early on the validation fold, like the XGBoost runs;
                                      pass n_rounds to refit with a fixed number of rounds
  score(X)                            review task: P(negative) as a vector; game task: an (n, 3) probability matrix
Every model gets the same class-imbalance treatment as the XGBoost runs (balanced weights), and the
same cross-validation, threshold and test rules (docs/modelling_protocol.md).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
MAX_ROUNDS = 400
EARLY_STOPPING_ROUNDS = 25

LABELS = {"dummy": "Dummy (prior)", "logreg": "Logistic Regression", "rf": "Random Forest", "lgbm": "LightGBM"}

# Small search grids, scored by cross-validation on the training part only.
GRIDS = {
    "review": {
        "dummy": [{}],
        "logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "rf": [{"min_samples_leaf": leaf, "max_features": mf} for leaf in (10, 50, 200) for mf in (0.3, "sqrt")],
        "lgbm": [{"num_leaves": n, "min_child_samples": m} for n in (15, 31, 63) for m in (20, 200)],
    },
    "game": {
        "dummy": [{}],
        "logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "rf": [{"min_samples_leaf": leaf, "max_features": mf} for leaf in (1, 3, 5, 10) for mf in (0.3, "sqrt")],
        "lgbm": [{"num_leaves": n, "min_child_samples": m} for n in (4, 8, 16) for m in (5, 10, 20)],
    },
}
DEFAULTS = {
    "review": {"dummy": {}, "logreg": {"C": 1.0}, "rf": {"min_samples_leaf": 50, "max_features": "sqrt"},
               "lgbm": {"num_leaves": 31, "min_child_samples": 100}},
    "game": {"dummy": {}, "logreg": {"C": 0.1}, "rf": {"min_samples_leaf": 3, "max_features": "sqrt"},
             "lgbm": {"num_leaves": 8, "min_child_samples": 10}},
}


class Model:
    """Uniform wrapper around the four estimators (see the module docstring)."""

    def __init__(self, name: str, task: str, params: dict | None = None):
        if name not in LABELS or task not in GRIDS:
            raise ValueError(f"unknown model {name!r} or task {task!r}")
        self.name, self.task = name, task
        self.params = {**DEFAULTS[task][name], **(params or {})}
        self.rounds_: int | None = None
        self.est = None

    def _build(self, y):
        p, review = self.params, self.task == "review"
        if self.name == "dummy":
            return DummyClassifier(strategy="prior")
        if self.name == "logreg":
            return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                 LogisticRegression(C=p["C"], class_weight="balanced", max_iter=300))
        if self.name == "rf":
            return RandomForestClassifier(
                n_estimators=150 if review else 300, min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                max_samples=0.25 if review else None, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
        # Early stopping uses the same metric as the XGBoost runs (PR-AUC for reviews, multi-class log-loss for games).
        # Judged on log-loss, a class-weighted review model stops at round 1, because weighting worsens log-loss at once.
        weight = ({"scale_pos_weight": float((y == 0).sum() / max((y == 1).sum(), 1)), "metric": "average_precision"} if review
                  else {"class_weight": "balanced", "metric": "multi_logloss"})
        return lgb.LGBMClassifier(
            n_estimators=MAX_ROUNDS, learning_rate=0.05, num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=SEED,
            verbose=-1, **weight)

    def fit(self, X, y, X_val=None, y_val=None, n_rounds: int | None = None) -> "Model":
        est = self._build(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.name == "lgbm":
                if n_rounds is None and X_val is not None:
                    est.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
                    self.rounds_ = int(est.best_iteration_)
                else:
                    est.set_params(n_estimators=int(n_rounds or MAX_ROUNDS))
                    est.fit(X, y)
                    self.rounds_ = int(n_rounds or MAX_ROUNDS)
            else:
                est.fit(X, y)
        self.est = est
        return self

    def score(self, X) -> np.ndarray:
        p = self.est.predict_proba(X)
        return p[:, 1] if self.task == "review" else p
