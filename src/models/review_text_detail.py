"""Test-set detail for the text model: accuracy, balanced accuracy, confusion matrix and the baseline accuracy.

  python -m src.models.review_text_detail

Refits the Logistic Regression on the full TF-IDF text exactly as review_text_models.py did (C = 3.0) and applies
the threshold that script chose on the out-of-fold predictions (0.788). Nothing is tuned here: the threshold is read
from reports/results/reviewtext_tfidf_lr.csv, and the test set is scored once. Accuracy alone is misleading on
10% positives, so the always-positive baseline is reported next to it.
Writes reports/results/reviewtext_tfidf_lr_detail.csv and figures/models/review_text_confusion.png
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

from .review_text_models import RESULTS, TARGET, _fit_score, load_subset, tfidf_matrix

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("review_text_detail")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    saved = pd.read_csv(RESULTS / "reviewtext_tfidf_lr.csv").iloc[0]["notes"]
    threshold = float(re.search(r"threshold ([\d.]+)", saved).group(1))
    c = float(re.search(r'"C": ([\d.]+)', saved).group(1))
    log.info("threshold %.3f, C %.1f (read from the saved results)", threshold, c)
    d, meta, _, vec = load_subset()
    X = tfidf_matrix(d, vec, meta)
    tr, te = np.where(d["split"] == "train")[0], np.where(d["split"] == "test")[0]
    y = d[TARGET].values
    prob = _fit_score(X, y, tr, te, c)
    pred = (prob >= threshold).astype(int)
    yt = y[te]
    cm = confusion_matrix(yt, pred)
    out = {"n_test": len(te), "base_rate": yt.mean(), "baseline_accuracy_always_positive": 1 - yt.mean(),
           "accuracy": accuracy_score(yt, pred), "balanced_accuracy": balanced_accuracy_score(yt, pred),
           "precision": precision_score(yt, pred), "recall": recall_score(yt, pred), "f1": f1_score(yt, pred),
           "specificity": cm[0, 0] / cm[0].sum(), "mcc": matthews_corrcoef(yt, pred), "threshold": threshold,
           "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}
    pd.DataFrame([out]).to_csv(RESULTS / "reviewtext_tfidf_lr_detail.csv", index=False)
    for k, v in out.items():
        log.info("%-36s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["positive", "negative"])
    ax.set_yticks([0, 1], ["positive", "negative"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual review")
    ax.set_title("Text model, test set (row-normalised)", fontsize=10)
    fig.tight_layout()
    (ROOT / "figures" / "models").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / "models" / "review_text_confusion.png", dpi=170)


if __name__ == "__main__":
    main()
