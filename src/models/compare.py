"""Collect every model's results file into one comparison: tables, CSVs and a figure.

  python -m src.models.compare

Reads reports/results/{review,game}_<model>.csv (protocol table format, written by review_xgboost.py,
game_xgboost.py, review_models.py and game_models.py) and writes:
  reports/results/comparison_review.csv, comparison_game.csv
  figures/models/model_comparison.png
  docs/model_comparison.md
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS, FIGURES, DOCS = ROOT / "reports" / "results", ROOT / "figures" / "models", ROOT / "docs"
ORDER = ["dummy", "logreg", "rf", "lgbm", "xgboost"]
NAMES = {"dummy": "Dummy (prior)", "logreg": "Logistic Regression", "rf": "Random Forest", "lgbm": "LightGBM", "xgboost": "XGBoost"}

# Written by hand after reading the numbers; keep it in step with the tables.
NOTES = Path(ROOT / "docs" / "model_comparison_notes.md")


def num(pattern: str, text: str, group: int = 1) -> float:
    m = re.search(pattern, text)
    return float(m.group(group)) if m else float("nan")


def cv_parts(s: str) -> tuple[float, float]:
    m = re.match(r"\s*([\d.]+)\s*\+/-\s*([\d.]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


def load(task: str) -> pd.DataFrame:
    rows = []
    for key in ORDER:
        path = RESULTS / f"{task}_{key}.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path).iloc[0]
        notes, (cv, sd) = str(r["notes"]), cv_parts(r["cv_mean"])
        row = {"key": key, "Model": NAMES[key], "CV mean": cv, "CV sd": sd, "Test": float(r["test"]),
               "Tuned": "tuned" if "tuned {" in notes.split("seed")[-1] else "default settings"}
        if task == "review":
            row |= {"Test ROC-AUC": num(r"test ROC-AUC ([\d.]+)", notes), "Per-game ROC-AUC": num(r"test per-game ROC-AUC ([\d.]+)", notes),
                    "Precision": num(r"precision ([\d.]+), recall", notes), "Recall": num(r"recall ([\d.]+), F1", notes),
                    "F1": num(r"F1 ([\d.]+)", notes)}
        else:
            rec = re.search(r"recall Str/Sol/Stg ([\d.]+)/([\d.]+)/([\d.]+)", notes)
            row |= {"Test bal-acc": num(r"test bal-acc ([\d.]+)", notes),
                    "Recall Struggling": float(rec.group(1)) if rec else np.nan, "Recall Solid": float(rec.group(2)) if rec else np.nan,
                    "Recall Strong": float(rec.group(3)) if rec else np.nan,
                    "OvR AUC Struggling": num(r"OvR AUC Str ([\d.]+)", notes), "CI low": num(r"boot 95% \[([\d.]+),", notes),
                    "CI high": num(r"boot 95% \[[\d.]+, ([\d.]+)\]", notes),
                    "Tuned-scale macro-F1": num(r"tuned class scales[^:]*: test macro-F1 ([\d.]+)", notes),
                    "Tuned-scale Struggling recall": num(r"tuned class scales[^:]*: test macro-F1 [\d.]+, Struggling recall ([\d.]+)", notes)}
        rows.append(row)
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, spec: list[tuple[str, callable]]) -> str:
    out = ["| " + " | ".join(h for h, _ in spec) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(spec) - 1)) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r) for _, f in spec) + " |")
    return "\n".join(out)


def figure(rev: pd.DataFrame, gm: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = ["#8C8C8C", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    for ax, df, title, ylabel in ((axes[0], rev, "Review level: will the review be negative?", "PR-AUC (higher is better)"),
                                  (axes[1], gm, "Game level: success tier (3 classes)", "macro-F1 (higher is better)")):
        x = np.arange(len(df))
        ax.bar(x, df["Test"], color=[colors[ORDER.index(k)] for k in df["key"]], width=0.62, label="test (scored once)")
        ax.errorbar(x, df["CV mean"], yerr=df["CV sd"], fmt="D", color="black", capsize=4, ms=5, label="cross-validation mean ± sd")
        if "CI low" in df:
            ax.errorbar(x + 0.22, df["Test"], yerr=[df["Test"] - df["CI low"], df["CI high"] - df["Test"]], fmt="none", ecolor="#333333",
                        capsize=3, lw=1, label="test 95% bootstrap interval")
        for xi, v in zip(x, df["Test"]):
            ax.text(xi, 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")
        ax.set_ylim(0, float(max(df["Test"].max(), (df["CV mean"] + df["CV sd"]).max(), df.get("CI high", df["Test"]).max())) * 1.32)
        ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n", 1) for n in df["Model"]], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Model comparison on the shared train/test split (the Dummy bar is the baseline to beat)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGURES / "model_comparison.png", dpi=150); plt.close(fig)


def main() -> None:
    rev, gm = load("review"), load("game")
    rev.drop(columns="key").to_csv(RESULTS / "comparison_review.csv", index=False)
    gm.drop(columns="key").to_csv(RESULTS / "comparison_game.csv", index=False)
    f3 = lambda c: (lambda r: f"{r[c]:.3f}")
    rev_t = table(rev, [("Model", lambda r: r["Model"]), ("Settings", lambda r: r["Tuned"]),
                        ("CV PR-AUC (mean ± sd)", lambda r: f"{r['CV mean']:.3f} ± {r['CV sd']:.3f}"), ("Test PR-AUC", f3("Test")),
                        ("Test ROC-AUC", f3("Test ROC-AUC")), ("Per-game ROC-AUC", f3("Per-game ROC-AUC")),
                        ("Precision / recall / F1 at the CV threshold", lambda r: f"{r['Precision']:.2f} / {r['Recall']:.2f} / {r['F1']:.2f}")])
    gm_t = table(gm, [("Model", lambda r: r["Model"]), ("Settings", lambda r: r["Tuned"]),
                      ("CV macro-F1 (mean ± sd)", lambda r: f"{r['CV mean']:.3f} ± {r['CV sd']:.3f}"),
                      ("Test macro-F1 [95% interval]", lambda r: f"{r['Test']:.3f} [{r['CI low']:.2f}, {r['CI high']:.2f}]"),
                      ("Test balanced accuracy", f3("Test bal-acc")),
                      ("Recall: Struggling / Solid / Strong", lambda r: f"{r['Recall Struggling']:.2f} / {r['Recall Solid']:.2f} / {r['Recall Strong']:.2f}"),
                      ("AUC Struggling vs rest", f3("OvR AUC Struggling")),
                      ("With tuned class scales: macro-F1 / Struggling recall", lambda r: f"{r['Tuned-scale macro-F1']:.3f} / {r['Tuned-scale Struggling recall']:.2f}")])
    figure(rev, gm)
    notes = NOTES.read_text(encoding="utf-8") if NOTES.exists() else ""
    DOCS.joinpath("model_comparison.md").write_text(
        "# Model comparison\n\n"
        "Generated by `python -m src.models.compare` from `reports/results/`. Every model uses the same features, the shared\n"
        "train/test split by game and the same 5 cross-validation folds (`docs/modelling_protocol.md`); the test set was scored once.\n"
        "The Dummy row is the baseline to beat.\n\n"
        "![Model comparison](../figures/models/model_comparison.png)\n\n"
        "## Review level: will the review be negative? (1,257,095 training / 316,341 test reviews)\n\n" + rev_t + "\n\n"
        "## Game level: success tier (1,191 training / 299 test games)\n\n" + gm_t + "\n\n" + notes, encoding="utf-8")
    print(rev_t); print(); print(gm_t)


if __name__ == "__main__":
    main()
