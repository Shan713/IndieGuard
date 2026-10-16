"""Blind manual check of the topic model (card #33): build the labelling sheet, then score it.

  python -m src.analysis.topic_manual_validation sheet     # writes the blind sheet (no model topic, no notes)
  python -m src.analysis.topic_manual_validation score     # reads the filled sheet, writes agreement tables

The sheet reuses the 200 review IDs of the first sample (25 per topic by the model's assignment), shuffled, with the model's topic
hidden. A person reads each review and writes the topic number (0 to 7) that is its main theme, or 8 for "other / unclear", plus a
short note in their own words. `score` joins the labels to the model's topic and writes per-topic agreement, a confusion matrix and
Cohen's kappa. The labeller must not look at `topic_manual_validation_200.csv`, which holds the model's topic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "reports" / "results"
MODEL_SAMPLE = RES / "topic_manual_validation_200.csv"       # holds the model's topic per review (first sample)
SHEET = RES / "topic_manual_labelling_sheet.csv"
OUT = RES / "topic_manual_validation_blind.csv"
SUMMARY = RES / "topic_manual_validation_blind_summary.csv"
CONFUSION = RES / "topic_manual_validation_blind_confusion.csv"
SEED = 7
OTHER = 8


def build_sheet() -> None:
    ids = pd.read_csv(MODEL_SAMPLE, usecols=["recommendationid"]).recommendationid
    parts = [pd.read_parquet(p, columns=["recommendationid", "review"]) for p in sorted((ROOT / "data/processed/reviews_clean").glob("*.parquet"))]
    text = pd.concat(parts)
    text = text[text.recommendationid.isin(ids)]
    sheet = text.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sheet["review"] = sheet.review.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 700)
    sheet.insert(0, "sample_id", np.arange(1, len(sheet) + 1))
    sheet["your_topic"] = ""      # 0 to 7, or 8 for other / unclear
    sheet["your_note"] = ""       # one short note in your own words
    sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"wrote {SHEET.relative_to(ROOT)} ({len(sheet)} reviews, shuffled, model topic hidden)")


def score() -> None:
    lab = pd.read_csv(SHEET)
    lab["your_topic"] = pd.to_numeric(lab.your_topic, errors="coerce")
    missing = int(lab.your_topic.isna().sum())
    if missing:
        sys.exit(f"{missing} reviews have no label yet: fill `your_topic` for all {len(lab)} first.")
    if not lab.your_topic.isin(range(0, 9)).all():
        sys.exit("`your_topic` must be a whole number from 0 to 8.")
    if lab.your_note.fillna("").str.strip().str.len().lt(3).any() or lab.your_note.nunique() < 0.5 * len(lab):
        sys.exit("Write a short, different note for each review (at least half of the notes must be distinct).")
    ref = pd.read_csv(MODEL_SAMPLE)[["recommendationid", "model_topic_id", "model_topic_label", "sentiment"]]
    d = lab.merge(ref, on="recommendationid", validate="one_to_one")
    d["agree"] = (d.your_topic == d.model_topic_id).astype(int)
    d.rename(columns={"your_topic": "manual_topic_id", "your_note": "manual_note"}).to_csv(OUT, index=False, encoding="utf-8-sig")

    by = d.groupby("model_topic_id").agg(label=("model_topic_label", "first"), sampled=("agree", "size"), agreed=("agree", "sum"))
    by["agreement_rate"] = by.agreed / by.sampled
    conf = pd.crosstab(d.model_topic_id, d.your_topic).reindex(index=range(8), columns=range(9), fill_value=0)
    po = d.agree.mean()
    pe = sum((conf.sum(axis=1)[k] / len(d)) * (conf.sum(axis=0)[k] / len(d)) for k in range(8))
    kappa = (po - pe) / (1 - pe)
    by.loc["overall"] = ["all topics", len(d), int(d.agree.sum()), po]
    by["cohen_kappa"] = [np.nan] * 8 + [kappa]
    by.to_csv(SUMMARY)
    conf.to_csv(CONFUSION)
    print(by.round(3).to_string())
    print(f"\nlabelled as other / unclear: {(d.your_topic == OTHER).sum()} of {len(d)}; overall agreement {po:.1%}, Cohen's kappa {kappa:.2f}")


if __name__ == "__main__":
    {"sheet": build_sheet, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sheet"]()
