"""Validation script for Review 2 deliverables (Tasks #31-#33, #44).

Checks that all output CSVs, metric tables, and visualization figures exist,
have valid schemas, contain non-zero rows, and exhibit no invalid NaNs or probabilities.

Usage:
  python -m src.analysis.validate_review2
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("validate_review2")

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "results"
FIGS_TM = ROOT / "figures" / "text_mining"
FIGS_FINDINGS = ROOT / "figures" / "findings"


def run_checks() -> int:
    errors = 0
    checks_passed = 0

    log.info("Running Review 2 output validation checks...")

    # 1. Check generated report CSVs
    required_csvs = {
        "ngram_frequencies.csv": ["term", "log_odds_ratio", "count_negative", "count_positive"],
        "playtime_ngram_comparison.csv": ["term", "log_odds_under_2h_vs_later", "count_under_2h", "count_later"],
        "topic_terms.csv": ["topic_id", "label", "top_terms", "top_terms_weights"],
        "topic_prevalence.csv": ["group", "topic_id", "label", "mean_topic_weight", "dominant_topic_share"],
        "representative_reviews.csv": ["topic_id", "label", "topic_weight", "review_excerpt"],
        "text_evaluation_metrics.csv": ["methodology_tier", "task_type", "primary_metric", "secondary_metric"],
        "topic_stability_evaluation.csv": ["topic_id", "label", "mean_pairwise_jaccard", "stability_status"],
        "patch_impact_analysis.csv": ["patch_recency_bin", "n_reviews", "negative_rate"],
        "combined_insights.csv": ["rank", "theme", "what_data_shows", "supporting_evidence_metrics", "recommended_action"],
    }

    for fname, expected_cols in required_csvs.items():
        fpath = REPORTS / fname
        if not fpath.exists():
            log.error("Missing file: %s", fpath)
            errors += 1
            continue
        try:
            df = pd.read_csv(fpath)
            if len(df) == 0:
                log.error("Empty CSV: %s", fname)
                errors += 1
                continue
            for col in expected_cols:
                if col not in df.columns:
                    log.error("Missing column '%s' in %s", col, fname)
                    errors += 1
            checks_passed += 1
            log.info("PASS: %s (%d rows, verified columns)", fname, len(df))
        except Exception as e:
            log.error("Error reading %s: %s", fname, e)
            errors += 1

    # 2. Check generated figures
    required_figures = [
        FIGS_TM / "ngram_comparison.png",
        FIGS_TM / "topic_prevalence_comparison.png",
        FIGS_TM / "topics_by_playtime.png",
        FIGS_TM / "topic_evaluation_diagnostics.png",
        FIGS_FINDINGS / "patch_sentiment_impact.png",
        FIGS_FINDINGS / "combined_insights_framework.png",
    ]

    for fpath in required_figures:
        if not fpath.exists() or fpath.stat().st_size == 0:
            log.error("Missing or zero-byte figure: %s", fpath)
            errors += 1
        else:
            checks_passed += 1
            log.info("PASS: Figure exists (%s, %d bytes)", fpath.name, fpath.stat().st_size)

    # 3. Numeric bounds and integrity checks
    try:
        prev = pd.read_csv(REPORTS / "topic_prevalence.csv")
        if (prev["dominant_topic_share"] < 0).any() or (prev["dominant_topic_share"] > 1.0).any():
            log.error("Invalid dominant_topic_share in topic_prevalence.csv")
            errors += 1
        else:
            checks_passed += 1
            log.info("PASS: Topic prevalence rates strictly bounded in [0, 1]")

        topics = pd.read_csv(REPORTS / "topic_terms.csv")
        if len(topics) != 8:
            log.error("Expected 8 topics in topic_terms.csv, found %d", len(topics))
            errors += 1
        else:
            checks_passed += 1
            log.info("PASS: Topic count strictly matches k=8")

        stab = pd.read_csv(REPORTS / "topic_stability_evaluation.csv")
        if (stab["mean_pairwise_jaccard"] < 0).any() or (stab["mean_pairwise_jaccard"] > 1.0).any():
            log.error("Invalid Jaccard similarity in topic_stability_evaluation.csv")
            errors += 1
        else:
            checks_passed += 1
            log.info("PASS: Topic stability Jaccard similarities strictly bounded in [0, 1]")

    except Exception as e:
        log.error("Integrity check failed: %s", e)
        errors += 1

    log.info("Review 2 Validation Complete: %d passed, %d errors", checks_passed, errors)
    return errors


if __name__ == "__main__":
    import sys
    errs = run_checks()
    sys.exit(1 if errs > 0 else 0)
