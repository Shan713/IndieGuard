"""Compare association-rule configurations on the same transactions (Review 2 rubric: compare relevant alternatives fairly).

  python -m src.analysis.association_rule_comparison

Mines the training games only and checks each configuration's rules on the held-out test games, exactly like
src.analysis.association_rules, but writes only reports/results/association_rule_configuration_comparison.csv
(the main rule tables are not touched). Support, confidence and lift are reported for the rules that pass FDR control.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import association_rules as ar

logging.getLogger("association_rules").setLevel(logging.WARNING)
OUT = ar.RESULTS / "association_rule_configuration_comparison.csv"

CONFIGS = [
    ("FP-Growth, min support 25 games, up to 3 items (chosen)", dict(method="fpgrowth", min_count=25, max_len=3, exclude_evaluative=True)),
    ("Apriori, same thresholds", dict(method="apriori", min_count=25, max_len=3, exclude_evaluative=True)),
    ("FP-Growth, min support 15 games (looser)", dict(method="fpgrowth", min_count=15, max_len=3, exclude_evaluative=True)),
    ("FP-Growth, min support 40 games (stricter)", dict(method="fpgrowth", min_count=40, max_len=3, exclude_evaluative=True)),
    ("FP-Growth, pairs only (max 2 items)", dict(method="fpgrowth", min_count=25, max_len=2, exclude_evaluative=True)),
    ("FP-Growth, keeping evaluative tags", dict(method="fpgrowth", min_count=25, max_len=3, exclude_evaluative=False)),
]


def run(cfg: dict, n_boot: int = 100) -> dict:
    t0 = time.time()
    df = ar.load_dataset()
    df_bool, df_meta = ar.build_transactions(df, 25, 0.85, cfg["exclude_evaluative"])
    tr, te = (df_meta["split"] == "train").values, (df_meta["split"] == "test").values
    trb, teb = df_bool[tr].reset_index(drop=True), df_bool[te].reset_index(drop=True)
    trt, tet = df_meta.loc[tr, "tier"].values, df_meta.loc[te, "tier"].values
    rules = ar.mine_rules(trb, trt, cfg["min_count"], cfg["max_len"], 10, cfg["method"])
    rules = ar.evaluate_stability(rules, trb, trt, teb, tet, n_boot=n_boot, min_test_n=10)
    strong = ar.filter_outcome_rules(rules, "Strong", min_lift=1.2, fdr=0.10, min_improvement=0.03)
    struggling = ar.filter_outcome_rules(rules, "Struggling", min_lift=1.2, fdr=0.10, min_improvement=0.03)
    sig = pd.concat([strong, struggling])
    ok = sig[sig.stability != "too few test games"]
    return {"candidate_rules": len(rules), "strong_rules": len(strong), "struggling_rules": len(struggling),
            "mean_support": sig.support.mean() if len(sig) else np.nan, "mean_confidence": sig.confidence.mean() if len(sig) else np.nan,
            "mean_lift": sig.lift.mean() if len(sig) else np.nan,
            "share_holding_on_test": float((ok.stability == "stable").mean()) if len(ok) else np.nan, "rules_with_enough_test_games": len(ok),
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    rows = []
    for name, cfg in CONFIGS:
        r = run(cfg)
        rows.append({"configuration": name, **r})
        print(f"{name}: {r}")
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ar.ROOT)}")


if __name__ == "__main__":
    main()
