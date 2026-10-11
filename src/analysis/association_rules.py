"""Association rule mining for IndieGuard (Review 2 ownership: Pradeep Sasidharan).

Addresses Research Question 3 and Hypothesis H6:
  - RQ3: Which tag, price and Early Access combinations go with strong reception?
  - H6: Test tag and launch combinations with reception tiers using association rules.
  - Deliverables:
      #34 - Transaction dataset generation (tags, price brackets, early access, success tiers)
      #35 - Frequent itemset mining via FP-Growth / Apriori
      #36 - Rule evaluation, filtering by consequence, stability testing across train/test splits

Usage:
  python -m src.analysis.association_rules
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Set up paths matching project layout
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FEATURES = PROCESSED / "features"
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "figures" / "findings"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("association_rules")

PRICE_CUTS = [5, 10, 20]  # tier 0: Free, 1: <$5, 2: $5-10, 3: $10-20, 4: $20+


def price_band(price_usd: float | None, is_free: bool) -> str:
    """Discretize price into standard IndieGuard price bands."""
    if is_free or price_usd is None or pd.isna(price_usd) or price_usd == 0:
        return "price:Free"
    if price_usd < 5.0:
        return "price:<$5"
    elif price_usd < 10.0:
        return "price:$5-10"
    elif price_usd < 20.0:
        return "price:$10-20"
    else:
        return "price:$20+"


def load_dataset() -> pd.DataFrame:
    """Load games joined with split and success tiers (eligible == 1 only)."""
    games = pd.read_parquet(PROCESSED / "games_clean.parquet")
    targets = pd.read_parquet(FEATURES / "game_targets.parquet")
    
    # Merge on appid
    df = games.merge(
        targets[["appid", "eligible", "tier", "tier_code", "high_risk"]],
        on="appid",
        how="inner",
        validate="one_to_one",
    )
    
    # Filter to eligible games with defined success tiers (>= 20 reviews)
    df = df[df["eligible"] == 1].copy()
    log.info("Loaded %d eligible games for association rule mining.", len(df))
    return df


def build_transactions(df: pd.DataFrame, min_tag_freq: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Issue #34: Transform game catalog into one-hot basket transactions.
    
    Items per game basket:
      - Frequent tags (present in >= min_tag_freq games)
      - Price bracket ('price:Free', 'price:<$5', 'price:$5-10', 'price:$10-20', 'price:$20+')
      - Early Access status ('early_access:yes', 'early_access:no')
      - Target tier ('tier:Strong', 'tier:Solid', 'tier:Struggling')
    """
    # Count tag frequencies to exclude rare noise
    all_tags = []
    for raw_tags in df["tags"].fillna(""):
        items = [t.strip() for t in str(raw_tags).split(";") if t.strip()]
        all_tags.extend(items)
    
    tag_counts = Counter(all_tags)
    frequent_tags = {tag for tag, count in tag_counts.items() if count >= min_tag_freq}
    log.info("Found %d unique tags; retained %d tags with frequency >= %d.",
             len(tag_counts), len(frequent_tags), min_tag_freq)
    
    baskets = []
    metadata = []
    
    for _, row in df.iterrows():
        basket = set()
        
        # 1. Tags
        raw_tags = [t.strip() for t in str(row.get("tags", "")).split(";") if t.strip()]
        for tag in raw_tags:
            if tag in frequent_tags:
                basket.add(f"tag:{tag}")
        
        # 2. Price
        p_band = price_band(row.get("price_usd"), bool(row.get("is_free", False)))
        basket.add(p_band)
        
        # 3. Early access
        launch_dt = row.get("launch_date")
        release_dt = row.get("release_date")
        is_ea = False
        if pd.notna(launch_dt) and pd.notna(release_dt):
            is_ea = (release_dt - launch_dt).total_seconds() / 86400.0 > 7.0
        basket.add("early_access:yes" if is_ea else "early_access:no")
        
        # 4. Target tier
        tier = row.get("tier")
        if pd.notna(tier):
            basket.add(f"tier:{tier}")
        
        baskets.append(basket)
        metadata.append({
            "appid": row["appid"],
            "name": row.get("name", ""),
            "split": row.get("split", "train"),
            "tier": row.get("tier", ""),
        })
    
    # Build unique item vocabulary
    vocab = sorted({item for b in baskets for item in b})
    vocab_idx = {item: i for i, item in enumerate(vocab)}
    
    # Convert to dense boolean matrix
    matrix = np.zeros((len(baskets), len(vocab)), dtype=bool)
    for row_i, b in enumerate(baskets):
        for item in b:
            matrix[row_i, vocab_idx[item]] = True
            
    df_bool = pd.DataFrame(matrix, columns=vocab)
    df_meta = pd.DataFrame(metadata)
    
    log.info("Built transaction matrix with %d transactions and %d distinct items.",
             df_bool.shape[0], df_bool.shape[1])
    return df_bool, df_meta


def mine_rules(
    df_bool: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.25,
    min_lift: float = 1.0,
    method: Literal["fpgrowth", "apriori"] = "fpgrowth",
) -> pd.DataFrame:
    """Issue #35: Mine frequent itemsets and extract association rules."""
    log.info("Mining frequent itemsets using %s (min_support=%.3f)...", method, min_support)
    
    if method == "fpgrowth":
        frequent_itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = apriori(df_bool, min_support=min_support, use_colnames=True)
        
    log.info("Extracted %d frequent itemsets.", len(frequent_itemsets))
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=min_lift,
        num_itemsets=len(frequent_itemsets),
    )
    
    # Filter by confidence
    rules = rules[rules["confidence"] >= min_confidence].copy()
    
    # String representations for readability
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    
    log.info("Generated %d raw association rules (min_confidence=%.2f, min_lift=%.2f).",
             len(rules), min_confidence, min_lift)
    return rules


def evaluate_stability(
    rules: pd.DataFrame,
    df_bool: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Issue #36: Evaluate rule stability across the train and test splits."""
    if rules.empty:
        return rules
        
    is_train = df_meta["split"] == "train"
    is_test = df_meta["split"] == "test"
    
    train_bool = df_bool[is_train]
    test_bool = df_bool[is_test]
    
    n_train = len(train_bool)
    n_test = len(test_bool)
    
    test_supports = []
    test_confidences = []
    test_lifts = []
    
    for _, row in rules.iterrows():
        ant = list(row["antecedents"])
        con = list(row["consequents"])
        
        # Test antecedent & consequent occurrence
        test_ant_mask = test_bool[ant].all(axis=1) if ant else pd.Series(True, index=test_bool.index)
        test_con_mask = test_bool[con].all(axis=1) if con else pd.Series(True, index=test_bool.index)
        test_both_mask = test_ant_mask & test_con_mask
        
        ant_count = test_ant_mask.sum()
        both_count = test_both_mask.sum()
        con_count = test_con_mask.sum()
        
        t_supp = both_count / n_test if n_test > 0 else 0.0
        t_conf = both_count / ant_count if ant_count > 0 else 0.0
        con_supp = con_count / n_test if n_test > 0 else 0.0
        t_lift = (t_conf / con_supp) if con_supp > 0 else 0.0
        
        test_supports.append(t_supp)
        test_confidences.append(t_conf)
        test_lifts.append(t_lift)
        
    rules["test_support"] = test_supports
    rules["test_confidence"] = test_confidences
    rules["test_lift"] = test_lifts
    
    # A rule is marked stable if confidence drop on held-out test is <= 0.15 and test support >= 0.015
    rules["conf_diff"] = (rules["confidence"] - rules["test_confidence"]).abs()
    rules["is_stable"] = (rules["conf_diff"] <= 0.15) & (rules["test_support"] >= 0.015)
    
    return rules


def filter_outcome_rules(rules: pd.DataFrame, target_tier: str) -> pd.DataFrame:
    """Filter rules where the consequent is specifically the target tier,
    and antecedents are actionable launch attributes (tags, price, early access).
    """
    if rules.empty:
        return pd.DataFrame()
        
    def is_target_consequent(c):
        return len(c) == 1 and list(c)[0] == target_tier
        
    def is_clean_antecedent(a):
        # Antecedents should not contain outcome tiers
        return not any(item.startswith("tier:") for item in a)
        
    mask = rules["consequents"].apply(is_target_consequent) & rules["antecedents"].apply(is_clean_antecedent)
    res = rules[mask].sort_values(by=["lift", "confidence"], ascending=False).copy()
    return res


def plot_rule_scatter(rules: pd.DataFrame, out_path: Path) -> None:
    """Generate Support vs Confidence scatter plot colored by Lift."""
    if rules.empty:
        return
        
    plt.figure(figsize=(9, 6), dpi=150)
    sc = plt.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        s=rules["lift"] * 40,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5,
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Lift", fontsize=11)
    
    plt.xlabel("Support", fontsize=11)
    plt.ylabel("Confidence", fontsize=11)
    plt.title("IndieGuard Association Rules: Support vs. Confidence by Lift", fontsize=13, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Annotate top 4 rules with highest lift
    top_rules = rules.nlargest(4, "lift")
    for _, r in top_rules.iterrows():
        plt.annotate(
            f"{r['antecedents_str']} -> {r['consequents_str']}",
            (r["support"], r["confidence"]),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="grey"),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="gray"),
        )
        
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log.info("Saved rule scatter plot to %s.", out_path)


def plot_top_rules_bar(strong_rules: pd.DataFrame, struggling_rules: pd.DataFrame, out_path: Path) -> None:
    """Generate comparison bar chart of top rules by lift for Strong vs Struggling tiers."""
    top_s = strong_rules.nlargest(5, "lift") if not strong_rules.empty else pd.DataFrame()
    top_w = struggling_rules.nlargest(5, "lift") if not struggling_rules.empty else pd.DataFrame()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150, sharex=False)
    
    if not top_s.empty:
        labels_s = [f"{row['antecedents_str']}" for _, row in top_s.iterrows()]
        axes[0].barh(range(len(top_s)), top_s["lift"], color="#2ca02c", alpha=0.85, edgecolor="black")
        axes[0].set_yticks(range(len(top_s)))
        axes[0].set_yticklabels(labels_s, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Lift (over baseline 35% Strong)", fontsize=10)
        axes[0].set_title("Top Recipes for Strong Reception (Tier: Strong)", fontsize=11, weight="bold")
        axes[0].axvline(1.0, color="gray", linestyle="--", linewidth=1)
        for i, lift in enumerate(top_s["lift"]):
            axes[0].text(lift + 0.02, i, f"{lift:.2f}x", va="center", fontsize=9, weight="bold")
    
    if not top_w.empty:
        labels_w = [f"{row['antecedents_str']}" for _, row in top_w.iterrows()]
        axes[1].barh(range(len(top_w)), top_w["lift"], color="#d62728", alpha=0.85, edgecolor="black")
        axes[1].set_yticks(range(len(top_w)))
        axes[1].set_yticklabels(labels_w, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Lift (over baseline 14.5% Struggling)", fontsize=10)
        axes[1].set_title("Top Risk Combinations (Tier: Struggling)", fontsize=11, weight="bold")
        axes[1].axvline(1.0, color="gray", linestyle="--", linewidth=1)
        for i, lift in enumerate(top_w["lift"]):
            axes[1].text(lift + 0.02, i, f"{lift:.2f}x", va="center", fontsize=9, weight="bold")
            
    plt.suptitle("IndieGuard Actionable Association Rules (Review 2)", fontsize=13, weight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log.info("Saved top rules comparison bar chart to %s.", out_path)


def run_pipeline(
    min_support: float = 0.025,
    min_confidence: float = 0.20,
    min_tag_freq: int = 25,
    method: Literal["fpgrowth", "apriori"] = "fpgrowth",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete association rule pipeline (#34–#36)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    df = load_dataset()
    
    # 2. Build transactions (#34)
    df_bool, df_meta = build_transactions(df, min_tag_freq=min_tag_freq)
    
    # Save transactions manifest for dashboard / verification
    df_bool.to_parquet(RESULTS / "transactions_matrix.parquet", index=False)
    
    # 3. Mine association rules (#35)
    rules = mine_rules(df_bool, min_support=min_support, min_confidence=min_confidence, method=method)
    
    if rules.empty:
        log.warning("No association rules found with the current thresholds.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # 4. Evaluate stability (#36)
    rules = evaluate_stability(rules, df_bool, df_meta)
    
    # 5. Extract outcome rules
    strong_rules = filter_outcome_rules(rules, "tier:Strong")
    struggling_rules = filter_outcome_rules(rules, "tier:Struggling")
    
    log.info("Found %d rules leading to Strong tier, %d leading to Struggling tier.",
             len(strong_rules), len(struggling_rules))
             
    # Save CSV outputs for report, slides, and Srihitha's dashboard
    save_cols = [
        "antecedents_str", "consequents_str", "support", "confidence", "lift",
        "test_support", "test_confidence", "test_lift", "is_stable",
    ]
    
    # Save top 25,000 rules sorted by lift to keep file size compact and within GitHub limits (<100MB)
    top_all_rules = rules[save_cols].sort_values(by=["lift", "confidence"], ascending=False).head(25000)
    top_all_rules.to_csv(RESULTS / "association_rules_all.csv", index=False)
    strong_rules[save_cols].to_csv(RESULTS / "association_rules_strong.csv", index=False)
    struggling_rules[save_cols].to_csv(RESULTS / "association_rules_struggling.csv", index=False)
    
    # Save JSON summary metrics
    summary = {
        "total_eligible_games": len(df),
        "total_rules_mined": len(rules),
        "strong_rules_count": len(strong_rules),
        "struggling_rules_count": len(struggling_rules),
        "stable_rules_share": float(rules["is_stable"].mean()) if not rules.empty else 0.0,
        "top_strong_rule": strong_rules.iloc[0].to_dict() if not strong_rules.empty else None,
        "top_struggling_rule": struggling_rules.iloc[0].to_dict() if not struggling_rules.empty else None,
    }
    # Clean non-serializable sets
    for k in ["top_strong_rule", "top_struggling_rule"]:
        if summary[k] and "antecedents" in summary[k]:
            summary[k]["antecedents"] = list(summary[k]["antecedents"])
            summary[k]["consequents"] = list(summary[k]["consequents"])
            
    with open(RESULTS / "association_rules_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
        
    # 6. Plot visualizations
    plot_rule_scatter(rules, FIGURES / "association_rules_scatter.png")
    plot_top_rules_bar(strong_rules, struggling_rules, FIGURES / "association_rules_top_lift.png")
    
    log.info("Pipeline completed successfully! Results written to %s and %s.", RESULTS, FIGURES)
    return rules, strong_rules, struggling_rules


def main():
    parser = argparse.ArgumentParser(description="IndieGuard Association Rule Mining (Review 2)")
    parser.add_argument("--min-support", type=float, default=0.025, help="Minimum support threshold")
    parser.add_argument("--min-confidence", type=float, default=0.20, help="Minimum confidence threshold")
    parser.add_argument("--min-tag-freq", type=int, default=25, help="Minimum tag frequency in dataset")
    parser.add_argument("--method", choices=["fpgrowth", "apriori"], default="fpgrowth", help="Mining algorithm")
    args = parser.parse_args()
    
    rules, strong, struggling = run_pipeline(
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_tag_freq=args.min_tag_freq,
        method=args.method,
    )
    
    print("\n" + "=" * 70)
    print("INDIEGUARD ASSOCIATION RULES (REVIEW 2 - PRADEEP SASIDHARAN)")
    print("=" * 70)
    if not strong.empty:
        print("\n--- TOP RECIPES FOR STRONG RECEPTION (Tier: Strong) ---")
        for _, r in strong.head(5).iterrows():
            stable_mark = "✔ Stable" if r["is_stable"] else "~ Sensitive"
            print(f"  • {r['antecedents_str']:<45} => Strong (Lift: {r['lift']:.2f}x, Conf: {r['confidence']*100:.1f}%, Supp: {r['support']*100:.1f}%) [{stable_mark}]")
            
    if not struggling.empty:
        print("\n--- TOP COMBINATIONS CARRYING HIGH RISK (Tier: Struggling) ---")
        for _, r in struggling.head(5).iterrows():
            stable_mark = "✔ Stable" if r["is_stable"] else "~ Sensitive"
            print(f"  • {r['antecedents_str']:<45} => Struggling (Lift: {r['lift']:.2f}x, Conf: {r['confidence']*100:.1f}%, Supp: {r['support']*100:.1f}%) [{stable_mark}]")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
