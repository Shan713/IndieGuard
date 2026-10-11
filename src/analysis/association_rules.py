"""Association rule mining for IndieGuard (Review 2 ownership: Pradeep Sasidharan).

Addresses Research Question 3 and Hypothesis H6:
  - RQ3: Which tag, price and Early Access combinations go with strong reception?
  - H6: Test tag and launch combinations with reception tiers using association rules.
  - Deliverables:
      #34 - Transaction dataset (tags, price brackets, early access, success tiers)
      #35 - Frequent itemset mining via FP-Growth / Apriori
      #36 - Rule evaluation, filtering by consequence, stability testing across train/test

Protocol (same as the models): everything is mined on TRAIN games only; the held-out TEST
games are used once, to check whether each rule still holds.

How the rules are built
  1. Frequent itemsets (FP-Growth) are mined over the launch-time items only: tags, price band,
     Early Access. Tier items are NOT mixed in, so rare tiers (Struggling is ~14%) are not lost
     to the support threshold.
  2. Each frequent itemset A is then scored against each target tier T:
        support = P(A and T)   confidence = P(T | A)   lift = confidence / P(T)
     plus an exact one-sided binomial p-value (is the tier more common inside A than overall?),
     Benjamini-Hochberg q-values, a Wilson interval for confidence, a bootstrap check on the
     training games, and a train -> test stability check.

Usage:
  python -m src.analysis.association_rules
  python -m src.analysis.association_rules --min-count 20 --fdr 0.10
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from scipy.stats import binom

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Set up paths matching project layout
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FEATURES = PROCESSED / "features"
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "figures" / "findings"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("association_rules")

SEED = 42
TARGET_TIERS = ("Strong", "Struggling")

# Tags that players vote AFTER playing, or that duplicate another item. A rule such as
# "Great Soundtrack => Strong" is circular (players tag the games they liked), and
# "Free to Play" / "Early Access" are already covered by the price band and EA items.
EVALUATIVE_TAGS = {
    "Great Soundtrack", "Replay Value", "Difficult", "Relaxing", "Funny", "Atmospheric",
    "Cute", "Beautiful", "Emotional",
}
DUPLICATE_TAGS = {"Free to Play", "Early Access"}


# -- small helpers ------------------------------------------------------------
def price_band(price_usd: float | None, is_free: bool) -> str:
    """Discretize price into the standard IndieGuard price bands."""
    if is_free or price_usd is None or pd.isna(price_usd) or price_usd == 0:
        return "price:Free"
    if price_usd < 5.0:
        return "price:<$5"
    if price_usd < 10.0:
        return "price:$5-10"
    if price_usd < 20.0:
        return "price:$10-20"
    return "price:$20+"


def _to_text(x) -> str:
    if isinstance(x, (list, tuple, np.ndarray)):
        return ";".join(map(str, x))
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _detect_tag_separator(series: pd.Series) -> str:
    """Pick the separator (; | ,) that appears most often in the tag column."""
    sample = series.dropna().head(200).map(_to_text)
    counts = {sep: sample.str.count(re.escape(sep)).sum() for sep in (";", "|", ",")}
    return max(counts, key=counts.get)


def parse_tags(raw, sep: str) -> list[str]:
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in _to_text(raw).split(sep) if t.strip()]


def detect_early_access(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Early Access flag per game, using the most reliable column that exists."""
    for col in ("early_access", "is_early_access"):
        if col in df.columns:
            return df[col].fillna(False).astype(bool), f"column '{col}'"
    for col in ("genres", "categories", "tags"):
        if col in df.columns:
            s = df[col].map(_to_text).str.contains("Early Access", case=False, regex=False)
            if s.any():
                return s, f"'{col}' mentions Early Access"
    if {"launch_date", "release_date"} <= set(df.columns):
        days = (pd.to_datetime(df["release_date"]) - pd.to_datetime(df["launch_date"])).dt.total_seconds() / 86400
        log.warning("No Early Access column found: falling back to release_date - launch_date > 7 days. "
                    "Check this against a column you trust.")
        return (days > 7.0).fillna(False), "date difference (release_date - launch_date > 7 days)"
    log.warning("No Early Access information found: every game is marked early_access:no.")
    return pd.Series(False, index=df.index), "none found"


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    n = np.maximum(n, 1)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    scaled = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# -- data loading -------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """Load games joined with split and success tiers (eligible == 1 only)."""
    games = pd.read_parquet(PROCESSED / "games_clean.parquet")
    targets = pd.read_parquet(FEATURES / "game_targets.parquet")

    df = games.merge(
        targets[["appid", "eligible", "tier", "tier_code", "high_risk"]],
        on="appid", how="inner", validate="one_to_one",
    )

    # The train/test split lives with the model features. Never default to "train".
    if "split" not in df.columns:
        gf = pd.read_parquet(FEATURES / "game_features.parquet")
        keep = [c for c in ("appid", "split", "cv_fold") if c in gf.columns]
        if "split" not in keep:
            raise KeyError("No 'split' column found in games_clean or game_features.")
        df = df.merge(gf[keep], on="appid", how="left", validate="one_to_one")
    if df["split"].isna().any():
        raise ValueError("Some games have no train/test split assigned.")

    df = df[df["eligible"] == 1].copy()
    df["tier"] = df["tier"].astype(str)
    log.info("Loaded %d eligible games (%d train / %d test).", len(df),
             (df["split"] == "train").sum(), (df["split"] == "test").sum())
    return df


# -- #34 transactions ---------------------------------------------------------
def build_transactions(
    df: pd.DataFrame,
    min_tag_freq: int = 25,
    max_item_support: float = 0.85,
    exclude_evaluative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Issue #34: one boolean basket per game.

    Items: frequent tags, price band, Early Access status, and the tier (tier:*).
    Tag frequency and near-universal items are decided on TRAIN games only.
    """
    is_train = (df["split"] == "train").values
    sep = _detect_tag_separator(df["tags"])
    tag_lists = [parse_tags(t, sep) for t in df["tags"]]
    log.info("Tag separator detected: %r", sep)

    skip = set(DUPLICATE_TAGS) | (EVALUATIVE_TAGS if exclude_evaluative else set())
    train_counts = Counter(t for tags, tr in zip(tag_lists, is_train) if tr for t in set(tags))
    frequent = {t for t, c in train_counts.items() if c >= min_tag_freq and t not in skip}
    removed = sorted(t for t in skip if t in train_counts)
    log.info("%d unique tags; kept %d with >= %d train games. Excluded evaluative/duplicate tags: %s",
             len(train_counts), len(frequent), min_tag_freq, removed or "none present")

    ea, ea_source = detect_early_access(df)
    log.info("Early Access detected from: %s (%d games).", ea_source, int(ea.sum()))
    bands = [price_band(p, bool(f)) for p, f in zip(df["price_usd"], df.get("is_free", pd.Series(False, index=df.index)))]

    baskets = []
    for tags, band, e, tier in zip(tag_lists, bands, ea.values, df["tier"]):
        basket = {f"tag:{t}" for t in tags if t in frequent}
        basket.add(band)
        basket.add("early_access:yes" if e else "early_access:no")
        basket.add(f"tier:{tier}")
        baskets.append(basket)

    vocab = sorted({item for b in baskets for item in b})
    idx = {item: i for i, item in enumerate(vocab)}
    matrix = np.zeros((len(baskets), len(vocab)), dtype=bool)
    for r, b in enumerate(baskets):
        for item in b:
            matrix[r, idx[item]] = True
    df_bool = pd.DataFrame(matrix, columns=vocab)

    # Drop near-universal design items (for example Roguelike, Indie): they carry no signal.
    support = df_bool.loc[is_train].mean()
    universal = [c for c in vocab if not c.startswith("tier:") and support[c] > max_item_support]
    if universal:
        log.info("Dropped near-universal items (> %.0f%% of train games): %s", 100 * max_item_support, universal)
        df_bool = df_bool.drop(columns=universal)

    df_meta = df[["appid", "name", "split", "tier"]].reset_index(drop=True) if "name" in df.columns \
        else df[["appid", "split", "tier"]].reset_index(drop=True)
    log.info("Transaction matrix: %d games x %d items.", *df_bool.shape)
    return df_bool, df_meta


# -- #35 mining ---------------------------------------------------------------
def _masks(df_bool: pd.DataFrame, itemsets: list[tuple[str, ...]]) -> np.ndarray:
    """Boolean matrix (games x itemsets): does the game contain every item of the itemset?"""
    arr = df_bool.to_numpy(dtype=bool)
    col = {c: i for i, c in enumerate(df_bool.columns)}
    return np.column_stack([arr[:, [col[i] for i in s]].all(axis=1) for s in itemsets])


def mine_rules(
    train_bool: pd.DataFrame,
    train_tier: np.ndarray,
    min_count: int = 25,
    max_len: int = 3,
    min_hits: int = 10,
    method: Literal["fpgrowth", "apriori"] = "fpgrowth",
) -> pd.DataFrame:
    """Issue #35: frequent itemsets on train, scored against each target tier."""
    design = [c for c in train_bool.columns if not c.startswith("tier:")]
    n = len(train_bool)
    # Half a game below the threshold, so itemsets with exactly min_count games are kept by both
    # algorithms (float rounding otherwise drops them in FP-Growth but not in Apriori).
    min_support = (min_count - 0.5) / n
    log.info("Mining frequent itemsets with %s on %d train games (>= %d games, max_len=%d)...",
             method, n, min_count, max_len)

    miner = fpgrowth if method == "fpgrowth" else apriori
    freq = miner(train_bool[design], min_support=min_support, use_colnames=True, max_len=max_len)
    log.info("Extracted %d frequent itemsets.", len(freq))
    if freq.empty:
        return pd.DataFrame()

    itemsets = [tuple(sorted(s)) for s in freq["itemsets"]]
    X = _masks(train_bool, itemsets)
    n_ant = X.sum(axis=0)

    frames = []
    for tier in TARGET_TIERS:
        t = (train_tier == tier)
        base = t.mean()
        if base == 0:
            continue
        n_hit = (X & t[:, None]).sum(axis=0)
        conf = n_hit / n_ant
        lo, hi = wilson_interval(n_hit, n_ant)
        p = binom.sf(n_hit - 1, n_ant, base)  # exact one-sided: P(X >= hits | base rate)
        conf_by_set = dict(zip(itemsets, conf))

        improvement = []
        for s, c in zip(itemsets, conf):
            if len(s) == 1:
                improvement.append(np.nan)
                continue
            subs = [conf_by_set[sub] for k in range(1, len(s)) for sub in combinations(s, k)]
            improvement.append(c - max(subs))

        frames.append(pd.DataFrame({
            "antecedents": itemsets,
            "tier": tier,
            "n_antecedent": n_ant,
            "n_hit": n_hit,
            "base_rate": base,
            "support": n_hit / n,
            "antecedent_support": n_ant / n,
            "confidence": conf,
            "conf_lo": lo,
            "conf_hi": hi,
            "lift": conf / base,
            "lift_lo": lo / base,
            "p_value": p,
            "q_value": bh_qvalues(p),
            "improvement": improvement,
        }))

    rules = pd.concat(frames, ignore_index=True)
    rules["consequents"] = rules["tier"].map(lambda t: (f"tier:{t}",))
    rules["antecedents_str"] = rules["antecedents"].map(lambda s: " + ".join(s))
    rules["consequents_str"] = rules["consequents"].map(lambda s: " + ".join(s))
    rules = rules[rules["n_hit"] >= min_hits].reset_index(drop=True)
    log.info("Scored %d candidate rules (antecedent -> tier) with at least %d hits.", len(rules), min_hits)
    return rules


# -- #36 stability ------------------------------------------------------------
def evaluate_stability(
    rules: pd.DataFrame,
    train_bool: pd.DataFrame,
    train_tier: np.ndarray,
    test_bool: pd.DataFrame,
    test_tier: np.ndarray,
    n_boot: int = 300,
    min_test_n: int = 10,
    seed: int = SEED,
) -> pd.DataFrame:
    """Issue #36: check each rule on held-out test games and on bootstrap resamples of train.

    Test:      rule is "stable" if >= min_test_n (default 10) test games match the antecedent, test lift > 1
               and the train confidence lies inside the 95% Wilson interval of the test confidence
               (the test games do not contradict the train result). With fewer test games the rule is
               labelled "too few test games", not "unstable".
    Bootstrap: share of resamples of the train games in which lift stays above 1.
    """
    if rules.empty:
        return rules
    rules = rules.copy()
    rng = np.random.RandomState(seed)
    n_tr = len(train_bool)

    cols = ["test_n_antecedent", "test_n_hit", "test_support", "test_confidence", "test_lift",
            "boot_lift_gt1", "boot_lift_lo", "boot_lift_hi"]
    out = pd.DataFrame(np.nan, index=rules.index, columns=cols)

    for tier in TARGET_TIERS:
        sel = rules.index[rules["tier"] == tier]
        if len(sel) == 0:
            continue
        ants = list(rules.loc[sel, "antecedents"])
        t_tr = (train_tier == tier).astype(np.float32)
        t_te = (test_tier == tier)

        # held-out test
        Xte = _masks(test_bool, ants)
        n_a = Xte.sum(axis=0)
        n_h = (Xte & t_te[:, None]).sum(axis=0)
        test_conf = np.where(n_a > 0, n_h / np.maximum(n_a, 1), np.nan)
        out.loc[sel, "test_n_antecedent"] = n_a
        out.loc[sel, "test_n_hit"] = n_h
        out.loc[sel, "test_support"] = n_h / max(len(test_bool), 1)
        out.loc[sel, "test_confidence"] = test_conf
        out.loc[sel, "test_lift"] = test_conf / max(t_te.mean(), 1e-12)

        # bootstrap over train games: weights are how many times each game is drawn
        Xtr = _masks(train_bool, ants).astype(np.float32)
        gt1 = np.zeros(len(sel))
        lifts = np.empty((n_boot, len(sel)), dtype=np.float32)
        for b in range(n_boot):
            w = rng.multinomial(n_tr, np.full(n_tr, 1.0 / n_tr)).astype(np.float32)
            nA = w @ Xtr
            nH = (w * t_tr) @ Xtr
            base_b = (w * t_tr).sum() / n_tr
            lift_b = np.where(nA > 0, nH / np.maximum(nA, 1) / max(base_b, 1e-12), 0.0)
            lifts[b] = lift_b
            gt1 += lift_b > 1
        out.loc[sel, "boot_lift_gt1"] = gt1 / n_boot
        out.loc[sel, "boot_lift_lo"] = np.percentile(lifts, 2.5, axis=0)
        out.loc[sel, "boot_lift_hi"] = np.percentile(lifts, 97.5, axis=0)

    rules = pd.concat([rules, out], axis=1)
    enough = rules["test_n_antecedent"] >= min_test_n
    t_lo, t_hi = wilson_interval(rules["test_n_hit"].fillna(0).values, rules["test_n_antecedent"].fillna(0).values)
    rules["test_conf_lo"], rules["test_conf_hi"] = t_lo, t_hi
    consistent = (rules["confidence"] >= t_lo) & (rules["confidence"] <= t_hi)
    holds = (rules["test_lift"] > 1) & consistent
    rules["conf_diff"] = (rules["confidence"] - rules["test_confidence"]).abs()
    rules["stability"] = np.where(~enough, "too few test games", np.where(holds, "stable", "unstable"))
    rules["is_stable"] = rules["stability"] == "stable"
    return rules


def filter_outcome_rules(
    rules: pd.DataFrame,
    target_tier: str,
    min_lift: float = 1.2,
    fdr: float = 0.10,
    min_improvement: float = 0.03,
    significant_only: bool = True,
) -> pd.DataFrame:
    """Rules whose consequent is the target tier and whose antecedent is a launch-time recipe.

    A longer rule is kept only if it beats every shorter version of itself by min_improvement
    (otherwise it is just a padded copy). Sorted by the lower confidence bound of the lift, so
    small samples cannot top the list.
    """
    if rules.empty:
        return pd.DataFrame()
    mask = (rules["tier"] == target_tier) & (rules["lift"] >= min_lift)
    mask &= rules["antecedents"].map(lambda a: not any(i.startswith("tier:") for i in a))
    mask &= rules["improvement"].isna() | (rules["improvement"] >= min_improvement)
    if significant_only:
        mask &= rules["q_value"] <= fdr
    return rules[mask].sort_values(by=["lift_lo", "lift"], ascending=False).copy()


# -- plots --------------------------------------------------------------------
def plot_rule_scatter(rules: pd.DataFrame, out_path: Path, fdr: float) -> None:
    """Support vs confidence for candidate rules, coloured by lift; FDR-significant rules outlined."""
    cand = rules[rules["lift"] >= 1.0]
    if cand.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    sig = cand["q_value"] <= fdr
    sc = ax.scatter(cand["support"], cand["confidence"], c=cand["lift"], cmap="viridis",
                    s=np.where(sig, 70, 22), alpha=0.75,
                    edgecolors=np.where(sig, "black", "none"), linewidth=0.8)
    fig.colorbar(sc, ax=ax).set_label("Lift", fontsize=11)
    ax.set_xlabel("Support (share of train games that match the rule)", fontsize=11)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title(f"IndieGuard Association Rules (train games)\noutlined = significant at FDR {fdr:.0%}",
                 fontsize=12, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    for _, r in cand[sig].nlargest(4, "lift_lo").iterrows():
        ax.annotate(f"{r['antecedents_str']} -> {r['tier']}", (r["support"], r["confidence"]),
                    xytext=(10, 5), textcoords="offset points", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="grey"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved rule scatter plot to %s.", out_path)


def plot_top_rules_bar(strong: pd.DataFrame, struggling: pd.DataFrame,
                       base_rates: dict[str, float], out_path: Path, n: int = 5) -> None:
    """Top rules by lift for Strong vs Struggling, with the test lift as a marker."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    panels = [
        (axes[0], strong, "Strong", "#2ca02c", "Top recipes for strong reception"),
        (axes[1], struggling, "Struggling", "#d62728", "Top risk combinations"),
    ]
    for ax, rules, tier, color, title in panels:
        top = rules.nlargest(n, "lift_lo") if not rules.empty else rules
        if top.empty:
            ax.text(0.5, 0.5, f"No FDR-significant rules for {tier}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.barh(range(len(top)), top["lift"], height=0.6, color=color, alpha=0.85, edgecolor="black")
        ax.set_yticks(range(len(top)), [a if len(a) < 55 else a[:52] + "..." for a in top["antecedents_str"]], fontsize=8)
        ax.set_ylim(n - 0.5, -0.5)  # same bar thickness whatever the number of rules
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Lift (baseline {100 * base_rates[tier]:.1f}% {tier} in train games)", fontsize=10)
        ax.set_title(title, fontsize=11, weight="bold")
        short = {"stable": "stable", "unstable": "unstable", "too few test games": "low test n"}
        for i, (lift, tl, stab, nn) in enumerate(zip(top["lift"], top["test_lift"], top["stability"],
                                                     top["n_antecedent"])):
            test_txt = f", test {tl:.2f}x {short.get(stab, stab)}" if pd.notna(tl) else ""
            ax.text(lift + 0.03, i, f"{lift:.2f}x  n={int(nn)}{test_txt}", va="center", fontsize=7.5)
        ax.set_xlim(0, top["lift"].max() * 2.3)
    fig.suptitle("IndieGuard actionable association rules (Review 2)", fontsize=13, weight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved top rules comparison bar chart to %s.", out_path)


# -- pipeline -----------------------------------------------------------------
def run_pipeline(
    min_count: int = 25,
    max_len: int = 3,
    min_hits: int = 10,
    min_lift: float = 1.2,
    fdr: float = 0.10,
    min_improvement: float = 0.03,
    min_tag_freq: int = 25,
    max_item_support: float = 0.85,
    exclude_evaluative: bool = True,
    n_boot: int = 300,
    min_test_n: int = 10,
    method: Literal["fpgrowth", "apriori"] = "fpgrowth",
    save_transactions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the complete association rule pipeline (#34-#36)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df_bool, df_meta = build_transactions(df, min_tag_freq, max_item_support, exclude_evaluative)

    is_train = (df_meta["split"] == "train").values
    is_test = (df_meta["split"] == "test").values
    train_bool, test_bool = df_bool[is_train].reset_index(drop=True), df_bool[is_test].reset_index(drop=True)
    train_tier, test_tier = df_meta.loc[is_train, "tier"].values, df_meta.loc[is_test, "tier"].values
    base_rates = {t: float((train_tier == t).mean()) for t in TARGET_TIERS}
    log.info("Train tier rates: %s", {k: f"{100 * v:.1f}%" for k, v in base_rates.items()})

    if save_transactions:
        out = pd.concat([df_meta[["appid", "split"]], df_bool], axis=1)
        out.to_parquet(RESULTS / "transactions_matrix.parquet", index=False)

    rules = mine_rules(train_bool, train_tier, min_count, max_len, min_hits, method)
    if rules.empty:
        log.warning("No frequent itemsets with the current thresholds. Lower --min-count.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rules = evaluate_stability(rules, train_bool, train_tier, test_bool, test_tier,
                                 n_boot=n_boot, min_test_n=min_test_n)

    kw = dict(min_lift=min_lift, fdr=fdr, min_improvement=min_improvement)
    strong = filter_outcome_rules(rules, "Strong", **kw)
    struggling = filter_outcome_rules(rules, "Struggling", **kw)
    log.info("FDR-significant rules: %d for Strong, %d for Struggling (of %d candidates).",
             len(strong), len(struggling), len(rules))

    save_cols = [
        "antecedents_str", "consequents_str", "n_antecedent", "n_hit", "support", "confidence",
        "conf_lo", "conf_hi", "lift", "lift_lo", "p_value", "q_value", "improvement",
        "test_n_antecedent", "test_support", "test_confidence", "test_conf_lo", "test_conf_hi", "test_lift",
        "conf_diff",
        "boot_lift_gt1", "boot_lift_lo", "boot_lift_hi", "stability", "is_stable",
    ]
    rules.sort_values("lift_lo", ascending=False)[save_cols].to_csv(RESULTS / "association_rules_all.csv", index=False)
    strong[save_cols].to_csv(RESULTS / "association_rules_strong.csv", index=False)
    struggling[save_cols].to_csv(RESULTS / "association_rules_struggling.csv", index=False)

    def top(r: pd.DataFrame):
        return None if r.empty else r.iloc[0][save_cols].to_dict()

    summary = {
        "n_games": int(len(df)), "n_train": int(is_train.sum()), "n_test": int(is_test.sum()),
        "train_tier_rates": base_rates,
        "settings": dict(min_count=min_count, max_len=max_len, min_hits=min_hits, min_lift=min_lift,
                         fdr=fdr, min_improvement=min_improvement, min_tag_freq=min_tag_freq,
                         max_item_support=max_item_support, exclude_evaluative=exclude_evaluative,
                         n_boot=n_boot, min_test_n=min_test_n, method=method),
        "n_items": int(df_bool.shape[1]),
        "candidate_rules": int(len(rules)),
        "strong_rules_count": int(len(strong)),
        "struggling_rules_count": int(len(struggling)),
        "stable_share_of_significant": float(pd.concat([strong, struggling])["is_stable"].mean())
        if len(strong) + len(struggling) else None,
        "top_strong_rule": top(strong),
        "top_struggling_rule": top(struggling),
    }
    (RESULTS / "association_rules_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    plot_rule_scatter(rules, FIGURES / "association_rules_scatter.png", fdr)
    plot_top_rules_bar(strong, struggling, base_rates, FIGURES / "association_rules_top_lift.png")
    log.info("Pipeline completed. Results in %s and %s.", RESULTS, FIGURES)
    return rules, strong, struggling


def main():
    p = argparse.ArgumentParser(description="IndieGuard Association Rule Mining (Review 2)")
    p.add_argument("--min-count", type=int, default=25, help="Min train games matching an itemset")
    p.add_argument("--max-len", type=int, default=3, help="Max items in a rule antecedent")
    p.add_argument("--min-hits", type=int, default=10, help="Min train games that match a rule's antecedent AND tier")
    p.add_argument("--min-lift", type=float, default=1.2, help="Min lift for the shortlisted rules")
    p.add_argument("--fdr", type=float, default=0.10, help="Benjamini-Hochberg q-value cutoff")
    p.add_argument("--min-improvement", type=float, default=0.03,
                   help="A longer rule must beat its shorter versions' confidence by this much")
    p.add_argument("--min-tag-freq", type=int, default=25, help="Min train games for a tag to be an item")
    p.add_argument("--max-item-support", type=float, default=0.85, help="Drop items present in more train games than this")
    p.add_argument("--keep-evaluative-tags", action="store_true",
                   help="Keep tags players vote after playing (Great Soundtrack, Difficult, ...)")
    p.add_argument("--n-boot", type=int, default=300, help="Bootstrap resamples for stability")
    p.add_argument("--min-test-n", type=int, default=10,
                   help="Min test games matching a rule before its test result counts as stable/unstable")
    p.add_argument("--method", choices=["fpgrowth", "apriori"], default="fpgrowth")
    p.add_argument("--save-transactions", action="store_true", help="Also save the transaction matrix (parquet)")
    a = p.parse_args()

    rules, strong, struggling = run_pipeline(
        min_count=a.min_count, max_len=a.max_len, min_hits=a.min_hits, min_lift=a.min_lift, fdr=a.fdr,
        min_improvement=a.min_improvement, min_tag_freq=a.min_tag_freq, max_item_support=a.max_item_support,
        exclude_evaluative=not a.keep_evaluative_tags, n_boot=a.n_boot, min_test_n=a.min_test_n,
        method=a.method,
        save_transactions=a.save_transactions,
    )
    if rules.empty:
        return

    def show(title: str, found: pd.DataFrame, tier: str):
        print(f"\n--- {title} ---")
        if found.empty:
            print("  No rule survives the FDR filter. Closest candidates (NOT significant):")
            found = filter_outcome_rules(rules, tier, min_lift=1.0, significant_only=False).head(3)
        for _, r in found.head(5).iterrows():
            print(f"  - {r['antecedents_str']:<48} => {tier} | lift {r['lift']:.2f} (>= {r['lift_lo']:.2f}), "
                  f"conf {100 * r['confidence']:.0f}%, n={int(r['n_antecedent'])}, q={r['q_value']:.3f}, "
                  f"test lift {r['test_lift']:.2f} [{r['stability']}], bootstrap lift>1 {100 * r['boot_lift_gt1']:.0f}%")

    print("\n" + "=" * 78)
    print("INDIEGUARD ASSOCIATION RULES (REVIEW 2)  -  mined on train games, checked on test games")
    print("=" * 78)
    show("TOP RECIPES FOR STRONG RECEPTION", strong, "Strong")
    show("TOP COMBINATIONS CARRYING HIGH RISK (Struggling)", struggling, "Struggling")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()