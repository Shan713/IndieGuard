"""Initial business findings for a studio, computed from the feature tables (descriptive, not causal).

  python -m src.analysis.business_findings

Writes reports/results/findings_*.csv and figures/findings/*.png. The write-up is docs/business_findings.md.
  1. Refund window: how much negative feedback arrives while a refund is still possible (under 2 hours played)
  2. Complaint themes: what negative reviews mention, and how that differs inside the refund window (keyword-based, English)
  3. Game level: which launch characteristics go with struggling games (training games only)
Complaint themes here are simple keyword groups, an initial look; Review 2's topic modelling replaces them.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The theme patterns group words with (...) only to alternate them; pandas warns that the groups are unused.
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "data" / "processed" / "features"
RES, FIG = ROOT / "reports" / "results", ROOT / "figures" / "findings"

THEMES = {
    "Bugs, crashes, performance": r"\b(bug|bugs|buggy|crash|crashes|crashed|crashing|glitch|glitches|glitchy|lag|laggy|freeze|freezes|freezing|stutter|stuttering|fps|performance|unplayable|broken)\b",
    "Boring, repetitive, grindy": r"\b(boring|repetitive|repetition|grind|grindy|tedious|monotonous|dull|samey|repeats)\b",
    "Price and value": r"\b(overpriced|too expensive|not worth|waste of (?:money|time)|cash ?grab|rip ?off|scam)\b",
    "Unfair, frustrating, balance": r"\b(unfair|too hard|frustrating|frustrated|unbalanced|rng|cheap)\b",
    "Controls, camera, interface": r"\b(controls|controller|clunky|camera|keybind|keybinds|ui|menu)\b",
    "Too little content, unfinished": r"\b(unfinished|lack of content|not enough content|too short|abandoned|dead game|no content|early access)\b",
    "Online, servers, multiplayer": r"\b(servers|matchmaking|netcode|disconnect|multiplayer|co-?op)\b",
    "Monetisation, ads, DLC": r"\b(microtransactions?|dlc|ads|pay to win|p2w|gacha|paywall)\b",
    "Refunds": r"\b(refund|refunded|refunding|money back)\b",
    "Praise (control group)": r"\b(fun|addictive|masterpiece|amazing|love|great)\b",
}


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def refund_window() -> pd.DataFrame:
    rf = pd.read_parquet(F / "review_features", columns=["appid", "target_is_negative", "in_refund_window", "playtime_bucket",
                                                          "price_tier", "launch_week", "written_during_early_access", "split"])
    neg = rf["target_is_negative"] == 1
    rows = []

    def add(group, label, mask):
        n, k = int(mask.sum()), int(neg[mask].sum())
        lo, hi = wilson([k], [n])
        rows.append({"group": group, "level": label, "reviews": n, "negative": k, "negative_rate": k / n, "ci_low": lo[0], "ci_high": hi[0]})

    add("all reviews", "all", pd.Series(True, index=rf.index))
    for b, name in {0: "under 2 h", 1: "2 to 10 h", 2: "10 to 50 h", 3: "50 h or more"}.items():
        add("playtime at review", name, rf["playtime_bucket"] == b)
    add("refund window", "inside (under 2 h)", rf["in_refund_window"] == 1)
    add("refund window", "outside", rf["in_refund_window"] == 0)
    tiers = {0: "free", 1: "under $5", 2: "$5 to 10", 3: "$10 to 20", 4: "$20 or more"}
    for t, name in tiers.items():
        add("price tier, inside window", name, (rf["price_tier"] == t) & (rf["in_refund_window"] == 1))
        add("price tier, outside window", name, (rf["price_tier"] == t) & (rf["in_refund_window"] == 0))
    add("launch week", "first 7 days", rf["launch_week"] == 1)
    add("launch week", "later", rf["launch_week"] == 0)
    add("Early Access", "written in Early Access", rf["written_during_early_access"] == 1)
    add("Early Access", "after 1.0", rf["written_during_early_access"] == 0)
    out = pd.DataFrame(rows)

    # share of all negative reviews that arrive inside the window, and a per-game view that big games cannot dominate
    share = float(rf.loc[neg, "in_refund_window"].mean())
    g = rf.groupby(["appid", "in_refund_window"])["target_is_negative"].agg(["mean", "size"]).unstack()
    ok = (g["size"] >= 50).all(axis=1)
    diff = (g["mean"][1] - g["mean"][0])[ok]
    extra = pd.DataFrame([{"group": "summary", "level": "share of all negative reviews written inside the refund window", "reviews": int(neg.sum()),
                           "negative": int(rf.loc[neg, "in_refund_window"].sum()), "negative_rate": share, "ci_low": np.nan, "ci_high": np.nan},
                          {"group": "summary", "level": f"share of games (n={int(ok.sum())}, 50+ reviews each side) where the window is more negative",
                           "reviews": int(ok.sum()), "negative": int((diff > 0).sum()), "negative_rate": float((diff > 0).mean()), "ci_low": np.nan, "ci_high": np.nan},
                          {"group": "summary", "level": "median per-game gap: negative rate inside minus outside the window",
                           "reviews": int(ok.sum()), "negative": 0, "negative_rate": float(diff.median()), "ci_low": np.nan, "ci_high": np.nan}])
    out = pd.concat([out, extra], ignore_index=True)
    out.to_csv(RES / "findings_refund_window.csv", index=False)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    pb = out[out["group"] == "playtime at review"]
    ax[0].bar(pb["level"], 100 * pb["negative_rate"], color=["#C44E52", "#8C8C8C", "#8C8C8C", "#8C8C8C"],
              yerr=[100 * (pb["negative_rate"] - pb["ci_low"]), 100 * (pb["ci_high"] - pb["negative_rate"])], capsize=3)
    ax[0].axhline(100 * out.loc[0, "negative_rate"], ls="--", c="black", lw=1, label="all reviews")
    ax[0].set_ylabel("negative reviews (%)"); ax[0].set_title("Negative share by playtime when the review was written"); ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)
    ins, outs = out[out["group"] == "price tier, inside window"], out[out["group"] == "price tier, outside window"]
    x = np.arange(len(ins)); w = 0.38
    ax[1].bar(x - w / 2, 100 * ins["negative_rate"], w, color="#C44E52", label="inside refund window (under 2 h)")
    ax[1].bar(x + w / 2, 100 * outs["negative_rate"], w, color="#8C8C8C", label="outside")
    ax[1].set_xticks(x); ax[1].set_xticklabels(ins["level"]); ax[1].set_title("Negative share by price tier"); ax[1].set_ylabel("negative reviews (%)")
    ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "refund_window.png", dpi=150); plt.close(fig)
    return out


def themes() -> pd.DataFrame:
    rf = pd.read_parquet(F / "review_features", columns=["recommendationid", "target_is_negative", "in_refund_window"])
    parts = [pd.read_parquet(p, columns=["recommendationid", "review", "text_mining_ok"]) for p in sorted((ROOT / "data/processed/reviews_clean").glob("*.parquet"))]
    txt = pd.concat(parts)
    txt = txt[txt["text_mining_ok"]].drop(columns="text_mining_ok")
    d = txt.merge(rf, on="recommendationid", how="inner")
    neg, win = d["target_is_negative"] == 1, d["in_refund_window"] == 1
    base = float(neg.mean())
    lower = d["review"].str.lower()
    rows = []
    for name, pat in THEMES.items():
        hit = lower.str.contains(pat, regex=True)
        rows.append({"theme": name, "share_of_reviews": hit.mean(), "negative_rate_when_mentioned": neg[hit].mean(),
                     "lift_vs_base": neg[hit].mean() / base, "share_of_negative_reviews": hit[neg].mean(),
                     "share_of_negative_inside_window": hit[neg & win].mean(), "share_of_negative_outside_window": hit[neg & ~win].mean(),
                     "mentions": int(hit.sum())})
    out = pd.DataFrame(rows)
    meta = pd.DataFrame([{"theme": "(English reviews analysed)", "share_of_reviews": 1.0, "negative_rate_when_mentioned": base, "lift_vs_base": 1.0,
                          "share_of_negative_reviews": 1.0, "share_of_negative_inside_window": 1.0, "share_of_negative_outside_window": 1.0, "mentions": len(d)}])
    pd.concat([meta, out], ignore_index=True).to_csv(RES / "findings_themes.csv", index=False)

    o = out[out["theme"] != "Praise (control group)"].sort_values("share_of_negative_inside_window")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(o)); h = 0.38
    ax.barh(y + h / 2, 100 * o["share_of_negative_inside_window"], h, color="#C44E52", label="negative reviews inside the refund window")
    ax.barh(y - h / 2, 100 * o["share_of_negative_outside_window"], h, color="#8C8C8C", label="negative reviews outside it")
    ax.set_yticks(y); ax.set_yticklabels(o["theme"]); ax.set_xlabel("share of negative reviews that mention the theme (%)")
    ax.set_title("What negative reviews mention (English, keyword groups)"); ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "complaint_themes.png", dpi=150); plt.close(fig)
    return out


def game_factors() -> tuple[pd.DataFrame, pd.DataFrame]:
    gf = pd.read_parquet(F / "game_features.parquet")
    gt = pd.read_parquet(F / "game_targets.parquet")[["appid", "tier", "eligible"]]
    games = pd.read_parquet(ROOT / "data/processed/games_clean.parquet", columns=["appid", "tags"])
    d = gf.merge(gt, on="appid").merge(games, on="appid")
    d = d[(d["eligible"] == 1) & (d["split"] == "train")]          # training games only
    base = float((d["tier"] == "Struggling").mean())
    rows = []

    def add(group, level, mask):
        n, k = int(mask.sum()), int((d.loc[mask, "tier"] == "Struggling").sum())
        s = int((d.loc[mask, "tier"] == "Strong").sum())
        lo, hi = wilson([k], [n])
        rows.append({"factor": group, "level": level, "games": n, "struggling_rate": k / n, "ci_low": lo[0], "ci_high": hi[0], "strong_rate": s / n})

    add("all training games", "all", pd.Series(True, index=d.index))
    for t, name in {0: "free", 1: "under $5", 2: "$5 to 10", 3: "$10 to 20", 4: "$20 or more"}.items():
        add("price tier", name, d["price_tier"] == t)
    add("launch", "launched in Early Access", d["launched_in_early_access"] == 1)
    add("launch", "no Early Access phase", d["launched_in_early_access"] == 0)
    for lo_, hi_, name in [(1, 1, "1"), (2, 3, "2 to 3"), (4, 7, "4 to 7"), (8, 999, "8 or more")]:
        add("interface languages", name, d["n_supported_languages"].between(lo_, hi_))
    add("achievements", "has achievements", d["has_achievements"] == 1)
    add("achievements", "none listed", d["has_achievements"] == 0)
    add("DLC", "at least one DLC", d["n_dlc"] > 0)
    add("DLC", "none", d["n_dlc"] == 0)
    add("controller support", "full", d["controller_support_level"] == 2)
    add("controller support", "none listed", d["controller_support_level"] == 0)
    factors = pd.DataFrame(rows)

    sets = d["tags"].fillna("").str.split(";").apply(lambda x: {t.strip() for t in x if t.strip()})
    counts = pd.Series([t for s in sets for t in s]).value_counts()
    trows = []
    for tag in counts[counts >= 40].index:
        m = sets.apply(lambda s, t=tag: t in s)
        n, k = int(m.sum()), int((d.loc[m, "tier"] == "Struggling").sum())
        lo, hi = wilson([k], [n])
        trows.append({"tag": tag, "games": n, "struggling_rate": k / n, "ci_low": lo[0], "ci_high": hi[0],
                      "strong_rate": float((d.loc[m, "tier"] == "Strong").mean())})
    tags = pd.DataFrame(trows).sort_values("struggling_rate")
    tags["base_rate"] = base
    factors.to_csv(RES / "findings_game_factors.csv", index=False)
    tags.to_csv(RES / "findings_tags.csv", index=False)
    return factors, tags


def review_patterns() -> pd.DataFrame:
    """Negative share by review length and by language (the two strongest metadata signals in the SHAP ranking)."""
    rf = pd.read_parquet(F / "review_features", columns=["target_is_negative", "review_n_chars", "language"])
    rows = []
    bands = pd.cut(rf["review_n_chars"], [0, 20, 60, 200, 1000, 1e9], labels=["under 20 characters", "20 to 60", "60 to 200", "200 to 1,000", "over 1,000"])
    for level, g in rf.groupby(bands, observed=True):
        rows.append({"pattern": "review length", "level": str(level), "reviews": len(g), "negative_rate": g["target_is_negative"].mean()})
    top = rf["language"].value_counts().head(8).index
    for lang in top:
        g = rf[rf["language"] == lang]
        rows.append({"pattern": "language", "level": lang, "reviews": len(g), "negative_rate": g["target_is_negative"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "findings_review_patterns.csv", index=False)
    return out


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    print(review_patterns().to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    r = refund_window(); print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    t = themes(); print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    f, g = game_factors(); print(f.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(g.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}")); print(g.tail(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
