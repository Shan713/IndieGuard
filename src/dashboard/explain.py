"""Explain a launch-risk prediction: why a game scores as it does, and which single changes would help.

Review 2 feedback: the scorer showed only a tier. This module adds
  explain(game)   the drivers of the prediction, from XGBoost's per-feature contributions, grouped into
                  readable factors; store tags are scored one by one (remove the tag, re-score)
  suggest(game)   single launch-time changes (price, languages, achievements, controller, platforms,
                  categories, tags) re-scored by the same model and ranked by how much they raise
                  the Strong score and lower the Struggling score
Both are associations learned from 1,191 training games, not causal effects.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.dashboard.featurize import _items, featurize_game
from src.dashboard.risk import ROOT, TIER_NAMES, load_scorer

STRUGGLING, STRONG = 0, 2
GROUPS = {
    "Price": ["price_usd", "price_tier", "is_free"],
    "Launch timing": ["launch_year", "launch_quarter", "launch_before_window"],
    "Early Access launch": ["launched_in_early_access"],
    "DLC at launch": ["n_dlc"],
    "Supported languages": ["n_supported_languages"],
    "Achievements": ["n_achievements", "has_achievements"],
    "macOS / Linux support": ["platform_mac", "platform_linux"],
    "Controller support": ["controller_support_level"],
    "Self-published": ["self_published"],
}
EXTRA_CATEGORIES = ["Steam Achievements", "Steam Cloud", "Steam Trading Cards", "Steam Leaderboards", "Full controller support", "Remote Play Together"]


def _rule_tags() -> list[str]:
    r = pd.read_csv(ROOT / "reports" / "results" / "association_rules_strong.csv")
    return sorted({t[4:] for s in r.antecedents_str for t in s.split(" + ") if t.startswith("tag:")})


def _frame(games: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([featurize_game(g) for g in games]).loc[:, features].astype(float)


def _scores(model, meta, x: pd.DataFrame) -> np.ndarray:
    """Class-scaled, normalised scores, exactly as the scorer shows them."""
    p = model.predict(xgb.DMatrix(x, feature_names=meta["feature_columns"])) * np.asarray(meta["class_scales"], dtype=float)
    return p / p.sum(axis=1, keepdims=True)


def _margin_gap(model, meta, x: pd.DataFrame) -> np.ndarray:
    """Raw-score gap Strong minus Struggling: positive pushes toward Strong."""
    m = model.predict(xgb.DMatrix(x, feature_names=meta["feature_columns"]), output_margin=True)
    return m[:, STRONG] - m[:, STRUGGLING]


def _describe(name: str, f: dict[str, float]) -> str:
    return {
        "Price": "free" if f["is_free"] else f"${f['price_usd']:.2f}",
        "Launch timing": f"{int(f['launch_year'])} Q{int(f['launch_quarter'])}",
        "Early Access launch": "yes" if f["launched_in_early_access"] else "no",
        "DLC at launch": f"{int(f['n_dlc'])}",
        "Supported languages": f"{int(f['n_supported_languages'])}",
        "Achievements": f"{int(f['n_achievements'])}",
        "macOS / Linux support": ", ".join(n for n, k in (("macOS", "platform_mac"), ("Linux", "platform_linux")) if f[k]) or "Windows only",
        "Controller support": {2: "full", 1: "partial"}.get(int(f["controller_support_level"]), "none"),
        "Self-published": "yes" if f["self_published"] else "no",
    }.get(name, "")


def explain(game: dict[str, Any], max_tags: int = 20) -> pd.DataFrame:
    """Drivers of the prediction. effect > 0 pushes toward Strong, < 0 toward Struggling (raw-score units)."""
    model, meta = load_scorer()
    feats = meta["feature_columns"]
    x = _frame([game], feats)
    contrib = model.predict(xgb.DMatrix(x, feature_names=feats), pred_contribs=True)[0]       # (3 classes, features + bias)
    gap = contrib[STRONG, :-1] - contrib[STRUGGLING, :-1]
    by_name = dict(zip(feats, gap))
    f = x.iloc[0].to_dict()
    rows = [{"factor": g, "value": _describe(g, f), "effect": float(sum(by_name[c] for c in cols)), "kind": "store setting"} for g, cols in GROUPS.items()]

    # store tags and categories: leave one out and re-score, so each tag gets its own effect
    base = _margin_gap(model, meta, x)[0]
    items = [("tags", t) for t in _items(game.get("tags"))[:max_tags]] + [("categories", c) for c in _items(game.get("categories"))]
    if items:
        variants = []
        for field, item in items:
            g = copy.deepcopy(game)
            g[field] = ";".join(i for i in _items(game.get(field)) if i != item)
            variants.append(g)
        gaps = _margin_gap(model, meta, _frame(variants, feats))
        rows += [{"factor": f"{'Tag' if field == 'tags' else 'Category'}: {item}", "value": "present", "effect": float(base - g_), "kind": field}
                 for (field, item), g_ in zip(items, gaps)]
    out = pd.DataFrame(rows)
    out = out[out.effect.abs() > 1e-3].sort_values("effect", ascending=False).reset_index(drop=True)
    return out


def suggest(game: dict[str, Any], top: int = 6) -> pd.DataFrame:
    """Single launch-time changes ranked by the gain in (Strong score minus Struggling score)."""
    model, meta = load_scorer()
    feats = meta["feature_columns"]
    tags, cats = _items(game.get("tags")), _items(game.get("categories"))
    cands: list[tuple[str, str, dict[str, Any]]] = []

    def add(kind, label, **changes):
        g = copy.deepcopy(game); g.update(changes); cands.append((kind, label, g))

    price = float(game.get("price_usd") or 0)
    for p in (4.99, 9.99, 14.99, 19.99):
        if abs(p - price) > 1:
            add("Price", f"Set the price to ${p:.2f}", price_usd=p, is_free=False)
    langs = int(game.get("n_supported_languages") or 1)
    for n in (8, 12, 16):
        if n > langs:
            add("Localisation", f"Support {n} languages (now {langs})", n_supported_languages=n); break
    ach = int(game.get("n_achievements") or 0)
    if ach < 20:
        add("Content", f"Add Steam achievements (about 30; now {ach})", n_achievements=30)
    ctrl = str(game.get("controller_support") or "none").lower()
    if ctrl != "full":
        add("Controls", "Add full controller support", controller_support="full")
    if not game.get("platform_mac"):
        add("Platforms", "Ship a macOS build", platform_mac=True)
    if not game.get("platform_linux"):
        add("Platforms", "Ship a Linux build", platform_linux=True)
    launch = pd.Timestamp(game["launch_date"])
    early = not pd.isna(pd.to_datetime(game.get("release_date"), errors="coerce")) and (pd.Timestamp(game["release_date"]) - launch).days > 7
    if early:
        add("Launch", "Launch as a full release instead of Early Access", release_date=str(launch.date()))
    for c in EXTRA_CATEGORIES:
        if c not in cats:
            add("Store page", f"Add the store feature '{c}'", categories=";".join(cats + [c]))
    for t in tags:
        add("Positioning", f"Drop the tag '{t}'", tags=";".join(i for i in tags if i != t))
    if len(tags) < 20:
        for t in _rule_tags():
            if t not in tags:
                add("Positioning", f"Add the tag '{t}' (only if it truly fits)", tags=";".join(tags + [t]))

    x = _frame([game] + [g for _, _, g in cands], feats)
    s = _scores(model, meta, x)
    base = s[0]
    rows = []
    for (kind, label, _), sc in zip(cands, s[1:]):
        gain = (sc[STRONG] - sc[STRUGGLING]) - (base[STRONG] - base[STRUGGLING])
        rows.append({"area": kind, "change": label, "strong": sc[STRONG], "struggling": sc[STRUGGLING],
                     "strong_delta": sc[STRONG] - base[STRONG], "struggling_delta": sc[STRUGGLING] - base[STRUGGLING],
                     "new_tier": TIER_NAMES[int(sc.argmax())], "gain": gain})
    out = pd.DataFrame(rows)
    out = out[out.gain > 0.005].sort_values("gain", ascending=False)
    # at most two suggestions per area keep the list varied (tags could otherwise fill it)
    out = out.groupby("area", sort=False).head(2).head(top).reset_index(drop=True)
    return out
