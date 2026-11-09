"""Explain a launch-risk prediction: why a game scores as it does, and which single changes would help.

Review 2 feedback: the scorer showed only a tier. This module adds
  explain(game)          each launch choice compared with its obvious alternative (achievements or none,
                         controller support or none, a tag or not, ...): the change in the Struggling score
                         the scorer shows, re-scored by the same model
  suggest(game, tier)    single launch-time changes ranked by how much they lower the Struggling score
                         (or, for a game already predicted Strong, raise the Strong score)
The scorer is about launch risk, so effects are measured on the Struggling score. A choice can lower that
risk and still move a game from Strong toward Solid; measuring "Strong minus Struggling" made such choices
look harmful. Both functions report associations learned from 1,191 training games, not causal effects.
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
EXTRA_CATEGORIES = ["Steam Achievements", "Steam Cloud", "Steam Trading Cards", "Steam Leaderboards", "Full controller support", "Remote Play Together"]


def _rule_tags() -> list[str]:
    r = pd.read_csv(ROOT / "reports" / "results" / "association_rules_strong.csv")
    return sorted({t[4:] for s in r.antecedents_str for t in s.split(" + ") if t.startswith("tag:")})


def _median_price() -> float:
    g = pd.read_csv(ROOT / "data" / "dashboard" / "games.csv", usecols=["price_usd"])
    return float(g.loc[g.price_usd > 0, "price_usd"].median())


def _scores(games: list[dict[str, Any]]) -> np.ndarray:
    """Class-scaled, normalised scores, exactly as the scorer shows them (one row per game)."""
    model, meta = load_scorer()
    feats = meta["feature_columns"]
    x = pd.DataFrame([featurize_game(g) for g in games]).loc[:, feats].astype(float)
    p = model.predict(xgb.DMatrix(x, feature_names=feats)) * np.asarray(meta["class_scales"], dtype=float)
    return p / p.sum(axis=1, keepdims=True)


def _early_access(game: dict[str, Any]) -> bool:
    rel = pd.to_datetime(game.get("release_date"), errors="coerce")
    return not pd.isna(rel) and (rel - pd.Timestamp(game["launch_date"])).days > 7


def _alternatives(game: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(factor as it stands, kind, the game with the obvious alternative instead)."""
    out = []

    def alt(label, kind, **changes):
        g = copy.deepcopy(game); g.update(changes); out.append((label, kind, g))

    ach = int(game.get("n_achievements") or 0)
    alt(f"{ach} achievements" if ach else "No achievements", "setting", n_achievements=0 if ach else 30)
    ctrl = str(game.get("controller_support") or "none").lower()
    alt(f"{ctrl.capitalize()} controller support" if ctrl in ("full", "partial") else "No controller support", "setting",
        controller_support="none" if ctrl in ("full", "partial") else "full")
    langs = int(game.get("n_supported_languages") or 1)
    alt(f"{langs} supported languages" if langs > 1 else "Only one language", "setting", n_supported_languages=1 if langs > 1 else 8)
    alt("macOS build" if game.get("platform_mac") else "No macOS build", "setting", platform_mac=not game.get("platform_mac"))
    alt("Linux build" if game.get("platform_linux") else "No Linux build", "setting", platform_linux=not game.get("platform_linux"))
    if int(game.get("n_dlc") or 0) > 0:
        alt(f"{int(game['n_dlc'])} DLC at launch", "setting", n_dlc=0)
    if _early_access(game):
        alt("Early Access launch", "setting", release_date=str(pd.Timestamp(game["launch_date"]).date()))
    price, med = float(game.get("price_usd") or 0), _median_price()
    if abs(price - med) > 0.5:
        alt("Free to play" if price == 0 else f"Price ${price:.2f} (typical is ${med:.2f})", "setting", price_usd=med, is_free=False)
    self_pub = bool(set(_items(game.get("developers"))) & set(_items(game.get("publishers"))))
    alt("Self-published" if self_pub else "Has a publisher", "setting", publishers="publisher" if self_pub else game.get("developers"))
    for field, word in (("tags", "Tag"), ("categories", "Store feature")):
        items = _items(game.get(field))
        for it in items:
            alt(f"{word}: {it}", field, **{field: ";".join(i for i in items if i != it)})
    return out


def explain(game: dict[str, Any]) -> pd.DataFrame:
    """effect = Struggling score with the alternative minus the score as it stands (percentage points).
    Positive: the game's current choice lowers the risk of Struggling (in its favour)."""
    alts = _alternatives(game)
    s = _scores([game] + [g for _, _, g in alts])
    base = s[0, STRUGGLING]
    rows = [{"factor": label, "kind": kind, "effect": float(sc[STRUGGLING] - base)} for (label, kind, _), sc in zip(alts, s[1:])]
    out = pd.DataFrame(rows)
    return out[out.effect.abs() >= 0.005].sort_values("effect", ascending=False).reset_index(drop=True)


def suggest(game: dict[str, Any], tier: str, top: int = 6) -> pd.DataFrame:
    """Single launch-time changes. For a game predicted Strong, ranked by the rise in the Strong score;
    otherwise by the fall in the Struggling score."""
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
    if int(game.get("n_achievements") or 0) < 20:
        add("Content", f"Add Steam achievements (about 30; now {int(game.get('n_achievements') or 0)})", n_achievements=30)
    if str(game.get("controller_support") or "none").lower() != "full":
        add("Controls", "Add full controller support", controller_support="full")
    if not game.get("platform_mac"):
        add("Platforms", "Ship a macOS build", platform_mac=True)
    if not game.get("platform_linux"):
        add("Platforms", "Ship a Linux build", platform_linux=True)
    if _early_access(game):
        add("Launch", "Launch as a full release instead of Early Access", release_date=str(pd.Timestamp(game["launch_date"]).date()))
    for c in EXTRA_CATEGORIES:
        if c not in cats:
            add("Store page", f"Add the store feature '{c}'", categories=";".join(cats + [c]))
    for t in tags:
        add("Positioning", f"Drop the tag '{t}'", tags=";".join(i for i in tags if i != t))
    if len(tags) < 20:
        for t in _rule_tags():
            if t not in tags:
                add("Positioning", f"Add the tag '{t}' (only if it truly fits)", tags=";".join(tags + [t]))

    s = _scores([game] + [g for _, _, g in cands])
    base = s[0]
    rows = []
    for (kind, label, _), sc in zip(cands, s[1:]):
        gain = (sc[STRONG] - base[STRONG]) if tier == "Strong" else (base[STRUGGLING] - sc[STRUGGLING])
        rows.append({"area": kind, "change": label, "strong": sc[STRONG], "struggling": sc[STRUGGLING],
                     "strong_delta": sc[STRONG] - base[STRONG], "struggling_delta": sc[STRUGGLING] - base[STRUGGLING],
                     "new_tier": TIER_NAMES[int(sc.argmax())], "gain": gain})
    out = pd.DataFrame(rows)
    out = out[out.gain > 0.005].sort_values("gain", ascending=False)
    # at most two suggestions per area keep the list varied (tags could otherwise fill it)
    return out.groupby("area", sort=False).head(2).head(top).reset_index(drop=True)
