"""Game-level target: three ordered success tiers on Steam's all-language review totals.

  data/processed/features/game_targets.parquet   appid, split, cv_fold, eligible, tier, tier_code, high_risk

Tiers (games with at least MIN_REVIEWS reviews; fixed in advance, nothing is fitted from the data):
  Struggling  30% or more of the game's reviews are negative          (tier_code 0)
  Strong      10% or less are negative (90%+ positive)                  (tier_code 2)
  Solid       everything in between                                     (tier_code 1)
`high_risk` is the binary version (Struggling vs the rest) used in the EDA notebook.

Boundaries use whole-number arithmetic (10 * negative >= 3 * total), because a float comparison such as
positive_share < 0.70 misclassifies games sitting exactly on a cut-off (5 games here).
Games under MIN_REVIEWS get no tier: their ratios are too noisy (371 games).
The tiers describe reception only; reach (review count, owners) depends on how long a game has been out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DOCS, OUT, get_logger

log = get_logger("features")

MIN_REVIEWS = 20
TIERS = ["Struggling", "Solid", "Strong"]


def assign_tiers(total: pd.Series, negative: pd.Series) -> pd.Series:
    """Tier code 0/1/2 (Struggling/Solid/Strong) from review totals; <NA> below MIN_REVIEWS."""
    total, negative = total.astype("int64"), negative.astype("int64")
    struggling = 10 * negative >= 3 * total
    strong = ~struggling & (10 * negative <= total)
    code = pd.Series(np.where(struggling, 0, np.where(strong, 2, 1)), index=total.index).astype("Int8")
    return code.mask(total < MIN_REVIEWS)


def build() -> pd.DataFrame:
    go = pd.read_parquet(OUT / "game_outcomes.parquet")
    code = assign_tiers(go["outcome_total_reviews"], go["outcome_total_negative"])
    t = pd.DataFrame({"appid": go["appid"], "split": go["split"], "cv_fold": go["cv_fold"],
                      "eligible": (go["outcome_total_reviews"] >= MIN_REVIEWS).astype("int8"),
                      "tier": code.map(dict(enumerate(TIERS))), "tier_code": code})
    t["high_risk"] = (t["tier_code"] == 0).astype("Int8").mask(t["tier_code"].isna())
    t.to_parquet(OUT / "game_targets.parquet", index=False)

    bal = (t[t["eligible"] == 1].groupby(["split", "tier"]).size().unstack("split").reindex(TIERS)
           .assign(total=lambda d: d.sum(axis=1)))
    bal["share"] = (bal["total"] / bal["total"].sum()).round(3)
    DOCS.mkdir(parents=True, exist_ok=True)
    bal.rename_axis("tier").to_csv(DOCS / "game_target_balance.csv")
    log.info("game_targets: %d eligible of %d games; tiers %s", int(t["eligible"].sum()), len(t), bal["total"].to_dict())
    return t


CATALOG = [
    ("game_targets", "tier, tier_code", "target", "game_outcomes", "Success tier for games with 20+ reviews: Struggling (30%+ negative, 0), Solid (1), Strong (10%- negative, 2); missing below 20 reviews"),
    ("game_targets", "high_risk", "target", "game_outcomes", "Binary version: Struggling vs the rest (same games as tier)"),
    ("game_targets", "eligible, appid, split, cv_fold", "key", "game_outcomes, game_split.csv", "Has 20+ reviews (gets a tier); key and shared split"),
]


if __name__ == "__main__":
    build()
