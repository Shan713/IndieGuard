"""Shared, leakage-safe train/test split by GAME, plus 5 cross-validation folds inside train.

Every team model uses this file so results are comparable:
  data/processed/splits/game_split.csv   appid, split ('train'/'test'), cv_fold (0-4 for train, -1 for test)
  docs/train_test_split.md               method + balance checks

Rules:
- Split unit is the game (appid): all reviews of a game are on the same side, so no game-specific
  signal leaks from train to test (brief: "train/test split grouped by game to avoid leakage").
- Stratified by release year x review-volume quartile x rating tertile so both sides have the same mix.
- The 5 capped games (reviews_capped) are forced into train, so the test set has only fully scraped games.
- Review-level balance: the top 20 games hold ~53% of all reviews, so which giants land in test swings the
  pooled review counts. Among SEED_SEARCH candidate seeds (stratified as above), the one whose test set holds
  closest to 20% of reviews with a negative-review rate closest to train's is used; folds likewise. This is
  chosen before any model is trained (no labels beyond class balance are used), so it does not leak.
- Deterministic: re-running gives the identical split.

Usage: python -m src.splits.make_split
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.collect.common import ROOT

PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "splits" / "game_split.csv"
DOC = ROOT / "docs" / "train_test_split.md"
SEED_SEARCH = range(500)
TEST_SIZE = 0.20
N_FOLDS = 5


def strata(g: pd.DataFrame) -> pd.Series:
    year = g["release_date"].dt.year.astype(str)
    volume = pd.qcut(g["steam_total_reviews"].rank(method="first"), 4, labels=["v1", "v2", "v3", "v4"]).astype(str)
    rating = pd.qcut((g["steam_total_positive"] / g["steam_total_reviews"]).rank(method="first"), 3,
                     labels=["r1", "r2", "r3"]).astype(str)
    return year + "_" + volume + "_" + rating


def imbalance(parts: list[pd.DataFrame], total_n: int, shares: list[float]) -> float:
    """Distance of each part from its target review share and from the overall negative-review rate."""
    overall_neg = sum(p["neg"].sum() for p in parts) / sum(p["n"].sum() for p in parts)
    return sum(abs(p["n"].sum() / total_n - sh) + abs(p["neg"].sum() / p["n"].sum() - overall_neg)
               for p, sh in zip(parts, shares))


def main() -> None:
    g = pd.read_parquet(PROCESSED / "games_clean.parquet")
    g["stratum"] = strata(g)
    per_game = pd.read_parquet(PROCESSED / "reviews_clean", columns=["appid", "voted_up"]).groupby("appid")["voted_up"]
    g = g.join(pd.DataFrame({"n": per_game.size(), "neg": per_game.apply(lambda x: (~x).sum())}), on="appid")

    forced = g[g["reviews_capped"]]
    pool = g[~g["reviews_capped"]]          # capped games always go to train and are left out of the balance score
    n_test = int(round(TEST_SIZE * len(g)))

    best = None
    for seed in SEED_SEARCH:
        _, te = train_test_split(pool["appid"], test_size=n_test, stratify=pool["stratum"], random_state=seed)
        t, tr = pool[pool["appid"].isin(te)], pool[~pool["appid"].isin(te)]
        score = imbalance([t, tr], pool["n"].sum(), [TEST_SIZE, 1 - TEST_SIZE])
        if best is None or score < best[0]:
            best = (score, seed, set(te))
    test_seed, test_ids = best[1], best[2]

    split = pd.DataFrame({"appid": g["appid"]})
    split["split"] = np.where(split["appid"].isin(test_ids), "test", "train")
    split["cv_fold"] = -1
    train = g[split["split"].eq("train").values].reset_index(drop=True)
    train_pool = train[~train["reviews_capped"]]

    best = None
    for seed in SEED_SEARCH:
        folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(train, train["stratum"]))
        parts = [train.iloc[idx] for _, idx in folds]
        score = imbalance([p[~p["reviews_capped"]] for p in parts], train_pool["n"].sum(), [1 / N_FOLDS] * N_FOLDS)
        if best is None or score < best[0]:
            best = (score, seed, folds)
    fold_seed = best[1]
    for k, (_, idx) in enumerate(best[2]):
        split.loc[split["appid"].isin(train["appid"].iloc[idx]), "cv_fold"] = k

    OUT.parent.mkdir(parents=True, exist_ok=True)
    split.sort_values("appid").to_csv(OUT, index=False)
    write_doc(g.merge(split, on="appid"), len(forced), test_seed, fold_seed)
    print(f"wrote {OUT} ({(split.split == 'train').sum()} train / {(split.split == 'test').sum()} test games; "
          f"test seed {test_seed}, fold seed {fold_seed})")


def write_doc(d: pd.DataFrame, n_forced: int, test_seed: int, fold_seed: int) -> None:
    reviews = pd.read_parquet(PROCESSED / "reviews_clean", columns=["appid", "voted_up", "text_mining_ok"])
    r = reviews.merge(d[["appid", "split", "cv_fold", "reviews_capped"]], on="appid")

    def row(name, gm, rv):
        return (f"| {name} | {len(gm):,} | {len(rv):,} | {1 - rv['voted_up'].mean():.1%} | "
                f"{rv['text_mining_ok'].sum():,} | {gm['price_usd'].median():.2f} | "
                f"{(gm['steam_total_positive'] / gm['steam_total_reviews']).median():.1%} |")

    head = ["| Part | Games | Reviews | Negative reviews | Text-mining reviews | Median price (USD) | Median game rating |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    parts = [row("train", d[d.split == "train"], r[r.split == "train"]),
             row("test", d[d.split == "test"], r[r.split == "test"])]
    parts_uncapped = [row("train (excl. capped)", d[(d.split == "train") & ~d.reviews_capped],
                          r[(r.split == "train") & ~r.reviews_capped])]
    folds = [row(f"fold {k}", d[d.cv_fold == k], r[r.cv_fold == k]) for k in range(N_FOLDS)]
    years = pd.crosstab(d["release_date"].dt.year, d["split"], normalize="columns").round(3) * 100

    lines = [
        "# Train / test split", "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC by `python -m src.splits.make_split` "
        f"(test seed {test_seed}, fold seed {fold_seed}). File: `data/processed/splits/game_split.csv`.", "",
        "**Every model in the team uses this split**, so results are directly comparable.", "",
        "## Method", "",
        f"- **Unit: game.** All reviews of a game are on the same side, so no game-specific signal leaks into the test set.",
        f"- **Test: {TEST_SIZE:.0%} of games**, held out. Use it **once**, for the final comparison. Never tune on it.",
        f"- **{N_FOLDS} cross-validation folds** inside train (`cv_fold` 0-{N_FOLDS - 1}), also by game, for tuning and model selection.",
        "- **Stratified** by release year × review-volume quartile × rating tertile (Steam totals), so both sides "
        "have the same mix of small/large and liked/disliked games.",
        f"- **The {n_forced} capped games are always in train** (`reviews_capped`): the test set contains only fully "
        "scraped games. For review-level models, drop them from train too if you use launch/patch timing features.",
        f"- **Review-level balance.** The top 20 games hold ~53% of all reviews, so a plain stratified split can put "
        f"very different review volumes and negative rates in test. Among {len(SEED_SEARCH)} stratified candidate "
        "splits, the one whose test set is closest to 20% of reviews with the same negative-review rate as train "
        "was kept (folds chosen the same way). This was decided before any model was trained.", "",
        "## How to use", "",
        "```python",
        "import pandas as pd",
        "split = pd.read_csv('data/processed/splits/game_split.csv')",
        "reviews = pd.read_parquet('data/processed/reviews_clean').merge(split, on='appid')",
        "train, test = reviews[reviews.split == 'train'], reviews[reviews.split == 'test']",
        "",
        "# tuning: GroupKFold-style CV with the shared folds",
        "for k in range(5):",
        "    fit_part, val_part = train[train.cv_fold != k], train[train.cv_fold == k]",
        "```", "",
        "Anything learned from data (scalers, encoders, PCA/SVD, TF-IDF, bucket edges from quantiles) must be "
        "fitted on the training part only and then applied to the test part.", "",
        "## Balance checks", "",
        *head, *parts, *parts_uncapped, "", "Cross-validation folds (train only):", "", *head, *folds, "",
        "Release year share (% of games in each part):", "",
        "| Year | train | test |", "| --- | ---: | ---: |",
        *[f"| {y} | {years.loc[y, 'train']:.1f} | {years.loc[y, 'test']:.1f} |" for y in years.index], "",
        "**Why the folds still differ at review level.** Three games dominate their folds: Dave the Diver "
        "(162k reviews, 3.5% negative) is ~56% of fold 2, Buckshot Roulette ~49% of fold 0 and Escape From Duckov "
        "~41% of fold 4. A game cannot be split across folds without reintroducing leakage, and putting the three "
        "giants in three different folds is the best possible. So for review-level models:", "",
        "- report CV results as **mean ± standard deviation across the 5 folds**, not a single fold;",
        "- report **per-game (macro) metrics** next to pooled ones, so a few huge games do not decide the result;",
        "- optionally weight reviews so each game contributes at most a fixed amount (e.g. `sample_weight` = "
        "min(1, 5000 / reviews_of_that_game)), and say so in the report.", "",
        "The held-out **test** set is balanced at review level (see table above), so the final comparison is fair.", "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
