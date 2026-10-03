"""Checks that the feature tables are complete, leak-free and fitted on training games only.

  python -m src.features.validate        (also run at the end of python -m src.features)
Exit code 0 = every check passed.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import tags
from .common import OUT, PROCESSED, get_logger, load_games

log = get_logger("features")

# Columns that exist only after the review is written or measure the outcome: never allowed as features
FORBIDDEN_REVIEW = {"votes_up", "votes_funny", "weighted_vote_score", "comment_count", "n_reactions", "has_dev_response",
                    "refunded", "author_playtime_forever", "author_playtime_last_two_weeks", "author_last_played", "voted_up"}


def run(check_text: bool = True) -> bool:
    results: list[tuple[bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((bool(ok), name))
        print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""))

    games = load_games()
    gf = pd.read_parquet(OUT / "game_features.parquet")
    go = pd.read_parquet(OUT / "game_outcomes.parquet")
    check("game_features: one row per game, same games as game_split.csv", gf["appid"].is_unique and set(gf["appid"]) == set(games["appid"]), f"{len(gf)} games")
    check("game_features: split and cv_fold copied from game_split.csv",
          gf.merge(games[["appid", "split", "cv_fold"]], on="appid", suffixes=("", "_s")).pipe(lambda d: (d["split"] == d["split_s"]).all() and (d["cv_fold"] == d["cv_fold_s"]).all()))
    num = gf.drop(columns=["split"])
    check("game_features: no missing values", not num.isna().any().any())
    check("game_features: no outcome columns among the features", not any(c.startswith(("outcome_", "steam_total", "spy_")) for c in gf.columns))
    check("game_outcomes: same games, outcome_ columns only", set(go["appid"]) == set(gf["appid"]) and all(c.startswith("outcome_") or c in ("appid", "split", "cv_fold") for c in go.columns))

    # tag PCA must have been fitted on training games only
    model = dict(np.load(OUT / "tag_pca_model.npz"))
    train = games[games["split"] == "train"]
    X_train, X_all = tags.multi_hot(train, list(model["columns"])), tags.multi_hot(games, list(model["columns"]))
    check("tag PCA mean equals the TRAINING-game mean", np.allclose(model["mean"], X_train.mean(axis=0), atol=1e-6))
    check("tag PCA mean differs from the all-games mean (test games were not used)", not np.allclose(model["mean"], X_all.mean(axis=0), atol=1e-6))
    pcs = [c for c in gf.columns if c.startswith("pc_")]
    again = tags.transform(games.set_index("appid").loc[gf["appid"]].reset_index(), model)
    check("tag components reproduce from the saved model", np.allclose(gf[pcs].to_numpy(), again, atol=1e-4), f"{len(pcs)} components")

    files = sorted(glob.glob(str(OUT / "review_features" / "*.parquet")))
    rf = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    clean_ids = pd.concat([pd.read_parquet(f, columns=["recommendationid", "appid"]) for f in sorted(glob.glob(str(PROCESSED / "reviews_clean" / "*.parquet")))])
    capped = set(games.loc[games["reviews_capped"], "appid"])
    clean_ids = clean_ids[~clean_ids["appid"].isin(capped)]
    check("review_features: exactly the clean reviews of the 1,856 fully scraped games", rf["recommendationid"].is_unique and set(rf["recommendationid"]) == set(clean_ids["recommendationid"]), f"{len(rf):,} reviews")
    check("review_features: no capped game", not rf["appid"].isin(capped).any())
    check("review_features: split matches the game's split", (rf["split"] == rf["appid"].map(games.set_index("appid")["split"])).all())
    check("review_features: no game in both train and test", rf.groupby("appid")["split"].nunique().max() == 1)
    check("review_features: no post-review or outcome columns", not (FORBIDDEN_REVIEW & set(rf.columns)) and not any(c.startswith(("steam_total", "outcome_")) for c in rf.columns))
    check("review_features: label has both classes in train and test", all(rf.loc[rf["split"] == s, "target_is_negative"].nunique() == 2 for s in ("train", "test")))

    if check_text:
        tf = pd.concat([pd.read_parquet(f, columns=["recommendationid"]) for f in sorted(glob.glob(str(OUT / "text_svd" / "*.parquet")))])
        check("text_svd: unique review ids, all present in review_features", tf["recommendationid"].is_unique and tf["recommendationid"].isin(rf["recommendationid"]).all(), f"{len(tf):,} reviews")
        first = pd.read_parquet(sorted(glob.glob(str(OUT / "text_svd" / "*.parquet")))[0])
        check("text_svd: no missing values", not first.isna().any().any())

    ok = all(r for r, _ in results)
    log.info("validate: %d/%d checks passed", sum(r for r, _ in results), len(results))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
