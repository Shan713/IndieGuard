"""Tag block: user tags + store categories -> multi-hot -> PCA (fitted on training games only).

Steps (all vocabulary and PCA decisions use the training games only):
  1. Multi-hot encode user tags (top 20 per game) and store categories. Genres are dropped: 11 of the 12
     genres are also user tags (Action, Strategy, RPG ...) and "Free To Play" duplicates `is_free`.
  2. Drop near-constant columns (in >= 95% of training games) and rare columns (in < 1%).
  3. Drop categories that repeat a tag (Co-op, PvP).
  4. PCA (centred). k = elbow of the cumulative-variance curve (src.features.common.elbow).

Outputs:
  data/processed/features/tag_components.parquet    appid, split, pc_01 .. pc_kk  (all games)
  data/processed/features/tag_pca_model.npz         vocabulary, mean, components: transform new games
  docs/feature_engineering/tag_columns.csv          every candidate column: prevalence and why kept/dropped
  docs/feature_engineering/tag_pca_variance.csv     variance per component (in-sample and held-out)
  docs/feature_engineering/tag_pca_loadings.csv     column x component loadings
  docs/figures/tag_pca_variance.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .common import DOCS, FIGS, OUT, SEED, elbow, get_logger, load_games, split_items

log = get_logger("features")

MIN_PREVALENCE = 0.01     # drop columns present in fewer than 1% of training games
MAX_PREVALENCE = 0.95     # drop columns present in at least 95% of training games (carry no information)


def _item_sets(games: pd.DataFrame) -> tuple[list[set], list[set]]:
    return ([set(split_items(v)) for v in games["tags"]], [set(split_items(v)) for v in games["categories"]])


def build_vocabulary(train: pd.DataFrame) -> pd.DataFrame:
    """Candidate columns with their training prevalence and the decision taken for each."""
    tags, cats = _item_sets(train)
    n = len(train)
    rows = []
    tag_counts = pd.Series([x for s in tags for x in s]).value_counts()
    tag_names = {x.lower() for x in tag_counts.index}
    for kind, counts in (("tag", tag_counts), ("category", pd.Series([x for s in cats for x in s]).value_counts())):
        for name, c in counts.items():
            prev = c / n
            if kind == "category" and name.lower() in tag_names:
                decision = "dropped: duplicates a tag"
            elif prev >= MAX_PREVALENCE:
                decision = "dropped: near-constant"
            elif prev < MIN_PREVALENCE:
                decision = "dropped: rare"
            else:
                decision = "kept"
            rows.append({"column": f"{kind}__{name}", "kind": kind, "train_games": int(c),
                         "train_prevalence": round(prev, 4), "decision": decision})
    # Full sort (ties broken by name): the column order must not depend on set iteration order, which changes
    # with the per-process hash seed and would change the PCA from run to run.
    return (pd.DataFrame(rows).sort_values(["decision", "train_games", "column"], ascending=[True, False, True])
            .reset_index(drop=True))


def multi_hot(games: pd.DataFrame, columns: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(columns)}
    tags, cats = _item_sets(games)
    X = np.zeros((len(games), len(columns)), dtype=np.float32)
    for r, (ts, cs) in enumerate(zip(tags, cats)):
        for kind, items in (("tag", ts), ("category", cs)):
            for item in items:
                j = idx.get(f"{kind}__{item}")
                if j is not None:
                    X[r, j] = 1.0
    return X


def transform(games: pd.DataFrame, model: dict) -> np.ndarray:
    """Project games onto the fitted tag components (use for new games, e.g. the launch-risk scorer)."""
    X = multi_hot(games, list(model["columns"]))
    return (X - model["mean"]) @ model["components"].T


def heldout_variance(X: np.ndarray, folds: np.ndarray, k: int) -> float:
    """Variance of held-out training games captured by k components fitted on the other cv folds."""
    captured = []
    for f in np.unique(folds):
        a, b = X[folds != f], X[folds == f]
        m = PCA(n_components=k, random_state=SEED).fit(a)
        rec = m.transform(b) @ m.components_ + m.mean_
        captured.append(1 - ((b - rec) ** 2).sum() / ((b - m.mean_) ** 2).sum())
    return float(np.mean(captured))


def build() -> pd.DataFrame:
    games = load_games()
    train = games[games["split"] == "train"].reset_index(drop=True)

    vocab = build_vocabulary(train)
    columns = vocab.loc[vocab["decision"] == "kept", "column"].tolist()
    X_train = multi_hot(train, columns)
    log.info("tags: %d candidate columns -> %d kept (%s)", len(vocab), len(columns),
             vocab["decision"].value_counts().to_dict())

    full = PCA(random_state=SEED).fit(X_train)
    cum = np.cumsum(full.explained_variance_ratio_)
    k = elbow(cum)
    pca = PCA(n_components=k, random_state=SEED).fit(X_train)
    held = heldout_variance(X_train, train["cv_fold"].to_numpy(), k)
    log.info("tags: elbow k=%d keeps %.1f%% of training variance (%.1f%% on held-out cv folds)", k, 100 * cum[k - 1], 100 * held)

    names = [f"pc_{i:02d}" for i in range(1, k + 1)]
    comps = pd.DataFrame(pca.transform(multi_hot(games, columns)), columns=names).astype("float32")
    comps.insert(0, "split", games["split"].values)
    comps.insert(0, "appid", games["appid"].values)

    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    comps.to_parquet(OUT / "tag_components.parquet", index=False)
    np.savez(OUT / "tag_pca_model.npz", columns=np.array(columns), mean=pca.mean_.astype("float32"),
             components=pca.components_.astype("float32"))
    vocab.to_csv(DOCS / "tag_columns.csv", index=False)
    pd.DataFrame({"component": range(1, len(cum) + 1), "explained_variance_ratio": full.explained_variance_ratio_,
                  "cumulative": cum}).to_csv(DOCS / "tag_pca_variance.csv", index=False)
    pd.DataFrame(pca.components_.T, index=columns, columns=names).rename_axis("column").to_csv(DOCS / "tag_pca_loadings.csv")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(np.arange(1, len(cum) + 1), 100 * cum, color="#4C72B0", lw=2)
    ax.axvline(k, color="#C44E52", ls="--")
    ax.scatter([k], [100 * cum[k - 1]], color="#C44E52", zorder=3)
    ax.annotate(f"elbow: k = {k}\n{100 * cum[k - 1]:.0f}% of training variance", (k, 100 * cum[k - 1]),
                xytext=(k + 12, 100 * cum[k - 1] - 22), arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("number of principal components"); ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title(f"Tag PCA on training games ({len(columns)} columns)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "tag_pca_variance.png", dpi=150); plt.close(fig)
    return comps


CATALOG = [
    ("game_features", "pc_01 .. pc_NN", "launch_time", "tags, categories", "Principal components of the multi-hot tag/category block, fitted on training games (k at the variance elbow)"),
]


if __name__ == "__main__":
    build()
