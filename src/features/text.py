"""Text block: TF-IDF on English review text -> Truncated SVD (LSA), fitted on training games only.

Rows: English reviews with analysable text (`text_mining_ok`) from the 1,856 fully scraped games, minus reviews
written in another language. `language` is the reviewer's chosen language, not detected from the text, so
about 1% of "english" reviews are not English. Two filters remove most of them:
  1. non-Latin script (Russian, Chinese ...): Latin letters < 90% of all letters;
  2. Latin-script foreign text: at least 2 function words that are common in Spanish, Portuguese, French,
     German, Italian, Indonesian, Dutch, Turkish or Polish but are not English words (FOREIGN_MARKERS), or
     1 such word in a review of 6 words or fewer.
Both are heuristics (a language-ID model was too slow and mislabelled short English such as "its dope"), so a
small amount of foreign text remains: see the limitations in docs/feature_engineering.md.
Fit: TF-IDF vocabulary/idf on a random sample of training reviews, SVD (300 components) on a sample of
those; k = elbow of the cumulative-variance curve. Then every row is transformed in chunks, so memory
stays small (the full TF-IDF matrix is never held).

Outputs:
  data/processed/features/text_svd/part-00k.parquet   recommendationid, appid, t_01 .. t_kk   (git-ignored: ~200 MB)
  data/interim/text_svd_model.joblib                  fitted vectoriser + SVD (git-ignored)
  docs/feature_engineering/text_svd_variance.csv      variance per component
  docs/feature_engineering/text_svd_top_terms.csv     highest and lowest weighted terms of the first 15 components
  docs/figures/text_svd_variance.png
Regenerate with: python -m src.features.text
"""
from __future__ import annotations

import argparse
import glob
import shutil

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import DOCS, FIGS, INTERIM, OUT, PROCESSED, SEED, elbow, get_logger

log = get_logger("features")

N_FIT_DOCS = 120_000       # training reviews used to learn the vocabulary and idf weights
N_SVD_DOCS = 100_000       # of those, reviews used to fit the SVD
MAX_COMPONENTS = 300       # the elbow k is chosen on this curve
CHUNK = 40_000
MIN_LATIN_SHARE = 0.9      # drop reviews whose letters are less than 90% A-Z (non-Latin scripts)
# Function words of other Latin-script languages that are not English words. Words that clash with English
# ("die", "me", "no", "ve" as in "I've", "gut", "est", "la", "las", "los") are left out on purpose.
FOREIGN_MARKERS = frozenset("""el del una pero para por que está esta juego juegazo bueno más gracias jugar recomiendo
muchas mucho excelente jogo muito você voce não nao uma bom melhor mais jogar vale pena les des très jeu une avec
pour mais bien beaucoup vraiment und der das ist nicht ein sehr spiel ich mit aber auch nur macht che molto gioco
bello davvero yang dan ini itu tidak bagus banget sangat untuk dengan kalau sudah het een van niet heel leuk spel maar
çok oyun güzel için jest bardzo gra nie się ale polecam""".split())


def load_text(limit: int | None = None) -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games_clean.parquet", columns=["appid", "reviews_capped"])
    split = pd.read_csv(PROCESSED / "splits" / "game_split.csv")
    capped = set(games.loc[games["reviews_capped"], "appid"])
    parts = []
    for path in sorted(glob.glob(str(PROCESSED / "reviews_clean" / "*.parquet"))):
        p = pd.read_parquet(path, columns=["recommendationid", "appid", "review", "text_mining_ok"])
        parts.append(p[p["text_mining_ok"] & ~p["appid"].isin(capped)].drop(columns="text_mining_ok"))
        if limit and sum(len(x) for x in parts) >= limit:
            break
    df = pd.concat(parts, ignore_index=True).merge(split[["appid", "split"]], on="appid", how="left")
    text = df["review"].astype(object)         # Python re: Unicode-aware, unlike pandas' Arrow strings
    latin_share = text.str.count(r"[A-Za-z]") / text.str.count(r"[^\W\d_]").clip(lower=1)
    log.info("text: dropping %d of %d 'english' reviews written in a non-Latin script (%.2f%%)",
             int((latin_share < MIN_LATIN_SHARE).sum()), len(df), 100 * (latin_share < MIN_LATIN_SHARE).mean())
    df = df[latin_share >= MIN_LATIN_SHARE].reset_index(drop=True)

    tokens = df["review"].astype(object).str.lower().str.findall(r"[a-zà-ÿœ]+")
    markers = np.array([sum(w in FOREIGN_MARKERS for w in t) for t in tokens])
    foreign = (markers >= 2) | ((markers >= 1) & (tokens.str.len().to_numpy() <= 6))
    log.info("text: dropping %d further Latin-script reviews with foreign function words (%.2f%%)",
             int(foreign.sum()), 100 * foreign.mean())
    df = df[~foreign].reset_index(drop=True)
    return df.head(limit) if limit else df


def build(limit: int | None = None) -> None:
    df = load_text(limit)
    train = df[df["split"] == "train"]
    log.info("text: %d English reviews (%d from training games)", len(df), len(train))

    fit_docs = train["review"].sample(min(N_FIT_DOCS, len(train)), random_state=SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10 if len(fit_docs) > 50_000 else 3, max_df=0.9,
                          max_features=50_000, sublinear_tf=True, stop_words="english", dtype=np.float32)
    X_fit = vec.fit_transform(fit_docs)
    log.info("text: TF-IDF vocabulary %d terms from %d training reviews", len(vec.vocabulary_), X_fit.shape[0])

    n_comp = min(MAX_COMPONENTS, X_fit.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, n_iter=5, random_state=SEED)
    svd.fit(X_fit[: min(N_SVD_DOCS, X_fit.shape[0])])
    cum = np.cumsum(svd.explained_variance_ratio_)
    k = elbow(cum)
    log.info("text: elbow k=%d of %d components keeps %.1f%% of the TF-IDF variance (%d components: %.1f%%)",
             k, n_comp, 100 * cum[k - 1], n_comp, 100 * cum[-1])

    names = [f"t_{i:02d}" for i in range(1, k + 1)]
    folder = OUT / "text_svd"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for i, start in enumerate(range(0, len(df), CHUNK)):
        chunk = df.iloc[start:start + CHUNK]
        Z = svd.transform(vec.transform(chunk["review"]))[:, :k].astype("float32")
        out = pd.DataFrame(Z, columns=names)
        out.insert(0, "appid", chunk["appid"].to_numpy())
        out.insert(0, "recommendationid", chunk["recommendationid"].to_numpy())
        out.to_parquet(folder / f"part-{i:03d}.parquet", index=False)
        log.info("text: transformed %d/%d reviews", min(start + CHUNK, len(df)), len(df))

    INTERIM.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "svd": svd, "k": k}, INTERIM / "text_svd_model.joblib")
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"component": range(1, n_comp + 1), "explained_variance_ratio": svd.explained_variance_ratio_,
                  "cumulative": cum}).to_csv(DOCS / "text_svd_variance.csv", index=False)

    terms = np.array(vec.get_feature_names_out())
    rows = []
    for c in range(15):
        w = svd.components_[c]
        order = np.argsort(w)
        rows.append({"component": f"t_{c + 1:02d}", "top_positive_terms": ", ".join(terms[order[::-1][:12]]),
                     "top_negative_terms": ", ".join(terms[order[:12]])})
    pd.DataFrame(rows).to_csv(DOCS / "text_svd_top_terms.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(np.arange(1, n_comp + 1), 100 * cum, color="#4C72B0", lw=2)
    ax.axvline(k, color="#C44E52", ls="--")
    ax.scatter([k], [100 * cum[k - 1]], color="#C44E52", zorder=3)
    ax.annotate(f"elbow: k = {k}\n{100 * cum[k - 1]:.1f}% of TF-IDF variance", (k, 100 * cum[k - 1]),
                xytext=(k + 30, 100 * cum[k - 1] - 4), arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("number of SVD components"); ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title(f"Review-text SVD ({len(vec.vocabulary_):,} TF-IDF terms, training sample)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "text_svd_variance.png", dpi=150); plt.close(fig)


CATALOG = [
    ("text_svd", "t_01 .. t_NN", "at_review", "review (English, text_mining_ok, language-filtered)", "TF-IDF (1-2 grams) then Truncated SVD, fitted on training games; k at the variance elbow. Git-ignored: regenerate with python -m src.features.text"),
    ("text_svd", "recommendationid, appid", "key", "reviews_clean", "Key columns; join to review_features on recommendationid"),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N reviews (quick test)")
    build(ap.parse_args().limit)
