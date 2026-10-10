"""Task #31: Text mining and topic modelling on Steam game reviews.

Part of Review 2 (Tasks #31-#33, #44).
Builds upon Review 1's cleaned dataset and preliminary keyword themes, replacing crude keyword
matching with rigorous exploratory text mining, n-gram analysis, and NMF topic modelling.

Pipeline:
  1. Multilingual evaluation & language subsetting (English subset: 557,202 reviews).
  2. Preprocessing with negation binding ("not_fun", "too_hard", "clunky_controls") and noise reduction.
  3. Exploratory n-gram and vocabulary distinctiveness (Positive vs Negative; <2h vs >=2h).
  4. NMF topic modelling with TF-IDF vectorisation (k=8 data-driven topics).
  5. Topic prevalence comparison across sentiment, playtime windows, and price tiers.
  6. Output tables to reports/results/ and figures to figures/text_mining/.

Usage:
  python -m src.analysis.text_mining
"""
from __future__ import annotations

import argparse
import glob
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("text_mining")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FEATURES = PROCESSED / "features"
REPORTS = ROOT / "reports" / "results"
FIGS = ROOT / "figures" / "text_mining"

SEED = 42
N_TOPICS = 8
MIN_LATIN_SHARE = 0.9

FOREIGN_MARKERS = frozenset("""el del una pero para por que está esta juego juegazo bueno más gracias jugar recomiendo
muchas mucho excelente jogo muito você voce não nao uma bom melhor mais jogar vale pena les des très jeu une avec
pour mais bien beaucoup vraiment und der das ist nicht ein sehr spiel ich mit aber auch nur macht che molto gioco
bello davvero yang dan ini itu tidak bagus banget sangat untuk dengan kalau sudah het een van niet heel leuk spel maar
çok oyun güzel için jest bardzo gra nie się ale polecam""".split())

# Core English stopwords without negation words ("not", "no", "never", "too", "against", "cannot", "hardly", etc.)
RAW_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me",
    "more", "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours",
    "yourself", "yourselves", "game", "games", "play", "played", "playing",
    # Conversational filler words
    "get", "gets", "got", "getting", "one", "two", "first", "like", "likes", "liked", "just", "really",
    "even", "dont", "do_not", "does_not", "did_not", "is_not", "are_not", "will", "make", "makes", "made",
    "making", "way", "much", "also", "every", "pretty", "kinda", "now", "still", "felt", "feels", "feel",
    "feeling", "looks", "look", "looking", "looked", "well", "thing", "things", "lot", "lots", "bit", "see",
    "many", "going", "goes", "gone", "take", "takes", "took", "something", "give", "gives", "gave", "say",
    "says", "said", "know", "think", "people", "actually", "probably", "definitely", "maybe", "always",
    "can_not", "cant", "wont", "cannot", "time", "hours"
}

BASE_STOPWORDS = {w for w in RAW_STOPWORDS if len(w) >= 2}

TOPIC_LABELS = {
    0: "Boss Design, Damage & Upgrade Difficulty",
    1: "Repetitive Gameplay, Grind & Pacing",
    2: "Flawed Design, Clunky Controls & RNG",
    3: "Early Access State & Content Scarcity",
    4: "Replayability & Co-op Friend Engagement",
    5: "Multiplayer Issues, Crashes & Dev Bugs",
    6: "Art & Presentation vs Gameplay Loop",
    7: "Unbalanced Mechanics & Frustration",
}


def load_corpus(limit: int | None = None) -> pd.DataFrame:
    """Load cleaned reviews and merge with feature metadata, applying Latin and foreign-word filters."""
    log.info("Loading cleaned review text and metadata...")
    games = pd.read_parquet(PROCESSED / "games_clean.parquet", columns=["appid", "reviews_capped"])
    capped_appids = set(games.loc[games["reviews_capped"], "appid"])

    parts = []
    for path in sorted(glob.glob(str(PROCESSED / "reviews_clean" / "*.parquet"))):
        p = pd.read_parquet(path, columns=[
            "recommendationid", "appid", "review", "voted_up", "author_playtime_at_review", "text_mining_ok"
        ])
        # Filter for text_mining_ok and exclude capped mega-games
        p = p[p["text_mining_ok"] & ~p["appid"].isin(capped_appids)].drop(columns="text_mining_ok")
        parts.append(p)
        if limit and sum(len(x) for x in parts) >= limit:
            break

    df = pd.concat(parts, ignore_index=True)
    n_initial = len(df)
    log.info("Loaded %d candidate reviews from uncapped games", n_initial)

    # 1. Non-Latin script filter
    text_obj = df["review"].astype(object)
    latin_share = text_obj.str.count(r"[A-Za-z]") / text_obj.str.count(r"[^\W\d_]").clip(lower=1)
    non_latin = latin_share < MIN_LATIN_SHARE
    log.info("Dropping %d non-Latin reviews (%.2f%%)", int(non_latin.sum()), 100 * non_latin.mean())
    df = df[~non_latin].reset_index(drop=True)

    # 2. Latin-script foreign word heuristic filter
    tokens = df["review"].astype(object).str.lower().str.findall(r"[a-zà-ÿœ]+")
    markers = np.array([sum(w in FOREIGN_MARKERS for w in t) for t in tokens])
    foreign = (markers >= 2) | ((markers >= 1) & (tokens.str.len().to_numpy() <= 6))
    log.info("Dropping %d foreign-marker reviews (%.2f%%)", int(foreign.sum()), 100 * foreign.mean())
    df = df[~foreign].reset_index(drop=True)

    # 3. Merge with review_features metadata
    rf_parts = []
    for p in sorted(glob.glob(str(FEATURES / "review_features" / "*.parquet"))):
        rf_parts.append(pd.read_parquet(p, columns=[
            "recommendationid", "target_is_negative", "in_refund_window",
            "playtime_bucket", "price_tier", "split", "cv_fold"
        ]))
    rf = pd.concat(rf_parts, ignore_index=True)
    df = df.merge(rf, on="recommendationid", how="inner")
    log.info("Final processed English corpus: %d reviews (%.1f%% negative, %.1f%% in refund window)",
             len(df), 100 * df["target_is_negative"].mean(), 100 * df["in_refund_window"].mean())
    return df.head(limit) if limit else df


def preprocess_text(text: str) -> str:
    """Normalize text, bind negations and key complaint expressions, and remove noise."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # Strip URLs
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    # Normalize repeated characters (e.g., "sooooo baaaad" -> "soo baad")
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    # Handle common contractions
    t = re.sub(r"can't|cannot", "can_not", t)
    t = re.sub(r"won't", "will_not", t)
    t = re.sub(r"n't\b", "_not", t)
    t = re.sub(r"\bnot\s+([a-z]+)", r"not_\1", t)
    t = re.sub(r"\btoo\s+([a-z]+)", r"too_\1", t)
    t = re.sub(r"\bno\s+([a-z]+)", r"no_\1", t)
    t = re.sub(r"\black\s+of\s+([a-z]+)", r"lack_of_\1", t)
    # Remove punctuation except underscores
    t = re.sub(r"[^a-z0-9_\s]", " ", t)
    # Collapse whitespace
    return re.sub(r"\s+", " ", t).strip()


def run_exploratory_text_mining(df: pd.DataFrame) -> pd.DataFrame:
    """Compute frequent unigrams, bigrams, and vocabulary comparisons between review groups."""
    log.info("Running exploratory text mining & n-gram frequency analysis...")
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    neg_df = df[df["target_is_negative"] == 1]
    pos_df = df[df["target_is_negative"] == 0]
    early_neg_df = neg_df[neg_df["in_refund_window"] == 1]
    late_neg_df = neg_df[neg_df["in_refund_window"] == 0]

    # Preprocess a sample for vocabulary analysis to maintain efficiency
    sample_size = min(30_000, len(neg_df))
    neg_sample = neg_df["review"].sample(sample_size, random_state=SEED).apply(preprocess_text)
    pos_sample = pos_df["review"].sample(sample_size, random_state=SEED).apply(preprocess_text)

    vec = CountVectorizer(ngram_range=(1, 2), min_df=15, max_df=0.6,
                          stop_words=list(BASE_STOPWORDS), token_pattern=r"\b[a-z_][a-z0-9_]{2,}\b")
    X_neg = vec.fit_transform(neg_sample)
    vocab = vec.get_feature_names_out()
    neg_counts = np.asarray(X_neg.sum(axis=0)).ravel()

    # Score positive sample on same vocabulary
    X_pos = vec.transform(pos_sample)
    pos_counts = np.asarray(X_pos.sum(axis=0)).ravel()

    total_neg_words = neg_counts.sum()
    total_pos_words = pos_counts.sum()

    # Calculate frequency rates and Log Odds Ratio with Laplace smoothing
    p_neg = (neg_counts + 1) / (total_neg_words + len(vocab))
    p_pos = (pos_counts + 1) / (total_pos_words + len(vocab))
    log_odds = np.log(p_neg / (1 - p_neg)) - np.log(p_pos / (1 - p_pos))

    ngram_df = pd.DataFrame({
        "term": vocab,
        "is_bigram": [("_" in w and not w.startswith(("not_", "too_", "no_", "lack_of_"))) or (" " in w) for w in vocab],
        "count_negative": neg_counts,
        "count_positive": pos_counts,
        "freq_negative_per_10k": 10000 * (neg_counts / total_neg_words),
        "freq_positive_per_10k": 10000 * (pos_counts / total_pos_words),
        "log_odds_ratio": log_odds,
    })
    ngram_df = ngram_df.sort_values("log_odds_ratio", ascending=False).reset_index(drop=True)
    ngram_df.to_csv(REPORTS / "ngram_frequencies.csv", index=False)
    log.info("Saved reports/results/ngram_frequencies.csv (%d terms)", len(ngram_df))

    # Compare Early (<2h) vs Late (>=2h) negative reviews
    early_sample_size = min(15_000, len(early_neg_df))
    late_sample_size = min(15_000, len(late_neg_df))
    early_sample = early_neg_df["review"].sample(early_sample_size, random_state=SEED).apply(preprocess_text)
    late_sample = late_neg_df["review"].sample(late_sample_size, random_state=SEED).apply(preprocess_text)

    vec_time = CountVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.6,
                               stop_words=list(BASE_STOPWORDS), token_pattern=r"\b[a-z_][a-z0-9_]{2,}\b")
    X_early = vec_time.fit_transform(early_sample)
    vocab_time = vec_time.get_feature_names_out()
    early_counts = np.asarray(X_early.sum(axis=0)).ravel()
    X_late = vec_time.transform(late_sample)
    late_counts = np.asarray(X_late.sum(axis=0)).ravel()

    t_early_words, t_late_words = early_counts.sum(), late_counts.sum()
    p_early = (early_counts + 1) / (t_early_words + len(vocab_time))
    p_late = (late_counts + 1) / (t_late_words + len(vocab_time))
    time_log_odds = np.log(p_early / (1 - p_early)) - np.log(p_late / (1 - p_late))

    time_ngram_df = pd.DataFrame({
        "term": vocab_time,
        "count_under_2h": early_counts,
        "count_later": late_counts,
        "rate_under_2h_per_10k": 10000 * (early_counts / t_early_words),
        "rate_later_per_10k": 10000 * (late_counts / t_late_words),
        "log_odds_under_2h_vs_later": time_log_odds
    }).sort_values("log_odds_under_2h_vs_later", ascending=False).reset_index(drop=True)
    time_ngram_df.to_csv(REPORTS / "playtime_ngram_comparison.csv", index=False)
    log.info("Saved reports/results/playtime_ngram_comparison.csv")

    # Generate Visualization: N-Gram Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top Negative Distinctive Terms
    top_neg_terms = ngram_df.head(15)
    axes[0].barh(range(len(top_neg_terms)), top_neg_terms["log_odds_ratio"], color="#C44E52")
    axes[0].set_yticks(range(len(top_neg_terms)))
    axes[0].set_yticklabels(top_neg_terms["term"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Log Odds Ratio (Negative vs Positive)")
    axes[0].set_title("Top Distinctive Terms in Negative Reviews")
    axes[0].grid(axis="x", alpha=0.3)

    # Top Early vs Late Negative Terms
    top_early_terms = time_ngram_df.head(15)
    axes[1].barh(range(len(top_early_terms)), top_early_terms["log_odds_under_2h_vs_later"], color="#DD8452")
    axes[1].set_yticks(range(len(top_early_terms)))
    axes[1].set_yticklabels(top_early_terms["term"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Log Odds Ratio (<2h vs >=2h Playtime)")
    axes[1].set_title("Terms Over-represented Inside Refund Window (<2h)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "ngram_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved figures/text_mining/ngram_comparison.png")

    return ngram_df


def fit_topic_model(df: pd.DataFrame) -> tuple[NMF, TfidfVectorizer, pd.DataFrame, pd.DataFrame]:
    """Fit NMF on negative reviews and evaluate document-topic weights across all groups."""
    log.info("Fitting NMF Topic Model (k=%d) with TF-IDF vectorizer...", N_TOPICS)

    train_neg = df[(df["split"] == "train") & (df["target_is_negative"] == 1)]
    log.info("Training negative reviews available: %d", len(train_neg))

    fit_sample = train_neg["review"].sample(min(40_000, len(train_neg)), random_state=SEED).apply(preprocess_text)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=20,
        max_df=0.4,
        max_features=15_000,
        sublinear_tf=True,
        stop_words=list(BASE_STOPWORDS),
        token_pattern=r"\b[a-z_][a-z0-9_]{2,}\b"
    )
    X_fit = vec.fit_transform(fit_sample)
    feature_names = np.array(vec.get_feature_names_out())
    log.info("TF-IDF matrix built: shape %s, vocabulary %d terms", X_fit.shape, len(feature_names))

    nmf = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=300,
        random_state=SEED
    )
    nmf.fit(X_fit)
    log.info("NMF successfully fitted with %d components", N_TOPICS)

    # Extract Top Terms per Topic
    topic_terms_records = []
    for topic_idx in range(N_TOPICS):
        weights = nmf.components_[topic_idx]
        top_indices = np.argsort(weights)[::-1][:15]
        top_terms = feature_names[top_indices]
        top_weights = weights[top_indices]

        topic_terms_records.append({
            "topic_id": topic_idx,
            "label": TOPIC_LABELS[topic_idx],
            "top_terms": ", ".join(top_terms[:10]),
            "all_15_terms": ", ".join(top_terms),
            "top_terms_weights": ", ".join(f"{w:.3f}" for w in top_weights[:10]),
        })

    topic_terms_df = pd.DataFrame(topic_terms_records)
    topic_terms_df.to_csv(REPORTS / "topic_terms.csv", index=False)
    log.info("Saved reports/results/topic_terms.csv")

    # Transform a representative evaluation sample across subsets (train and test)
    eval_sample = pd.concat([
        df[df["target_is_negative"] == 1].sample(min(25_000, (df["target_is_negative"] == 1).sum()), random_state=SEED),
        df[df["target_is_negative"] == 0].sample(min(25_000, (df["target_is_negative"] == 0).sum()), random_state=SEED),
    ]).reset_index(drop=True)

    eval_clean = eval_sample["review"].apply(preprocess_text)
    X_eval = vec.transform(eval_clean)
    W_eval = nmf.transform(X_eval)  # Document-topic matrix

    # Normalize weights per document so they sum to 1 (topic proportions)
    row_sums = W_eval.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W_eval / row_sums

    # Assign dominant topic
    eval_sample["dominant_topic"] = np.argmax(W_norm, axis=1)
    for k in range(N_TOPICS):
        eval_sample[f"topic_{k}_weight"] = W_norm[:, k]

    # Calculate Topic Prevalence
    prevalence_records = []
    groups = {
        "All Sampled Reviews": pd.Series(True, index=eval_sample.index),
        "Negative Reviews": eval_sample["target_is_negative"] == 1,
        "Positive Reviews": eval_sample["target_is_negative"] == 0,
        "Negative (<2h, Refund Window)": (eval_sample["target_is_negative"] == 1) & (eval_sample["in_refund_window"] == 1),
        "Negative (>=2h, Post-Window)": (eval_sample["target_is_negative"] == 1) & (eval_sample["in_refund_window"] == 0),
        "Price Tier < $5 (Negative)": (eval_sample["target_is_negative"] == 1) & (eval_sample["price_tier"] == 1),
        "Price Tier $10-$20 (Negative)": (eval_sample["target_is_negative"] == 1) & (eval_sample["price_tier"] == 3),
        "Price Tier $20+ (Negative)": (eval_sample["target_is_negative"] == 1) & (eval_sample["price_tier"] == 4),
    }

    for group_name, mask in groups.items():
        sub = eval_sample[mask]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        for topic_idx in range(N_TOPICS):
            mean_weight = sub[f"topic_{topic_idx}_weight"].mean()
            dominant_share = (sub["dominant_topic"] == topic_idx).mean()
            prevalence_records.append({
                "group": group_name,
                "n_reviews": n_sub,
                "topic_id": topic_idx,
                "label": TOPIC_LABELS[topic_idx],
                "mean_topic_weight": mean_weight,
                "dominant_topic_share": dominant_share,
            })

    prevalence_df = pd.DataFrame(prevalence_records)
    prevalence_df.to_csv(REPORTS / "topic_prevalence.csv", index=False)
    log.info("Saved reports/results/topic_prevalence.csv")

    # Extract Representative Reviews (Anonymized, top scoring for each topic)
    rep_records = []
    for topic_idx in range(N_TOPICS):
        topic_col = f"topic_{topic_idx}_weight"
        top_docs = eval_sample[eval_sample["target_is_negative"] == 1].sort_values(topic_col, ascending=False).head(3)
        for _, row in top_docs.iterrows():
            clean_excerpt = row["review"].replace("\n", " ").strip()
            if len(clean_excerpt) > 280:
                clean_excerpt = clean_excerpt[:280] + "..."
            rep_records.append({
                "topic_id": topic_idx,
                "label": TOPIC_LABELS[topic_idx],
                "topic_weight": row[topic_col],
                "author_playtime_hours": round(row["author_playtime_at_review"] / 60, 2),
                "in_refund_window": bool(row["in_refund_window"]),
                "review_excerpt": clean_excerpt
            })

    rep_df = pd.DataFrame(rep_records)
    rep_df.to_csv(REPORTS / "representative_reviews.csv", index=False)
    log.info("Saved reports/results/representative_reviews.csv")

    # Visualizations:
    # 1. Topic Prevalence in Negative vs Positive Reviews
    fig, ax = plt.subplots(figsize=(13, 6))
    neg_prev = prevalence_df[prevalence_df["group"] == "Negative Reviews"].sort_values("topic_id")
    pos_prev = prevalence_df[prevalence_df["group"] == "Positive Reviews"].sort_values("topic_id")

    x = np.arange(N_TOPICS)
    width = 0.38
    ax.bar(x - width / 2, 100 * neg_prev["dominant_topic_share"], width, label="Negative Reviews", color="#C44E52")
    ax.bar(x + width / 2, 100 * pos_prev["dominant_topic_share"], width, label="Positive Reviews", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}: {TOPIC_LABELS[i]}" for i in range(N_TOPICS)], rotation=30, ha="right")
    ax.set_ylabel("Dominant Topic Share (%)")
    ax.set_title("Topic Distribution: Negative vs Positive Reviews (NMF k=8)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "topic_prevalence_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved figures/text_mining/topic_prevalence_comparison.png")

    # 2. Topic Prevalence: Early (<2h) vs Later (>=2h) Negative Reviews
    fig, ax = plt.subplots(figsize=(13, 6))
    early_prev = prevalence_df[prevalence_df["group"] == "Negative (<2h, Refund Window)"].sort_values("topic_id")
    late_prev = prevalence_df[prevalence_df["group"] == "Negative (>=2h, Post-Window)"].sort_values("topic_id")

    ax.bar(x - width / 2, 100 * early_prev["dominant_topic_share"], width, label="Inside Refund Window (<2h)", color="#DD8452")
    ax.bar(x + width / 2, 100 * late_prev["dominant_topic_share"], width, label="Outside Refund Window (>=2h)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}: {TOPIC_LABELS[i]}" for i in range(N_TOPICS)], rotation=30, ha="right")
    ax.set_ylabel("Dominant Topic Share (%)")
    ax.set_title("Complaint Themes by Playtime: Inside vs Outside Refund Window (<2h)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "topics_by_playtime.png", dpi=150)
    plt.close(fig)
    log.info("Saved figures/text_mining/topics_by_playtime.png")

    return nmf, vec, topic_terms_df, prevalence_df


def main():
    parser = argparse.ArgumentParser(description="Task #31: Text mining and topic modelling")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of reviews for quick testing")
    args = parser.parse_args()

    df = load_corpus(limit=args.limit)
    run_exploratory_text_mining(df)
    fit_topic_model(df)
    log.info("Task #31 Text mining and topic modelling complete!")


if __name__ == "__main__":
    main()
