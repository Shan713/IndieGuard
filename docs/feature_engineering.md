# Feature engineering and dimensionality reduction

Closes the work in issue #4. Code: `src/features/`. Rebuild everything and validate it:

```bash
python -m src.features              # about 3 minutes; add --skip-text to skip the text block
python -m src.features.validate     # 21 checks: completeness, targets, no leakage, fitted on training games only
                                    # (the 2 text checks are skipped until the text matrix has been built)
```

It needs only files that are committed (`data/processed/`), not the raw data. The authoritative list of every
feature, with its source, group and definition, is `docs/feature_engineering/feature_catalog.csv`.

## 1. What was built

| Table | Rows | Columns | Where | Use |
| --- | ---: | ---: | --- | --- |
| `game_features` | 1,861 games | 90 (68 tag components) | `data/processed/features/game_features.parquet` | Game-level success-tier model, association rules |
| `game_outcomes` | 1,861 games | 10 (all `outcome_*`) | `.../game_outcomes.parquet` | Targets for the game-level model. Never inputs |
| `game_targets` | 1,861 games (1,490 with a tier) | 7 | `.../game_targets.parquet` | The game-level label (`tier`, `tier_code`, `high_risk`); see `docs/game_target.md` |
| `review_features` | 1,573,436 reviews | 29 | `.../review_features/` (5 parquet parts) | Review-level "is it negative?" model |
| `tag_components` | 1,861 games | 70 | `.../tag_components.parquet` | The tag PCA scores on their own |
| `tag_pca_model.npz` | | | `.../tag_pca_model.npz` | Vocabulary, mean and loadings to score a new game |
| `text_svd` | 557,202 English reviews | 80 (78 components) | `.../text_svd/` (233 MB, **git-ignored**) | Text features. Regenerate with `python -m src.features.text` (about 1 minute) |

```python
import pandas as pd
rf = pd.read_parquet("data/processed/features/review_features")        # reads all parts as one table
train, test = rf[rf.split == "train"], rf[rf.split == "test"]            # shared split, by game
gf = pd.read_parquet("data/processed/features/game_features.parquet")
go = pd.read_parquet("data/processed/features/game_outcomes.parquet")    # targets, join on appid
txt = pd.read_parquet("data/processed/features/text_svd")                # after running the text step; join on recommendationid
```

## 2. Rules that keep the results honest

1. **Fit on training games only.** The vocabularies, the tag PCA, the TF-IDF vocabulary and idf weights, and the
   text SVD are all learned from the training games in `data/processed/splits/game_split.csv`, then applied to
   every game. `validate.py` proves it for the tag PCA: its mean equals the training-game mean and differs from
   the all-games mean.
2. **Outcomes are kept apart.** Steam's totals, rating and score live only in `game_outcomes` (columns prefixed
   `outcome_`). `game_features` and `review_features` contain none of them, nor SteamSpy or Metacritic
   figures, which are consequences of reception.
3. **No post-review columns in the review table.** Left out on purpose: `votes_up`, `votes_funny`,
   `weighted_vote_score`, `comment_count`, `n_reactions`, `has_dev_response`, `refunded`,
   `author_playtime_forever`, `author_playtime_last_two_weeks`, `author_last_played`. They are only known after
   the review exists, or were measured at collection time.
4. **Feature groups.** The catalog tags each feature `launch_time` (known when a developer decides on the
   launch), `at_review` (known when the review is written), `post_launch` (patch cadence after launch: fine for a
   post-launch monitor, not for a launch-risk scorer), `target`, `outcome` or `key`.
5. **The five capped games are left out of `review_features` and `text_svd`.** Megabonk, Cult of the Lamb,
   Hades II, Balatro and Vampire Survivors only have their newest ~15,000 reviews, so their timing features would
   be wrong (see `docs/data_documentation.md`, section 8). They stay in the game-level tables, whose columns are
   complete.

## 3. Tag block: PCA on tags and categories (replaces the first SVD)

The first version (PR #2 and #49, `src/dimensionality_reduction/`) reduced 455 columns to a fixed 154. That is
kept for history but **superseded** by `src/features/tags.py`, which fixes the issues listed on #4.

| Step | Decision | Numbers (training games, n = 1,489) |
| --- | --- | --- |
| Encode | Multi-hot of user tags (top 20 per game) and store categories | 439 candidate columns |
| Genres | Dropped: 11 of 12 genres are also tags, "Free To Play" repeats `is_free` | |
| Near-constant | Dropped if in at least 95% of games | `Indie`, `Single-player`, `Family Sharing` |
| Duplicates | Category dropped when a tag has the same name | `Co-op`, `Online Co-op`, `PvP` |
| Rare | Dropped if in under 1% of games (fewer than 15) | 199 columns (182 tags, 17 categories) |
| Kept | | **234 columns** (200 tags, 34 categories) |
| Method | PCA, centred (the brief's method; the old version used uncentred Truncated SVD) | |
| k | The elbow of the cumulative-variance curve: the point farthest above the straight line between the first and last point | **k = 68**, 75.3% of training variance |
| Check | The same k on games not used to fit it (5 cv folds): 70.5% of their variance | the gap is small, so no overfitting |

Variance is spread thinly across tags (90% would need 124 components), so PCA gives a modest reduction here,
234 columns to 68. Files: `docs/feature_engineering/tag_columns.csv` (every column and why it was kept or
dropped), `tag_pca_variance.csv`, `tag_pca_loadings.csv`; figure `docs/figures/tag_pca_variance.png`.

The first components, named from their loadings (`+` and `-` are the two ends of each axis):

| Component | Variance | Reads as |
| --- | ---: | --- |
| `pc_01` | 7.6% | Action roguelike, bullet hell, action (+) versus card game, deckbuilding, strategy (-) |
| `pc_02` | 4.4% | Polished store page: Steam Cloud, controller support, achievements (+) versus 3D and free-to-play (-) |
| `pc_03` | 3.6% | 2D, pixel graphics, retro, casual (+) versus 3D, adventure, multiplayer (-) |
| `pc_04` | 3.1% | RPG, fantasy, adventure (+) versus casual, shooter, arcade (-) |
| `pc_05` to `pc_10` | 2.3% to 1.7% each | Mixed combinations (for example co-op multiplayer roguelite versus accessibility options in `pc_08`); no single clean theme |

The first four components are interpretable; later ones are weak and mixed, which is normal for binary tag data.

## 4. Game-level features

All in `game_features` (full list in the catalog). Price tiers use fixed cut-offs (free, under $5, $5-10,
$10-20, $20+), not quantiles, so nothing is fitted.

| Group | Features |
| --- | --- |
| Price | `price_usd`, `price_tier` (0 to 4), `is_free` |
| Launch | `launch_year`, `launch_quarter`, `launched_in_early_access` (1.0 date more than 7 days after the original launch), `launch_before_window` |
| Content and support | `n_dlc`, `n_supported_languages`, `n_achievements` (none listed = 0), `has_achievements`, `platform_mac`, `platform_linux`, `controller_support_level` (0 none, 2 full: no game lists "partial"), `self_published` |
| Tags | `pc_01` to `pc_68` |
| Post-launch | `patches_first_30d`, `patches_first_90d`, `sales_first_90d` (developer announcements after `launch_date`) |
| Keys | `appid`, `split`, `cv_fold` (-1 = test), `reviews_capped` |

Windows support is true for every game and is omitted. DLC and achievement counts are as of the store snapshot,
so they can include post-launch additions.

**The game-level target is defined separately**: three success tiers in `game_targets` (see `docs/game_target.md`), built from
`game_outcomes`. Rules for using these tables in models are in `docs/modelling_protocol.md`.

## 5. Review-level features

`review_features` has one row per review of the 1,856 fully scraped games (1,257,095 train, 316,341 test; 10.6% negative in both).

| Group | Features |
| --- | --- |
| Label | `target_is_negative` (1 = not recommended) |
| Playtime | `playtime_at_review_min_capped` (winsorised at the 99.5% quantile), `playtime_bucket` (0 under 2 h, 1 2-10 h, 2 10-50 h, 3 50 h+; -1 unknown for 12 reviews), `in_refund_window` (under 2 hours, 14.5% of reviews) |
| Timing | `days_since_launch`, `is_prerelease` (1.9%), `launch_week` (12.4%) |
| Patches and sales | `days_since_last_patch`, `n_patches_before`, `has_prior_patch`, `days_since_last_sale`, `n_sales_before` |
| Flags | `steam_purchase`, `received_for_free`, `written_during_early_access`, `primarily_steam_deck` |
| Reviewer | `author_num_games_owned_log1p`, `author_num_reviews_log1p` |
| Text length | `review_n_chars`, `review_n_words` (words are unreliable for Chinese and Japanese) |
| Language | `language`, `is_english` |
| Game | `price_usd`, `price_tier`, `is_free` |
| Keys | `recommendationid`, `appid`, `split`, `cv_fold` |

Join `game_features` on `appid` for more game attributes. Missing values: `days_since_last_patch` is missing for
4.3% of reviews (before the game's first patch) and `days_since_last_sale` for 58.1% (before the first sale); use
the `has_prior_patch` flag and let tree models handle the rest, or impute for linear models. `language` is a
string: one-hot encode the top languages.

A first look (not a model): **21.2% of reviews written within the 2-hour refund window are negative, against
8.8% after it**, which supports the lost-revenue angle in the brief.

## 6. Text block: TF-IDF and Truncated SVD

| Step | Decision | Numbers |
| --- | --- | --- |
| Rows | English (`text_mining_ok`) reviews of fully scraped games | 563,961 |
| Foreign text | Dropped: non-Latin script (Latin letters under 90%) | 4,305 (0.76%) |
| | Dropped: at least 2 function words of Spanish, Portuguese, French, German, Italian, Indonesian, Dutch, Turkish or Polish that are not English words, or 1 in a review of 6 words or fewer | 2,454 (0.44%) |
| Rows kept | | **557,202** (436,715 from training games) |
| TF-IDF | Word 1-2 grams, English stop words removed, sublinear tf, `min_df` 10, `max_df` 0.9, 50,000 terms at most; fitted on a random 120,000 training reviews | 41,891 terms |
| SVD | Truncated SVD, 300 components fitted on 100,000 of those reviews; first k kept | k = 78 at the elbow |
| Variance | | 14.6% of the TF-IDF variance at k = 78 (26.2% at 300) |

The curve is shallow (`docs/figures/text_svd_variance.png`): short reviews share few words, so TF-IDF variance
spreads thinly and the elbow is gentle. That is typical of review text. The top terms of the first 15 components
are in `docs/feature_engineering/text_svd_top_terms.csv`. The leading components mostly separate kinds of generic praise
("good game", "fun", "great", "10/10", "love"); the negative end of `t_01` is rare, game-specific phrasing. They carry
sentiment-like signal for models, but they are **not topics**: topic modelling (M5) should use its own pipeline.

**Limitations of the text block**
- `language` is the reviewer's chosen language, not detected from the text. The two filters above are heuristics; a
  language-ID model was tried and rejected (about 16 minutes, and it mislabelled short English such as "its dope" and
  "Broken game"). A small amount of Latin-script foreign text still remains, probably under 1%. If topic models show
  foreign clusters, add a language-ID step for the long reviews.
- The vocabulary is learned from a 120,000-review training sample, so very rare terms are missing.
- Only English reviews are covered; other languages (more than 60% of all reviews) have no text features.
- `text_svd` is not committed (233 MB). Rebuild it with `python -m src.features.text`.

## 7. Choices to revisit

- **k by elbow rule** (68 and 78). Compare nearby values by model performance with M4 if there is time.
- **Review-level negative rate by game size.** The shared split is balanced overall, but a few huge games dominate
  review counts; report per-game as well as pooled metrics (see `docs/train_test_split.md`).
- **Cross-validation inside train.** The tag and text components were fitted on all training games, including each
  validation fold. This is a small, unsupervised leak; mention it in the report. For a stricter check, refit per fold.
- **Reviewer counts are measured at collection time.** `author_num_games_owned` and `author_num_reviews` include the
  reviewer's activity after the review was written. A weak signal (about 0.58 AUC on its own) and unrelated to the
  outcome, so the leak risk is low, but mention it in the report or drop the two columns for a strict version.
- **Sanity check before modelling** (throwaway baselines, not committed): a gradient-boosted review-level model
  reaches a test AUC of about 0.73 (PR-AUC 0.27 against a base rate of 0.11; within-game AUC 0.73), and a random
  forest on launch-time game features reaches about 0.65 AUC for "high-risk game" (negative ratio of 30% or more).
  No single feature is above 0.66 AUC alone, which is what we expect when nothing leaks.
- **Tag components use top-20 user tags**, which is all the store page shows. Games with few votes have noisier tags.

## 8. Validation

`python -m src.features.validate` runs 21 checks (19 if the git-ignored text matrix has not been built; it says so), all passing at the time of writing:
one game row per game and the same games as `game_split.csv`; split and fold copied correctly; no missing values;
the game-level tiers match an exact-fraction recomputation and every tier appears in train and test;
no outcome columns among the features; the tag PCA fitted on training games only and reproducible from its saved
model; review features cover exactly the clean reviews of the fully scraped games with no capped game, no game in
both train and test, and no post-review column; the label has both classes in train and test; text features have
unique review ids that all exist in `review_features`.

Beyond the script, the time-based review features were recomputed by brute force for a random 2,500 reviews each
(patch and sale counts and recency, days since launch, launch week, refund window) and matched exactly.
