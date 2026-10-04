# Problem statement

**Project:** IndieGuard: Indie Game Launch Analytics. Predicting negative-review risk and identifying winning feature
combinations on Steam. Business Analytics capstone (Units 1 to 3), team of five.

## Context

Steam publishes thousands of indie games a year and a small studio has little margin for a bad launch. Reviews decide a
game's visibility and reputation, and Steam lets a player refund a game within 14 days if they have played under 2 hours. A
negative review written in those first two hours therefore costs a studio twice: a worse rating and, often, the sale itself.
Studios today decide what to build, what to fix before launch and when to patch mostly by instinct.

## The problem

A small studio or publisher cannot tell, from public information, (1) which combinations of game features give the best chance
of a well-received launch, (2) what is driving negative feedback, and above all what to fix before the refund window closes, and
(3) when a patch is worth the effort.

## Business objective

Give a small studio or publisher evidence to decide **which feature combinations to back, what to fix before the refund window
closes, and when to patch**, using data collected from Steam's own public sources.

## Questions we answer

| # | Question | Method | Review |
| --- | --- | --- | --- |
| 1 | Will a given review be negative, and how does that depend on how long the player has played? | Classification at review level (Logistic Regression, Random Forest, XGBoost, LightGBM), refund-window analysis | 1 |
| 2 | Which success tier will a game land in (Struggling, Solid, Strong)? | Classification at game level, split by game to avoid leakage | 1 |
| 3 | Which tag, price and Early Access combinations go with strong reception? | Association rule mining (support, confidence, lift, stability across game subsets) | 2 |
| 4 | What do players complain about, and how does it differ between negative and positive reviews? | Sentiment and topic modelling (LDA or BERTopic), validated by coherence and a manual check of about 200 reviews | 2 |
| 5 | How do ratings change after a patch or a discount? | Before and after analysis around Steam news dates (time series only as a fallback) | 2 |

## Scope

- **Niche:** indie roguelike and roguelite games on Steam released 2022 to 2025 with at least 10 reviews: **1,861 games**.
- **Data:** **1,652,999 reviews** in 31 languages, plus store details, user tags, owner estimates and news for every game, all collected by us from four public sources (Steam reviews, appdetails, SteamSpy, Steam news). No pre-built dataset is used. See `docs/data_documentation.md`.
- **Out of scope:** causal claims, revenue figures (reviews are not purchases), other genres, and games released before 2022 or after 2025.

## How success is measured

| Level | Measure | Target |
| --- | --- | --- |
| Models | PR-AUC for reviews, macro-F1 for game tiers, on a held-out test set of whole games, scored once | Clearly above a dummy baseline, with cross-validation spread reported |
| Findings | Quantified, evidence-linked recommendations a studio can act on | Each recommendation points to a table or chart and states its limits |
| Process | Reproducible pipeline, documented sources and cleaning, one shared split | `python -m src.features` rebuilds the features from the committed data and passes its checks |
| Team | Every task has one owner, a due date and evidence on the board | Contribution visible in the board history |

## What we expect (hypotheses), and what the first results say

| Hypothesis | Initial evidence (`docs/business_findings.md`) |
| --- | --- |
| H1. Reviews written inside the refund window are more negative than later ones | **Supported:** 21.2% negative against 8.8%, and in 94% of games with enough reviews |
| H2. The effect is larger for pricier games | **Supported:** 31.8% negative inside the window for games at $20 or more, against 11.4% under $5 |
| H3. Early complaints differ from later ones | **Supported:** controls, camera and interface in 13.1% of early negative reviews against 7.2% later; repetition and balance dominate later |
| H4. Store metadata alone predicts a game's tier well | **Not supported:** macro-F1 about 0.42 against 0.22 for guessing; the signal is weak |
| H5. The review text predicts whether the review is negative far better than metadata | **Supported:** PR-AUC 0.785 against 0.360 on the same English reviews |
| H6. Some tag combinations go with strong reception | **Open:** single tags show differences, but they are uncorrected hypotheses; association rules (Review 2) test it properly |

## Risks and limits

- Tags, prices and review totals are snapshots from 1 October 2026, not from launch; older games have had longer to collect negative reviews.
- A few very large games dominate review counts, so results are reported per game as well as pooled, and five very large games have only their newest reviews.
- About 60% of reviews are not in English, so text methods cover a minority unless a multilingual approach is added.
- The relationships are associations; they do not prove that a change in a game causes a change in its reviews.

## Plan

Review 1 (units 1 and 2): data, cleaning, EDA, features, models, evaluation and initial findings. Review 2 (unit 3): association
rules, text mining, the interactive dashboard (market explorer, tag-combination and complaint explorer, launch-risk scorer),
patch-impact analysis and the final report. Task owners and dates are on the project board.
