# Panel Q&A briefing: what each member must be able to explain

The panel verifies individual contribution through the project board and by asking each member about their assigned work (rubric: 1 of 3 marks). Read your own section; read the three "hard questions" at the end, because anyone can be asked them.

## Thotakura Srihitha (EDA, dashboard wireframe and downloads, co-author of the tuning work)

**Your work:** 29 EDA figures and association tests (PR #1, `notebooks/02_EDA_Visualizations.ipynb`, cards #20 and #21); dashboard wireframe, the "showing X of Y games" caption, CSV downloads and the empty-tag fix (PR #101, cards #37, #40, #41).

**Be ready to say:**
- The refund window finding: 21.2% of reviews written under 2 hours of play are negative against 8.8% later, and these are 29% of all negative reviews. Inside that window, negativity rises from 11% (under $5) to 32% ($20 or more).
- What each dashboard control updates: release year, price band, minimum reviews and "must have tags" filter the games shown, the KPIs, the tier mix and the tag chart together. Pixel Graphics gives 550 games.
- Tuning (you are co-author; the runs were done on Shantharam's machine): a grid search tries every combination of settings. Each setting is scored on the same 5 cross-validation folds inside the training games, the default setting is grid point 0 so default and tuned are compared the same way, and the test games are scored once at the end. Code: `src/models/grid_search.py`; tables: `reports/results/tuning_*.csv`.

## Gayas Sheik (features, patch impact, launch-risk scorer)

**Your work:** features and tag PCA (68 components from 439 tag columns), patch impact with matched controls (PR #80), launch-risk scorer (PR #71), reproducibility (cards #4, #18, #19, #38, #39, #42).

**Be ready to say:**
- Everything learned from data (PCA, imputers) is fitted on training games only, so test games do not leak into features.
- Patch impact: 1,685 patch comparisons on 617 games. Game level: positive share falls 2.06 points (95% interval -2.75 to -1.37) in the 30 days after a patch against the 30 days before.
- Why that is not a patch effect: matched no-patch controls (164 games, 282 windows) show +0.27 points (interval -0.82 to +1.36). The control interval includes zero, so there is no clear patch effect once launch timing and regression to the mean are handled.
- The scorer returns a tier from launch-time features only; scores are class-scaled, not calibrated probabilities.

## Shantharam P (data, cleaning, split, models, tuning, dashboard, report)

**Your work:** own scrapers and cleaning (1,861 games, 1,648,387 reviews), split by game with 5 folds, models and tuning (cards #5 to #17, #23 to #27, #43, #45, #83 to #94), dashboard deployment, final report.

**Be ready to say:** the 32 cleaning rules, why the split is by game (a game's reviews never sit in both train and test), how thresholds and class scales come only from out-of-fold predictions, and the three hard questions below.

## Pradeep Sasidharan (XGBoost with SHAP, association rules)

**Your work:** XGBoost review and game models with SHAP (PRs #55, #70); association rule mining (cards #34, #35, #36).

**Be ready to say:**
- Support, confidence and lift in one sentence each. Example rule: Pixel Graphics + Retro + Shooter, 25 games, 80% reach the Strong tier (lift 2.26), and 3 of 4 test games did.
- Why FP-Growth and Apriori give the same 95 rules (same thresholds, same answer; FP-Growth is faster), and why we correct for multiple testing (4,715 candidate rules; Benjamini-Hochberg at q below 0.10).
- No rule predicts the Struggling tier: risk is not a matter of tags alone. Supports are small (30 to 60 games), so rules are hypotheses.
- SHAP at review level: review length, Chinese-language reviews and playtime; at game level: a few tag components and number of achievements. Associations, not causes.

## Valikala Tejaswini (text mining, topic modelling, combined insights)

**Your work:** text mining and topic modelling (PRs #66, #81; cards #31, #32, #44); the blind topic check (PR #99, card #33).

**Be ready to say:**
- Text detector: logistic regression on TF-IDF reaches PR-AUC 0.838 on 120,487 English test reviews (guessing: 0.115); precision 0.75, recall 0.79. It detects a negative review from its words; it does not forecast one.
- Topics: NMF with 8 topics. LDA scores better on coherence (-1.732 against -2.380), but NMF gives more distinct topics (diversity 90.0% against 86.25%) and is easier to read; we chose NMF for interpretability.
- The blind check was labelled by an AI assistant with the model topic hidden: 35.5% agreement. Only Early Access, repetition and the crash/multiplayer topics are reliable; boss and balance topics are exploratory. An earlier 96% figure was withdrawn because those labels were not independent.

## The three hard questions (everyone)

**1. Tuning raised cross-validation but lowered test for Random Forest and LightGBM. Why?**
Cross-validation after a search is optimistic: the search picks the best of 55 to 97 settings on the same validation folds, so some of the gain is luck. The cross-validation gains (Random Forest 0.354 to 0.434, LightGBM 0.417 to 0.445) are not much larger than the fold-to-fold spread (about 0.03), and the test set is only 299 games (51 Struggling), so test intervals are wide and overlap. XGBoost (0.437) and Logistic Regression (0.423) held up. We reported the test numbers as they came out and did not re-tune after seeing them, because that would make the test set no longer independent.

**2. Game-level macro-F1 is only about 0.42. Is the model useful?**
It is about twice the guessing baseline (0.22), but it is a modest result, and that is the finding: launch-time store data say little about how a game will be received, and only 51 test games are Struggling. We use the model as a risk indicator in the dashboard (class-scaled scores, not calibrated probabilities), not as a forecast. The reliable results are the review-level text detector and the refund-window and tag-rule findings.

**3. Patches did not help. Did you measure anything?**
Yes. Raw before-and-after shows a fall of 2.06 points, but matched controls with no patch fall nothing like that (+0.27, interval -0.82 to +1.36). The interval means we can rule out a gain larger than about 1.4 points (and a loss worse than about 0.8), so patching for its own sake does not lift sentiment; patch for specific problems (crashes, controls, balance).
