# Review 2 Presentation Guide: Text Mining, Topic Modelling & Combined Insights

**Presenter:** Valikala Tejaswini (CB.SC.U4CSE23752)  
**Project:** IndieGuard: Indie Game Launch Analytics on Steam  
**Assigned Tasks:** Tasks #31–#33 (Text Mining & Topic Evaluation) & Task #44 (Combined Business Insights)  
**Presentation Context:** Review 2 — Data Analysis, Text Mining, Evaluation & Strategic Synthesis  

---

## 1. Slide Outline & Visual Specifications

### Slide 1: Review 2 Focus — Text Mining & Combined Strategic Insights
* **Title:** IndieGuard Review 2: Text Mining, Topic Modelling & Combined Insights
* **Presenter Information:** Valikala Tejaswini (CB.SC.U4CSE23752) | Group 1
* **Core Agenda:**
  1. Multilingual Assessment & Text Preprocessing Pipeline
  2. Exploratory N-Gram Text Mining & Distinctiveness
  3. Data-Driven Topic Modelling (NMF $k=8$)
  4. Mathematical Topic Evaluation (Coherence, Diversity, Stability, NMF vs. LDA)
  5. Distinction: Supervised Sentiment Triage vs. Unsupervised Thematic Discovery
  6. Task #44: Ranked Combined Business Insights & Studio Action Playbook
* **Suggested Visual:** Project workflow diagram illustrating raw reviews flowing into language filtering, NMF topic decomposition, and integration with Review 1 predictive models.

---

### Slide 2: Multilingual Assessment & Preprocessing Pipeline
* **Slide Title:** Dataset Preprocessing & Multilingual Evaluation
* **Key Content:**
  * **Multilingual Assessment:** 1,648,387 cleaned reviews across 31 languages. Over 60.5% are non-English (Simplified Chinese: 32.0%, Russian: 7.5%).
  * **English Subset Justification:** Non-segmented scripts (CJK) lack uniform stopword/negation binding without heavy neural models; Steam language tag has ~1% error. Filtering for verified Latin script ($\ge 90\%$) and removing Latin foreign markers isolates a pristine corpus of **557,202 English reviews** (436,715 train / 120,487 test).
  * **Negation Binding:** Preserved critical failure expressions by binding negations before stopword removal (`not_fun`, `too_hard`, `clunky_controls`, `not_worth`).
* **Suggested Table:**
  | Pipeline Stage | Rows Before | Rows After | Excluded | Justification |
  | :--- | :---: | :---: | :---: | :--- |
  | Raw Cleaned Corpus | 1,648,387 | 1,641,628 | 6,759 | Excluded 5 capped mega-games |
  | Text Mining Quality Filter | 1,641,628 | 563,961 | 1,077,667 | Kept high-content English reviews |
  | Non-Latin Script Filter | 563,961 | 559,656 | 4,305 | Removed mislabeled CJK/Cyrillic |
  | Foreign Function Word Filter | 559,656 | **557,202** | 2,454 | Stripped non-English Latin text |

---

### Slide 3: Exploratory Text Mining — The Language of Dissatisfaction
* **Slide Title:** Exploratory N-Grams & Vocabulary Distinctiveness
* **Key Findings:**
  * **Negative vs. Positive Reviews:** Dirichlet-smoothed Log Odds Ratios across 10,643 terms reveal that negative reviews are dominated by price regret (`not_worth` +1.86, `waste` +1.57), core boredom (`boring` +1.66), and mechanical friction (`unbalanced` +1.88, `crashes` +1.62).
  * **Early (<2h) vs. Late (>=2h) Playtime:**
    * Inside the refund window (<2h): `refund` (+2.15 Log Odds), `clunky_controls` (+1.42 Log Odds), `crashes` (+1.38 Log Odds).
    * Outside the refund window (>=2h): `repetitive` (+1.28 Log Odds), `rng` / `unbalanced` (+1.15 Log Odds).
* **Suggested Visual:** Side-by-side horizontal bar chart from `figures/text_mining/ngram_comparison.png` displaying top Log Odds ratios.

---

### Slide 4: Data-Driven Topic Modelling (NMF $k=8$)
* **Slide Title:** Topic Modelling: Identifying Latent Failure Modes
* **Method:** Non-Negative Matrix Factorization (NMF) on TF-IDF vectors ($k=8$ components) fitted on negative reviews.
* **The 8 Discovered Topics:**
  1. **T0: Boss Design, Damage & Upgrade Difficulty** (`run`, `boss`, `damage`, `upgrades`) — 30.86% prevalence
  2. **T1: Repetitive Gameplay, Grind & Pacing** (`boring`, `repetitive`, `slow`, `grind`) — 4.64% prevalence
  3. **T2: Flawed Design, Clunky Controls & RNG** (`bad`, `controls`, `design`, `bad_rng`) — 4.21% prevalence
  4. **T3: Early Access State & Content Scarcity** (`early_access`, `content`, `abandoned`) — 4.56% prevalence
  5. **T4: Replayability & Co-op Friend Engagement** (`fun`, `runs`, `friends`, `content`) — 9.34% prevalence
  6. **T5: Multiplayer Issues, Crashes & Dev Bugs** (`multiplayer`, `crashes`, `fix`, `bugs`) — 25.40% prevalence
  7. **T6: Art & Presentation vs Gameplay Loop** (`gameplay`, `art`, `loop`, `style`) — 19.34% prevalence
  8. **T7: Unbalanced Mechanics & Frustration** (`not_fun`, `unbalanced`, `buggy`, `waste`) — 1.65% prevalence
* **Suggested Visual:** `figures/text_mining/topic_prevalence_comparison.png` and `figures/text_mining/topics_by_playtime.png`.

---

### Slide 5: Rigorous Topic Model Evaluation (Task #32)
* **Slide Title:** Model Evaluation: Coherence, Diversity & Baseline Benchmark
* **Key Content:**
  * **Intrinsic Metric Performance:**
    * **UMass Coherence:** -2.380 (tight semantic co-occurrence across top terms).
    * **Topic Diversity:** **90.00%** (72 unique terms across 80 top words; zero generic redundancy).
    * **Seed Stability:** **Mean Jaccard = 1.00** across seeds 42, 101, 2024 (fully deterministic).
  * **NMF vs. LDA Benchmark:** NMF is **45x faster** (2.3s vs. 105s), achieves higher topic diversity (90.0% vs. 86.2%), and avoids LDA's topic blending.
  * **Supervised vs. Unsupervised Distinction:**
    * **Supervised Sentiment Model (Review 1):** Full TF-IDF Logistic Regression achieves **PR-AUC 0.785, ROC-AUC 0.955** (post-hoc triage of written reviews).
    * **Unsupervised NMF:** Discovers actionable root causes (controls, netcode, progression), bridging post-launch triage with pre-launch design.
* **Suggested Visual:** 3-panel diagnostic plot from `figures/text_mining/topic_evaluation_diagnostics.png`.

---

### Slide 6: Combined Business Insights — Ranked Developer Playbook (Task #44)
* **Slide Title:** Task #44: Synthesized Business Findings & Developer Playbook
* **The 5 Ranked Strategic Insights:**
  1. **The 2-Hour Refund Trap:** 21.2% of reviews <2h are negative (vs. 8.8% later), representing 29.0% of all negatives. Controls and crashes dominate. *Action: Treat first 60 minutes as standalone product.*
  2. **The Price-Expectation Disconnect:** Negativity under 2h rises from 11.4% (<$5) to 31.8% ($20+). *Action: Price roguelikes below $15 at launch; discount Early Access entry.*
  3. **Temporal Complaint Bifurcation:** Early reviews complain of controls/crashes; late reviews complain of repetition and RNG balance. *Action: Two-stage patch strategy (hotfix onboarding Day 1–7; patch balance Week 2+).*
  4. **Low-Cost Store Polish Signals:** Controller support cuts struggling from 15.5% to 11.5%; achievements cut struggling from 19.6% to 12.5%. *Action: Non-negotiable launch requirements.*
  5. **Scope Discipline:** 2D Platformers (3.2%) and Deckbuilders (5–9%) have low risk; Online Co-op (26.6%) and 3D PvE (33%) suffer high failure rates. *Action: Small teams should stick to polished 2D single-player scopes.*
* **Suggested Visual:** `figures/findings/combined_insights_framework.png` and `figures/findings/patch_sentiment_impact.png`.

---

## 2. Professional Speaking Script

### Slide 1: Introduction (approx. 45 seconds)
> "Good morning, respected professors and evaluation committee. In Review 1, our team established that while store metadata provides modest signals about game success, the richest diagnostic signal lies in the actual text of player reviews. Review 1 achieved a strong PR-AUC of 0.785 with full-text models, but relied on basic keyword dictionaries to guess what players were complaining about.
> 
> In Review 2, I was assigned Tasks #31 through #33—covering automated text mining, topic modelling, and rigorous evaluation—and Task #44, which synthesizes these text findings with our predictive models, refund window dynamics, and post-launch patch events into an actionable developer playbook. Today, I will walk you through our methodology, mathematical evaluations, and data-driven business insights."

### Slide 2: Multilingual Preprocessing Pipeline (approx. 60 seconds)
> "To begin with Task #31, our cleaned dataset contains over 1.64 million reviews across 31 distinct languages. Over 60% of these reviews are non-English—predominantly Simplified Chinese and Russian. However, text mining on unsegmented East Asian scripts requires fundamentally different morphological tokenizers, and Steam's language tags carry a 1% error rate where foreign reviews are mislabeled as English.
> 
> To guarantee linguistic consistency and directly align with Review 1’s benchmark models, we established a rigorous multi-stage filtering pipeline. We filtered for Latin-script character integrity above 90% and eliminated Latin-script foreign function words. This isolated exactly 557,202 pristine English reviews. 
> 
> Crucially, standard NLP pipelines strip punctuation and stopwords, destroying phrases like 'not fun' or 'too hard'. We engineered a custom negation-binding preprocessor that locks phrases like `not_fun`, `not_worth`, and `clunky_controls` into single analytical tokens before pruning conversational fillers."

### Slide 3: Exploratory Text Mining (approx. 60 seconds)
> "Next, we conducted exploratory text mining using Dirichlet-smoothed Log Odds Ratios across more than 10,000 n-grams. Comparing negative reviews to positive reviews revealed that player dissatisfaction is overwhelmingly anchored in price regret—with terms like `not_worth` and `waste` exhibiting log odds ratios of +1.86 and +1.57—followed by pacing issues like `boring` and mechanical friction like `unbalanced` and `clunky`.
> 
> Even more revealing is the comparison by playtime: reviews submitted inside the 2-hour refund window are dominated by `refund`, `clunky_controls`, and `crashes`. But once players surpass the 2-hour mark, complaints shift completely toward `repetitive`, `grind`, and `rng`. This provided the first statistical evidence that player expectations bifurcate across game lifecycles."

### Slide 4: Topic Modelling (NMF $k=8$) (approx. 75 seconds)
> "To move beyond individual words, we implemented Non-Negative Matrix Factorization (NMF) on TF-IDF vectors using 8 components, trained on negative reviews. NMF naturally decomposed player complaints into 8 distinct, data-driven themes.
> 
> As you can see on the slide, the model successfully isolated Boss Design and Difficulty (Topic 0), Repetitive Gameplay (Topic 1), Clunky Controls and RNG (Topic 2), Early Access Scarcity (Topic 3), and Multiplayer Crashes and Bugs (Topic 5).
> 
> When we examine topic prevalence, Topic 5—multiplayer bugs and crashes—represents 29.2% of negative reviews inside the refund window, while controls represent nearly 5%. However, in post-2-hour reviews, Topic 0—boss design and progression scaling—expands dramatically from 24% to 34%. This directly quantifies our core finding: players refund early due to technical and control barriers, but leave negative reviews late due to balance and replayability fatigue."

### Slide 5: Topic Model Evaluation (Task #32) (approx. 75 seconds)
> "In Task #32, we did not simply generate word clouds; we conducted a rigorous mathematical evaluation of topic quality. 
> 
> First, our NMF model achieved a Mean UMass Topic Coherence of -2.380 and an outstanding Topic Diversity of 90.0%, meaning 72 of the 80 top terms are completely unique across topics. Second, testing stability across independent random seeds yielded a pairwise Jaccard similarity of 1.00, demonstrating that NMF with NNDSVD initialization is fully deterministic and robust.
> 
> Third, benchmarking against Latent Dirichlet Allocation (LDA) showed that NMF converged 45 times faster—2.3 seconds versus 105 seconds—while avoiding LDA's vocabulary bleeding.
> 
> Finally, I must clarify an essential distinction for our project: Review 1's supervised Logistic Regression achieved a PR-AUC of 0.785 on review text, but that model performs post-hoc triage—detecting sentiment on an already-written review. It cannot forecast pre-launch game risk because reviews do not exist before launch. Our unsupervised NMF topic model uncovers the underlying failure modes that explain why reviews turn negative in the first place, bridging post-launch monitoring with pre-launch design."

### Slide 6: Combined Business Insights (Task #44) (approx. 90 seconds)
> "Finally, in Task #44, we integrated these text findings with Review 1’s predictive models, refund window dynamics, and 28,875 post-launch patch events into a ranked 5-point studio playbook.
> 
> Priority 1 is the 2-Hour Refund Trap: 21.2% of reviews under 2 hours are negative versus 8.8% later. Inside the window, controls and crashes dominate. A studio must treat the first 60 minutes as a standalone product.
> 
> Priority 2 is the Price-Expectation Disconnect: inside the refund window, negativity jumps from 11.4% for games under $5 to 31.8% for games at $20 or more, while outside the window price has virtually no effect. Higher launch prices severely amplify initial scrutiny. We recommend pricing indie roguelikes below $15.
> 
> Priority 3 is Complaint Bifurcation: developers must adopt a Two-Stage Patch Strategy. Day 1 to 7 hotfixes must exclusively target controls, camera, and crash bugs. Only in Week 2 should engineering resources shift toward endgame balance and item variety.
> 
> Priority 4 highlights Store Polish Signals: listing Steam achievements cuts struggling risk from 19.6% to 12.5%, and controller support cuts risk from 15.5% to 11.5%. These are low-cost, high-return launch signals.
> 
> Priority 5 emphasizes Scope Discipline: tight 2D single-player scopes like Deckbuilders and Twin-stick shooters face only 2% to 9% failure rates, whereas ambitious Online Co-Op scopes suffer a 26.6% failure rate.
> 
> All code, tests, and tables are fully committed, validated with 18 automated checks, and reproducible from our documented commands. Thank you, and I look forward to your questions."

---

## 3. Likely Faculty Viva Questions & Answers

### Q1: Why did you choose NMF over LDA for topic modelling in Task #31?
* **Answer:**  
  "We benchmarked both algorithms on our actual dataset. NMF with TF-IDF vectorization operates exceptionally well on short-to-medium customer reviews because TF-IDF naturally downweights ubiquitous general gaming terms (like 'game' or 'play') while magnifying specific failure words (like 'clunky', 'crash', or 'unbalanced'). In our evaluation (Task #32), NMF achieved a higher Topic Diversity of **90.0%** compared to **86.3%** for LDA, converged **45 times faster** (2.3 seconds versus 105 seconds), and exhibited zero topic blending, whereas LDA repeatedly mixed generic roguelike vocabulary across multiple topics."

### Q2: Why did you restrict your in-depth text analysis to English reviews when 60% of the dataset is non-English?
* **Answer:**  
  "That was a deliberate methodological decision based on linguistic rigor. The dataset contains 31 languages, with Simplified Chinese accounting for 32% and Russian 7.5%. However, standard lexical preprocessing, compound negation binding (like `not_fun` or `too_hard`), and dictionary stopword removal fail on unsegmented CJK scripts without specialized neural morphological analyzers. Furthermore, Steam's reviewer language tag has a ~1% misclassification rate. By rigorously filtering for Latin script purity and stripping foreign function words, we isolated 557,202 verified English reviews. This maintains complete semantic interpretability, exact negation preservation, and direct comparability with Review 1's benchmark models."

### Q3: What is the mathematical meaning of UMass coherence and why is it negative?
* **Answer:**  
  "UMass coherence measures how frequently the top representative words of a topic co-occur within documents across the corpus:
  $$C_{\text{UMass}}(w_i, w_j) = \log \frac{D(w_i, w_j) + 1}{D(w_j)}$$
  Because the co-document frequency $D(w_i, w_j)$ is always less than or equal to the single-word document frequency $D(w_j)$, the ratio inside the logarithm is always less than or equal to 1. The logarithm of a fraction is negative, so UMass coherence values are naturally negative. A higher score—closer to zero—indicates stronger empirical co-occurrence and higher semantic consistency. Our NMF score of -2.380 reflects strong, coherent vocabulary clusters."

### Q4: In Review 1, your team achieved a PR-AUC of 0.785 on full review text. Why can't a studio use that model to predict which games will fail before launch?
* **Answer:**  
  "That is a fundamental distinction between post-hoc detection and pre-launch forecasting. Review 1's text model operates on an already-written review. It asks: *'Given what this player wrote, is this a negative review?'* It is a post-launch automated triage tool for filtering incoming customer support and feedback. Before a game launches, zero reviews exist, so that model has no inputs. Pre-launch risk prediction must rely on launch-time store features, which our team showed achieves a macro-F1 of 0.42. The role of Review 2's topic modelling is to bridge this gap: it extracts the root-cause failure modes from post-launch text so developers know what to fix before they release their next game."

### Q5: How do you know that early negative reviews are caused by controls rather than just random player preference?
* **Answer:**  
  "We evaluated this using both statistical Log Odds Ratios and topic prevalence across playtimes. Across 228,899 reviews written inside the 2-hour refund window, Topic 5 (Crashes/Bugs) and Topic 2 (Controls/Flawed Design) appear at their highest rates, with specific n-grams like `clunky_controls` showing a +1.42 Log Odds lift. Furthermore, in 94.3% of games with at least 50 reviews on each side, the under-2-hour window is significantly more negative, with a median gap of +13.1 percentage points. While observational data cannot prove strict mechanical causality, this consistent pattern across 1,861 games proves that onboarding friction is an empirical platform-wide vulnerability."

### Q6: Can a studio fix its review score simply by releasing more patches?
* **Answer:**  
  "No, our patch impact analysis (`patch_impact_analysis.csv`) proves that releasing patches does not automatically guarantee sentiment recovery. In fact, in the first 0 to 3 days after a patch, review negativity actually rises slightly to 11.8% due to player friction with balance changes or new bugs. Sentiment only stabilizes to 9.2% around days 15 to 30. More importantly, patches only work if they target the specific failure modes identified by our text mining: hotfixing controls and crashes in Week 1, and rebalancing RNG and boss curves in Week 2+."
