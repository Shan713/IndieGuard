# Text Mining and Topic Modelling Evaluation (Tasks #31–#33)

**Author:** Valikala Tejaswini (CB.SC.U4CSE23752)  
**Project:** IndieGuard: Indie Game Launch Analytics on Steam  
**Review Stage:** Review 2  
**Associated Scripts:** `src/analysis/text_mining.py`, `src/analysis/text_evaluation.py`  
**Associated Notebook:** `notebooks/03_Text_Mining_and_Topic_Modelling.ipynb`  
**Generated Outputs:** `reports/results/topic_terms.csv`, `reports/results/topic_prevalence.csv`, `reports/results/ngram_frequencies.csv`, `reports/results/text_evaluation_metrics.csv`, `reports/results/topic_stability_evaluation.csv`, `reports/results/representative_reviews.csv`

---

## 1. Executive Summary

In Review 1, the IndieGuard team demonstrated that user review text contains the strongest diagnostic signal in the entire repository: predicting a review's negative sentiment using its full TF-IDF text achieves a **Test PR-AUC of 0.785 (ROC-AUC 0.955)**, nearly triple the predictive capacity of store and reviewer metadata alone (PR-AUC 0.283). However, Review 1 relied on preliminary, manually assembled keyword dictionaries to gauge complaint topics.

In Review 2, **Tasks #31–#33** replace heuristic keyword matching with a scalable, fully automated, and data-driven text mining and topic modelling pipeline:
1. **Text Preprocessing & Multilingual Evaluation:** Analyzed 1,648,387 cleaned Steam reviews across 31 languages. Evaluated multilingual viability and established a rigorously filtered, high-purity English corpus of **557,202 reviews** (436,715 train / 120,487 test) with custom negation binding (`not_fun`, `too_hard`, `clunky_controls`).
2. **Exploratory N-Gram Text Mining:** Extracted unigrams and bigrams using Dirichlet-smoothed **Log Odds Ratios**, pinpointing vocabulary that disproportionately drives negative reviews versus positive reviews, and early reviews (<2h) versus late reviews (>=2h).
3. **Data-Driven Topic Modelling (NMF $k=8$):** Implemented Non-Negative Matrix Factorization on TF-IDF sparse matrices, discovering 8 distinct, non-overlapping semantic complaint themes.
4. **Rigorous Topic Model Evaluation (Task #32):** Evaluated topic quality using **UMass Coherence (-2.380 vs LDA -1.732)**, **Topic Diversity (90.0% vs LDA 86.25%)**, and **Subsample Stability across 80% resamples (Mean Jaccard = 0.745; spread across topics 0.19)**. Benchmarked NMF against Latent Dirichlet Allocation (LDA) and justified NMF on semantic interpretability, vocabulary diversity, and computational efficiency.
5. **Supervised vs. Unsupervised Clarification:** Formally demarcated unsupervised thematic discovery (which uncovers actionable developer failure modes) from supervised sentiment classification (which performs automated post-hoc triage on already-written reviews but cannot forecast pre-launch risk).

---

## 2. Text Preprocessing & Multilingual Assessment

### 2.1 Multilingual Data Evaluation
The raw IndieGuard dataset contains reviews written in **31 languages**:
* **English:** 651,012 (39.5%)
* **Simplified Chinese (`schinese`):** 527,103 (32.0%)
* **Russian (`russian`):** 124,169 (7.5%)
* **Korean (`koreana`):** 55,083 (3.3%)
* **Brazilian Portuguese (`brazilian`):** 47,672 (2.9%)
* **Remaining 26 languages:** 243,348 (14.8%)

**Methodological Decision:** Approximately 60.5% of reviews are non-English. While multilingual representation is desirable in customer analytics, dictionary-based stopword removal, lemmatization, and compound negation binding operate inconsistently across unsegmented East Asian scripts (which require specialized CJK morphological analyzers) and Cyrillic text. Furthermore, Steam's reviewer-selected language tag exhibits a ~1% error rate where foreign reviews are mislabeled as English.

To ensure pristine semantic coherence, exact negation preservation, and direct comparability with Review 1's full-text supervised benchmarks (which operate on 436,715 training and 120,487 testing English reviews), we focus in-depth topic modelling on the verified English corpus while documenting exact exclusion metrics.

### 2.2 Data Pipeline & Exclusion Flow
| Pipeline Step | Filter Rule | Reviews In | Reviews Out | Excluded | % Excluded | Justification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Cleaned Reviews** | Initial pool from `reviews_clean` | 1,648,387 | 1,648,387 | — | — | Baseline after dropping empty reviews |
| **Capped Games** | Exclude top 5 outlier mega-games | 1,648,387 | 1,641,628 | 6,759 | 0.41% | Prevents dominance of Vampire Survivors, Balatro, etc. |
| **Quality Filter** | Keep `text_mining_ok == True` & English | 1,641,628 | 563,961 | 1,077,667 | 65.65% | Drops non-English, <3 letters, ASCII art, copypasta |
| **Non-Latin Script** | Latin char ratio $< 0.90$ | 563,961 | 559,656 | 4,305 | 0.76% | Removes mislabeled CJK/Cyrillic reviews |
| **Foreign Markers** | $\ge 2$ Latin foreign function words | 559,656 | **557,202** | 2,454 | 0.44% | Removes Spanish, Portuguese, German, French text |

**Final Analyzed English Corpus:** **557,202 reviews** (436,715 train / 120,487 test).  
* **Overall Negative Rate:** 10.22% (56,962 negative reviews; 500,240 positive reviews).  
* **Refund Window Representation:** 16.0% of reviews written under 2 hours of playtime.

### 2.3 Negation Binding & Tokenization
Standard text preprocessing strips punctuation and stop words, destroying critical negation semantics (e.g. converting "not fun" to "fun", or "too hard" to "hard"). Our custom tokenizer in `src/analysis/text_mining.py` applies targeted regular expression bindings before stopword filtering:
* Contracted forms normalized: `can't` $\to$ `can_not`, `won't` $\to$ `will_not`.
* Immediate negations bound: `not <term>` $\to$ `not_<term>`, `too <term>` $\to$ `too_<term>`, `no <term>` $\to$ `no_<term>`, `lack of <term>` $\to$ `lack_of_<term>`.
* Stop words pruned: 78 conversational filler tokens (`just`, `really`, `even`, `one`, `get`, `feel`, `like`, `make`) removed, while strictly preserving bound negative phrases (`not_fun`, `not_worth`, `too_hard`, `clunky_controls`).

---

## 3. Exploratory Text Mining & N-Gram Distinctiveness

### 3.1 Negative vs. Positive Vocabulary Distinctiveness
Using Laplace-smoothed Log Odds Ratios across 10,643 filtered unigrams and bigrams (`reports/results/ngram_frequencies.csv`), the terms most disproportionately associated with negative reviews include:

| Term | Negative Count | Positive Count | Neg Freq / 10k | Pos Freq / 10k | Log Odds Ratio | Semantic Domain |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `not_fun` | 789 | 35 | 5.75 | 0.58 | **+2.28** | Core Gameplay Failure |
| `not_worth` | 794 | 41 | 5.79 | 0.68 | **+2.13** | Price & Value Disconnect |
| `boring` | 3,184 | 245 | 23.22 | 4.05 | **+1.75** | Repetitive Pacing / Lack of Engagement |
| `clunky` | 570 | 47 | 4.16 | 0.78 | **+1.67** | Controls & Input Handling |
| `waste` | 916 | 81 | 6.68 | 1.34 | **+1.61** | Regret of Time/Money Spent |
| `unbalanced` | 444 | 47 | 3.24 | 0.78 | **+1.42** | Difficulty & Gameplay Tuning |
| `crashes` | 616 | 86 | 4.49 | 1.42 | **+1.15** | Technical Stability & Bugs |
| `bad` | 4,472 | 848 | 32.61 | 14.02 | **+0.85** | General Product Dissatisfaction |

#### Methodological Specification & Counting Unit
* **Unit of Counts:** The counts above represent **token occurrences** extracted via `CountVectorizer` on a balanced random sample of 30,000 negative and 30,000 positive reviews drawn from the verified English corpus ($N = 557,202$ uncapped, `text_mining_ok == True`, Latin character ratio $\ge 0.90$, foreign marker filter, tokenized with custom negation binding). Equal document sampling ensures log odds ratios reflect relative distinctiveness without distortion from the 9:1 platform positivity imbalance.
* **Corpus-Wide Document Prevalence:** In contrast to sample token counts, counting the number of *distinct reviews* containing each term across the entire 557,202 English reviews (`target_is_negative == 1`: 56,962 reviews; `target_is_negative == 0`: 500,240 reviews) shows:
  * `boring`: occurs in **5,183 negative reviews** (9.10% of negatives) vs **3,992 positive reviews** (0.80% of positives) — an 11.4x relative representation lift in negative reviews.
  * `waste`: occurs in **1,649 negative reviews** (2.89% of negatives) vs **1,327 positive reviews** (0.27% of positives) — a 10.9x relative representation lift in negative reviews.
  * `crashes`: occurs in **952 negative reviews** (1.67% of negatives) vs **1,449 positive reviews** (0.29% of positives) — a 5.8x relative representation lift in negative reviews.

Both token-frequency log odds and full-corpus review counts align: `boring` is the single most prevalent quality complaint in this dataset, while `crashes` and `clunky` form the sharpest early onboarding blockers.

### 3.2 Early (<2h) vs. Later (>=2h) Playtime Comparison
Examining negative reviews written inside the 2-hour refund window versus later reviews (`reports/results/playtime_ngram_comparison.csv`):
* **Over-represented Inside Refund Window (<2h):**
  * `refund` / `refunded` (+2.15 Log Odds): Direct financial exit.
  * `clunky_controls` / `controls` (+1.42 Log Odds): Input friction prevents early progression.
  * `crashes` / `black_screen` / `not_work` (+1.38 Log Odds): Technical launch barrier.
* **Over-represented Later (>=2h):**
  * `repetitive` / `grind` (+1.28 Log Odds): Emerges after core gameplay loop becomes stale.
  * `rng` / `unbalanced` / `boss` (+1.15 Log Odds): Late-game difficulty spikes and poor scaling.
  * `lack_of_content` / `abandoned` (+1.10 Log Odds): Player exhausts early content.

---

## 4. Topic Modelling (NMF $k=8$)

### 4.1 Topic Extraction & Representative Terms
We fitted **Non-Negative Matrix Factorization (NMF)** on TF-IDF vectors generated from 40,000 negative training reviews. The 8 emergent topics, their extracted weights, and data-driven labels (`reports/results/topic_terms.csv`) are:

| Topic ID | Human-Readable Label | Top Representative Words (Ranked by Weight) |
| :---: | :--- | :--- |
| **T0** | **Boss Design, Damage & Upgrade Difficulty** | `run`, `enemies`, `boss`, `damage`, `level`, `upgrades`, `enemy`, `character`, `difficulty`, `runs`, `items`, `build` |
| **T1** | **Repetitive Gameplay, Grind & Pacing** | `boring`, `repetitive`, `boring_fast`, `boring_repetitive`, `fast`, `quickly`, `slow`, `boring_gameplay`, `pacing` |
| **T2** | **Flawed Design, Clunky Controls & RNG** | `bad`, `not_a_bad`, `rng`, `design`, `bad_design`, `sucks`, `controls`, `bad_rng`, `terrible`, `gambling`, `bad_controls` |
| **T3** | **Early Access State & Content Scarcity** | `early`, `access`, `early_access`, `abandoned`, `content`, `release`, `full`, `state`, `potential`, `wait`, `full_release` |
| **T4** | **Replayability & Co-op Friend Engagement** | `fun`, `rng`, `runs`, `recommend`, `not_that_fun`, `friends`, `content`, `replayability`, `less_fun`, `frustrating` |
| **T5** | **Multiplayer Issues, Crashes & Dev Bugs** | `multiplayer`, `steam`, `update`, `work`, `devs`, `crashes`, `fix`, `money`, `bugs`, `recommend`, `support`, `refund` |
| **T6** | **Art & Presentation vs Gameplay Loop** | `good`, `gameplay`, `repetitive`, `art`, `loop`, `content`, `price`, `style`, `gameplay_loop`, `story`, `concept`, `art_style` |
| **T7** | **Unbalanced Mechanics & Frustration** | `not_fun`, `sucks`, `not_for`, `rng`, `buggy`, `unbalanced`, `simply_not_fun`, `waste`, `solo`, `slow`, `lame`, `update` |

### 4.2 Topic Prevalence Across Review Cohorts
Measuring the dominant topic share across cohorts (`reports/results/topic_prevalence.csv`):

```
+---------------------------------------------------+-------------------+-------------------+--------------------+--------------------+
| Topic Label                                       | Negative Reviews  | Positive Reviews  | Neg (<2h Playtime) | Neg (>=2h Playtime)|
+---------------------------------------------------+-------------------+-------------------+--------------------+--------------------+
| T0: Boss Design, Damage & Upgrade Difficulty     |      30.86%       |      24.02%       |       24.28%       |       34.13%       |
| T1: Repetitive Gameplay, Grind & Pacing           |       4.64%       |       0.41%       |       5.83%        |        4.05%       |
| T2: Flawed Design, Clunky Controls & RNG          |       4.21%       |       1.15%       |       4.95%        |        3.84%       |
| T3: Early Access State & Content Scarcity         |       4.56%       |       3.26%       |       4.48%        |        4.60%       |
| T4: Replayability & Co-op Friend Engagement       |       9.34%       |      20.94%       |       7.08%        |       10.47%       |
| T5: Multiplayer Issues, Crashes & Dev Bugs        |      25.40%       |      18.30%       |      29.17%        |       23.52%       |
| T6: Art & Presentation vs Gameplay Loop           |      19.34%       |      31.84%       |      22.64%        |       17.69%       |
| T7: Unbalanced Mechanics & Frustration            |       1.65%       |       0.08%       |       1.55%        |        1.70%       |
+---------------------------------------------------+-------------------+-------------------+--------------------+--------------------+
```

#### Key Thematic Findings:
1. **The Early Onboarding Vulnerability:** Inside the refund window (<2h), **Topic 5 (Crashes & Bugs)** accounts for **29.17%** of dominant negative topics, and **Topic 2 (Controls & Flawed Design)** reaches its peak. Players drop out immediately when input or crash barriers block them.
2. **The Late Depth Transition:** After 2 hours of play, **Topic 0 (Boss Design & Upgrades)** expands from 24.28% to **34.13%**, and **Topic 4 (Replayability Fatigue)** expands from 7.08% to **10.47%**. Once technical barriers are cleared, critique shifts strictly to balance curves, difficulty fairness, and endgame content variety.

---

## 5. Topic Model Evaluation (Tasks #32 & #33)

Rather than treating topic models as black boxes or stopping at word clouds, we executed intrinsic, stability, comparative benchmarking, and qualitative ground-truth manual evaluations (`reports/results/text_evaluation_metrics.csv`):

### 5.1 Intrinsic Metric Results
1. **Topic Coherence (UMass):**
   $$C_{\text{UMass}}(w_i, w_j) = \log \frac{D(w_i, w_j) + 1}{D(w_j)} \le 0$$
   * NMF achieved a Mean UMass score of **-2.380**, compared to LDA's **-1.732**. Because UMass is negative, values closer to 0 represent higher pairwise co-occurrence (meaning LDA scores higher on raw co-occurrence). However, LDA's higher score is driven by high-frequency general gaming vocabulary, whereas NMF penalizes non-specific co-occurrences through TF-IDF weighting.
2. **Topic Diversity:**
   $$\text{Diversity} = \frac{|\bigcup_{k=1}^K Top10(T_k)|}{10 \times K}$$
   * NMF achieved a Topic Diversity of **90.00%** (72 unique tokens across 80 top terms), substantially higher than LDA (**86.25%**). NMF produces distinct, non-overlapping vocabularies across all 8 components.
3. **Subsample Stability (Replacing Deterministic Seed Tests):**
   * Testing topic stability across random seeds on the same data with deterministic `init="nndsvda"` trivially produces Jaccard = 1.00. To genuinely test stability against data perturbation, we evaluated topic recovery across **5 independent 80% subsamples** of the negative training reviews (seeds 101, 2024, 7, 99, 123) against the full reference model (`reports/results/topic_stability_evaluation.csv`):
     * **T0 (Boss Design & Upgrades):** Mean Jaccard = **1.000 $\pm$ 0.000** (High Stability)
     * **T1 (Repetitive Gameplay & Pacing):** Mean Jaccard = **0.733 $\pm$ 0.133** (Moderate Stability)
     * **T2 (Controls, Flawed Design & RNG):** Mean Jaccard = **0.768 $\pm$ 0.156** (Moderate Stability)
     * **T3 (Early Access & Content Scarcity):** Mean Jaccard = **0.855 $\pm$ 0.073** (High Stability)
     * **T4 (Replayability & Co-op):** Mean Jaccard = **0.697 $\pm$ 0.061** (Moderate Stability)
     * **T5 (Multiplayer & Crash Bugs):** Mean Jaccard = **0.855 $\pm$ 0.073** (High Stability)
     * **T6 (Art & Presentation vs Loop):** Mean Jaccard = **0.723 $\pm$ 0.226** (Moderate Stability)
     * **T7 (Unbalanced Mechanics & Frustration):** Mean Jaccard = **0.329 $\pm$ 0.341** (Lower Stability)
   * **Overall Mean Pairwise Jaccard:** **0.745** (the mean of the eight topic means; the spread across topics is $\pm$ 0.19 and the mean within-topic standard deviation over resamples is 0.13). The core diagnostic topics (Boss Design, Crashes, Early Access, Controls) remain highly stable under resampling, while low-prevalence residual clusters (T7) exhibit expected variance.

### 5.2 Comparative Baseline: NMF vs. LDA
| Evaluation Metric | NMF (TF-IDF) | LDA (Count Vectors) | Justification & Takeaway |
| :--- | :---: | :---: | :--- |
| **Topic Coherence (UMass)** | -2.380 | **-1.732** | LDA achieves higher raw co-occurrence (closer to 0) because Count vectors favor ubiquitous general gaming words (`game`, `play`, `time`) |
| **Topic Diversity** | **90.00%** | 86.25% | NMF achieves cleaner, less redundant topic vocabulary across top-10 terms |
| **Fitting Execution Time** | **2.32 seconds** | 105.12 seconds | NMF is **45x faster**, critical for reproducible pipelines on large corpora |
| **Semantic Separability** | **High** | Moderate | LDA blended general roguelike tokens across multiple topics; NMF cleanly isolated controls, crashes, and early access |
| **Algorithmic Selection Verdict** | **Selected** | Baseline | **NMF selected:** superior interpretability, higher diversity, clean failure-mode isolation, and 45x speedup justify NMF despite LDA's higher raw UMass co-occurrence |

### 5.3 Manual Qualitative Validation: Check of 200 Labelled Reviews (Task #33)

To move beyond statistical metrics and verify that NMF topics correspond to genuine player experiences, we performed a rigorous human evaluation on **200 stratified review excerpts** (`reports/results/topic_manual_validation_200.csv` and `reports/results/topic_manual_validation_summary.csv`).

#### Sampling & Evaluation Protocol
* **Sample Size & Balance:** Exactly **25 reviews per topic** across all 8 NMF topics ($N = 200$).
* **Cohort Representation:** Strata include both **negative reviews (126 reviews, 63%)** and **positive reviews (74 reviews, 37%)**, as well as reviews written inside the 2-hour refund window ($<2\text{h}$, 58 reviews) and post-window ($\ge 2\text{h}$, 142 reviews).
* **Evaluation Method:** Each review was independently inspected without model bias using structured semantic rubrics to determine ground-truth theme (`manual_topic_id`). Topic-level agreement ($1$ for match, $0$ for discrepancy) and explicit semantic rationale were recorded.

#### Topic-Level Agreement & Accuracy Table
| Topic ID | Topic Label | Total Sampled | Negative Reviews | Positive Reviews | Agreed | Disagreed | Agreement Rate | Primary Boundary Confusion | Qualitative Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **T0** | Boss Design, Damage & Upgrade Difficulty | 25 | 15 | 10 | 25 | 0 | **100.0%** | None (100% agreement) | Exceptional precision; distinctive vocabulary around damage, boss phases, and progression. |
| **T1** | Repetitive Gameplay, Grind & Pacing | 25 | 18 | 7 | 25 | 0 | **100.0%** | None (100% agreement) | Perfect precision; terms like `boring`, `repetitive`, and `grind` isolate pacing fatigue. |
| **T2** | Flawed Design, Clunky Controls & RNG | 25 | 16 | 9 | 24 | 1 | **96.0%** | T7 (Unbalanced Mechanics) | High agreement; minor boundary overlap with general game frustration. |
| **T3** | Early Access State & Content Scarcity | 25 | 15 | 10 | 25 | 0 | **100.0%** | None (100% agreement) | Unambiguous detection of Early Access, content drought, and roadmap critique. |
| **T4** | Replayability & Co-op Friend Engagement | 25 | 14 | 11 | 23 | 2 | **92.0%** | T1 (Repetitive Gameplay) | Strong agreement; negative co-op reviews occasionally emphasize repetitive gameplay. |
| **T5** | Multiplayer Issues, Crashes & Dev Bugs | 25 | 15 | 10 | 23 | 2 | **92.0%** | T2 (Flawed Design) | High precision on crash bugs; occasional overlap when control lockups accompany crash reports. |
| **T6** | Art & Presentation vs Gameplay Loop | 25 | 14 | 11 | 24 | 1 | **96.0%** | T1 (Repetitive Gameplay) | Clean identification of visual aesthetic and soundtrack contrasts with gameplay. |
| **T7** | Unbalanced Mechanics & Frustration | 25 | 19 | 6 | 23 | 2 | **92.0%** | T0 (Boss Design) | Captures acute player dissatisfaction; occasional overlap with specific boss difficulty spikes. |
| **Overall** | **Macro Average (All 8 Topics)** | **200** | **126** | **74** | **192** | **8** | **96.0%** | **Adjacent themes** | **Robust empirical validation of NMF topics; confirms high real-world semantic fidelity.** |

#### Key Qualitative Takeaways:
1. **High Real-World Fidelity:** With an overall agreement rate of **96.0% (192/200)**, the unsupervised NMF model accurately partitions unstructured player feedback into distinct, actionable engineering and game-design categories.
2. **Deterministic Complaint Isolation:** Technical failure modes (Topic T5: Crashes, Topic T2: Controls, Topic T3: Early Access) show near-zero leakage into aesthetic or progression topics, confirming that automated triage accurately alerts developers to critical launch bugs.
3. **Boundary Ambiguity:** The small fraction of disagreements ($4.0\%$) occurs along natural semantic borders: reviews discussing co-op fatigue (T4 vs T1) or blaming boss damage scaling as unfair design (T7 vs T0).

### 5.4 Distinction: Supervised Sentiment Detection vs. Unsupervised Topic Modelling
It is vital to distinguish between:
* **Supervised Sentiment Classification (Review 1's full-text Logistic Regression model):**
   * Evaluated on test set: **PR-AUC 0.785, ROC-AUC 0.955, Precision 0.72, Recall 0.72**.
   * **Role:** Automated post-hoc triage. It ingests an already-written review and classifies whether the player gave a thumbs-down.
   * **Crucial Limitation:** A sentiment classifier *cannot forecast pre-launch game risk*, because no reviews exist prior to release.
* **Unsupervised Topic Modelling (Review 2's NMF model):**
   * Evaluated via Coherence (-2.380), Diversity (90%), Subsample Stability (mean Jaccard 0.745; topic-to-topic spread 0.19), and **Manual Qualitative Validation (96.0% agreement across 200 reviews)**.
   * **Role:** Causal failure mode discovery. It decomposes the unstructured text of dissatisfied players into actionable categories (controls, netcode, RNG, progression), explaining *why* negative reviews occur and what developers must fix.

---

## 6. Limitations & Threats to Validity

1. **Language Coverage:** Covers English reviews only (557,202 of 1,648,387; ~39.5%). Non-English communities (Simplified Chinese 32%, Russian 7.5%) may experience different complaint patterns (e.g. localization quality, translation bugs).
2. **Reviewer Selection Bias:** Players write Steam reviews under extreme emotional states (exceptional satisfaction or severe frustration). The corpus represents vocal players, not silent churners who uninstall without reviewing.
3. **Snapshot Timing:** Reviews span 2022 to 2025; older titles had longer exposure to accumulate post-launch patches, shifting complaints from launch bugs to late-game content scarcity.
