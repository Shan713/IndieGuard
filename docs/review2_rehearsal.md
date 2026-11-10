# Review 2 rehearsal sheet (cards #46 and #47)

Deck: `slides/review2.pptx` (20 slides on the official template, speaker notes on every slide). Target 15 minutes including questions; the plan below is 14:30 of talking, so about 0:30 is left to absorb slips and one question.

## Timing and speakers

| Slide | Title | Speaker | Time | Cumulative |
| ---: | --- | --- | ---: | ---: |
| 1 | IndieGuard: Indie Game Launch Analytics | Shantharam | 0:20 | 0:20 |
| 2 | Business problem and objectives | Tejaswini | 0:40 | 1:00 |
| 3 | Complete project workflow | Shantharam | 0:30 | 1:30 |
| 4 | Overall dataset description | Shantharam | 0:45 | 2:15 |
| 5 | Sample dataset and key variables | Shantharam | 0:35 | 2:50 |
| 6 | Dataset validation and preparation | Shantharam | 0:45 | 3:35 |
| 7 | Exploratory analysis and preprocessing | Srihitha | 0:55 | 4:30 |
| 8 | Predictive modelling and final evaluation | Pradeep | 0:50 | 5:20 |
| 9 | Model comparison at a glance | Pradeep | 0:20 | 5:40 |
| 10 | Selected advanced analytics methods | Shantharam | 0:30 | 6:10 |
| 11 | Method 1 implementation | Tejaswini | 0:45 | 6:55 |
| 12 | Method 1 results at a glance | Tejaswini | 0:20 | 7:15 |
| 13 | Method 1 evaluation and comparison | Tejaswini | 0:55 | 8:10 |
| 14 | Method 2 implementation | Pradeep | 0:45 | 8:55 |
| 15 | Method 2 rules at a glance | Pradeep | 0:20 | 9:15 |
| 16 | Method 2 evaluation and comparison | Pradeep | 0:55 | 10:10 |
| 17 | Combined insights and recommendations | Shantharam | 0:55 | 11:05 |
| 18 | Interactive dashboard or website | Srihitha | 2:00 | 13:05 |
| 19 | Completion and individual contribution | all five, about 9 seconds each | 0:45 | 13:50 |
| 20 | Conclusion and limitations | Shantharam | 0:40 | 14:30 |

The split is a suggestion; swap slides if someone prefers another. The slide notes name the same speaker and time. Slides 9, 12 and 15 are chart slides (20 seconds each): say the one-line reading and move on.

## Hand-over lines

- 1 to 2: "That was indieguard: indie game launch analytics. Next: business problem and objectives." (Shantharam to Tejaswini)
- 2 to 3: "That was business problem and objectives. Next: complete project workflow." (Tejaswini to Shantharam)
- 6 to 7: "That was dataset validation and preparation. Next: exploratory analysis and preprocessing." (Shantharam to Srihitha)
- 7 to 8: "That was exploratory analysis and preprocessing. Next: predictive modelling and final evaluation." (Srihitha to Pradeep)
- 9 to 10: "That was model comparison at a glance. Next: selected advanced analytics methods." (Pradeep to Shantharam)
- 10 to 11: "That was selected advanced analytics methods. Next: method 1 implementation." (Shantharam to Tejaswini)
- 13 to 14: "That was method 1 evaluation and comparison. Next: method 2 implementation." (Tejaswini to Pradeep)
- 16 to 17: "That was method 2 evaluation and comparison. Next: combined insights and recommendations." (Pradeep to Shantharam)
- 17 to 18: "That was combined insights and recommendations. Next: interactive dashboard or website." (Shantharam to Srihitha)
- 18 to 19: "That was interactive dashboard or website. Next: completion and individual contribution." (Srihitha to all five, about 9 seconds each)
- 19 to 20: "That was completion and individual contribution. Next: conclusion and limitations." (all five, about 9 seconds each to Shantharam)

## Dashboard demo (slide 18, about two minutes; open https://indieguard.streamlit.app a few minutes earlier so it is awake)

1. `streamlit run src/dashboard/app.py` before the session; keep the browser on the Market explorer.
2. Market explorer: tick the tag "Pixel Graphics" in the sidebar and point at the tag chart and the Struggling share.
3. Tag rules and complaints: set stability to "stable", lift to 1.8, read one rule with its train and test counts.
4. Launch-risk scorer: leave the default game, press "Score this game" and read the tier and the three scores. Say they are not calibrated probabilities.
5. If the app will not start, fall back to the screenshot on the slide and the local copy (`streamlit run src/dashboard/app.py`) and say so.

## Questions to expect

| Question | Short answer |
| --- | --- |
| Did tuning change your conclusions? | No at review level: cross-validation moved by 0.000 to 0.007 and the three tree models still tie. At game level the searches raised cross-validation by up to 0.08, but Random Forest and LightGBM fell to 0.37 on test while XGBoost (0.437) and Logistic Regression (0.423) held up, so we quote test intervals, not cross-validation. The text model's jump is mostly a vocabulary refit on all training reviews. |
| Why NMF if LDA has better coherence? | UMass favours LDA, but NMF gives more distinct, interpretable themes (diversity 90.0% against 86.25%) and fits about 45 times faster. We report both. |
| Is the manual topic check done? | Done blind, but labelled by an AI assistant, not a team member: 35.5% agreement, kappa 0.30, 73 of 200 with no clear theme. Early Access (88%), repetition (64%) and multiplayer and crashes (60%) hold up; boss and balance topics (8% to 16%) do not. An earlier 96% figure was withdrawn. A human re-label of a subset is the next step. |
| Do patches work? | Not in general. The drop after a patch is explained by launch timing and regression to the mean; against matched controls the difference is +0.27 points with an interval that includes zero. |
| Can the rules be trusted? | They are tested on training games with FDR control and checked once on test games: about two-thirds (45 of the 66 testable rules) hold. Supports are small, so they are hypotheses. No rule predicts Struggling. |
| How good is the scorer? | About 0.43 macro-F1 in cross-validation against 0.22 for guessing; Struggling games are hard to find. It compares launch plans; it does not promise an outcome. |
| Is there leakage? | The split is by game, everything learned from data is fitted on training rows, outcome columns are never features, and the test set is scored once. 21 validation checks run in `python -m src.features.validate`. |
| Why are game tiers so hard? | Store metadata does not see gameplay quality, marketing or luck; a flat learning curve says more games would not help. |

## Checklist before the session

- [ ] Deck opens, fonts fine on the presenting laptop.
- [ ] Dashboard starts and the default scorer page loads.
- [ ] `git pull` on main so the dashboard tables match the deck.
- [ ] Each speaker has read their slide notes once, aloud, with a timer.
- [ ] A full run-through with a timer; trim any slide that runs over 1:30.
