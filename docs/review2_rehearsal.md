# Review 2 rehearsal sheet (cards #46 and #47)

Deck: `slides/review2.pptx` (12 slides, speaker notes on every slide). Target 15 minutes including questions; the plan below is 13.5 minutes of talking and 1.5 minutes of buffer.

## Timing and speakers

| Slide | Title | Speaker | Time | Cumulative |
| ---: | --- | --- | ---: | ---: |
| 1 | Title | Shantharam | 0:30 | 0:30 |
| 2 | Review 1 in three numbers, and what Review 2 adds | Shantharam | 1:15 | 1:45 |
| 3 | Reviewer comment: tuning by grid search | Shantharam | 1:15 | 3:00 |
| 4 | Text mining: from keywords to topics | Tejaswini | 1:15 | 4:15 |
| 5 | What players complain about | Tejaswini | 1:15 | 5:30 |
| 6 | Which tag combinations go with Strong reception | Pradeep | 1:15 | 6:45 |
| 7 | Do patches help? | Gayas | 1:15 | 8:00 |
| 8 | Launch-risk scorer | Gayas | 1:15 | 9:15 |
| 9 | The dashboard (live demo) | Srihitha | 1:15 | 10:30 |
| 10 | What a studio should do | Shantharam | 1:15 | 11:45 |
| 11 | Limits and honest results | Pradeep | 0:45 | 12:30 |
| 12 | Contributions, tracking and next steps | all five, then Shantharam | 1:00 | 13:30 |

The split is a suggestion; swap slides if someone prefers another. The slide notes name the same speaker and time.

## Hand-over lines

- 3 to 4: "That was the reviewer comment. Next, what the review text says." (Shantharam to Tejaswini)
- 5 to 6: "That was complaints. Which tag combinations go with strong games?" (Tejaswini to Pradeep)
- 6 to 7: "Rules say what to build; Gayas will show what patching does." (Pradeep to Gayas)
- 8 to 9: "The scorer and the other pages are in one dashboard; Srihitha will show it." (Gayas to Srihitha)
- 9 to 10: "So what should a studio do?" (Srihitha to Shantharam)

## Dashboard demo (slide 9, about one minute)

1. `streamlit run src/dashboard/app.py` before the session; keep the browser on the Market explorer.
2. Market explorer: tick the tag "Pixel Graphics" in the sidebar and point at the tag chart and the Struggling share.
3. Tag rules and complaints: set stability to "stable", lift to 1.8, read one rule with its train and test counts.
4. Launch-risk scorer: leave the default game, press "Score this game" and read the tier and the three scores. Say they are not calibrated probabilities.
5. If the app will not start, fall back to slide 8 (the same example) and say so.

## Questions to expect

| Question | Short answer |
| --- | --- |
| Did tuning change your conclusions? | No. Cross-validation moved by 0.002 to 0.02 and the test scores did not follow consistently; the three tree models still tie. The text model's jump is mostly a vocabulary refit on all training reviews. |
| Why NMF if LDA has better coherence? | UMass favours LDA, but NMF gives more distinct, interpretable themes (diversity 90.0% against 86.25%) and fits about 45 times faster. We report both. |
| Is the manual topic check done? | It is being redone with blind labels; we do not quote an agreement figure. |
| Do patches work? | Not in general. The drop after a patch is explained by launch timing and regression to the mean; against matched controls the difference is +0.27 points with an interval that includes zero. |
| Can the rules be trusted? | They are tested on training games with FDR control and checked once on test games: 45 of 95 hold. Supports are small, so they are hypotheses. No rule predicts Struggling. |
| How good is the scorer? | About 0.43 macro-F1 in cross-validation against 0.22 for guessing; Struggling games are hard to find. It compares launch plans; it does not promise an outcome. |
| Is there leakage? | The split is by game, everything learned from data is fitted on training rows, outcome columns are never features, and the test set is scored once. 21 validation checks run in `python -m src.features.validate`. |
| Why are game tiers so hard? | Store metadata does not see gameplay quality, marketing or luck; a flat learning curve says more games would not help. |

## Checklist before the session

- [ ] Deck opens, fonts fine on the presenting laptop.
- [ ] Dashboard starts and the default scorer page loads.
- [ ] `git pull` on main so the dashboard tables match the deck.
- [ ] Each speaker has read their slide notes once, aloud, with a timer.
- [ ] A full run-through with a timer; trim any slide that runs over 1:30.
