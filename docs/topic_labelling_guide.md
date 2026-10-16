# Blind topic labelling guide (card #33)

**Goal.** Check by hand whether the topic model's themes match what reviewers wrote. You label 200 reviews **without seeing the model's topic**, then `score` compares your labels with it.

**Do not open** `reports/results/topic_model_assignments_200.csv` while labelling: it holds the model's answer.

## How

1. Open `reports/results/topic_manual_labelling_sheet.csv` in a spreadsheet (UTF-8).
2. For each review, fill `your_topic` with a number from the table below, and `your_note` with a few words of your own (for example "complains about boss damage").
3. Choose the **main** theme of the review. If none fits, or it is too short to tell, use **8**.
4. When all 200 are filled, run `python -m src.analysis.topic_manual_validation score`. It refuses to run if rows are empty, labels are out of range, or the notes are copy-pasted.

## The topics

| Number | Topic | The review is mainly about |
| ---: | --- | --- |
| 0 | Boss design, damage and upgrade difficulty | bosses, enemies, damage numbers, upgrades, builds, difficulty of runs |
| 1 | Repetitive gameplay, grind and pacing | boring, repetitive, grindy, slow or tedious loops |
| 2 | Flawed design, clunky controls and RNG | bad design, controls and camera, unfair randomness, "sucks" |
| 3 | Early Access state and content scarcity | early access, abandoned, not enough content, waiting for the full release |
| 4 | Replayability and co-op | replay value, playing with friends, whether it stays fun |
| 5 | Multiplayer issues, crashes and developer bugs | netcode, lobbies, crashes, bugs, updates, developer support |
| 6 | Art and presentation against the gameplay loop | art style, music, story, against the core loop |
| 7 | Unbalanced mechanics and frustration | specific mechanics that feel unfair or unbalanced |
| 8 | Other / unclear | none of the above, or too short to tell |

Positive reviews are included on purpose: label the theme they talk about, not whether they are positive.

## What the result means

`score` writes per-topic agreement, a confusion matrix and Cohen's kappa to `reports/results/topic_manual_validation_blind*.csv`. Report the agreement together with how many reviews you put in "other / unclear". Agreement is a check on interpretability, not on accuracy against a ground truth, and with 25 reviews per topic each topic's rate has a wide interval.
