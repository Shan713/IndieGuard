# Game universe: selection rules

Built by `python -m src.collect.universe` (owner: M1). Output: `docs/app_ids.csv`.
Run statistics are in `data/raw/universe/summary.json` and are copied below after each run.

## Niche
Roguelike / roguelite indie games on Steam. A game is a candidate when it carries the Steam user tag
**Indie** (tagid 492) **and** at least one roguelike-family tag:

| Tag | tagid |
| --- | --- |
| Roguelike | 1716 |
| Roguelite | 3959 |
| Action Roguelike | 42804 |
| Roguelike Deckbuilder | 1091588 |
| Traditional Roguelike | 454187 |

## Source
Steam store search (`store.steampowered.com/search/results/`, `category1=998` = games only, `cc=us`, `l=english`),
sorted newest first (`sort_by=Released_DESC`). Paging stops once a full page is dated before 2022.
Every raw result page is saved untouched under `data/raw/search/<tagid>_released_desc/`.
A check against default-sort pages showed no in-window game missing from the newest-first crawl.

## Selection rules
1. Release year (store release date) 2022 to 2025 inclusive. Undated / "Coming soon" titles are excluded.
2. At least 10 user reviews in the store search tooltip, so every game has a Steam review score.
3. Deduplicated by appid across the five tags; `matched_tags` lists which tags matched.

The team agreed to scrape **every** game passing these rules instead of a hand-picked 150 to 300,
to maximise data. Modelling subsets (for example games with at least 50 reviews) are defined later
in cleaning, and are documented in `docs/cleaning_log.md`.

## Known limitations
- User tags are crowd-sourced and change over time; this is a snapshot taken on the run date.
- Search review counts reflect store display preferences; exact totals come from
  `query_summary.total_reviews` (all languages) in the review scrape.
- SteamSpy is blocked by the collection network's ISP (connection reset), so it was reached through
  Cloudflare WARP. User tags are taken from the Steam store page (top 20 tags with vote counts);
  SteamSpy's tag votes are kept alongside as a second source.

## Run statistics
Run on 2026-10-01 (built 13:52 UTC, 112 search requests). Source: `data/raw/universe/summary.json`.

| Step | Games |
| --- | --- |
| Candidates (Indie + any roguelike-family tag) | 5,917 |
| Released 2022 to 2025 | 3,529 |
| With at least 10 reviews: **selected** | **1,861** |

Selected games by release year: 2022: 286, 2023: 372, 2024: 557, 2025: 646.
Search review counts for the selection sum to 1,136,972 (reviewer's-language basis); the full
all-language scrape returned 1,652,999 reviews (see `docs/data_documentation.md`).
