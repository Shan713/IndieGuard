# Dataset documentation

**Project:** IndieGuard: Indie Game Launch Analytics (Business Analytics capstone, Units 1 to 3)
**Data owner:** M1 (data lead). **Collected:** 1 to 2 October 2026.
**Status:** raw snapshot frozen; cleaned tables produced. Feature engineering is out of scope here (M2).

All data was collected by the team from public Steam and SteamSpy endpoints. No pre-built dataset
(Kaggle, UCI, GitHub or similar) was used.

## 1. At a glance

| | |
| --- | --- |
| Niche | Indie roguelike / roguelite games on Steam |
| Games | 1,861, released 2022 to 2025 (286 / 372 / 557 / 646 per year) |
| Reviews (raw) | 1,652,999 in 31 languages |
| Reviews (cleaned) | 1,648,387; 10.4% negative; 597,777 in the English text-mining subset |
| Patch / sale events | 28,875 patches across 1,597 games; 1,043 sale announcements |
| Review coverage | 99.95% of Steam's reported total for the 1,856 fully scraped games; 5 largest hits capped at their newest ~15,000 reviews |
| Collection window | 2026-10-01 13:46 UTC to 2026-10-02 01:21 UTC (19:16 to 06:51 IST) |
| Raw snapshot hash | SHA256 of `data/raw/manifest.csv`: `03f5022713884ca9bc3564a8010deb486401a97dc143a005ea3009edf228241e` |

## 2. Sources

| Source | Endpoint | Used for | Key needed |
| --- | --- | --- | --- |
| Steam store search | `store.steampowered.com/search/results/` | Building the game universe (tags, release date, review count) | No |
| Steam reviews | `store.steampowered.com/appreviews/<appid>` | Every review: text, recommend flag, playtime, timestamps, votes, flags | No |
| Steam appdetails | `store.steampowered.com/api/appdetails` | Price, genres, categories, release date, developer/publisher, DLC, platforms | No |
| Steam store page | `store.steampowered.com/app/<appid>/` | Top 20 user tags with vote counts | No |
| Steam news | `api.steampowered.com/ISteamNews/GetNewsForApp/v2/` | Developer announcements, used for patch and sale dates | No |
| SteamSpy | `steamspy.com/api.php?request=appdetails` | Owner estimates, concurrent users, SteamSpy tag votes | No |

SteamSpy is blocked by the collection network's ISP (TLS connection reset), so it was reached through
Cloudflare WARP. All other sources were reached directly or through WARP; the data returned does not depend on it.

## 3. Collection method

Code: `src/collect/`. Every collector saves the untouched API response to `data/raw/` first and is
resumable: re-running skips files that already exist.

### 3.1 Game universe (`universe.py`)
Store search was queried for games (`category1=998`) carrying the **Indie** tag (492) **and** one of five
roguelike-family tags: Roguelike (1716), Roguelite (3959), Action Roguelike (42804), Roguelike Deckbuilder
(1091588), Traditional Roguelike (454187). Results were sorted newest first and paging stopped once a full
page was dated before 2022. This produced 5,917 candidates, of which 3,529 were released 2022 to 2025 and
**1,861 had at least 10 reviews**. Full rules: `docs/game_universe.md`; the list: `docs/app_ids.csv`.

The brief suggested 150 to 300 games; the team chose to keep every qualifying game to maximise data.

### 3.2 Game metadata (`metadata.py`)
For each game: appdetails (`cc=us`, `l=english`, so prices are USD), the store page (user tags parsed
from the embedded tag list), Steam news (up to 1,000 items, contents truncated to 600 characters) and SteamSpy.

### 3.3 Reviews (`reviews.py`)
Every review was paged with the cursor using `filter=recent`, `language=all`, `review_type=all`,
`purchase_type=all`, `filter_offtopic_activity=0` (review-bomb periods included), 100 per page.

- **Rate limiting.** Steam throttles the store host per IP. In practice it allowed about 150 requests
  before returning HTTP 429, regardless of pace (1.25 to 1.84 s per request tested). The scraper waits at
  least 1.25 s between requests and backs off 45 s, 90 s, 135 s… on a 429. appdetails and store-page
  requests share the same throttle because Steam counts them in the same bucket.
- **Checkpointing.** Each page is appended to disk and the cursor saved, so a crash resumes mid-game.
  A supervisor (`run_forever.ps1`) restarts the scraper after a crash.
- **Completion check.** A game ends when Steam returns an empty page or the reported total is reached;
  early empty pages are retried before being accepted.
- **Cap.** To finish within the time available, the five largest games were capped at their newest
  ~15,000 reviews (`--max-per-game`); see section 8.

### 3.4 Timeline (UTC)

| Step | Start | End |
| --- | --- | --- |
| Universe search | 2026-10-01 13:44 | 13:52 |
| News and SteamSpy | 2026-10-01 13:46 | 14:34 |
| appdetails and store tags | 2026-10-01 13:46 | 15:57 |
| Reviews | 2026-10-01 13:52 | 2026-10-02 01:21 |
| Raw snapshot frozen (manifest) | 2026-10-02 01:26 | |

## 4. Files and lineage

```
data/raw/          untouched API responses (never edited, not committed)
  search/          store search result pages
  universe/        candidates.csv, summary.json
  appdetails/ tags/ news/ steamspy/   one JSON per game
  reviews/         <appid>.jsonl.gz (reviews) + <appid>.done.json (totals, stop reason)
  manifest.csv     path, size, row count, SHA256 for every raw file
data/interim/      typed tables, nothing dropped      (src/clean/flatten.py)
  reviews.parquet  games.parquet  news.parquet
data/processed/    cleaned tables, committed to git    (src/clean/clean.py)
  reviews_clean/   part-000..004.parquet (one table split under GitHub's 100 MB file limit)
  games_clean.parquet  events.parquet  cleaning_log.json
```

Rebuild everything after raw: `python -m src.clean` (about 2 minutes). Verify the raw snapshot:
`python -m src.collect.manifest` and compare the hash above. Raw and interim data are git-ignored;
counts and hashes are committed in `docs/data_manifest.md`. The cleaned tables in `data/processed/` are
committed so teammates can start from a clone:

```python
import pandas as pd
reviews = pd.read_parquet("data/processed/reviews_clean")      # reads all parts as one table
games = pd.read_parquet("data/processed/games_clean.parquet")
events = pd.read_parquet("data/processed/events.parquet")
```

## 5. Data dictionary

### 5.1 `reviews_clean/` (1,648,387 rows, one per review)

| Column | Type | Description |
| --- | --- | --- |
| `recommendationid` | int | Steam's unique review ID (primary key) |
| `appid` | int | Game ID, joins to `games_clean.appid` |
| `language` | str | Language the reviewer selected (31 values; english 39.5%, schinese 32.0%, russian 7.5%) |
| `review` | str | Review text, as written |
| `voted_up` | bool | True = recommended, False = not recommended (negative review) |
| `votes_up` / `votes_funny` | int / float | Helpful and funny votes from other users |
| `weighted_vote_score` | float | Steam's helpfulness score, 0 to 1 |
| `comment_count` | int | Comments on the review |
| `steam_purchase` | bool | Bought on Steam (False = key activated elsewhere) |
| `received_for_free` | bool | Reviewer marked the game as received for free (2.1%) |
| `refunded` | bool | Reviewer refunded the game (0.47%) |
| `written_during_early_access` | bool | Written while the game was in Early Access (22.7%) |
| `primarily_steam_deck` | bool | Reviewer played mostly on Steam Deck |
| `n_reactions` | int | Total award reactions on the review |
| `has_dev_response` | bool | Developer replied to the review |
| `timestamp_dev_responded` | float | Unix time of the developer reply (null when none) |
| `author_steamid_hash` | str | Salted SHA256 of the reviewer's Steam ID (see section 9) |
| `author_num_games_owned` / `author_num_reviews` | int | Reviewer's library size and review count |
| `author_playtime_forever` | int | Reviewer's total minutes played, at collection time |
| `author_playtime_last_two_weeks` | int | Minutes played in the two weeks before collection |
| `author_playtime_at_review` | float | Minutes played when the review was written (12 nulls) |
| `author_last_played` | int | Unix time the reviewer last played |
| `created` / `updated` | datetime (UTC) | When the review was written / last edited |
| `author_playtime_at_review_capped` | float | Playtime at review winsorised at the 99.5% quantile (363 h); raw column kept |
| `playtime_outlier` | bool | Playtime at review above that cap |
| `no_playtime` | bool | Zero or missing playtime at review |
| `is_prerelease` | bool | Written before the game's launch (`games_clean.launch_date`) |
| `low_content` | bool | Under 3 letters or under 2 words (CJK: under 4 characters), e.g. "good", "10/10" |
| `is_ascii_art` | bool | 50+ characters, over half symbols |
| `is_copypasta` | bool | Same 50+ character text appears 10+ times in the dataset |
| `text_mining_ok` | bool | English and none of the three flags above: the text-mining subset |

### 5.2 `games_clean.parquet` (1,861 rows, one per game)

| Column | Type | Description |
| --- | --- | --- |
| `appid`, `name` | int, str | Steam app ID and name |
| `search_released_raw`, `search_release_date` | str | Release date as listed in store search |
| `search_review_count`, `search_pct_positive` | int | Review count and % positive from the search tooltip (reviewer's-language basis) |
| `search_tagids` | str | Top tag IDs shown in search |
| `matched_tags` | str | Which roguelike-family tags matched (`;`-separated) |
| `appdetails_ok`, `appdetails_fetched_utc` | bool, str | appdetails success and fetch time |
| `type` | str | Steam app type (all "game") |
| `is_free` | bool | Free to play (308 games) |
| `price_currency`, `price_initial_cents`, `price_final_cents`, `discount_percent` | | US store price at fetch time; null for free games |
| `price_usd` | float | Regular (undiscounted) price in USD; 0 for free games |
| `price_final_usd_snapshot` | float | Price paid at fetch time, after any discount running then |
| `price_missing` | bool | No price available (0 games) |
| `store_release_date_raw`, `coming_soon` | str, bool | Store release date as published |
| `release_date` | datetime | Parsed store release date (the 1.0 date for games that left Early Access) |
| `launch_date` | datetime | Original launch including Early Access start (from reviews' `app_release_date`); differs from `release_date` for 318 games |
| `launch_before_window` | bool | Early Access launch before 2022 though 1.0 is in 2022 to 2025 (86 games) |
| `developers`, `publishers` | str | `;`-separated names |
| `genres`, `categories` | str | Steam genres (e.g. Indie, Action) and features (e.g. Single-player, Steam Achievements), `;`-separated |
| `tags` | str | Top 20 user tags in vote order, `;`-separated |
| `tags_votes_json` | str (JSON) | `{tag: votes}` for those tags |
| `n_dlc` | int | Number of DLCs |
| `platform_windows`, `platform_mac`, `platform_linux` | bool | Supported platforms |
| `metacritic_score` | float | Metacritic score (40 games; null otherwise) |
| `recommendations_total` | float | Steam's recommendation count (null when Steam omits it, 1,320 games) |
| `required_age` | int | Age gate |
| `controller_support` | str | "full" / "partial"; null = none listed |
| `n_supported_languages` | int | Number of interface languages |
| `n_achievements` | float | Number of achievements (null = none) |
| `short_description` | str | Store blurb |
| `spy_owners_raw`, `spy_owners_low`, `spy_owners_high` | str, int | SteamSpy owner range, e.g. "20,000 .. 50,000" |
| `spy_ccu`, `spy_positive`, `spy_negative` | int | SteamSpy peak concurrent users and review counts (may lag Steam) |
| `spy_tags_votes_json` | str (JSON) | SteamSpy `{tag: votes}`; null when SteamSpy has none (550 games) |
| `spy_missing` | bool | No SteamSpy record (0 games) |
| `reviews_scraped`, `review_scrape_stop` | int, str | Reviews collected and why the scrape stopped (`end` or `capped`) |
| `steam_total_reviews`, `steam_total_positive`, `steam_total_negative` | int | Steam's own all-language totals at collection time (complete for every game, including capped ones) |
| `steam_review_score`, `steam_review_score_desc` | int, str | Steam's rating band, e.g. 8 / "Very Positive" |
| `review_scrape_complete` | bool | At least 99% of Steam's total collected |
| `reviews_capped` | bool | Reviews capped at the newest ~15,000 (5 games) |

### 5.3 `events.parquet` (29,918 rows, one per event)

| Column | Type | Description |
| --- | --- | --- |
| `appid` | int | Game ID |
| `gid` | str | Steam news item ID |
| `date` | datetime (UTC) | Announcement time |
| `event_type` | str | `patch` or `sale` |
| `title`, `news_tags`, `url` | str | Announcement title, Steam news tags, link |

**Patch:** a developer announcement tagged `patchnotes` by Steam, or titled like a shipped patch
(patch, hotfix, update, version number), excluding teasers ("coming soon", roadmap, preview), demos,
sales, awards and dev blogs. **Sale:** a developer announcement titled sale, discount, deal or "% off".
Press coverage from external feeds is excluded.

## 6. Cleaning summary

Full rule-by-rule table with before/after counts and justifications: `docs/cleaning_log.md`.

- **Dropped:** 4,612 reviews with empty text. No duplicate review IDs, impossible timestamps,
  out-of-window games or non-game apps were found.
- **Fixed:** whitespace in tag/genre/developer lists (56 games); release date replaced by the original
  launch date for launch timing (318 games); overflowed or negative counters set to missing (none found).
- **Flagged, kept:** playtime outliers (8,242), zero playtime (12), pre-launch reviews (30,018),
  low content (208,724), ASCII art (4,951), copypasta (2,285), free copies (34,394), capped games (5).

Principle: rows are dropped only when unusable; anything suspicious is kept with a flag so each analysis
decides for itself.

## 7. Coverage and quality checks

- All 1,861 game scrapes finished; every game has reviews (10 to 162,452; median 50; 941 games with 50+).
- 1,856 fully scraped games: 1,577,704 of 1,578,440 reviews Steam reports (99.95%). The gap is reviews
  Steam still counts but no longer returns (deleted or hidden).
- Raw to cleaned row counts reconcile exactly with the cleaning log (1,652,999 − 4,612 = 1,648,387).
- No duplicate `recommendationid`; 6 reviews share an author and game (Steam allows one per account,
  so this indicates edits/re-posts, kept).

## 8. Limitations and known biases

1. **Five capped games.** Megabonk, Cult of the Lamb, Hades II, Balatro and Vampire Survivors contain
   only their newest ~15,000 reviews (Sep 2025 to Oct 2026). These five hold 34% of all Steam reviews
   in the niche but 4.5% of this dataset, and their recent reviews are more negative than their lifetime
   average (e.g. Vampire Survivors 4.5% vs 1.7%). **Rules:** exclude them (`reviews_capped`) from
   launch-timing, refund-window and patch-impact analyses; use `steam_total_*` for their ratings; report
   per-game rather than review-weighted market figures.
2. **Snapshot timing.** Prices, tags, owner estimates and totals are as of 1 October 2026, not as of
   launch. No price history is available, so discount effects rely on sale announcement dates.
3. **Language field.** `language` is the reviewer's chosen language, not detected from the text, so
   some reviews may be labelled with a language they are not written in. This was not measured.
4. **Playtime.** Steam counts idle time, hence the capped copy. Playtime is missing or zero for
   pre-release keys and family sharing.
5. **SteamSpy.** Owner figures are wide ranges and its review counts lag Steam. Its playtime fields
   were zero for all 1,861 games, so they were dropped; playtime comes from the reviews instead.
6. **Patch detection** is rule-based on announcement titles and tags. A manual check of samples found
   high precision, but silent patches (no announcement) are missed.
7. **Scope.** Tag-based selection depends on crowd-sourced tags at collection time; games that
   relabelled themselves may be in or out. The 10-review minimum excludes 1,668 very small games.
8. **Survivorship.** Games delisted before collection are absent.

## 9. Ethics and privacy

- Only public data was collected, at a throttled rate, with a descriptive User-Agent identifying the
  academic project. No logins, cookies (other than Steam's public age-gate cookie) or paid APIs were used.
- Reviewer persona names, profile URLs and avatars were **discarded at collection time**. Steam IDs are
  replaced by a salted SHA256 (`author_steamid_hash`), so reviewers can be deduplicated but not
  identified. The salt is stored only in `data/raw/.steamid_salt`, which is never committed or shared.
- Review text is kept because it is the object of study; quoted examples in reports should not name reviewers.
