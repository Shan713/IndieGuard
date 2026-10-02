# Cleaning log

Generated 2026-10-02T01:57:52+00:00 UTC by `python -m src.clean.clean` from `data/interim/` (see `src/clean/flatten.py`).
Raw data is never modified. Rows are dropped only when unusable; suspicious rows are flagged.

| Table | Rule | Action | Rows before | Rows after | Affected | Justification |
| --- | --- | --- | ---: | ---: | ---: | --- |
| games | Duplicate appid | drop | 1,861 | 1,861 | 0 | Same game matched by several roguelike tags; one row per game. |
| games | No store details / not a game | drop | 1,861 | 1,861 | 0 | Delisted or non-game apps have no price, genre or release data. |
| games | Release date from store API (search date as fallback) | fix | 1,861 | 1,861 | 0 | appdetails is the authoritative date; search date only used where the store date is unparseable. |
| games | Release outside 2022-2025 | drop | 1,861 | 1,861 | 0 | Scope of the study; store date can differ from the search listing used to build the universe. |
| games | launch_date = original launch (Early Access start) where earlier than store date | fix | 1,861 | 1,861 | 318 | Store date is the 1.0 date for games that left Early Access; launch_date is the true launch. |
| games | Early Access launch before 2022 (1.0 inside window) | flag | 1,861 | 1,861 | 86 | Kept per the store-date scope rule; can be excluded in sensitivity checks. |
| games | No price (not purchasable in US store at snapshot) | flag | 1,861 | 1,861 | 0 | Kept: reviews are valid; price-based analyses must exclude or impute these. |
| games | Strip whitespace in tag/genre/developer lists | fix | 1,861 | 1,861 | 56 | Store tag names sometimes carry trailing spaces ('Dystopian '). |
| games | No SteamSpy record | flag | 1,861 | 1,861 | 0 | Owner estimates unavailable; kept. |
| games | Review scrape < 99% of Steam total | flag | 1,861 | 1,861 | 22 | Steam totals include hidden/deleted reviews the API no longer returns; per-game coverage is reported. |
| games | Reviews capped (newest 15,000 only) | flag | 1,861 | 1,861 | 5 | Largest hits capped to finish the scrape on time and stop them dominating review-level models. Their launch period is missing: exclude from launch/refund-window timing and patch before/after analyses, and use the steam_total_* columns (full Steam totals) for their rating. |
| games | Review scrape not finished | flag | 1,861 | 1,861 | 0 | Game still in progress when cleaning ran; its reviews are absent from reviews_clean. |
| news | Duplicate news item | drop | 45,183 | 45,183 | 0 | Same gid listed twice. |
| news | Developer announcement (not press coverage) | flag | 45,183 | 45,183 | 44,232 | Only the developer's own Steam announcements are used as event dates. |
| news | Patch event | flag | 45,183 | 45,183 | 28,875 | Steam 'patchnotes' tag, or title like patch/hotfix/update/version number, excluding teasers ('coming soon', roadmap), demos, sales, awards and dev blogs. |
| news | Sale/discount event | flag | 45,183 | 45,183 | 1,043 | Developer announcement titled sale/discount/deal/% off; used for price-effect analysis. |
| reviews | Game not in cleaned game table | drop | 1,652,999 | 1,652,999 | 0 | Follows the game-level scope rules above. |
| reviews | Duplicate recommendationid | drop | 1,652,999 | 1,652,999 | 0 | A page can be saved twice after a crash/resume, or shift when a new review is posted mid-scrape; the latest edit is kept. |
| reviews | Empty review text | drop | 1,652,999 | 1,648,387 | 4,612 | No text to analyse; Steam allows submitting a blank review. |
| reviews | Impossible creation timestamp | drop | 1,648,387 | 1,648,387 | 0 | Before 2015 or in the future. |
| reviews | votes_funny overflow -> missing | fix | 1,648,387 | 1,648,387 | 0 | Known Steam API counter bug. |
| reviews | Negative playtime -> missing | fix | 1,648,387 | 1,648,387 | 0 | Impossible values. |
| reviews | weighted_vote_score outside [0,1] -> missing | fix | 1,648,387 | 1,648,387 | 0 | Score is a probability. |
| reviews | Playtime at review above 99.5% quantile (363 h), capped copy added | flag | 1,648,387 | 1,648,387 | 8,242 | Steam counts idle/AFK time; raw minutes kept, author_playtime_at_review_capped is the winsorised copy. |
| reviews | Zero or missing playtime at review | flag | 1,648,387 | 1,648,387 | 12 | Pre-release keys, family sharing or playtime not recorded. |
| reviews | Written before launch (incl. Early Access start) | flag | 1,648,387 | 1,648,387 | 30,018 | Playtests/early keys; kept but can be excluded from launch-timing analyses. |
| reviews | Low content (<3 letters or <2 words; CJK: <4 characters) | flag | 1,648,387 | 1,648,387 | 208,724 | e.g. 'good', '10/10', '好玩', emoji only: valid votes, little text signal. |
| reviews | ASCII art / symbol spam (>50% symbols, 50+ chars) | flag | 1,648,387 | 1,648,387 | 4,951 | Meme reviews with no analysable text. |
| reviews | Copypasta (same 50+ char text >= 10 times) | flag | 1,648,387 | 1,648,387 | 2,285 | Copied meme/template text or coordinated posting. |
| reviews | Same author, same game, >1 review (info only) | flag | 1,648,387 | 1,648,387 | 6 | Steam allows one review per account per game; non-zero values would point to scrape problems. |
| reviews | Text-mining subset (English, not low-content/ASCII/copypasta) | flag | 1,648,387 | 1,648,387 | 597,777 | Steam's language field is the reviewer's chosen language; topic models are run on English only. |
| reviews | Received for free | flag | 1,648,387 | 1,648,387 | 34,394 | Kept; free copies may bias sentiment, so analyses can control for it. |

## Output

- `data/processed/games_clean.parquet`: 1,861 games
- `data/processed/reviews_clean/` (parquet parts): 1,648,387 reviews (10.4% negative, 39.5% English, 597,777 in the text-mining subset)
- `data/processed/events.parquet`: 28,875 patch and 1,043 sale announcements across 1,607 games
