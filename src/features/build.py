"""Build every feature table, then write the feature catalog.

  python -m src.features              build all + validate
  python -m src.features --skip-text  skip the TF-IDF/SVD text block (the slowest, ~1 minute)

Order matters: tags -> game_features (needs tag components) -> targets -> review_features (needs game_features) -> text.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import game_features, review_features, tags, targets, text, validate
from .common import DOCS, get_logger

log = get_logger("features")


def write_catalog() -> None:
    rows = [r for mod in (game_features, review_features, tags, targets, text) for r in mod.CATALOG]
    cat = pd.DataFrame(rows, columns=["table", "feature", "group", "source", "definition"])
    cat = cat.drop_duplicates(["table", "feature"]).sort_values(["table", "group", "feature"])
    DOCS.mkdir(parents=True, exist_ok=True)
    cat.to_csv(DOCS / "feature_catalog.csv", index=False)
    log.info("feature_catalog.csv: %d entries", len(cat))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    tags.build()
    game_features.build()
    targets.build()
    review_features.build()
    if not args.skip_text:
        text.build()
    write_catalog()
    if not args.no_validate:
        raise SystemExit(0 if validate.run(check_text=not args.skip_text) else 1)


if __name__ == "__main__":
    main()
