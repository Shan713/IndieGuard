import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard.featurize import featurize_game
from src.dashboard.risk import predict_launch_risk, predict_raw_game

ROOT = Path(__file__).resolve().parents[1]


class LaunchRiskTests(unittest.TestCase):
    def test_raw_features_match_stored_features(self):
        features = pd.read_parquet(
            ROOT / "data/processed/features/game_features.parquet"
        )
        raw = pd.read_parquet(ROOT / "data/processed/games_clean.parquet")

        appid = 794350
        stored_row = features.loc[features["appid"] == appid].iloc[0]
        raw_row = raw.loc[raw["appid"] == appid].iloc[0]

        feature_columns = json.loads(
            (ROOT / "models/launch_risk/feature_columns.json").read_text(
                encoding="utf-8"
            )
        )

        converted = featurize_game(raw_row.to_dict())
        self.assertEqual(set(converted), set(feature_columns))
        np.testing.assert_allclose(
            [converted[c] for c in feature_columns],
            [stored_row[c] for c in feature_columns],
            rtol=1e-5,
            atol=1e-5,
        )

        stored_result = predict_launch_risk(stored_row[feature_columns].to_dict())[0]
        raw_result = predict_raw_game(raw_row.to_dict())

        self.assertEqual(stored_result["tier_code"], raw_result["tier_code"])
        self.assertAlmostEqual(sum(raw_result["scores"].values()), 1.0, places=6)

    def test_all_eligible_games_score_successfully(self):
        features = pd.read_parquet(
            ROOT / "data/processed/features/game_features.parquet"
        )
        targets = pd.read_parquet(
            ROOT / "data/processed/features/game_targets.parquet"
        )
        eligible = targets.loc[targets["eligible"] == True, ["appid"]]
        data = eligible.merge(features, on="appid", validate="one_to_one")

        feature_columns = json.loads(
            (ROOT / "models/launch_risk/feature_columns.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(feature_columns), 83)

        results = predict_launch_risk(data[feature_columns])
        self.assertEqual(len(results), len(data))
        self.assertTrue(all(r["tier"] in {"Struggling", "Solid", "Strong"}
                            for r in results))
        scores = np.asarray(
            [
                [r["scores"][tier] for tier in ("Struggling", "Solid", "Strong")]
                for r in results
            ],
            dtype=float,
        )
        self.assertEqual(scores.shape, (len(data), 3))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
