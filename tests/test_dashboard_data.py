import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "data" / "dashboard"


class DashboardDataTests(unittest.TestCase):
    def test_game_table(self):
        g = pd.read_csv(DASH / "games.csv")
        self.assertEqual(len(g), 1861)
        self.assertTrue({"appid", "name", "release_year", "price_band", "neg_share", "tier", "tags"} <= set(g.columns))
        self.assertTrue(g.neg_share.dropna().between(0, 1).all())
        self.assertEqual(g.appid.nunique(), len(g))

    def test_monthly_and_refund_tables(self):
        m = pd.read_csv(DASH / "monthly.csv")
        self.assertTrue(m.neg_share.between(0, 1).all())
        self.assertGreater(m.reviews.sum(), 1_000_000)
        s = pd.read_csv(DASH / "refund_summary.csv").iloc[0]
        self.assertGreater(s.neg_share_inside_window, 2 * s.neg_share_outside_window)
        h = pd.read_csv(DASH / "refund_heatmap.csv")
        self.assertEqual(len(h), 20)

    def test_result_tables_have_the_columns_the_pages_use(self):
        r = pd.read_csv(ROOT / "reports/results/association_rules_strong.csv")
        self.assertTrue({"antecedents_str", "lift", "stability", "n_antecedent"} <= set(r.columns))
        p = pd.read_csv(ROOT / "reports/results/patch_impact_summary.csv")
        self.assertTrue({"eligible_patch_events", "games", "bootstrap_95_ci_low"} <= set(p.columns))


if __name__ == "__main__":
    unittest.main()
