"""Exercise the browser calculation module through Node, like pricing tests."""
import csv
import io
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_js(expression):
    script = f"require({json.dumps(str(ROOT / 'web/custom_leaderboard.js'))}); const api = globalThis.DHAIBenchCustomLeaderboard; console.log(JSON.stringify({expression}));"
    return json.loads(subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout)


def record(task, score, key="model", cost=1, predictions=100, **kwargs):
    return dict(task=task, metric=score, key=key, label=key, cost=cost, predictions=predictions, path=f"{key}-{task}.json", **kwargs)


def calculate(records, tasks=("A", "B"), weights=None):
    settings = {"weights": {key: {"weight": value} for key, value in (weights or {}).items()}}
    return run_js(f"api.calculate({json.dumps(records)}, {json.dumps(tasks)}, {json.dumps(settings)})")


class CustomLeaderboardTests(unittest.TestCase):
    def test_weighted_score_and_cost_balance_repeats_first(self):
        result = calculate([record("A", 100, cost=1), record("A", 80, cost=3), record("B", 70, cost=4)], weights={"A": 3, "B": 1})
        row = result["rows"][0]
        self.assertEqual(row["score"], 85)
        self.assertEqual(row["cost"], 25)
        self.assertEqual(row["breakdown"][0]["contribution"], 67.5)

    def test_missing_task_is_unranked_without_renormalization(self):
        rows = calculate([record("A", 100, "partial"), record("A", 80), record("B", 80)], weights={"A": 3, "B": 1})["rows"]
        self.assertEqual(rows[0]["rank"], 1)
        self.assertIsNone(rows[1]["score"])
        self.assertIsNone(rows[1]["rank"])
        self.assertEqual(rows[1]["coverage"], .75)

    def test_missing_or_zero_prediction_counts_make_cost_unknown(self):
        for kwargs in [{"predictions": None}, {"predictions": 0}, {"cost": None}]:
            with self.subTest(kwargs=kwargs):
                row = calculate([record("A", 90, **kwargs), record("A", 80), record("B", 70)])["rows"][0]
                self.assertTrue(row["complete"])
                self.assertIsNone(row["cost"])

    def test_zero_cost_is_a_known_price(self):
        row = calculate([record("A", 90, cost=0)], tasks=["A"])["rows"][0]
        self.assertEqual(row["cost"], 0)

    def test_cost_averages_run_rates_not_total_costs(self):
        row = calculate([record("A", 90, cost=1, predictions=100), record("A", 90, cost=10, predictions=1000)], tasks=["A"])["rows"][0]
        self.assertEqual(row["cost"], 10)

    def test_zero_weight_tasks_do_not_require_coverage(self):
        row = calculate([record("A", 90)], weights={"A": 1, "B": 0})["rows"][0]
        self.assertTrue(row["complete"])
        self.assertEqual(row["score"], 90)
        self.assertEqual(calculate([], weights={"A": 0, "B": 0})["tasks"], [])

    def test_disabling_task_retains_its_weight(self):
        value = run_js('api.normalizeSettings({weights: {A: {weight: 3, enabled: false}}})')
        self.assertEqual(value["weights"]["A"], {"weight": 3, "enabled": False})

    def test_extreme_weights_are_normalized_without_overflow(self):
        result = calculate([record("A", 90), record("B", 70)], weights={"A": 1e308, "B": 1e308})
        self.assertEqual(result["rows"][0]["score"], 80)

    def test_known_partial_runs_do_not_contribute(self):
        result = calculate([record("A", 100, partial=True), record("A", 80), record("B", 70)])
        row = result["rows"][0]
        self.assertEqual(row["score"], 75)
        self.assertEqual(len(row["breakdown"][0]["excludedRuns"]), 1)

    def test_conflicting_known_datasets_are_not_averaged(self):
        row = calculate([record("A", 90, dataset="one.csv"), record("A", 70, dataset="two.csv")], tasks=["A"])["rows"][0]
        self.assertFalse(row["complete"])
        self.assertIn("Multiple datasets", row["breakdown"][0]["reason"])

    def test_other_models_datasets_do_not_remove_task_coverage(self):
        rows = calculate([
            record("A", 90, "kimi-k3", dataset="normalization3.csv", protocol="same"),
            record("A", 92, "kimi-k3", dataset="normalization3.csv", protocol="same"),
            record("B", 96, "kimi-k3", dataset="POS_2.csv"),
            record("A", 75, "other", dataset="normalization.csv"),
            record("B", 80, "other", dataset="POS_2_s.csv"),
        ])["rows"]
        kimi = next(row for row in rows if row["key"] == "kimi-k3")
        self.assertTrue(kimi["complete"])
        self.assertEqual(kimi["coveredTasks"], 2)
        self.assertEqual(kimi["score"], 93.5)
        self.assertEqual(kimi["rank"], 1)

    def test_excluded_runs_cannot_create_dataset_conflicts(self):
        row = calculate([
            record("A", 90, dataset="current.csv"),
            record("A", 100, dataset="stopped.csv", partial=True),
            record("A", None, dataset="unscored.csv"),
        ], tasks=["A"])["rows"][0]
        self.assertTrue(row["complete"])
        self.assertEqual(row["score"], 90)

    def test_same_model_groups_across_providers_and_settings(self):
        identities = run_js('[{provider:"openai",model:"m"}, {provider:"other",model:"m"}, {provider:"openai",model:"m",runConfig:{reasoning_effort:"high"}}].map(api.modelIdentity)')
        self.assertEqual(len({item["key"] for item in identities}), 1)
        self.assertEqual(len({item["runLabel"] for item in identities}), 3)

    def test_provider_runs_are_averaged_within_tasks_before_weighting(self):
        result = run_js('api.calculate([{...api.modelIdentity({model:"m",provider:"p1"}),task:"A",metric:100,cost:1,predictions:100}, {...api.modelIdentity({model:"m",provider:"p2"}),task:"A",metric:80,cost:3,predictions:100}, {...api.modelIdentity({model:"m",provider:"p2"}),task:"B",metric:70,cost:4,predictions:100}], ["A","B"], {weights:{A:{weight:3},B:{weight:1}}})')
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["score"], 85)
        self.assertEqual(row["cost"], 25)
        self.assertEqual(row["providers"], ["p1", "p2"])
        self.assertTrue(row["complete"])

    def test_different_known_system_prompts_are_not_averaged(self):
        row = calculate([record("A", 90, protocol="first prompt"), record("A", 70, protocol="second prompt")], tasks=["A"])["rows"][0]
        self.assertFalse(row["complete"])
        self.assertIn("prompts", row["breakdown"][0]["reason"])

    def test_ties_share_rank_and_order_is_deterministic(self):
        rows = calculate([record("A", 90, "z"), record("A", 90, "a"), record("A", 80, "b")], tasks=["A"])["rows"]
        self.assertEqual([row["key"] for row in rows], ["a", "z", "b"])
        self.assertEqual([row["rank"] for row in rows], [1, 1, 3])

    def test_invalid_settings_are_sanitized(self):
        value = run_js('api.normalizeSettings({metric:"bad", weights:{A:{weight:-1}, B:{weight:Infinity}, C:{weight:"2"}, D:{weight:0}}})')
        self.assertEqual(value["metric"], "accuracy")
        self.assertEqual(list(value["weights"]), ["D"])

    def test_csv_contains_contributions_provenance_and_escaped_labels(self):
        records = [record("A", 90, '=SUM(1,2)\n"model"', samples=100)]
        text = run_js(f'api.csv(api.calculate({json.dumps(records)}, ["A"], {{}}), "2026-09-09", "https://example.test/#tab=custom")')
        row = list(csv.DictReader(io.StringIO(text)))[0]
        self.assertEqual(row["Model"], '\'=SUM(1,2)\n"model"')
        self.assertEqual(row["Contribution (percentage points)"], "90")
        self.assertEqual(row["Pricing updated"], "2026-09-09")
        self.assertEqual(row["Evaluated examples by run"], "100")


if __name__ == "__main__":
    unittest.main()
