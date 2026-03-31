import json
import subprocess
import textwrap
import unittest
from pathlib import Path


METRICS_AGGREGATION_JS_PATH = Path(__file__).resolve().parents[1] / "web" / "metrics_aggregation.js"


def _run_node_metrics_aggregation(items: list[dict]) -> dict:
    script = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const code = fs.readFileSync({json.dumps(str(METRICS_AGGREGATION_JS_PATH))}, "utf8");
        const context = {{}};
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(code, context);
        const items = {json.dumps(items)};
        const result = context.DHAIBenchMetricsAggregation.buildBalancedAggregate(items, {{
          getUnitKey: (item) => item.task,
          getMetricValue: (item) => item.metric,
          getPriceValue: (item) => item.price,
        }});
        process.stdout.write(JSON.stringify(result));
        """
    )
    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class WebMetricsAggregationTests(unittest.TestCase):
    def test_balanced_aggregate_averages_repeated_task_runs_before_group_mean(self) -> None:
        result = _run_node_metrics_aggregation(
            [
                {"task": "adv-ing", "metric": 90.0, "price": 4.0},
                {"task": "adv-ing", "metric": 70.0, "price": 8.0},
                {"task": "like_interrater", "metric": 60.0, "price": 10.0},
            ]
        )
        self.assertEqual(result["itemCount"], 3)
        self.assertEqual(result["unitCount"], 2)
        self.assertAlmostEqual(result["metricValue"], 70.0)

    def test_balanced_aggregate_averages_price_per_task_before_group_price(self) -> None:
        result = _run_node_metrics_aggregation(
            [
                {"task": "adv-ing", "metric": 80.0, "price": 4.0},
                {"task": "adv-ing", "metric": 80.0, "price": 8.0},
                {"task": "like_interrater", "metric": 80.0, "price": 10.0},
            ]
        )
        self.assertAlmostEqual(result["priceValue"], 8.0)
        self.assertEqual(result["pricedUnitCount"], 2)


if __name__ == "__main__":
    unittest.main()
