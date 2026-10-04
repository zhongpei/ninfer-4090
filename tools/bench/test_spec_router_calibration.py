"""Focused host regressions for pairing, correctness and actual batch accounting."""
import copy
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bench.run_spec_router_calibration import (
    frontier_bucket, main, metrics, percentile, qualify_pairs, validate_report,
)


def report(k=0, wall=1000, latency=1000, actual_batch=1):
    result = {
        "artifact_type": "ninfer_calibration_measurement", "schema_version": 1,
        "model": "/explicit/model.ninfer", "device": 0, "client_concurrency": 2,
        "engine_concurrency": 8, "draft_tokens": k, "max_context": 32768,
        "proposal_head": "full", "repeats": 2,
        "kv_capacity": 32768, "max_tokens": 512, "kv": "int8", "graph": True,
        "cache": True, "greedy": True, "presence_penalty": 0, "frequency_penalty": 0,
        "wave_wall_ns": [wall, wall],
        "rounds": [{"active_batch": actual_batch, "max_execution_frontier": 8193,
                    "elapsed_ns": 100, "committed_tokens": 2,
                    "draft_tokens": k, "verify_width": k + 1,
                    "proposal_width": k + 1 if k else 0,
                    "backend": 3 if k else 0, "neural_drafter_executed": bool(k)}],
        "requests": [],
    }
    for repeat in range(2):
        for slot in range(2):
            result["requests"].append({
                "repeat": repeat, "slot": slot, "latency_ns": latency,
                "generated_token_ids": [42, 7], "content": " x\n", "reasoning": "",
                "tool_calls": [], "finish_reason": 1, "matched_stop_string": None,
            })
    return result


class CalibrationTest(unittest.TestCase):
    def test_actual_batch_is_not_client_concurrency(self):
        cell = metrics(report(actual_batch=1))["actual_batch_frontier_cells"][0]
        self.assertEqual(cell["active_batch"], 1)
        self.assertEqual(cell["frontier_bucket"], "8193-32768")
        self.assertEqual(cell["round_tokens_per_second"], 20_000_000)

    def test_all_frontier_boundaries(self):
        self.assertEqual([frontier_bucket(x) for x in (1, 1024, 1025, 8192, 8193, 32768)],
                         ["1-1024"] * 2 + ["1025-8192"] * 2 + ["8193-32768"] * 2)
        for x in (0, 32769):
            with self.assertRaises(ValueError):
                frontier_bucket(x)

    def test_pairing_survives_large_time_drift(self):
        result = qualify_pairs([(report(wall=1000), report(7, wall=900)),
                                (report(wall=10000), report(7, wall=9000))])
        self.assertAlmostEqual(result["median_speedup"], 10 / 9)
        self.assertTrue(result["qualified_performance"])

    def test_full_whitespace_and_token_equality_gate(self):
        for field, value in (("content", "x\n"), ("generated_token_ids", [42, 8]),
                             ("reasoning", "hidden"), ("finish_reason", 3),
                             ("tool_calls", [{"name": "tool", "arguments_json": "{}"}])):
            candidate = report(7, wall=900)
            candidate["requests"][0][field] = value
            result = qualify_pairs([(report(), candidate)])
            self.assertFalse(result["correct"])
            self.assertIsNone(result["median_speedup"])
            self.assertFalse(result["qualified_performance"])

    def test_every_pair_must_gain_and_latency_limit(self):
        self.assertFalse(qualify_pairs([(report(), report(7, wall=900)),
                                        (report(), report(7, wall=1001))])["qualified_performance"])
        self.assertFalse(qualify_pairs([(report(), report(7, wall=900, latency=1051))])
                         ["qualified_performance"])

    def test_missing_duplicate_failed_or_empty_measurements_rejected(self):
        for mutation in (
                lambda r: r["requests"].pop(),
                lambda r: r["requests"].append(copy.deepcopy(r["requests"][0])),
                lambda r: r["requests"][0].update(finish_reason=5),
                lambda r: r.update(rounds=[]),
                lambda r: r.update(cache=False)):
            value = report()
            mutation(value)
            with self.assertRaises(ValueError):
                validate_report(value)

    def test_percentiles_are_request_metrics(self):
        self.assertEqual(percentile([1, 3], 50), 2)
        self.assertAlmostEqual(percentile([1, 3], 95), 2.9)

    def test_cross_pair_baseline_instability_is_rejected(self):
        baseline2, candidate2 = report(wall=10000), report(7, wall=9000)
        for arm in (baseline2, candidate2):
            for request in arm["requests"]:
                request["generated_token_ids"] = [99, 7]
        result = qualify_pairs([(report(), report(7, wall=900)), (baseline2, candidate2)])
        self.assertFalse(result["correct"])
        self.assertFalse(result["qualified_performance"])

    def test_round_percentiles_and_action_are_per_actual_cell(self):
        value = report(7)
        second = copy.deepcopy(value["rounds"][0])
        second["elapsed_ns"] = 300
        value["rounds"].append(second)
        cell = metrics(value)["actual_batch_frontier_cells"][0]
        self.assertEqual(cell["round_p50_ns"], 200)
        self.assertEqual(cell["round_p95_ns"], 290)
        self.assertEqual((cell["draft_tokens"], cell["verify_width"], cell["proposal_width"]), (7, 8, 8))

    def test_wrong_observer_action_width_proposal_or_skip_is_rejected(self):
        for k in (0, 7):
            for field, value in (("draft_tokens", 11), ("verify_width", 16),
                                 ("proposal_width", 16), ("backend", 1),
                                 ("neural_drafter_executed", not bool(k))):
                r = report(k)
                r["rounds"][0][field] = value
                with self.assertRaises(ValueError, msg=f"{k}: {field}"):
                    validate_report(r)

    def test_repeats_and_proposal_head_are_verified(self):
        for field, value in (("repeats", 3), ("proposal_head", "optimized")):
            r = report()
            r[field] = value
            with self.assertRaises(ValueError):
                validate_report(r)

    def test_context_and_capacity_cli_forwarding(self):
        for flags, context, capacity in (
                ([], 8192, 8192),
                (["--max-context", "32768", "--kv-capacity", "262144"], 32768, 262144)):
            commands = []

            def measurement(command, directory, gpu, interval):
                commands.append(command)
                action = int(command[command.index("--draft-tokens") + 1])
                value = report(action, wall=900 if action else 1000)
                value.update(max_context=context, kv_capacity=capacity)
                return value

            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                exe, model = root / "exe", root / "model.ninfer"
                exe.touch()
                model.touch()
                argv = ["--exe", str(exe), "--model", str(model), "--out", str(root / "out"),
                        "--workloads", "chat", "--concurrency", "2", "--draft-tokens", "7",
                        "--nvml-device", "GPU-explicit", "--cooldown", "0", *flags]
                with patch("tools.bench.run_spec_router_calibration.run_measurement", side_effect=measurement), \
                     patch("tools.bench.run_spec_router_calibration.subprocess.run",
                           return_value=SimpleNamespace(stdout="hardware")):
                    self.assertEqual(main(argv), 0)
            self.assertEqual(len(commands), 4)
            self.assertEqual([c[c.index("--draft-tokens") + 1] for c in commands], ["0", "7", "7", "0"])
            for command in commands:
                self.assertEqual(command[command.index("--max-context") + 1], str(context))
                self.assertEqual(command[command.index("--kv-capacity") + 1], str(capacity))

    def test_unpaired_context_rejected(self):
        candidate = report(7)
        candidate["max_context"] = 8192
        with self.assertRaises(ValueError):
            qualify_pairs([(report(), candidate)])


if __name__ == "__main__":
    unittest.main()
