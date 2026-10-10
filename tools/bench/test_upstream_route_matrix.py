"""Pure route A/B statistical and schema regressions; no CUDA/GPU required."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from tools.bench.run_upstream_route_matrix import (
    calculate, main, percentile, read_cases, streaming_sha256, summary_values,
)


class RouteMatrixContractTests(unittest.TestCase):
    def test_percentile_and_singleton(self):
        self.assertEqual(percentile([7.0], .95), 7.0)
        self.assertAlmostEqual(percentile([0.0, 100.0], .05), 5.0)
        self.assertEqual(summary_values([]), None)

    def test_reject_missing_baseline_or_non_env(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            path.write_text(json.dumps({"fast": {"NINFER_PROMPT_FAST": "1"}}))
            with self.assertRaises(ValueError):
                read_cases(path)
            path.write_text(json.dumps({"baseline": {"PATH": "invalid"}}))
            with self.assertRaises(ValueError):
                read_cases(path)

    def test_pair_math_and_memory(self):
        rows = []
        for pair in range(3):
            rows.append({"case": "baseline", "pair": pair, "warmup": False,
                         "metrics": {"4K": 100.0}, "resolved_chunk": 1024,
                         "runtime_reservation_bytes": 100, "workspace_capacity_bytes": 60})
            rows.append({"case": "fast", "pair": pair, "warmup": False,
                         "metrics": {"4K": 110.0}, "resolved_chunk": 1536,
                         "runtime_reservation_bytes": 110, "workspace_capacity_bytes": 65})
        report = calculate(rows, "fast", 3)
        self.assertAlmostEqual(report["tests"]["4K"]["paired_improvement_pct"]["median"], 10)
        self.assertTrue(report["performance_only_screen"])
        self.assertEqual(report["resolved_chunk_values"], ["1536"])
        self.assertEqual(report["runtime_reservation_peak_bytes"], 110)

    def test_multicase_runs_use_separate_baseline_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "bench"
            model = root / "model.ninfer"
            executable.write_bytes(b"benchmark")
            model.write_bytes(b"model")
            output = root / "out"

            def fake_run(command, **kwargs):
                if command[0] == "nvidia-smi":
                    return CompletedProcess(command, 0, "RTX 4090, 610, 24564 MiB\n", "")
                report = Path(command[command.index("--output-file") + 1])
                report.write_text(json.dumps({
                    "tests": [{"label": "1024", "prefill_tok_s_mean": 100.0}],
                    "config": {"prefill_chunk": 1024},
                    "memory": {"runtime_reservation_bytes": 10,
                               "workspace": {"capacity_bytes": 5}},
                }))
                return CompletedProcess(command, 0, "", "")

            argv = [
                "--exe", str(executable), "--model", str(model), "--out", str(output),
                "--device", "1", "--max-context", "1024", "--prompts", "1024",
                "--spec", "none", "--cases", "prompt_fast,gdn_two_stage,all_candidates",
                "--warmup", "1", "--pairs", "3",
            ]
            with patch("tools.bench.run_upstream_route_matrix.subprocess.run",
                       side_effect=fake_run):
                self.assertEqual(main(argv), 0)

            baseline_dirs = []
            for invocation in output.rglob("invocation.json"):
                record = json.loads(invocation.read_text())
                if record["case"] == "baseline":
                    baseline_dirs.append(invocation.parent)
            self.assertEqual(len(baseline_dirs), 12)
            self.assertEqual(len(set(baseline_dirs)), 12)
            candidate_routes = []
            for invocation in output.rglob("invocation.json"):
                record = json.loads(invocation.read_text())
                if record["case"] == "all_candidates":
                    candidate_routes.append(record["route_env"])
            self.assertEqual(len(candidate_routes), 4)
            expected_only = "attn_prompt_fast,gdn_two_stage/h32,gdn_two_stage/h48,t2_a16,prefill_align"
            self.assertTrue(all(route["NINFER_DEVICE_ROUTE_ONLY"] == expected_only
                                for route in candidate_routes))
            self.assertTrue((output / "summary.json").is_file())

    def test_missing_pair_not_silently_reused(self):
        rows = [{"case": "baseline", "pair": 0, "warmup": False,
                 "metrics": {"4K": 100}},
                {"case": "fast", "pair": 0, "warmup": False,
                 "metrics": {"4K": 110}}]
        with self.assertRaises(RuntimeError):
            calculate(rows, "fast", 3)

    def test_stream_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.bin"
            path.write_bytes(b"X" * (1024 * 1024 + 5))
            self.assertEqual(streaming_sha256(path),
                             hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
