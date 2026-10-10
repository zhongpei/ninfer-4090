"""CPU-only contract tests for resident-weight AB/BA routing.

The integration fake is a real, long-running JSONL child (no GPU needed).
It deliberately fails if the runner starts more than one process/model owner.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.bench.run_upstream_route_matrix import (
    build_plan, calculate, first_divergent_token, main, percentile,
    primary_measure, read_cases, streaming_sha256, summary_values,
    validated_measurement,
)


class RouteMatrixContractTests(unittest.TestCase):
    def test_percentile_and_singleton(self):
        self.assertEqual(percentile([7.0], .95), 7.0)
        self.assertAlmostEqual(percentile([0.0, 100.0], .05), 5.0)
        self.assertIsNone(summary_values([]))

    def test_reject_invalid_case_and_missing_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            path.write_text(json.dumps({"fast": {"NINFER_PROMPT_FAST": "1"}}))
            with self.assertRaises(ValueError):
                read_cases(path)
            path.write_text(json.dumps({"baseline": {"PATH": "invalid"}}))
            with self.assertRaises(ValueError):
                read_cases(path)
            path.write_text(json.dumps({"baseline": {"NINFER_DEVICE_ROUTE_MODE": "builtin"}}))
            with self.assertRaises(ValueError):
                read_cases(path)

    def test_both_pair_orders_and_disjoint_ids(self):
        plan = build_plan(["prompt_fast", "gdn_two_stage_approx"], pairs=5, warmup=1)
        self.assertEqual(len(plan), 2 * (2 + 10))
        self.assertEqual(len({item["tag"] for item in plan}), len(plan))
        pairs = [p for p in plan if not p["warmup"] and p["experiment"] == "prompt_fast"]
        self.assertEqual([a["case"] for a in pairs[:4]],
                         ["baseline", "prompt_fast", "prompt_fast", "baseline"])
        self.assertTrue(all(p["experiment"] == p["case"] or p["case"] == "baseline"
                            for p in plan))

    def test_pair_math_memory_and_quality(self):
        rows = []
        for pair in range(3):
            for case, tps, bytes_ in [("baseline", 100, 100), ("fast", 110, 110)]:
                rows.append({"case": case, "pair": pair, "warmup": False,
                             "metrics": {"pp4096": float(tps)},
                             "output_token_hashes": {"pp4096": [12345]},
                             "prefix_hashes": {"pp4096": [[12345]]},
                             "objectives": {"pp4096": {"metric": "prefill_tok_s", "unit": "tokens/s"}},
                             "resolved_chunk": 1024,
                             "runtime_reservation_bytes": bytes_,
                             "workspace_capacity_bytes": 60})
        report = calculate(rows, "fast", 3)
        self.assertAlmostEqual(report["tests"]["pp4096"]["paired_improvement_pct"]["median"], 10)
        self.assertTrue(report["performance_only_screen"])
        self.assertEqual(report["runtime_reservation_peak_bytes"], 110)
        self.assertEqual(report["quality_gate"], "not_executed")
        self.assertEqual(report["output_token_gate"], "pass")
        rows[-1]["output_token_hashes"] = {"pp4096": [12346]}
        rows[-1]["prefix_hashes"] = {"pp4096": [[12346]]}
        divergent = calculate(rows, "fast", 3)
        self.assertEqual(divergent["output_token_gate"], "fail")
        self.assertEqual(divergent["tests"]["pp4096"]["first_divergent_token_by_pair"][-1], 0)

    def test_custom_named_approx_case_still_blocks_quality_gate(self):
        rows = []
        for pair in range(3):
            for name in ("baseline", "optimized"):
                rows.append({
                    "case": name, "pair": pair, "warmup": False,
                    "metrics": {"pp1024": 100.0 if name == "baseline" else 110.0},
                    "output_token_hashes": {"pp1024": [777]},
                    "prefix_hashes": {"pp1024": [[777]]},
                    "objectives": {"pp1024": {"metric": "prefill_tok_s", "unit": "tokens/s"}},
                    "route_env": ({
                        "NINFER_GDN_TWO_STAGE_NUMERICS": "approx"
                    } if name == "optimized" else {
                        "NINFER_DEVICE_ROUTE_MODE": "off"
                    }),
                    "resolved_chunk": 1024, "runtime_reservation_bytes": 1,
                    "workspace_capacity_bytes": 1,
                })
        result = calculate(rows, "optimized", 3)
        self.assertEqual(result["output_token_gate"], "pass")
        self.assertEqual(result["quality_gate"], "blocked_approx_state_semantics")

    def test_full_request_objective_not_prefill_only(self):
        base = {"kind": "pp+tg", "label": "pp4096+tg256",
                "prefill_tok_s_mean": 2000.0, "total_seconds_mean": 3.0}
        candidate = {"kind": "pp+tg", "label": "pp4096+tg256",
                     "prefill_tok_s_mean": 2200.0, "total_seconds_mean": 3.5}
        old_prefill_only_gain = candidate["prefill_tok_s_mean"] / base["prefill_tok_s_mean"] - 1
        self.assertGreater(old_prefill_only_gain, 0)
        baseline_rate, metric, unit = primary_measure(base)
        candidate_rate, _, _ = primary_measure(candidate)
        self.assertEqual((metric, unit), ("full_request_rate", "requests/s"))
        self.assertLess(candidate_rate / baseline_rate - 1, 0)
        self.assertEqual(primary_measure({"kind": "pp", "label": "pp4096",
                                         "prefill_tok_s_mean": 1100})[0], 1100)

    def test_earliest_divergence_fingerprint(self):
        self.assertEqual(first_divergent_token([[1, 2, 3]], [[1, 9, 10]]), 1)
        self.assertEqual(first_divergent_token([[1, 2, 3]], [[1, 2]]), 2)
        self.assertIsNone(first_divergent_token([[1, 2]], [[1, 2]]))

    def test_exact_prefetch_case_has_explicit_isolated_flag(self):
        from tools.bench.run_upstream_route_matrix import DEFAULT_CASES, CLEAN_ENV
        arm = DEFAULT_CASES["gdn_exact_prefetch"]
        self.assertEqual(arm["NINFER_DEVICE_ROUTE_MODE"], "off")
        self.assertEqual(arm["NINFER_GDN_EXACT_PREFETCH"], "1")
        self.assertIn("NINFER_GDN_EXACT_PREFETCH", CLEAN_ENV)

    def test_missing_pair_and_rejected_fake_load_count(self):
        rows = [{"case": "baseline", "pair": 0, "warmup": False,
                 "metrics": {"pp4096": 100},
                 "output_token_hashes": {"pp4096": [1]}},
                {"case": "fast", "pair": 0, "warmup": False,
                 "metrics": {"pp4096": 110},
                 "output_token_hashes": {"pp4096": [1]}}]
        with self.assertRaises(RuntimeError):
            calculate(rows, "fast", 3)
        with self.assertRaises(RuntimeError):
            validated_measurement({
                "id": "foo", "event": "measurement", "ok": True,
                "model_load_count": 2, "report": {}
            }, "foo")

    def test_reject_artifact_reloads_even_with_load_count_one(self):
        import tools.bench.run_upstream_route_matrix as route
        base = {
            "event": "measurement", "id": "x", "ok": True, "model_load_count": 1,
            "report": {
                "residency": {"model_load_count": 1, "program_create_seconds": 0.1,
                              "artifact_bytes_read_this_arm": 4096,
                              "weight_bytes_uploaded_this_arm": 0},
                "tests": [{"label": "pp1024", "prefill_tok_s_mean": 100}],
                "generated_token_hashes": {"pp1024": [42]},
            },
        }
        with self.assertRaises(RuntimeError):
            route.validated_measurement(base, "x")

    def test_real_persistent_protocol_fake(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "fake_resident_bench"
            model = root / "model.ninfer"
            model.write_bytes(b"artifact")
            startup_log = root / "model-load-count.txt"
            exe.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["FAKE_ROUTE_STARTUPS"], "a", encoding="utf8") as f:
    f.write("1\\n")
print(json.dumps({"event": "ready", "ok": True, "model_load_count": 1,
                  "resident_weight_bytes": 123}), flush=True)
for line in sys.stdin:
    packet = json.loads(line)
    if packet.get("stop"):
        print(json.dumps({"event": "bye", "id": packet["id"], "ok": True,
                          "model_load_count": 1}), flush=True)
        break
    if (os.environ.get("FAKE_ROUTE_FAIL_CASE") and
            packet["id"].endswith("-" + os.environ["FAKE_ROUTE_FAIL_CASE"])):
        print(json.dumps({"event": "error", "id": packet["id"], "ok": False,
                          "error": "simulated illegal memory access"}), flush=True)
        break
    route = packet["route_env"]
    score = 100.0 if route.get("NINFER_DEVICE_ROUTE_MODE") == "off" else 110.0
    data = {"tests": [{"label": "pp1024", "prefill_tok_s_mean": score}],
            "config": {"prefill_chunk": 1024},
            "memory": {"runtime_reservation_bytes": 1000,
                       "workspace": {"capacity_bytes": 200}},
            "residency": {"model_load_count": 1, "program_create_seconds": 0.001,
                          "artifact_bytes_read_this_arm": 0,
                          "weight_bytes_uploaded_this_arm": 0},
            "generated_token_hashes": {"pp1024": [12345]},
            "generated_token_prefix_hashes": {"pp1024": [[12345]]}}
    print(json.dumps({"event": "measurement", "id": packet["id"], "ok": True,
                      "model_load_count": 1, "report": data}), flush=True)
""", encoding="utf-8")
            exe.chmod(0o755)
            out = root / "out"
            argv = ["--exe", str(exe), "--model", str(model), "--out", str(out),
                    "--max-context", "1024", "--prompts", "1024",
                    "--cases", "prompt_fast,gdn_two_stage_approx,all_candidates",
                    "--warmup", "1", "--pairs", "3", "--timeout", "10"]
            with patch.dict(os.environ, {"FAKE_ROUTE_STARTUPS": str(startup_log)}):
                self.assertEqual(main(argv), 0)
            # The old runner spawned (pairs+warmup) x 2 x cases models. New runner: exactly one.
            self.assertEqual(startup_log.read_text().splitlines(), ["1"])
            self.assertEqual(json.loads((out / "summary.json").read_text())["model_load_count"], 1)
            self.assertEqual(json.loads((out / "summary.json").read_text())["program_creation_count"], 24)
            self.assertEqual(len((out / "records.jsonl").read_text().splitlines()), 24)
            records = [json.loads(p.read_text()) for p in out.rglob("invocation.json")]
            self.assertEqual(len(records), 24)
            baselines = [r for r in records if r["case"] == "baseline"]
            self.assertEqual(len(baselines), 12)
            self.assertEqual(len({r["tag"] for r in baselines}), len(baselines))
            combinations = [r["route_env"] for r in records
                            if r["case"] == "all_candidates"]
            self.assertEqual(len(combinations), 4)
            only = "attn_prompt_fast,gdn_two_stage/h32,gdn_two_stage/h48,t2_a16,prefill_align"
            self.assertTrue(all(r["NINFER_DEVICE_ROUTE_ONLY"] == only for r in combinations))
            approx = json.loads((out / "summary-gdn_two_stage_approx.json").read_text())
            self.assertEqual(approx["quality_gate"], "blocked_approx_state_semantics")
            self.assertTrue((out / "summary.md").is_file())

    def test_failed_later_candidate_keeps_completed_pair_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "fake-resident"
            model = root / "model.ninfer"
            model.write_bytes(b"model")
            exe.write_text("""#!/usr/bin/env python3
import json, os, sys
print(json.dumps({"event": "ready", "ok": True, "model_load_count": 1}), flush=True)
for line in sys.stdin:
    packet = json.loads(line)
    if packet.get("stop"):
        print(json.dumps({"event": "bye", "id": packet["id"], "ok": True,
                          "model_load_count": 1}), flush=True)
        break
    if packet["id"].endswith("-gdn_two_stage_approx"):
        print(json.dumps({"event": "error", "id": packet["id"], "ok": False,
                          "error": "simulated CUDA failure"}), flush=True)
        break
    data = {"tests": [{"label": "pp1024", "prefill_tok_s_mean": 100.0}],
            "config": {"prefill_chunk": 1024},
            "memory": {"runtime_reservation_bytes": 1,
                       "workspace": {"capacity_bytes": 1}},
            "residency": {"model_load_count": 1, "program_create_seconds": 0.01,
                          "artifact_bytes_read_this_arm": 0,
                          "weight_bytes_uploaded_this_arm": 0},
            "generated_token_hashes": {"pp1024": [777]},
            "generated_token_prefix_hashes": {"pp1024": [[777]]}}
    print(json.dumps({"event": "measurement", "id": packet["id"], "ok": True,
                      "model_load_count": 1, "report": data}), flush=True)
""", encoding="utf-8")
            exe.chmod(0o755)
            out = root / "campaign"
            with self.assertRaises(RuntimeError):
                main(["--exe", str(exe), "--model", str(model), "--out", str(out),
                      "--cases", "prompt_fast,gdn_two_stage_approx",
                      "--pairs", "3", "--warmup", "0", "--prompts", "1024",
                      "--max-context", "1024", "--timeout", "10"])
            self.assertTrue((out / "summary-prompt_fast.json").is_file())
            self.assertEqual(
                len((out / "records.jsonl").read_text().splitlines()), 7)
            self.assertTrue((out / "ERROR.txt").is_file())
            self.assertFalse((out / "summary-gdn_two_stage_approx.json").exists())

    def test_stream_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            path.write_bytes(b"z" * (1024 * 1024 + 5))
            self.assertEqual(streaming_sha256(path),
                             hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
