"""Mocked native executables and memory queries; no model inference or GPU execution."""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bench.run_resident_spec_router_calibration import main
from tools.bench.test_calibrated_router_profile import identity, record
from tools.bench.calibrated_router_profile import REQUEST_FIELDS, build_profile


class FakeNative:
    def __init__(self):
        self.commands = []
        self.failure_action = None
        self.mutation = None
        self.seed_mutation = None
        self.actual_batch_one = False

    def run(self, command, **kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        self.commands.append(command)
        options = {command[i]: command[i + 1] for i in range(1, len(command) - 1)
                   if command[i].startswith("--") and not command[i + 1].startswith("--")}
        resolved = identity()
        resolved.update(max_context=int(options["--max-context"]),
                        resolved_kv_capacity=int(options["--kv-capacity"]),
                        max_concurrency=int(options["--engine-concurrency"]))
        output = Path(options["--output"])
        if "--identity-only" in command:
            payload = {"artifact_type": "ninfer_routing_identity", "schema_version": 1,
                       "identity": resolved}
            if self.seed_mutation:
                self.seed_mutation(payload)
        else:
            action = int(options["--draft-tokens"])
            if action == self.failure_action:
                return SimpleNamespace(returncode=4, stdout="", stderr="native failure")
            control = json.loads(Path(options["--spec-router-profile"]).read_text())
            assert control["identity"] == resolved
            assert len(control["cells"]) == 24
            assert all(cell["draft_tokens"] == action for cell in control["cells"])
            measured = record(action, wall=9000 if action else 10000)
            measured["identity"] = resolved
            measured["configuration"].update(
                prompt=Path(options["--prompt-file"]).read_bytes().decode("utf-8"),
                max_tokens=int(options["--max-tokens"]),
                client_concurrency=int(options["--client-concurrency"]),
                repeats=int(options["--repeats"]))
            clients = measured["configuration"]["client_concurrency"]
            measured["requests"] = [dict(copy.deepcopy(measured["requests"][0]), repeat=repeat, slot=slot)
                                    for repeat in range(measured["configuration"]["repeats"])
                                    for slot in range(clients)]
            for event in measured["rounds"]:
                event.update(active_batch=clients, committed_tokens=clients)
            if self.actual_batch_one:
                split = []
                for event in measured["rounds"]:
                    for half in range(clients):
                        e = copy.deepcopy(event)
                        e.update(round_index=len(split), active_batch=1, committed_tokens=1,
                                 max_execution_frontier=event["max_execution_frontier"] + half * 100,
                                 elapsed_ns=event["elapsed_ns"] // clients + (event["elapsed_ns"] % clients if half == 0 else 0))
                        split.append(e)
                measured["rounds"] = split
            priming = {"enabled": "--prime-prefix" in command}
            if priming["enabled"]:
                request = copy.deepcopy(measured["requests"][0])
                request["generated_token_ids"] = list(range(16))
                request["latency_ns"] = 10000
                events = []
                for i in range(15):
                    events.append({"round_index": i, "active_batch": 1,
                                   "max_execution_frontier": 64 + i,
                                   "draft_tokens": action, "verify_width": action + 1,
                                   "proposal_width": 16 if action else 0, "backend": 3,
                                   "neural_drafter_executed": bool(action),
                                   "committed_tokens": 1, "elapsed_ns": 1000})
                priming.update(requested_output_tokens=16, request=request, rounds=events,
                               wave_wall_ns=16000, settled_committed_tokens=15)
            payload = {
                "artifact_type": "ninfer_resident_router_measurement", "schema_version": 1,
                "record": measured,
                "resources": {
                    "memory": {"workspace_logical_peak_bytes": 185393152,
                               "workspace_allocator_peak_bytes": 185393152,
                               "runtime_reservation_bytes": 7869881856,
                               "cuda_graph_definition_count": 2, "cuda_graph_executable_count": 2},
                    "routing_counters_scope": "published_engine_snapshot_including_priming",
                    "runtime_stats": {}},
                "priming": priming,
            }
            if self.mutation:
                self.mutation(payload, action)
        output.write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class DriverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.exe, self.model = self.root / "fake-native", self.root / "model.ninfer"
        self.exe.touch()
        self.model.touch()
        self.fake = FakeNative()

    def tearDown(self):
        self.temp.cleanup()

    def argv(self, *extra):
        return ["--exe", str(self.exe), "--model", str(self.model),
                "--out", str(self.root / "out"), "--workloads", "chat",
                "--workload-label-prefix", "test-suite",
                "--concurrency", "2", "--draft-tokens", "7", "--max-tokens", "8",
                "--max-context", "32768", "--kv-capacity", "262144",
                "--nvml-device", "GPU-explicit", "--cooldown", "0", *extra]

    def run_driver(self, *extra):
        with patch("tools.bench.run_resident_spec_router_calibration.subprocess.run",
                   side_effect=self.fake.run):
            return main(self.argv(*extra))

    def test_identity_profiles_abba_complete_reports_and_explicit_capacities(self):
        self.assertEqual(self.run_driver("--no-prime-prefix"), 0)
        commands = self.fake.commands
        self.assertEqual(len(commands), 5)
        self.assertIn("--identity-only", commands[0])
        actions = [int(c[c.index("--draft-tokens") + 1]) for c in commands[1:]]
        self.assertEqual(actions, [0, 7, 7, 0])
        for command in commands:
            self.assertEqual(command[command.index("--kv-capacity") + 1], "262144")
            self.assertEqual(command[command.index("--max-context") + 1], "32768")
            self.assertNotIn("--prime-prefix", command)
        out = self.root / "out"
        controls = list((out / "controls").glob("*.json"))
        self.assertEqual(len(controls), 4)
        for path in controls:
            control = json.loads(path.read_text())
            self.assertEqual(control["identity"]["startup_draft_tokens"], 15)
            self.assertFalse(control["provenance"]["qualified"])
        doc = json.loads((out / "measurements.json").read_text())
        self.assertEqual(doc["comparisons"][0]["pairs"][0]["baseline"]["requested_action"], 0)
        self.assertEqual(doc["comparisons"][0]["pairs"][0]["baseline"]["rounds"][0]["backend"], 3)
        summary = json.loads((out / "qualification-summary.json").read_text())
        self.assertEqual(summary["cells"][3]["draft_tokens"], 7)
        self.assertEqual(len(list((out / "runs").glob("*/native.json"))), 4)
        for path in (out / "runs").glob("*/memory.json"):
            memory = json.loads(path.read_text())
            self.assertEqual(memory["kind"], "device_memory_observed_lower_bound")
            self.assertEqual(memory["gpu"], "GPU-explicit")
            self.assertEqual(memory["peak_used_mib"], 12345)

    def test_prompt_variant_bytes_saved_and_not_used_as_cell_estimates(self):
        directory = self.root / "prompts"
        directory.mkdir()
        original = "文档变体\r\n  whitespace preserved  \r\n"
        (directory / "chat.txt").write_bytes(original.encode("utf-8"))
        self.fake.actual_batch_one = True
        self.assertEqual(self.run_driver("--no-prime-prefix", "--prompt-dir", str(directory)), 0)
        out = self.root / "out"
        self.assertEqual((out / "prompts" / "chat.txt").read_bytes(), original.encode("utf-8"))
        doc = json.loads((out / "measurements.json").read_text())
        self.assertEqual(doc["comparisons"][0]["pairs"][0]["baseline"]["configuration"]["prompt"], original)
        summary = json.loads((out / "qualification-summary.json").read_text())
        chosen = {(c["active_batch"], c["frontier_upper"]): c["draft_tokens"] for c in summary["cells"]}
        self.assertEqual(chosen[1, 1024], 7)
        self.assertEqual(chosen[2, 1024], 0)

    def test_priming_is_explicit_and_audited_outside_record(self):
        self.assertEqual(self.run_driver("--prime-prefix"), 0)
        for command in self.fake.commands[1:]:
            self.assertIn("--prime-prefix", command)
        doc = json.loads((self.root / "out" / "measurements.json").read_text())
        self.assertNotIn("priming", doc["comparisons"][0]["pairs"][0]["baseline"])
        for path in (self.root / "out" / "runs").glob("*/native.json"):
            wrapper = json.loads(path.read_text())
            self.assertEqual(wrapper["priming"]["settled_committed_tokens"], 15)
            self.assertEqual(len(wrapper["priming"]["request"]["generated_token_ids"]), 16)

    def test_choice_of_priming_is_required(self):
        with self.assertRaises(SystemExit):
            self.run_driver()

    def test_native_failure_is_nonzero_and_retains_commands_and_memory(self):
        self.fake.failure_action = 7
        self.assertEqual(self.run_driver("--no-prime-prefix"), 1)
        out = self.root / "out"
        self.assertTrue((out / "failure.json").exists())
        self.assertGreaterEqual(len(list((out / "runs").glob("*/command.json"))), 2)
        self.assertGreaterEqual(len(list((out / "runs").glob("*/memory.json"))), 2)

    def test_mismatched_identity_wrong_resident_and_action_fallback_rejected(self):
        mutations = [
            lambda p, a: p["record"]["identity"].update(startup_draft_tokens=7),
            lambda p, a: p["record"]["identity"].update(hardware_class="other"),
            lambda p, a: p["record"]["rounds"][0].update(
                draft_tokens=0, verify_width=1, proposal_width=0, neural_drafter_executed=False) if a else None,
        ]
        for i, mutation in enumerate(mutations):
            self.fake.mutation = mutation
            argv = self.argv("--no-prime-prefix")
            argv[argv.index("--out") + 1] = str(self.root / f"bad-{i}")
            with patch("tools.bench.run_resident_spec_router_calibration.subprocess.run", side_effect=self.fake.run):
                self.assertEqual(main(argv), 1)

    def test_full_output_mismatch_is_not_qualified(self):
        self.fake.mutation = lambda p, a: p["record"]["requests"][0].update(content="changed") if a else None
        self.assertEqual(self.run_driver("--no-prime-prefix"), 1)
        summary = json.loads((self.root / "out" / "qualification-summary.json").read_text())
        self.assertTrue(all(c["draft_tokens"] == 0 for c in summary["cells"]))

    def test_seed_identity_missing_wrong_resident_or_startup_configuration_rejected(self):
        mutations = [
            lambda p: p.pop("identity"),
            lambda p: p["identity"].update(startup_draft_tokens=7),
            lambda p: p["identity"].update(max_context=8192),
            lambda p: p["identity"].update(resolved_kv_capacity=32768),
            lambda p: p["identity"].update(proposal_head="optimized"),
            lambda p: p["identity"].update(unrecognized=True),
        ]
        for i, mutation in enumerate(mutations):
            with self.subTest(case=i):
                self.fake.seed_mutation = mutation
                argv = self.argv("--no-prime-prefix")
                out = self.root / f"bad-seed-{i}"
                argv[argv.index("--out") + 1] = str(out)
                self.fake.commands.clear()
                with patch("tools.bench.run_resident_spec_router_calibration.subprocess.run", side_effect=self.fake.run):
                    self.assertEqual(main(argv), 1)
                self.assertEqual(len(self.fake.commands), 1)
                self.assertTrue((out / "failure.json").exists())
                self.assertFalse((out / "controls").exists())

    def test_nonfinite_timing_options_rejected_before_native_execution(self):
        for option in ("--cooldown", "--memory-sample-seconds"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(option=option, value=value):
                    argv = self.argv("--no-prime-prefix", option + "=" + value)
                    out = self.root / (option[2:] + "-" + value)
                    argv[argv.index("--out") + 1] = str(out)
                    self.fake.commands.clear()
                    with patch("tools.bench.run_resident_spec_router_calibration.subprocess.run", side_effect=self.fake.run):
                        with self.assertRaises(SystemExit):
                            main(argv)
                    self.assertEqual(self.fake.commands, [])
                    self.assertFalse(out.exists())

    def test_suite_and_requested_concurrency_labels_merge_without_actual_cell_collision(self):
        self.fake.actual_batch_one = True
        documents = []
        for suite in ("short", "long"):
            directory = self.root / suite
            directory.mkdir()
            prompt = (suite + " prompt\r\n").encode("utf-8")
            (directory / "chat.txt").write_bytes(prompt)
            argv = self.argv("--no-prime-prefix", "--prompt-dir", str(directory))
            out = self.root / (suite + "-out")
            argv[argv.index("--out") + 1] = str(out)
            argv[argv.index("--workload-label-prefix") + 1] = suite
            argv[argv.index("--concurrency") + 1] = "1,2"
            with patch("tools.bench.run_resident_spec_router_calibration.subprocess.run", side_effect=self.fake.run):
                self.assertEqual(main(argv), 0)
            doc = json.loads((out / "measurements.json").read_text())
            self.assertEqual([c["workload"] for c in doc["comparisons"]],
                             [suite + "/chat/C1", suite + "/chat/C2"])
            environment = json.loads((out / "environment.json").read_text())
            self.assertEqual(environment["workload_label_prefix"], suite)
            for comparison in doc["comparisons"]:
                baseline = comparison["pairs"][0]["baseline"]
                self.assertEqual(baseline["configuration"]["prompt"], prompt.decode("utf-8"))
                self.assertEqual({e["active_batch"] for e in baseline["rounds"]}, {1})
            documents.append(doc)
        merged = copy.deepcopy(documents[0])
        merged["comparisons"].extend(documents[1]["comparisons"])
        profile = build_profile(merged)
        self.assertEqual(next(c["draft_tokens"] for c in profile["cells"]
                              if c["active_batch"] == 1 and c["frontier_upper"] == 1024), 7)
        self.assertTrue(all(c["draft_tokens"] == 0 for c in profile["cells"] if c["active_batch"] != 1))

    def test_suite_prefix_required_and_separator_ambiguity_rejected(self):
        for value in (None, "", "suite/chat", "suite name", "中文"):
            with self.subTest(value=value):
                argv = self.argv("--no-prime-prefix")
                index = argv.index("--workload-label-prefix")
                if value is None:
                    del argv[index:index + 2]
                else:
                    argv[index + 1] = value
                with patch("tools.bench.run_resident_spec_router_calibration.subprocess.run", side_effect=self.fake.run):
                    with self.assertRaises(SystemExit):
                        main(argv)
                self.assertFalse((self.root / "out").exists())
        self.assertEqual(self.fake.commands, [])

    def test_wrong_priming_accounting_is_rejected(self):
        self.fake.mutation = lambda p, a: p["priming"].update(settled_committed_tokens=14)
        self.assertEqual(self.run_driver("--prime-prefix"), 1)


@unittest.skipUnless(os.environ.get("NINFER_CALIBRATION_NATIVE_EXE"),
                     "native CPU parser checks require an explicitly selected built binary")
class NativeOptionsTest(unittest.TestCase):
    def run_native(self, *args):
        return subprocess.run([os.environ["NINFER_CALIBRATION_NATIVE_EXE"], *args],
                              capture_output=True, text=True, timeout=30)

    def test_help_describes_identity_resident_and_priming(self):
        result = self.run_native("--help")
        self.assertEqual(result.returncode, 0)
        for fragment in ("--identity-only", "--spec-router-profile", "--prime-prefix",
                         "resident K15", "separate 16-token"):
            self.assertIn(fragment, result.stdout)

    def test_invalid_mode_combinations_fail_before_model_loading(self):
        for args, fragment in (
                (["--identity-only", "--spec-router-profile", "absent"], "identity-only requires Fixed"),
                (["--prime-prefix"], "prime-prefix requires a profile"),
                (["--identity-only", "--identity-only"], "duplicate flag"),
                (["--identity-only", "--model", "absent", "--output", "absent-output",
                  "--draft-tokens", "7"], "identity-only requires Fixed DFlash2 K15"),
                (["--model", "absent", "--prompt-file", "absent", "--output", "absent-output",
                  "--spec-router-profile", ""], "profile must be an explicit existing file"),
                (["--model", "absent", "--prompt-file", "absent", "--output", "absent-output",
                  "--spec-router-profile", "absent-profile"], "profile must be an explicit existing file")):
            with self.subTest(args=args):
                result = self.run_native(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(fragment, result.stderr)

    def test_identity_only_does_not_require_prompt_or_generate(self):
        result = self.run_native("--identity-only", "--model", "absent", "--output", "absent-output")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model must be an explicit existing artifact", result.stderr)
        self.assertNotIn("required: --prompt-file", result.stderr)


if __name__ == "__main__":
    unittest.main()
