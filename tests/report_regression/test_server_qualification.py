"""CPU regression checks for the 2026-10-02 report. No model or CUDA needed."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import socket
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tools.dflash2_training import server_validation as v

from tools.dflash2_training import ab_suite, server_ab as server


def response(content="answer", reasoning="", finish="stop", count=3):
    return {"id": "random-id", "created": 123, "choices": [{"index": 0,
            "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
            "finish_reason": finish}], "usage": {"completion_tokens": count}}


def row(rep=0, content="answer", workload="prose", **kwargs):
    output = v.canonical_response(response(content=content, **kwargs), thinking="model-default")
    return {"ok": True, "workload": workload, "rep": rep, "output": output,
            "sha256": v.digest(output), "completion_tokens": 3, "latency_s": 1.0}


def run(rows=None, speed=10):
    return {"rows": [row(0), row(1)] if rows is None else rows, "thinking": "off",
            "concurrency": 2, "cache_mode": "disabled", "cuda_graph": True,
            "engine_concurrency": 8, "max_context": 32768, "kv_capacity": 32768,
            "kv_dtype": "int8", "tok_s": speed,
            "runtime": {"status": "complete", "counters": {}}}


EXPECTED = {("prose", 0), ("prose", 1)}


class ResponseTests(unittest.TestCase):
    def test_hash_includes_reasoning(self):
        a = v.canonical_response(response("", "one"), thinking="model-default")
        b = v.canonical_response(response("", "two"), thinking="model-default")
        self.assertNotEqual(v.digest(a), v.digest(b))

    def test_hash_includes_finish(self):
        a = v.canonical_response(response())
        b = v.canonical_response(response(finish="length"))
        self.assertNotEqual(v.digest(a), v.digest(b))

    def test_transport_ids_not_compared(self):
        a, b = response(), response()
        b.update(id="different", created=456)
        self.assertEqual(v.digest(v.canonical_response(a)), v.digest(v.canonical_response(b)))

    def test_whitespace_not_normalized(self):
        a = v.canonical_response(response("hello"))
        b = v.canonical_response(response("hello\n"))
        self.assertNotEqual(v.digest(a), v.digest(b))
        self.assertEqual(v.first_difference(a, b)["char_offset"], 5)

    def test_optional_text_missing_is_empty_not_null_string(self):
        a = response()
        del a["choices"][0]["message"]["reasoning_content"]
        self.assertEqual(v.canonical_response(a)["reasoning_content"], "")

    def test_empty_content_cannot_pass_no_thinking(self):
        for content in (None, ""):
            with self.assertRaises(ValueError):
                v.canonical_response(response(content))

    def test_reasoning_disallowed_when_thinking_off(self):
        with self.assertRaises(ValueError):
            v.canonical_response(response("answer", "hidden reasoning"))

    def test_empty_all_channels_model_default_rejected(self):
        with self.assertRaises(ValueError):
            v.canonical_response(response("", ""), thinking="model-default")

    def test_bad_choices_finish_and_role_rejected(self):
        cases = [{}, {"error": {}}, {"choices": []}, {"choices": [{}, {}]}, response(finish=None)]
        bad = response()
        bad["choices"][0]["message"]["role"] = "user"
        cases.append(bad)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                v.canonical_response(value)

    def test_unrequested_tool_call_not_ignored(self):
        value = response()
        value["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "write"}}]
        with self.assertRaises(ValueError):
            v.canonical_response(value)

    def test_usage_strict(self):
        for value in (None, True, 0, -1, 3.0, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                v.completion_tokens(response(count=value))
        self.assertEqual(v.completion_tokens(response(count=3)), 3)

    def test_p95_nearest_rank_of_14_is_last(self):
        self.assertEqual(v.nearest_rank(range(1, 15), .95), 14)
        self.assertEqual(v.nearest_rank([2], .95), 2)
        self.assertIsNone(v.nearest_rank([], .95))

    def test_p95_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            v.nearest_rank([float("nan")], .95)


class GateTests(unittest.TestCase):
    def test_identical_stable_complete_qualifies(self):
        verdict = v.compare_runs(run(), run(speed=20), EXPECTED)
        self.assertTrue(verdict["qualified"])
        self.assertEqual(verdict["qualified_speedup"], 2)

    def test_baseline_self_instability_blocks_same_cross_pairs(self):
        a = run([row(0), row(1, "different")])
        verdict = v.compare_runs(a, copy.deepcopy(a), EXPECTED)
        self.assertTrue(verdict["exact"])
        self.assertFalse(verdict["qualified"])
        self.assertIn("baseline_self_unstable", verdict["reasons"])
        self.assertIsNone(verdict["qualified_speedup"])

    def test_candidate_self_instability_identified(self):
        verdict = v.compare_runs(run(), run([row(0), row(1, "different")]), EXPECTED)
        self.assertIn("candidate_self_unstable", verdict["reasons"])
        self.assertFalse(verdict["qualified"])

    def test_changed_output_fast_result_is_diagnostic_only(self):
        verdict = v.compare_runs(run(), run([row(0, "wrong"), row(1, "wrong")], speed=30), EXPECTED)
        self.assertEqual(verdict["diagnostic_speedup"], 3)
        self.assertIsNone(verdict["qualified_speedup"])
        self.assertEqual(verdict["differences"][0]["field"], "content")

    def test_no_requests_and_no_expected_is_not_pass(self):
        self.assertFalse(v.compare_runs(run([]), run([]), set())["qualified"])

    def test_missing_duplicate_extra_or_failed_row_blocks(self):
        cases = [[], [row(0)], [row(0), row(0)], [row(0), row(1), row(2)],
                 [row(0), {"ok": False, "rep": 1, "workload": "prose", "error": "HTTP 500"}]]
        for rows in cases:
            with self.subTest(rows=rows):
                self.assertFalse(v.compare_runs(run(), run(rows), EXPECTED)["qualified"])

    def test_wrong_saved_sha_not_trusted(self):
        rows = [row(0), row(1)]
        rows[1]["sha256"] = "forged"
        self.assertFalse(v.inspect_run(run(rows), EXPECTED)["complete"])

    def test_single_repeat_insufficient_not_success(self):
        expected = {("prose", 0)}
        verdict = v.compare_runs(run([row(0)]), run([row(0)]), expected)
        self.assertTrue(verdict["exact"])
        self.assertFalse(verdict["qualified"])

    def test_aa_control_uses_identical_runtime_flags(self):
        baseline = types.SimpleNamespace(name="baseline", compare_to="baseline", args=())
        with patch.object(server, "select_arms", return_value=(baseline,)):
            arms = server.select_test_arms("ignored", True)
        self.assertEqual(arms[0].args, arms[1].args)
        self.assertEqual(arms[1].compare_to, "baseline")
        self.assertEqual(arms[1].name, "baseline-aa")

    def test_config_mismatch_blocks(self):
        a, b = run(), run()
        b["cache_mode"] = "enabled"
        self.assertIn("configuration_mismatch", v.compare_runs(a, b, EXPECTED)["reasons"])

    def test_nonfinite_or_missing_speed_unqualified(self):
        for value in (None, float("nan"), float("inf"), 0):
            self.assertFalse(v.compare_runs(run(), run(speed=value), EXPECTED)["qualified"])

    def test_first_discarded_pair_still_fails_correctness(self):
        arm = types.SimpleNamespace(name="candidate", compare_to="baseline")
        workload = types.SimpleNamespace(name="prose")
        rows = []
        for pair in range(2):
            for name in ("baseline", "candidate"):
                content = "wrong" if pair == 0 and name == "candidate" else "answer"
                r = run([row(0, content), row(1, content)])
                r.update(comparison="candidate", pair=pair, arm=name)
                rows.append(r)
        result = server.summarize(rows, [arm], [workload], [2], 2, 1, 2)
        self.assertEqual(result["candidate"]["levels"][0]["qualification"], "FAIL")
        self.assertIsNone(result["candidate"]["levels"][0]["qualified_median_speedup"])

    def test_cross_pair_instability_is_not_noise(self):
        arm = types.SimpleNamespace(name="candidate", compare_to="baseline")
        workload = types.SimpleNamespace(name="prose")
        rows = []
        for pair in range(2):
            for name in ("baseline", "candidate"):
                r = run([row(0, str(pair)), row(1, str(pair))])
                r.update(comparison="candidate", pair=pair, arm=name)
                rows.append(r)
        level = server.summarize(rows, [arm], [workload], [2], 2, 0, 2)["candidate"]["levels"][0]
        self.assertTrue(level["exact"])
        self.assertEqual(level["qualification"], "FAIL")


def event(request_id=1):
    return {"event": "request_done", "server_instance_id": "one",
            "request": {"request_id": request_id, "enable_thinking": False,
                        "sampling": {"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0}},
            "speculative": {"rounds": 7, "accepted_tokens": 12, "drafted_tokens": 30,
                            "tree": {"rounds": 1, "fallback_rounds": 6, "nodes": 15, "accepted_drafts": 2},
                            "lookup": {"rounds": 0, "head_skip_rounds": 0}}}


class RuntimeTests(unittest.TestCase):
    def test_actual_fallback_counts(self):
        result = v.runtime_counters([event(1), event(2)], 2, "off")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counters"]["tree_rounds"], 2)
        self.assertEqual(result["counters"]["tree_fallback_rounds"], 12)
        self.assertEqual(result["tree_round_share"], 1/7)

    def test_missing_counter_is_unknown_not_zero(self):
        item = event()
        del item["speculative"]["tree"]["rounds"]
        result = v.runtime_counters([item], 1, "off")
        self.assertEqual(result["status"], "unqualified")
        self.assertIsNone(result["counters"]["tree_rounds"])

    def test_duplicates_or_wrong_instance_rejected(self):
        a, b = event(1), event(2)
        b["server_instance_id"] = "two"
        self.assertEqual(v.runtime_counters([a, b], 2, "off")["status"], "unqualified")
        self.assertEqual(v.runtime_counters([a, a], 2, "off")["status"], "unqualified")

    def test_thinking_and_penalty_mismatch_rejected(self):
        a = event()
        a["request"]["enable_thinking"] = True
        self.assertEqual(v.runtime_counters([a], 1, "off")["status"], "unqualified")
        a = event()
        a["request"]["sampling"]["presence_penalty"] = 1.5
        self.assertEqual(v.runtime_counters([a], 1, "off")["status"], "unqualified")


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(body)
        prompt = payload["messages"][0]["content"]
        if prompt == "http-error":
            status, raw = 500, b'{"error":"planned"}'
        elif prompt == "malformed":
            status, raw = 200, b'{"choices":'
        elif prompt == "reasoning":
            status, raw = 200, v.json_bytes(response("", "thinking only"))
        else:
            status, raw = 200, v.json_bytes(response(prompt))
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()

    def test_exchange_archives_full_body(self):
        result = server.one_request(self.base, "fake", "answer", 8, 2)
        self.assertTrue(result["ok"])
        self.assertIn(b'reasoning_content', result["_raw"])
        self.assertEqual(result["output"]["finish_reason"], "stop")
        self.assertEqual(result["request"]["presence_penalty"], 0)

    def test_malformed_and_http_error_preserve_wire_body(self):
        for prompt, raw in [("http-error", b'{"error":"planned"}'), ("malformed", b'{"choices":')]:
            result = server.one_request(self.base, "fake", prompt, 8, 2)
            self.assertFalse(result["ok"])
            self.assertEqual(result["_raw"], raw)

    def test_reasoning_only_response_is_policy_sensitive(self):
        a = server.one_request(self.base, "fake", "reasoning", 8, 2)
        b = server.one_request(self.base, "fake", "reasoning", 8, 2, thinking="model-default")
        self.assertFalse(a["ok"])
        self.assertTrue(b["ok"])

    def test_loopback_does_not_use_host_proxy(self):
        with patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}):
            self.assertTrue(server.one_request(self.base, "fake", "ok", 8, 2)["ok"])

    def test_parallel_rows_and_archives(self):
        workload = types.SimpleNamespace(name="prose", prompt="hello")
        result = server.run_level(self.base, "fake", [workload], 2, 2, 2, max_tokens=8)
        self.assertEqual(result["ok"], 2)
        self.assertGreater(result["tok_s"], 0)
        self.assertTrue(v.inspect_run(result, EXPECTED)["stable"])
        with tempfile.TemporaryDirectory() as directory:
            for i, r in enumerate(result["rows"]):
                r["archive_index"] = i
            server.archive_rows(Path(directory), result["rows"])
            self.assertEqual(len(list(Path(directory).glob("*.body"))), 2)
            self.assertNotIn("_raw", result["rows"][0])

    def test_existing_port_is_never_reused(self):
        with self.assertRaises(RuntimeError):
            server.assert_free_port(self.httpd.server_port)

    def test_server_command_explicit_policies(self):
        args = argparse.Namespace(serve=Path("/bin/serve"), model=Path("/model"), max_context=32768,
                                  kv_capacity=32768, engine_concurrency=8, kv_dtype="int8",
                                  thinking="off", cache_mode="disabled", no_cuda_graph=False)
        arm = types.SimpleNamespace(args=("--spec", "dflash2", "--draft-tokens", "15"))
        command = server.build_command(args, arm, 18080, Path("/runtime.log"))
        for value in ("--no-thinking", "--no-prefix-reuse", "--kv-capacity"):
            self.assertIn(value, command)
        self.assertEqual(command[command.index("--max-concurrency") + 1], "8")


class EndToEndTests(unittest.TestCase):
    def _invoke(self, *, diverge=False, port_override=None, occupied=False):
        import subprocess
        import sys
        import os
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "fixture.ninfer").write_bytes(b"NOT A MODEL: CPU protocol fixture")
        while True:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
                if port + 7 <= 65535:
                    break
        if port_override is not None:
            port = port_override
        if occupied:
            listener = socket.socket()
            listener.bind(("127.0.0.1", port))
            listener.listen()
            self.addCleanup(listener.close)
        # Isolate only the workload selectors. The actual modified CLI driver,
        # HTTP client, process lifecycle, logging and qualification code all run.
        bootstrap = '''
import sys, types
from dataclasses import dataclass
m=types.ModuleType("tools.dflash2_training.ab_suite")
@dataclass
class A:
 name:str
 compare_to:str
 args:tuple=()
m.select_arms=lambda _: (A("baseline","baseline"), A("candidate","baseline",("--spec","dflash2")))
m.select_workloads=lambda _: (types.SimpleNamespace(name="prose",prompt="fixture prompt"),)
sys.modules[m.__name__]=m
from tools.dflash2_training import server_ab
import os,json
original_exchange=server_ab.http_exchange
original_ready=server_ab.wait_ready
def exchange(url,payload=None,timeout=300):
 with open(os.environ["NINFER_FIXTURE_PORT_TRACE"],"a") as trace:
  trace.write(json.dumps({"url":url,"post":payload is not None})+"\\n")
 return original_exchange(url,payload,timeout)
def ready(port,proc,timeout):
 with open(os.environ["NINFER_FIXTURE_PORT_TRACE"],"a") as trace:
  trace.write(json.dumps({"ready_port":port})+"\\n")
 return original_ready(port,proc,timeout)
server_ab.http_exchange=exchange
server_ab.wait_ready=ready
main=server_ab.main
sys.argv=["server_ab"]+sys.argv[1:]
main()
'''
        executable = root / "fake-ninfer-serve"
        executable.write_bytes(Path(__file__).with_name("fake_ninfer_server.py").read_bytes().replace(
            b"server = ThreadingHTTPServer", b"ThreadingHTTPServer.allow_reuse_address = False\nserver = ThreadingHTTPServer"))
        executable.chmod(0o755)
        command = [sys.executable, "-c", bootstrap, "--serve", str(executable),
                   "--model", str(root / "fixture.ninfer"), "--out", str(root / "results"),
                   "--port", str(port), "--concurrency", "1,2", "--pairs", "2", "--discard", "1",
                   "--repeats", "2", "--startup-timeout", "5", "--request-timeout", "2"]
        env = os.environ.copy()
        env["NINFER_FIXTURE_PORT_TRACE"] = str(root / "port-trace.jsonl")
        if diverge:
            env["NINFER_FIXTURE_DIVERGE"] = "1"
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)
        result_path = root / "results/server-results.json"
        if port_override == 65535:
            return proc, None, root
        self.assertTrue(result_path.exists(), proc.stdout + proc.stderr)
        result = json.loads(result_path.read_text())
        return proc, result, root

    def test_complete_isolated_abba_lifecycle(self):
        proc, result, root = self._invoke()
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        runs = result["runs"]
        self.assertEqual(len(runs), 8)
        ports = [r["port"] for r in runs]
        self.assertEqual(ports, list(range(ports[0], ports[0] + 8)))
        self.assertEqual([int(r["command"][r["command"].index("--port") + 1]) for r in runs], ports)
        trace = [json.loads(line) for line in (root / "port-trace.jsonl").read_text().splitlines()]
        self.assertEqual([event["ready_port"] for event in trace if "ready_port" in event], ports)
        from urllib.parse import urlparse
        self.assertEqual({urlparse(event["url"]).port for event in trace if event.get("post")}, set(ports))
        self.assertEqual([r["arm"] for r in runs[:4]], ["baseline", "candidate", "candidate", "baseline"])
        logfiles = list((root / "results").glob("*/runtime.requests.jsonl"))
        self.assertEqual(len(logfiles), 8)
        instances = {json.loads(path.read_text().splitlines()[0])["server_instance_id"] for path in logfiles}
        self.assertEqual(len(instances), 8)
        self.assertEqual(len(list((root / "results").glob("*/response-*.body"))), 16)
        for level in result["comparisons"]["candidate"]["levels"]:
            self.assertEqual(level["qualification"], "PASS")

    def test_port_plan_overflow_rejected_before_any_output_or_process(self):
        proc, result, root = self._invoke(port_override=65535)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("port range", proc.stderr)
        self.assertIsNone(result)
        self.assertFalse((root / "results").exists())
        self.assertFalse((root / "port-trace.jsonl").exists())

    def test_occupied_instance_port_is_failure_not_reused_or_skipped(self):
        proc, result, root = self._invoke(occupied=True)
        self.assertEqual(proc.returncode, 2)
        first = result["runs"][0]
        self.assertIn("already in use", first["error"])
        self.assertEqual(first["rows"], [])
        self.assertEqual(len(result["runs"]), 8)
        trace = [json.loads(line) for line in (root / "port-trace.jsonl").read_text().splitlines()]
        self.assertNotIn(first["port"], [event["ready_port"] for event in trace if "ready_port" in event])
        self.assertEqual(result["comparisons"]["candidate"]["levels"][0]["qualification"], "FAIL")

    def test_end_to_end_different_output_fails_without_speedup(self):
        proc, result, _ = self._invoke(diverge=True)
        self.assertEqual(proc.returncode, 2, proc.stderr + proc.stdout)
        for level in result["comparisons"]["candidate"]["levels"]:
            self.assertEqual(level["qualification"], "FAIL")
            self.assertIsNone(level["qualified_median_speedup"])


if __name__ == "__main__":
    unittest.main()
