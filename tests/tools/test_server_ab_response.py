"""Regression tests for the failures in the 2026-10-02 server qualification.

Pure CPU; the process test launches a local HTTP fixture, never the NInfer model or a GPU.
Run from the repository root: python3.11 -m unittest discover -s tests/tools -p test_server_ab_response.py
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.dflash2_training import ab_response as R
from tools.dflash2_training import server_ab as S


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    args: tuple[str, ...] = ()
    compare_to: str = "baseline"


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str = "long-context"
    prompt: str = "Return a fixed answer."
    max_new: int = 512


def response(content="answer", reasoning=None, finish="stop"):
    return {"id": "request-random", "created": 1,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content,
                                     "reasoning_content": reasoning}}],
            "usage": {"completion_tokens": 10, "prompt_tokens": 20}}


def row(rep=0, content="answer", reasoning=None, finish="stop"):
    raw = response(content, reasoning, finish)
    return {"workload": "long-context", "rep": rep, "ok": True, "response": raw,
            **R.response_record(raw, "model-default")}


def done(request_id=1):
    return {"event": "request_done", "server_instance_id": "fixture",
            "request": {"request_id": request_id},
            "speculative": {"rounds": 4, "drafted_tokens": 60, "accepted_tokens": 16,
                            "tree": {"rounds": 1, "fallback_rounds": 3,
                                     "nodes": 15, "accepted_drafts": 4},
                            "lookup": {"rounds": 0, "head_skip_rounds": 0}}}


def trials(pairs=4, candidate=None):
    candidate = candidate or Arm("dflash2-k15")
    return [{"comparison": candidate.name, "concurrency": 1, "pair": p, "side": side,
             "arm": "baseline" if side == "base" else candidate.name,
             "tok_s": 100.0 if side == "base" else 150.0,
             "runtime": {"complete": True, "tree_rounds": 1, "head_skip_rounds": 1},
             "rows": [row(0), row(1)]}
            for p in range(pairs) for side in ("base", "candidate")]


def evaluate(data, candidate=None, pairs=4, discard=1):
    candidate = candidate or Arm("dflash2-k15")
    return S.evaluate(data, [(candidate.name, Arm("baseline"), candidate)], [Workload()],
                      [1], pairs, 2, discard)[candidate.name]["levels"][0]


class ResponseTests(unittest.TestCase):
    def test_reasoning_is_compared_when_content_is_empty(self):
        left = R.response_signature(response("", "first"))
        right = R.response_signature(response("", "other"))
        self.assertEqual(R.first_difference(left, right)["field"], "reasoning_content")

    def test_no_thinking_does_not_accept_reasoning(self):
        with self.assertRaises(ValueError):
            R.response_record(response("answer", "hidden answer"), "off")

    def test_completely_empty_response_is_not_success(self):
        with self.assertRaises(ValueError):
            R.response_signature(response(""))

    def test_finish_reason_is_semantic(self):
        a = R.response_signature(response(finish="stop"))
        b = R.response_signature(response(finish="length"))
        self.assertEqual(R.first_difference(a, b)["field"], "finish_reason")

    def test_whitespace_is_not_normalized(self):
        a = R.response_signature(response("hello"))
        b = R.response_signature(response("hello\n"))
        diff = R.first_difference(a, b)
        self.assertEqual((diff["field"], diff["character_offset"]), ("content", 5))

    def test_provider_ids_and_usage_are_not_output(self):
        a, b = response(), response()
        b.update(id="different", created=999)
        b["usage"]["completion_tokens"] = 11
        self.assertEqual(R.response_signature(a), R.response_signature(b))

    def test_null_and_absent_reasoning_are_equivalent(self):
        a, b = response(), response()
        b["choices"][0]["message"].pop("reasoning_content")
        self.assertEqual(R.response_signature(a), R.response_signature(b))

    def test_tool_arguments_and_refusal_are_retained(self):
        a = response("")
        a["choices"][0]["message"]["tool_calls"] = [
            {"id": "call-1", "type": "function",
             "function": {"name": "read", "arguments": '{"offset":40}'}}]
        b = copy.deepcopy(a)
        b["choices"][0]["message"]["tool_calls"][0]["id"] = "call-2"
        self.assertEqual(R.response_signature(a), R.response_signature(b))
        b["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"offset":"40"}'
        self.assertIsNotNone(R.first_difference(R.response_signature(a), R.response_signature(b)))
        refused = response("")
        refused["choices"][0]["message"]["refusal"] = "Declined"
        self.assertEqual(R.response_signature(refused)["refusal"], "Declined")

    def test_invalid_usage_does_not_become_zero_tokens(self):
        for value in (None, 0, -1, True, "10", float("nan")):
            raw = response()
            raw["usage"]["completion_tokens"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                R.response_record(raw, "off")

    def test_missing_terminal_fields_fail(self):
        for raw in ({}, {"choices": []}, response(finish=None)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                R.response_signature(raw)

    def test_nearest_rank_small_sample(self):
        self.assertEqual(R.nearest_rank(list(range(1, 15)), .95), 14)
        self.assertEqual(R.nearest_rank(list(range(1, 21)), .95), 19)
        self.assertIsNone(R.nearest_rank([], .95))
        with self.assertRaises(ValueError):
            R.nearest_rank([float("nan")], .95)


class GateTests(unittest.TestCase):
    def test_two_failed_sides_do_not_match_by_absence(self):
        expected = {("long-context", 0), ("long-context", 1)}
        self.assertFalse(R.compare_responses([], [], expected)["passed"])
        bad = [dict(row(0), ok=False), dict(row(1), ok=False)]
        self.assertFalse(R.compare_responses(bad, bad, expected)["passed"])

    def test_duplicate_row_is_rejected(self):
        self.assertFalse(R.trial_integrity([row(), row()], {("long-context", 0)})["passed"])

    def test_signature_hash_must_match(self):
        a = row()
        a["signature"]["content"] = "tampered"
        self.assertFalse(R.trial_integrity([a], {("long-context", 0)})["passed"])

    def test_signature_without_raw_evidence_fails(self):
        a = row()
        del a["response"]
        self.assertFalse(R.trial_integrity([a], {("long-context", 0)})["passed"])

    def test_all_pairs_and_repeats_pass(self):
        result = evaluate(trials())
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["qualified_speedup"], 1.5)
        self.assertEqual(result["performance_status"], "consistent_gain")

    def test_discarded_pair_remains_correctness_evidence(self):
        data = trials()
        data[1]["rows"][0] = row(content="wrong warmup")
        result = evaluate(data)
        self.assertFalse(result["gate_passed"])
        self.assertIsNone(result["qualified_speedup"])
        self.assertEqual(result["diagnostic_median_speedup"], 1.5)

    def test_matching_but_self_unstable_baseline_is_failure(self):
        data = trials()
        # Each A/B pair agrees, but both sides change within repeated requests.
        for trial in data:
            trial["rows"][1] = row(1, "unstable")
        result = evaluate(data)
        self.assertTrue(result["exact"])
        self.assertFalse(result["stable"])
        self.assertIsNone(result["qualified_speedup"])

    def test_changes_between_fresh_processes_are_also_caught(self):
        data = trials()
        for trial in data:
            if trial["pair"] == 3:
                trial["rows"] = [row(0, "another"), row(1, "another")]
        self.assertFalse(evaluate(data)["gate_passed"])

    def test_one_good_pair_is_not_a_release_speed_claim(self):
        result = evaluate(trials(1), pairs=1, discard=0)
        self.assertTrue(result["gate_passed"])
        self.assertIsNone(result["qualified_speedup"])
        self.assertEqual(result["performance_status"], "insufficient_pairs")

    def test_duplicate_or_missing_trial_fails(self):
        data = trials()
        self.assertFalse(evaluate(data[:-1])["gate_passed"])
        self.assertFalse(evaluate(data + [data[0]])["gate_passed"])

    def test_missing_runtime_evidence_blocks_qualification(self):
        data = trials()
        data[0]["runtime"] = {"complete": False}
        self.assertFalse(evaluate(data)["gate_passed"])

    def test_tree_flag_without_measured_tree_work_is_not_tree_speedup(self):
        tree = Arm("tree15", ("--spec-tree", "lattice"))
        data = trials(candidate=tree)
        for trial in data:
            if trial["pair"] > 0:
                trial["runtime"]["tree_rounds"] = 0
        result = evaluate(data, candidate=tree)
        self.assertTrue(result["gate_passed"])
        self.assertFalse(result["optimization_observed"])
        self.assertIsNone(result["qualified_speedup"])

    def test_baseline_self_keeps_side_identity(self):
        candidate = Arm("baseline")
        result = evaluate(trials(candidate=candidate), candidate=candidate)
        self.assertTrue(result["complete"])
        self.assertTrue(result["stable"])


class EvidenceTests(unittest.TestCase):
    def read_log(self, events, expected=2):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in events) + "\n", encoding="utf-8")
            return R.runtime_evidence(path, expected)

    def test_actual_counters_exclude_startup_and_throughput(self):
        got = self.read_log([{"event": "server_start"}, done(1),
                             {"event": "throughput", "speculative": {"rounds": 999}}, done(2)])
        self.assertTrue(got["complete"])
        self.assertEqual(got["tree_rounds"], 2)
        self.assertEqual(got["tree_execution_fraction"], .25)

    def test_duplicate_internal_request_id_fails(self):
        self.assertFalse(self.read_log([done(1), done(1)])["complete"])

    def test_missing_or_malformed_counter_fails(self):
        event = done(1)
        del event["speculative"]["tree"]
        self.assertFalse(self.read_log([event, done(2)])["complete"])
        self.assertFalse(self.read_log([None])["complete"])

    def test_missing_log_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(R.runtime_evidence(Path(directory) / "absent", 2)["complete"])


class RunnerTests(unittest.TestCase):
    def args(self, directory):
        return SimpleNamespace(serve=Path("/bin/true"), model=Path("/tmp/model"),
                               out=Path(directory), server_capacity="8", max_context=32768,
                               kv_capacity=32768, kv_dtype="int8", port=18080,
                               thinking="off", prefix_reuse="on", cuda_graph="on",
                               startup_timeout=5, request_timeout=5, repeats=2, seed=7, max_new=512)

    def test_explicit_model_policies_and_startup_capacity(self):
        args = self.args("/tmp")
        command = S.server_command(args, Arm("baseline"), 1, "unique", Path("requests.jsonl"))
        self.assertIn("--no-thinking", command)
        self.assertNotIn("--no-cuda-graph", command)
        self.assertEqual(command[command.index("--max-concurrency") + 1], "8")
        args.server_capacity = "level"
        args.prefix_reuse = args.cuda_graph = "off"
        command = S.server_command(args, Arm("baseline"), 2, "unique", Path("requests.jsonl"))
        self.assertEqual(command[command.index("--max-concurrency") + 1], "2")
        self.assertIn("--no-prefix-reuse", command)
        self.assertIn("--no-cuda-graph", command)
        args.thinking = "model-default"
        self.assertNotIn("--no-thinking", S.server_command(args, Arm("baseline"), 1, "id", Path("log")))

    def test_response_retained_even_when_validation_fails(self):
        raw = response("", "not permitted in off mode")
        with patch.object(S, "request_json", return_value=raw):
            got = S.one_request("http://local", "id", "prompt", 10, 5)
        self.assertFalse(got["ok"])
        self.assertEqual(got["response"], raw)

    def test_existing_server_is_not_mistaken_for_the_trial(self):
        proc = SimpleNamespace(poll=lambda: None)
        with patch.object(S, "request_json", side_effect=[{"status": "ok"}, {"data": [{"id": "wrong"}]}]):
            with self.assertRaisesRegex(RuntimeError, "different server"):
                S.wait_ready(12345, proc, 1, "ours")

    @unittest.skipUnless(os.name == "posix", "Linux process lifecycle integration")
    def test_fresh_server_per_level_and_side_with_raw_archive(self):
        with tempfile.TemporaryDirectory(prefix="ninfer ab ") as directory:
            args = self.args(directory)
            script = Path(directory) / "fake serve"
            script.write_text("#!" + sys.executable + "\n" + textwrap.dedent('''\
                import http.server, json, os, sys, threading
                argv = sys.argv
                get = lambda key: argv[argv.index(key)+1]
                name, logfile = get('--model-id'), get('--request-log-jsonl')
                lock = threading.Lock()
                count = 0
                class Handler(http.server.BaseHTTPRequestHandler):
                    def reply(self, value):
                        data = json.dumps(value).encode()
                        self.send_response(200)
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    def log_message(self, *args): pass
                    def do_GET(self):
                        self.reply({'status':'ok'} if self.path=='/health' else {'data':[{'id':name}]})
                    def do_POST(self):
                        global count
                        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                        with lock:
                            count += 1
                            event = {'event':'request_done', 'server_instance_id':str(os.getpid()),
                                     'request':{'request_id':count},
                                     'speculative':{'rounds':0, 'drafted_tokens':0, 'accepted_tokens':0,
                                                    'tree':{'rounds':0,'fallback_rounds':0,'nodes':0,'accepted_drafts':0},
                                                    'lookup':{'rounds':0,'head_skip_rounds':0}}}
                            with open(logfile,'a') as f: f.write(json.dumps(event)+'\\n')
                        self.reply({'id':name, 'choices':[{'message':{'role':'assistant','content':'answer'},
                                                         'finish_reason':'stop'}],
                                    'usage':{'completion_tokens':1,'prompt_tokens':5}})
                http.server.ThreadingHTTPServer(('127.0.0.1', int(get('--port'))), Handler).serve_forever()
                '''), encoding="utf-8")
            script.chmod(0o755)
            args.serve = script
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                args.port = probe.getsockname()[1]
            seen = []
            for level, side in ((1, "base"), (2, "candidate")):
                result = S.run_trial(args, Arm("baseline"), [Workload()], "baseline-self", level, 0, side)
                self.assertNotIn("error", result)
                self.assertTrue(result["runtime"]["complete"])
                self.assertEqual(len(result["rows"]), 2)
                path = Path(result["directory"])
                archived = [json.loads(line) for line in (path / "raw-responses.jsonl").read_text().splitlines()]
                self.assertEqual(archived[0]["response"]["choices"][0]["message"]["content"], "answer")
                seen.append(result["command"][result["command"].index("--model-id") + 1])
                with socket.socket() as probe:
                    self.assertNotEqual(probe.connect_ex(("127.0.0.1", args.port)), 0)
            self.assertNotEqual(seen[0], seen[1])
            with self.assertRaises(FileExistsError):
                S.run_trial(args, Arm("baseline"), [Workload()], "baseline-self", 1, 0, "base")


@unittest.skipUnless(os.name == "posix", "Linux wrapper contract")
class ShellTests(unittest.TestCase):
    def test_wrapper_keeps_cases_separate_and_collects_failures(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="ninfer shell ") as temporary:
            tmp = Path(temporary)
            model = tmp / "model artifact.ninfer"
            model.touch()
            out = tmp / "new results"
            log = tmp / "commands.jsonl"
            fake_python = tmp / "python fixture"
            fake_python.write_text("#!" + sys.executable + "\n" +
                "import json,os,sys\n" +
                "with open(os.environ['MOCK_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n" +
                "sys.exit(2 if 'baseline-production' in sys.argv[sys.argv.index('--out')+1] else 0)\n")
            fake_python.chmod(0o755)
            env = dict(os.environ, NINFER_PYTHON=str(fake_python), NINFER_SERVE_EXE="/bin/true",
                       MOCK_LOG=str(log))
            command = ["bash", str(root / "scripts/sweeps/dflash2-server-repro.sh"), str(model), str(out)]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(commands), 6)
            paths = [argv[argv.index("--out") + 1] for argv in commands]
            self.assertEqual(len(set(paths)), 6)
            self.assertTrue(all(argv[argv.index("--model") + 1] == str(model) for argv in commands))
            self.assertEqual(len((out / "exit-status.tsv").read_text().splitlines()), 6)
            again = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(again.returncode, 1)
            self.assertIn("already exists", again.stderr)


if __name__ == "__main__":
    unittest.main()
