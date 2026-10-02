import contextlib
import csv
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.dflash2_training import ab_suite, server_ab, ninfer_gate, cli_validation


class Gates(unittest.TestCase):
    def cli(self, responses, *extra):
        with tempfile.TemporaryDirectory() as out:
            root = Path(out)
            model = root / 'mock.ninfer'
            model.write_bytes(b'CPU fixture, not a model')
            executable = root / 'mock-cli'
            executable.write_text('#!' + sys.executable + '\n')
            executable.chmod(0o755)
            argv = ['ab_suite', '--exe', str(executable), '--model', str(model), '--out', str(root / 'results'),
                    '--pairs', '2', '--discard', '1', '--cooldown', '0',
                    '--arms', 'dflash2-k7', '--workloads', 'prose', *extra]
            rc = 0
            with patch.object(sys, 'argv', argv), patch.object(ab_suite.subprocess, 'run', side_effect=responses), contextlib.redirect_stdout(io.StringIO()):
                try:
                    ab_suite.main()
                except SystemExit as exc:
                    rc = exc.code
            result = json.loads((root / 'results' / 'ab-results.json').read_text()) if (root / 'results' / 'ab-results.json').exists() else None
            return rc, result

    def process(self, text=b'same', rc=0, speed=100):
        metrics = dict.fromkeys(cli_validation.COUNTERS, 0)
        metrics.update(schema=1, token_ids=[10, 20, 30], generated=3, decoded=2,
                       finish_reason=1, decode_seconds=2 / speed, decode_tok_s=speed,
                       prefill_seconds=0.01, temperature=0, presence_penalty=0,
                       frequency_penalty=0, acceptance_pct=None, tok_per_round=None)
        return subprocess.CompletedProcess([], rc, text,
                                           (cli_validation.PREFIX + json.dumps(metrics)).encode())

    def test_warmup_exit_fails(self):
        rc, result = self.cli([self.process(rc=1), self.process(), self.process(), self.process()])
        self.assertEqual(rc, 2)
        self.assertFalse(result['comparisons']['dflash2-k7']['correct'])
        self.assertEqual(len(result['runs']), 4)
        self.assertEqual(result['runs'][0]['returncode'], 1)

    def test_warmup_output_mismatch_fails(self):
        rc, _ = self.cli([self.process(b'different'), self.process(), self.process(), self.process()])
        self.assertEqual(rc, 2)

    def test_discard_only_excludes_performance(self):
        rc, result = self.cli([self.process(speed=1), self.process(speed=999), self.process(speed=120), self.process(speed=100)])
        self.assertEqual(rc, 0)
        perf = result['comparisons']['dflash2-k7']['workloads']['prose']['performance']
        self.assertEqual(perf['pairs'], 1)
        self.assertEqual(perf['median_speedup'], 1.2)

    def test_cli_baseline_only_rejected(self):
        rc, _ = self.cli([], '--arms', 'baseline')
        self.assertNotEqual(rc, 0)

    def test_missing_pair_fails(self):
        with tempfile.TemporaryDirectory() as out:
            args = type('Args', (), {'out': out, 'exe': 'mock', 'model': 'mock', 'kv_dtype': 'int8'})()
            with patch.object(ab_suite.subprocess, 'run', return_value=self.process()):
                row = ab_suite.run_once(args, ab_suite.ARMS[0], ab_suite.WORKLOADS[0], 0, 0)
        result = ab_suite.judge([row], ab_suite.WORKLOADS[:1], ab_suite.ARMS[:2], 1)
        self.assertFalse(result['dflash2-k7']['correct'])

    def test_missing_whole_pair_fails(self):
        _, result = self.cli([self.process() for _ in range(4)])
        rows = [r for r in result['runs'] if r['pair'] == 0]
        judged = ab_suite.judge(rows, ab_suite.WORKLOADS[:1], ab_suite.ARMS[:2], 0, 2)
        self.assertFalse(judged['dflash2-k7']['correct'])

    def test_chained_comparisons_keep_original_runs(self):
        responses = [self.process(), self.process(rc=1)] + [self.process() for _ in range(6)]
        rc, result = self.cli(responses, '--arms', 'tree7')
        self.assertEqual(rc, 2)
        self.assertFalse(result['comparisons']['dflash2-k7']['correct'])
        self.assertTrue(result['comparisons']['tree7']['correct'])

    def test_cli_tree_matrix_uses_raw_greedy_for_each_actual_pair(self):
        calls = []
        def transport(command, **kwargs):
            calls.append((command, kwargs['env']['NINFER_AB_ARM']))
            return self.process()
        rc, result = self.cli(transport, '--arms', 'tree7')
        self.assertEqual(rc, 0)
        self.assertEqual([arm for _, arm in calls],
                         ['baseline', 'dflash2-k7', 'dflash2-k7', 'baseline',
                          'dflash2-k7', 'tree7', 'tree7', 'dflash2-k7'])
        self.assertEqual(result['comparisons']['dflash2-k7']['base'], 'baseline')
        self.assertEqual(result['comparisons']['tree7']['base'], 'dflash2-k7')
        for command, arm in calls:
            with self.subTest(arm=arm):
                for flag in ('--presence-penalty', '--frequency-penalty'):
                    self.assertIn(flag, command)
                    self.assertEqual(float(command[command.index(flag) + 1]), 0)
                for flag in ('--greedy', '--no-thinking', '--raw-output'):
                    self.assertIn(flag, command)
                if arm == 'baseline':
                    self.assertNotIn('--spec', command)
                elif arm == 'tree7':
                    self.assertEqual(command[command.index('--spec-tree') + 1], 'lattice')

    def test_empty_output_gate_fails(self):
        self.assertFalse(ab_suite.output_gate([])['passed'])

    def test_server_failed_responses_counted_and_both_sides_rejected(self):
        def exchange(url, payload, timeout):
            if payload['messages'][0]['content'] == ab_suite.WORKLOADS[0].prompt:
                return 500, {}, b'{"error":"fixture failure"}'
            response = {'choices': [{'message': {'role': 'assistant', 'content': 'same'},
                                      'finish_reason': 'stop'}],
                        'usage': {'completion_tokens': 1}}
            return 200, {}, json.dumps(response).encode()
        with patch.object(server_ab, 'http_exchange', side_effect=exchange):
            level = server_ab.run_level('http://fixture', 'fixture',
                                        ab_suite.WORKLOADS[:2], 1, 2, 2)
        self.assertEqual(level['errors'], 2)
        self.assertEqual(level['ok'], 2)
        expected = {(w.name, rep) for w in ab_suite.WORKLOADS[:2] for rep in range(2)}
        verdict = server_ab.compare_runs(level, level, expected)
        self.assertFalse(verdict['exact'])
        self.assertFalse(verdict['qualified'])

    def test_server_baseline_only_rejected(self):
        with patch.object(sys, 'argv', ['server_ab', '--model', 'mock', '--arms', 'baseline']), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                server_ab.main()
        self.assertNotEqual(error.exception.code, 0)

    def test_server_tree_matrix_sends_zero_penalties_on_every_arm(self):
        # Exercise the real fresh-process lifecycle and HTTP client, using the
        # same CPU protocol fixture as the report qualification tests.
        with tempfile.TemporaryDirectory() as out:
            root = Path(out)
            model = root / 'mock.ninfer'
            model.write_bytes(b'CPU fixture, not a model')
            executable = root / 'mock-server'
            fixture = Path(__file__).parent / 'report_regression' / 'fake_ninfer_server.py'
            executable.write_text('#!' + sys.executable + '\n' +
                                  fixture.read_text().split('\n', 1)[1])
            executable.chmod(0o755)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            argv = ['server_ab', '--serve', str(executable), '--model', str(model),
                    '--out', str(root / 'results'), '--port', str(port),
                    '--arms', 'tree7', '--workloads', 'prose,chat', '--repeats', '2',
                    '--pairs', '2', '--discard', '1', '--concurrency', '1',
                    '--startup-timeout', '5', '--request-timeout', '2']
            requests = []
            exchange = server_ab.http_exchange
            def transport(url, payload=None, timeout=300):
                if payload is not None:
                    requests.append(payload)
                return exchange(url, payload, timeout)
            with patch.object(sys, 'argv', argv), \
                    patch.object(server_ab, 'http_exchange', side_effect=transport), \
                    contextlib.redirect_stdout(io.StringIO()):
                server_ab.main()
            result = json.loads((root / 'results/server-results.json').read_text())
            self.assertEqual(result['comparisons']['dflash2-k7']['base'], 'baseline')
            self.assertEqual(result['comparisons']['tree7']['base'], 'dflash2-k7')
            runs = result['runs']
            self.assertEqual([r['arm'] for r in runs],
                             ['baseline', 'dflash2-k7', 'dflash2-k7', 'baseline',
                              'dflash2-k7', 'tree7', 'tree7', 'dflash2-k7'])
            self.assertEqual(len(requests), 32)
            self.assertEqual(len({r['command'][r['command'].index('--request-log-jsonl') + 1] for r in runs}), 8)
            for run in runs:
                self.assertEqual(run['runtime']['status'], 'complete')
                self.assertEqual({row['workload'] for row in run['rows']}, {'prose', 'chat'})
                self.assertEqual(len(run['rows']), 4)
                self.assertIn('--no-thinking', run['command'])
            for payload in requests:
                self.assertEqual(payload['presence_penalty'], 0)
                self.assertEqual(payload['frequency_penalty'], 0)
                self.assertEqual(payload['temperature'], 0)
            for comparison in result['comparisons'].values():
                self.assertEqual(comparison['levels'][0]['qualification'], 'PASS')


class ArtifactGate(unittest.TestCase):
    def run_gate(self, returncodes, outputs=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / 'prompts.jsonl'
            prompts.write_text('"first"\n"second"\n', encoding='utf-8')
            artifact = root / 'mock.ninfer'
            artifact.touch()
            report = root / 'gate.csv'
            argv = ['ninfer_gate', '--ninfer', 'mock', '--artifact', f'a={artifact}', '--artifact', f'b={artifact}', '--prompts', str(prompts), '--out', str(report)]
            responses = [subprocess.CompletedProcess([], rc, (outputs or [b'same'] * 4)[i], b'decode speed 100 tok/s') for i, rc in enumerate(returncodes)]
            status = 0
            output = io.StringIO()
            with patch.object(sys, 'argv', argv), patch.object(ninfer_gate.subprocess, 'run', side_effect=responses) as run, contextlib.redirect_stdout(output):
                try:
                    ninfer_gate.main()
                except SystemExit as exc:
                    status = exc.code
            with report.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            return status, rows, output.getvalue(), run.call_count

    def test_failed_run_exits_nonzero_after_recording_all_results(self):
        for codes in ([1, 0, 0, 0], [0, 0, 0, -9]):
            with self.subTest(returncodes=codes):
                status, rows, output, calls = self.run_gate(codes)
                self.assertEqual(status, 2)
                self.assertEqual(calls, 4)
                self.assertEqual([int(r['returncode']) for r in rows], codes)
                self.assertIn('fail=1', output)

    def test_valid_runs_pass(self):
        status, rows, output, _ = self.run_gate([0] * 4)
        self.assertEqual(status, 0)
        self.assertEqual(len(rows), 4)
        self.assertIn('fail=0', output)

    def test_hash_difference_remains_diagnostic(self):
        status, _, output, _ = self.run_gate([0] * 4, [b'a', b'same', b'b', b'same'])
        self.assertEqual(status, 0)
        self.assertIn('output-hash differences: 1/2', output)


if __name__ == '__main__':
    unittest.main()
