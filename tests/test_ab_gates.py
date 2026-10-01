import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.dflash2_training import ab_suite, server_ab, ninfer_gate


class Gates(unittest.TestCase):
    def cli(self, responses, *extra):
        with tempfile.TemporaryDirectory() as out:
            argv = ['ab_suite', '--model', 'mock.ninfer', '--out', out,
                    '--pairs', '2', '--discard', '1', '--cooldown', '0',
                    '--arms', 'dflash2-k7', '--workloads', 'prose', *extra]
            rc = 0
            with patch.object(sys, 'argv', argv), patch.object(ab_suite.subprocess, 'run', side_effect=responses), contextlib.redirect_stdout(io.StringIO()):
                try:
                    ab_suite.main()
                except SystemExit as exc:
                    rc = exc.code
            result = json.loads((Path(out) / 'ab-results.json').read_text()) if (Path(out) / 'ab-results.json').exists() else None
            return rc, result

    def process(self, text=b'same', rc=0, speed=100):
        return subprocess.CompletedProcess([], rc, text, f'decode speed {speed} tok/s'.encode())

    def test_warmup_exit_fails(self):
        rc, result = self.cli([self.process(rc=1), self.process(), self.process(), self.process()])
        self.assertEqual(rc, 2)
        self.assertFalse(result['comparisons']['dflash2-k7']['correct'])

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

    def server(self, response, *extra, level_mutator=None):
        with tempfile.TemporaryDirectory() as out:
            argv = ['server_ab', '--model', 'mock', '--out', out, '--cooldown', '0', '--arms', 'dflash2-k7', '--workloads', 'prose,chat', '--repeats', '1', '--concurrency', '1', *extra]
            rc = 0
            original = server_ab.run_level
            def run_level(*args):
                level = original(*args)
                return level_mutator(level, args[0]) if level_mutator else level
            with patch.object(sys, 'argv', argv), patch.object(server_ab.subprocess, 'Popen'), patch.object(server_ab, 'wait_ready', return_value='mock'), patch.object(server_ab, 'stop_server'), patch.object(server_ab, 'request_json', side_effect=response), patch.object(server_ab, 'run_level', side_effect=run_level), contextlib.redirect_stdout(io.StringIO()):
                try:
                    server_ab.main()
                except SystemExit as exc:
                    rc = exc.code
            result = json.loads((Path(out) / 'server-results.json').read_text()) if (Path(out) / 'server-results.json').exists() else None
            return rc, result

    def response(self, url, payload, timeout):
        return {'choices': [{'message': {'content': 'same'}}], 'usage': {'completion_tokens': 1}}

    def test_server_both_sides_errors_fail(self):
        def response(url, payload, timeout):
            if payload['messages'][0]['content'] == server_ab.WORKLOADS[0].prompt:
                raise RuntimeError('mock response failure')
            return self.response(url, payload, timeout)
        rc, result = self.server(response)
        self.assertEqual(rc, 2)
        self.assertFalse(result['comparisons']['dflash2-k7']['levels'][0]['exact'])

    def test_server_baseline_errors_counted(self):
        def response(url, payload, timeout):
            if ':18080/' in url:
                raise RuntimeError('baseline failure')
            return self.response(url, payload, timeout)
        rc, result = self.server(response)
        self.assertEqual(rc, 2)
        self.assertEqual(result['comparisons']['dflash2-k7']['levels'][0]['errors'], 2)

    def test_server_missing_requests_fail(self):
        def missing(level, base):
            level['rows'].pop()
            return level
        rc, _ = self.server(self.response, level_mutator=missing)
        self.assertEqual(rc, 2)

    def test_server_duplicate_requests_fail(self):
        def duplicate(level, base):
            level['rows'][-1] = dict(level['rows'][0])
            return level
        rc, _ = self.server(self.response, level_mutator=duplicate)
        self.assertEqual(rc, 2)

    def test_server_baseline_only_rejected(self):
        rc, _ = self.server(self.response, '--arms', 'baseline')
        self.assertNotEqual(rc, 0)

    def test_server_tree_matrix_sends_zero_penalties_on_every_arm(self):
        requests = []
        def transport(url, payload, timeout):
            requests.append((url, payload))
            return self.response(url, payload, timeout)
        rc, result = self.server(transport, '--arms', 'tree7')
        self.assertEqual(rc, 0)
        self.assertEqual(result['comparisons']['dflash2-k7']['base'], 'baseline')
        self.assertEqual(result['comparisons']['tree7']['base'], 'dflash2-k7')
        self.assertEqual(len(requests), 6)
        for port in (18080, 18081, 18082):
            selected = [payload for url, payload in requests if f':{port}/' in url]
            self.assertEqual(len(selected), 2)
            self.assertEqual({p['messages'][0]['content'] for p in selected},
                             {w.prompt for w in server_ab.WORKLOADS[:2]})
            for payload in selected:
                with self.subTest(port=port):
                    self.assertEqual(payload.get('presence_penalty'), 0)
                    self.assertEqual(payload.get('frequency_penalty'), 0)
                    self.assertEqual(payload['temperature'], 0)

    def test_server_valid_passes(self):
        rc, result = self.server(self.response)
        self.assertEqual(rc, 0)
        self.assertTrue(result['comparisons']['dflash2-k7']['levels'][0]['exact'])


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
