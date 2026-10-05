"""Host qualification for mixed routes and true independent baseline arms."""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bench import run_auto_spec_router_comparison as bench
from tools.bench.test_calibrated_router_profile import identity, record


def profile():
    return {'schema_version': 1, 'artifact_type': 'ninfer_spec_router_profile',
            'identity': identity(), 'cells': [
                {'active_batch': 2, 'frontier_upper': 1024, 'draft_tokens': 7},
                {'active_batch': 2, 'frontier_upper': 8192, 'draft_tokens': 11}]}


def configuration():
    return record()['configuration']


def native_report(arm='auto', wall=9000, latency=6000):
    measured = record(7 if arm != 'none' else 0, wall=wall, latency=latency)
    if arm == 'auto':
        measured.pop('requested_action')
        for i, event in enumerate(measured['rounds']):
            if i == 1:
                event.update(max_execution_frontier=1025, draft_tokens=11, verify_width=12)
            elif i in (2, 3):
                event.update(draft_tokens=0, verify_width=1, proposal_width=0, neural_drafter_executed=False)
                if i == 2:
                    event['max_execution_frontier'] = 8193
                else:
                    event.update(active_batch=1, committed_tokens=1)
            elif i == 4:
                event['committed_tokens'] = 3
        return {'artifact_type': 'ninfer_auto_router_measurement', 'schema_version': 1,
                'record': measured, 'priming': {'enabled': False},
                'routing_profile': {'path': '/control/profile.json', 'cells': bench.profile_table(profile())},
                'resources': {'memory': {'workspace_logical_peak_bytes': 100,
                                        'workspace_allocator_peak_bytes': 100,
                                        'runtime_reservation_bytes': 1000,
                                        'cuda_graph_allowance_bytes': 100,
                                        'cuda_graph_definition_count': 2, 'cuda_graph_executable_count': 2,
                                        'cuda_graph_prepare_peak_device_delta_bytes': 50,
                                        'cuda_graph_prepare_device_delta_bytes': 50},
                              'routing_counters_scope': 'published_engine_snapshot_including_priming',
                              'runtime_stats': dict.fromkeys(bench.COUNTERS, 0)}}
    k = 7 if arm == 'fixed' else 0
    for event in measured['rounds']:
        event.pop('round_index')
        event.update(backend=3 if k else 0, proposal_width=k+1 if k else 0)
    return {'artifact_type': 'ninfer_calibration_measurement', 'schema_version': 1,
            'model': '/explicit/model.ninfer', 'model_name': 'test', 'device': 0,
            'client_concurrency': 2, 'engine_concurrency': 8, 'max_context': 32768,
            'kv_capacity': 262144, 'draft_tokens': k, 'proposal_head': 'full', 'repeats': 2,
            'max_tokens': 8, 'kv': 'int8', 'cache': True, 'graph': True, 'greedy': True,
            'presence_penalty': 0, 'frequency_penalty': 0,
            'requests': measured['requests'], 'rounds': measured['rounds'],
            'wave_wall_ns': measured['wave_wall_ns'],
            'memory': {'workspace_logical_peak_bytes': 100, 'workspace_allocator_peak_bytes': 100,
                       'runtime_reservation_bytes': 300 if arm == 'none' else 600}}


def analyse(raw, arm):
    return bench.validate_measurement(raw, arm, profile(), configuration(),
                                      '/explicit/model.ninfer', 0, 7)


def triple(wall=9000, latency=6000):
    return {arm: analyse(native_report(arm, wall if arm == 'auto' else 10000,
                                     latency if arm == 'auto' else 6000), arm)
            for arm in ('none', 'fixed', 'auto')}


def baseline_profile():
    return {'schema_version': 3, 'artifact_type': 'ninfer_spec_router_profile',
            'identity': identity(), 'proposal_compute': 'selected', 'default_action': 7,
            'cells': [
                {'active_batch': 2, 'frontier_upper': 8192, 'draft_tokens': 11},
                {'active_batch': 8, 'frontier_upper': 32768, 'draft_tokens': 0}]}


class AutoComparisonTest(unittest.TestCase):
    def test_schema3_sparse_table_expands_missing_cells_to_k7(self):
        table = bench.profile_table(baseline_profile())
        actions = {(c['active_batch'], c['frontier_upper']): c['draft_tokens'] for c in table}
        self.assertEqual(len(table), 24)
        self.assertEqual(actions[2, 1024], 7)
        self.assertEqual(actions[2, 8192], 11)
        self.assertEqual(actions[8, 32768], 0)
        bad = baseline_profile(); bad['default_action'] = 0
        with self.assertRaisesRegex(ValueError, 'default action'):
            bench.profile_table(bad)
        bad = baseline_profile(); bad['cells'][0]['draft_tokens'] = 15
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            bench.profile_table(bad)

    def test_sparse_table_normalized_and_mixed_actual_actions(self):
        table = bench.profile_table(profile())
        self.assertEqual(len(table), 24)
        self.assertEqual(sum(c['draft_tokens'] != 0 for c in table), 2)
        result = analyse(native_report(), 'auto')
        self.assertEqual(result['raw_round_action_counts'], {'0': 2, '7': 11, '11': 1, '15': 0})
        self.assertEqual(result['observed_route_switches'], 3)
        self.assertEqual(result['published_routing_counters'], dict.fromkeys(bench.COUNTERS, 0))
        self.assertEqual({c['active_batch'] for c in result['actual_cells']}, {1, 2})
        self.assertEqual(result['committed_tokens'], 28)

    def test_wrong_physical_actual_cell_or_table_rejected(self):
        for mutation in (
                lambda p: p['record']['rounds'][0].update(verify_width=16),
                lambda p: p['record']['rounds'][0].update(proposal_width=8),
                lambda p: p['record']['rounds'][0].update(neural_drafter_executed=False),
                lambda p: p['record']['rounds'][0].update(active_batch=1),
                lambda p: p['record']['rounds'][0].update(max_execution_frontier=8193),
                lambda p: p['routing_profile']['cells'][0].update(draft_tokens=7),
                lambda p: p['routing_profile']['cells'][0].update(active_batch=True),
                lambda p: p['record']['identity'].update(hardware_class='other'),
                lambda p: p['record']['rounds'][0].update(round_index=1),
                lambda p: p['resources']['runtime_stats'].update(calibrated_fixed_fallback_rounds=1)):
            raw = native_report()
            mutation(raw)
            with self.assertRaises(ValueError):
                analyse(raw, 'auto')

    def test_none_is_backend_none_not_resident_zero_and_fixed_is_standalone_width(self):
        self.assertEqual(analyse(native_report('none'), 'none')['execution_kind'], 'true_none')
        raw = native_report('none')
        raw['rounds'][0]['backend'] = 3
        with self.assertRaises(ValueError):
            analyse(raw, 'none')
        raw = native_report('fixed')
        raw['rounds'][0]['proposal_width'] = 16
        with self.assertRaises(ValueError):
            analyse(raw, 'fixed')
        self.assertEqual(analyse(native_report('fixed'), 'fixed')['execution_kind'], 'standalone_fixed_k7')
        raw = native_report('none'); raw['graph'] = 1
        with self.assertRaises(ValueError): analyse(raw, 'none')

    def test_complete_response_and_repeat_stability_gate(self):
        for field, value in (('content', 'hello'), ('generated_token_ids', [99]*8),
                             ('reasoning', 'hidden'), ('tool_calls', [{'name':'tool','arguments_json':'{}'}]),
                             ('matched_stop_string', 'stop')):
            pairs = [triple(), triple()]
            raw = native_report()
            raw['record']['requests'][0][field] = value
            pairs[1]['auto'] = analyse(raw, 'auto')
            result = bench.qualify_triples(pairs)
            self.assertFalse(result['correct'])
            self.assertFalse(result['vs_true_none']['qualified_performance'])
            self.assertIsNone(result['vs_standalone_fixed']['median_speedup'])

    def test_all_pairs_gain_median_and_each_p95_gate(self):
        self.assertTrue(bench.qualify_triples([triple(), triple()])['vs_true_none']['qualified_performance'])
        for pairs in ([triple(), triple(wall=10001)], [triple(wall=9900), triple(wall=9900)],
                      [triple(latency=6400), triple()]):
            result = bench.qualify_triples(pairs)
            self.assertTrue(result['correct'])
            self.assertFalse(result['vs_true_none']['qualified_performance'])
            self.assertFalse(result['vs_standalone_fixed']['qualified_performance'])

    def test_uncovered_target_only_auto_legal_and_fullround_after_wave_legal(self):
        p = profile(); p['cells'] = []
        raw = native_report()
        raw['routing_profile']['cells'] = bench.profile_table(p)
        for e in raw['record']['rounds']:
            e.update(active_batch=2, committed_tokens=2, draft_tokens=0, verify_width=1,
                     proposal_width=0, neural_drafter_executed=False, elapsed_ns=20000)
        result = bench.validate_measurement(raw, 'auto', p, configuration(), '/explicit/model.ninfer', 0, 7)
        self.assertEqual(result['raw_round_action_counts']['0'], 14)
        self.assertGreater(result['full_round_elapsed_ns'], result['wave_wall_ns'])

    def test_profile_duplicates_unknown_fields_and_old_identity_rejected(self):
        for mutation in (lambda p: p['cells'].append(copy.deepcopy(p['cells'][0])),
                         lambda p: p['cells'][0].update(extra=1),
                         lambda p: p['identity'].update(startup_draft_tokens=7)):
            p = profile(); mutation(p)
            with self.assertRaises(ValueError): bench.profile_table(p)

    def test_duplicate_requests_zero_commit_tail_and_latency_bounds(self):
        raw = native_report()
        event = copy.deepcopy(raw['record']['rounds'][-1])
        event.update(round_index=len(raw['record']['rounds']), committed_tokens=0)
        raw['record']['rounds'].append(event)
        self.assertEqual(analyse(raw, 'auto')['committed_tokens'], 28)
        for mutation in (lambda p: p['record']['requests'].append(copy.deepcopy(p['record']['requests'][0])),
                         lambda p: p['record']['requests'][0].update(latency_ns=10000)):
            raw = native_report();mutation(raw)
            with self.assertRaises(ValueError):analyse(raw,'auto')


class AutoDriverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.exe, self.model, self.profile = (self.root/name for name in ('native', 'model.ninfer', 'profile.json'))
        self.exe.touch(); self.model.touch()
        self.profile_bytes = json.dumps(profile(), indent=3).encode() + b'\n'
        self.profile.write_bytes(self.profile_bytes)
        self.commands = []
        self.failure_arm = None
        self.mutate_auto = None

    def tearDown(self):
        self.temp.cleanup()

    def argv(self):
        return ['--exe', str(self.exe), '--model', str(self.model), '--profile', str(self.profile),
                '--out', str(self.root/'out'), '--workloads', 'chat', '--concurrency', '2',
                '--fixed-draft-tokens', '7', '--max-tokens', '8', '--kv-capacity', '262144',
                '--nvml-device', 'GPU-explicit', '--cooldown', '0']

    def fake_run(self, command, **kwargs):
        if command[0] == 'nvidia-smi':
            return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
        self.commands.append(command)
        opts = {command[i]:command[i+1] for i in range(1,len(command)-1)
                if command[i].startswith('--') and not command[i+1].startswith('--')}
        arm = 'auto' if '--allow-route-switching' in command else 'none' if opts['--draft-tokens'] == '0' else 'fixed'
        if arm == self.failure_arm:
            return SimpleNamespace(returncode=9, stderr='failure')
        raw = native_report(arm, 9000 if arm == 'auto' else 10000)
        if arm == 'auto':
            raw['record']['configuration']['prompt'] = Path(opts['--prompt-file']).read_bytes().decode('utf-8')
            raw['routing_profile']['path'] = opts['--spec-router-profile']
            if self.mutate_auto: self.mutate_auto(raw)
        else:
            raw['model'] = opts['--model']
        Path(opts['--output']).write_text(json.dumps(raw))
        return SimpleNamespace(returncode=0)

    def run_driver(self):
        with patch('tools.bench.run_resident_spec_router_calibration.subprocess.run', side_effect=self.fake_run):
            return bench.main(self.argv())

    def test_fresh_three_arm_abba_profile_bytes_commands_resources_and_qualification(self):
        self.assertEqual(self.run_driver(), 0)
        kinds = ['auto' if '--allow-route-switching' in c else 'none' if c[c.index('--draft-tokens')+1]=='0' else 'fixed'
                 for c in self.commands]
        self.assertEqual(kinds, ['none','fixed','auto','auto','fixed','none'])
        for command, arm in zip(self.commands, kinds):
            self.assertNotIn('--prime-prefix', command)
            self.assertEqual(command[command.index('--kv-capacity')+1], '262144')
            if arm == 'auto':
                self.assertNotIn('--draft-tokens', command)
                self.assertIn('--spec-router-profile', command)
            else:
                self.assertNotIn('--spec-router-profile', command)
        out = self.root/'out'
        self.assertEqual((out/'profile-input.json').read_bytes(), self.profile_bytes)
        import hashlib
        summary = json.loads((out/'comparison.json').read_text())
        self.assertEqual(summary['profile']['sha256'], hashlib.sha256(self.profile_bytes).hexdigest())
        qualification = summary['comparisons'][0]['qualification']
        self.assertTrue(qualification['correct'])
        self.assertTrue(qualification['vs_true_none']['qualified_performance'])
        self.assertTrue(qualification['vs_standalone_fixed']['qualified_performance'])
        self.assertEqual(len(list((out/'runs').glob('*/native.json'))), 6)
        for path in (out/'runs').glob('*/memory.json'):
            memory = json.loads(path.read_text())
            self.assertEqual(memory['kind'], 'device_memory_observed_lower_bound')
            self.assertEqual(memory['gpu'], 'GPU-explicit')
            self.assertEqual(memory['peak_used_mib'], 12345)

    def test_nonfinite_timing_arguments_rejected_without_native(self):
        for option in ('--cooldown', '--memory-sample-seconds'):
            for value in ('nan', 'inf', '-inf'):
                with self.subTest(option=option, value=value):
                    with self.assertRaises(SystemExit): bench.main(self.argv()+[option+'='+value])
                    self.assertEqual(self.commands, [])
                    self.assertFalse((self.root/'out').exists())

    def test_nvml_sampling_failure_is_audited_as_missing_lower_bound(self):
        original = self.fake_run
        def run(command, **kwargs):
            if command[0] == 'nvidia-smi' and '--query-gpu=memory.used' in command:
                raise subprocess.CalledProcessError(2, command, stderr='sampling denied')
            return original(command, **kwargs)
        with patch('tools.bench.run_resident_spec_router_calibration.subprocess.run', side_effect=run):
            self.assertEqual(bench.main(self.argv()), 0)
        for path in (self.root/'out'/'runs').glob('*/memory.json'):
            memory = json.loads(path.read_text())
            self.assertIsNone(memory['peak_used_mib'])
            self.assertEqual(memory['samples'], [])
            self.assertTrue(memory['errors'])

    def test_native_failure_retains_raw_command_and_memory(self):
        self.failure_arm = 'auto'
        self.assertEqual(self.run_driver(), 1)
        out=self.root/'out'
        self.assertTrue((out/'failure.json').exists())
        self.assertEqual(len(list((out/'runs').glob('*/command.json'))), 3)
        self.assertEqual(len(list((out/'runs').glob('*/memory.json'))), 3)

    def test_correctness_failure_returns_nonzero_not_a_performance_claim(self):
        self.mutate_auto=lambda r: r['record']['requests'][0].update(content='changed')
        self.assertEqual(self.run_driver(), 1)
        summary=json.loads((self.root/'out'/'comparison.json').read_text())
        self.assertFalse(summary['correct'])
        self.assertFalse(summary['comparisons'][0]['qualification']['vs_true_none']['qualified_performance'])

    def test_latency_no_gain_reports_false_without_claiming_error(self):
        self.mutate_auto=lambda r: [request.update(latency_ns=6400) for request in r['record']['requests']]
        self.assertEqual(self.run_driver(), 0)
        summary=json.loads((self.root/'out'/'comparison.json').read_text())
        self.assertTrue(summary['correct'])
        self.assertFalse(summary['comparisons'][0]['qualification']['vs_standalone_fixed']['qualified_performance'])



@unittest.skipUnless(os.environ.get('NINFER_AUTO_NATIVE_EXE'), 'explicit native CPU binary required')
class NativeAutoOptionsTest(unittest.TestCase):
    def test_auto_flag_help(self):
        exe = os.environ['NINFER_AUTO_NATIVE_EXE']
        help_result = subprocess.run([exe, '--help'], capture_output=True, text=True, timeout=30)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('--allow-route-switching', help_result.stdout)

    def test_invalid_auto_flag_combinations(self):
        exe = os.environ['NINFER_AUTO_NATIVE_EXE']
        for args in (['--allow-route-switching'],
                     ['--allow-route-switching', '--spec-router-profile', 'absent', '--draft-tokens', '7'],
                     ['--allow-route-switching', '--spec-router-profile', 'absent', '--identity-only']):
            with self.subTest(args=args):
                result = subprocess.run([exe, *args], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('route switching', result.stderr)


if __name__ == '__main__':
    unittest.main()
