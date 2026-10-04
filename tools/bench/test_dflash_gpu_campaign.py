"""Campaign qualification protects held-out comparisons and public timing evidence."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.bench.run_dflash_gpu_campaign import (
    CANDIDATES, Campaign, parse_session_packet, prompts, rank_pairs, timing_metrics,
    validate_correctness_evidence, validate_prompt_split, validate_session_packet,
)
from tools.bench.run_auto_spec_router_comparison import validate_measurement
from tools.bench.test_auto_spec_router_comparison import configuration, native_report, profile


def pairs():
    result = []
    for index in range(2):
        runs = {}
        for name, rate in zip(CANDIDATES, (100, 120, 130, 125, 140, 145)):
            runs[name] = {'configuration_key': 'same-public-configuration', 'response_signature': 'complete-response',
                          'committed_tokens_per_round': 3,
                          'metrics': {'cold': {'throughput': rate, 'request_p95_ns': 1e9,
                                               'first_token_p50_seconds': .2,
                                               'request_decode_p50_tokens_per_second': rate,
                                               'prompt_tokens': [64]}}}
        result.append({'order': list(CANDIDATES if index == 0 else reversed(CANDIDATES)), 'runs': runs})
    return result


def timings(raw):
    return {'artifact_type': 'ninfer_generation_timings', 'schema_version': 1,
            'requests': [{**{k: r[k] for k in ('repeat', 'slot', 'latency_ns', 'prompt_tokens', 'reused_prompt_tokens')},
                          'generated_tokens': len(r['generated_token_ids']), 'first_token_seconds': .000001,
                          'generation_wall_seconds': .000004, 'decode_seconds': .000003,
                          'prompt_wall_seconds': .000001, 'queue_wait_seconds': 0}
                         for r in raw['requests']]}


class CampaignTests(unittest.TestCase):
    def test_single_client_command_fits_native_capacity_limit(self):
        campaign = Campaign.__new__(Campaign)
        campaign.args = SimpleNamespace(model=Path('/explicit/model.ninfer'), max_context=32768,
                                        kv_capacity=131072, max_tokens=512)
        campaign.exe = Path('/explicit/build/bench/ninfer_spec_router_calibration_bench')
        for clients, expected in ((1, 32768), (2, 131072), (4, 131072), (8, 131072)):
            command = campaign.base(clients)
            capacity = int(command[command.index('--engine-concurrency') + 1])
            requested = int(command[command.index('--kv-capacity') + 1])
            self.assertLessEqual(requested, campaign.args.max_context * capacity,
                                 'native rejects KV capacity above the usable per-slot total')
            self.assertEqual(requested, expected)

    def test_capacity_identity_and_prompt_bounds_match_requested_engine(self):
        campaign = Campaign.__new__(Campaign)
        campaign.args = SimpleNamespace(max_context=32768, kv_capacity=131072, max_tokens=512)
        identity = {'max_concurrency': 1, 'max_context': 32768,
                    'resolved_kv_capacity': 32768, 'proposal_head': 'full'}
        campaign.check_identity_capacity(identity, 1)
        identity['resolved_kv_capacity'] = 32767
        with self.assertRaisesRegex(ValueError, 'requested capacity'):
            campaign.check_identity_capacity(identity, 1)
        metrics = {'cold': {'prompt_tokens': [15872]}, 'warm': {'prompt_tokens': [15872]}}
        campaign.check_prompt(metrics, {'tier': 'long'}, 8)
        metrics['warm']['prompt_tokens'] = metrics['cold']['prompt_tokens'] = [15873]
        with self.assertRaisesRegex(ValueError, 'context/capacity contract'):
            campaign.check_prompt(metrics, {'tier': 'long'}, 8)

    def test_reuse_only_accepts_completed_same_model_and_build_suite(self):
        args = SimpleNamespace(model=Path('/explicit/model.ninfer'), build_dir=Path('/explicit/build'))
        suite = {'id': 'correctness-suite', 'ok': True, 'cases_passed': 18, 'ownership_checks': 5,
                 'scope': 'common_weight_residency', 'model_load_count': 1, 'resident_weight_bytes': 1234}
        evidence = {'artifact_type': 'ninfer_dflash_gpu_campaign', 'schema_version': 1,
                    'complete': False, 'configuration': {'model': str(args.model), 'build_dir': str(args.build_dir)},
                    'correctness_suite': suite, 'failure': 'later calibration failed'}
        self.assertEqual(validate_correctness_evidence(evidence, args), suite)
        for name in ('model', 'build_dir'):
            changed = copy.deepcopy(evidence)
            changed['configuration'][name] += '-other'
            with self.assertRaisesRegex(ValueError, 'model/build directory mismatch'):
                validate_correctness_evidence(changed, args)
        for field, value in (('ok', False), ('cases_passed', 17), ('ownership_checks', 4),
                             ('model_load_count', 2), ('scope', 'other')):
            changed = copy.deepcopy(evidence)
            changed['correctness_suite'][field] = value
            with self.assertRaises((ValueError, RuntimeError)):
                validate_correctness_evidence(changed, args)
        evidence.pop('correctness_suite')
        with self.assertRaisesRegex(ValueError, 'missing completed'):
            validate_correctness_evidence(evidence, args)

    def test_explicit_reuse_preserves_new_session_load_evidence_and_records_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(model=root / 'model.ninfer', build_dir=root / 'build',
                                   out=root / 'output', reuse_correctness=root / 'prior.json')
            suite = {'id': 'correctness-suite', 'ok': True, 'cases_passed': 18, 'ownership_checks': 5,
                     'scope': 'common_weight_residency', 'model_load_count': 1, 'resident_weight_bytes': 1234}
            args.reuse_correctness.write_text(json.dumps({
                'artifact_type': 'ninfer_dflash_gpu_campaign', 'schema_version': 1, 'complete': False,
                'configuration': {'model': str(args.model), 'build_dir': str(args.build_dir)},
                'correctness_suite': suite}), encoding='utf-8')
            campaign = Campaign(args)
            ready = {'event': 'ready', 'model_load_count': 1, 'resident_weight_bytes': 5678,
                     'scope': 'common_weight_residency'}
            campaign.summary['resident_sessions'] = {'0': ready}
            # No native session exists: explicit reuse must not execute the suite again.
            campaign.run_correctness()
            saved = json.loads((args.out / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['resident_sessions']['0'], ready)
            self.assertEqual(saved['correctness_suite'], suite)
            self.assertEqual(saved['correctness_evidence_origin'], str(args.reuse_correctness.resolve()))
            self.assertIn('suite not executed in this session', saved['stages'][-1])

    def test_auto_must_beat_best_fixed_not_just_none(self):
        measured = pairs()
        for pair in measured:
            pair['runs']['auto_selected']['metrics']['cold']['throughput'] = 129
        ranked = rank_pairs(measured, 'cold')
        self.assertEqual(ranked['best_fixed'], 'k11')
        self.assertFalse(ranked['comparisons']['auto_selected']['qualified_vs_best_fixed'])
        self.assertEqual(ranked['recommended'], 'auto_full')

    def test_reverse_pair_regression_and_near_tie_do_not_promote(self):
        measured = pairs()
        for pair in measured:
            pair['runs']['auto_full']['metrics']['cold']['throughput'] = 131
        measured[1]['runs']['auto_selected']['metrics']['cold']['request_p95_ns'] = 1.06e9
        ranked = rank_pairs(measured, 'cold')
        self.assertFalse(ranked['comparisons']['auto_full']['qualified_vs_best_fixed'])
        self.assertFalse(ranked['comparisons']['auto_selected']['qualified_vs_best_fixed'])
        self.assertEqual(ranked['recommended'], 'k11')

    def test_output_mismatch_invalidates_fast_run(self):
        measured = pairs()
        measured[1]['runs']['auto_selected']['response_signature'] = 'different-stop-or-token'
        with self.assertRaisesRegex(ValueError, 'output/configuration mismatch'):
            rank_pairs(measured, 'cold')

    def test_mixed_prompt_coverage_is_not_a_valid_comparison(self):
        measured = pairs()
        measured[1]['runs']['k11']['metrics']['cold']['prompt_tokens'] = [4096]
        with self.assertRaisesRegex(ValueError, 'mixed prompt-token coverage'):
            rank_pairs(measured, 'cold')

    def test_missing_sidecar_or_mismatched_request_cannot_estimate_ttft(self):
        raw = native_report('none')
        sidecar = timings(raw)
        self.assertEqual(timing_metrics(raw, sidecar)['cold']['sample_count'], 2)
        sidecar['requests'].pop()
        with self.assertRaisesRegex(ValueError, 'missing timing requests'):
            timing_metrics(raw, sidecar)
        sidecar = timings(raw)
        sidecar['requests'][0]['generated_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'timing/raw response mismatch'):
            timing_metrics(raw, sidecar)

    def test_first_wave_retains_actual_prefix_reuse(self):
        raw = native_report('none')
        raw['requests'][1]['reused_prompt_tokens'] = 32
        result = timing_metrics(raw, timings(raw))
        self.assertEqual(result['cold']['reused_prompt_tokens'], [0, 32])
        self.assertEqual(result['cold']['wave_scope'], 'unprimed first wave')
        self.assertEqual(result['warm']['wave_scope'], 'later natural reuse waves')

    def test_wrong_profile_identity_rejected_by_native_wrapper_validator(self):
        wrong = copy.deepcopy(profile())
        wrong['identity']['artifact_id'] = 'abcdef0123456789abcdef0123456789'
        with self.assertRaisesRegex(ValueError, 'startup identity/configuration mismatch'):
            validate_measurement(native_report(), 'auto', wrong, configuration(), '/explicit/model.ninfer', 0, 7)

    def test_calibration_and_validation_prompts_are_disjoint(self):
        workloads = prompts()
        validate_prompt_split(workloads)
        workloads['validation'][0]['prompt'] = workloads['train'][0]['prompt']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_prompt_split(workloads)


class SessionProtocolTests(unittest.TestCase):
    def test_serialized_ready_and_job_require_single_weight_load(self):
        ready = parse_session_packet('{"event":"ready","model_load_count":1,"resident_weight_bytes":1234,"scope":"common_weight_residency"}')
        validate_session_packet(ready, ready=True)
        response = parse_session_packet('{"id":"C1-K7","ok":true,"model_load_count":1,"resident_weight_bytes":1234}')
        validate_session_packet(response, 'C1-K7')
        response['model_load_count'] = 2
        with self.assertRaisesRegex(ValueError, 'load model exactly once'):
            validate_session_packet(response, 'C1-K7')

    def test_shutdown_and_correctness_suite_use_existing_resident_owner(self):
        bye = parse_session_packet('{"id":"stop","ok":true,"event":"bye","teardown_checks":1,"model_load_count":1,"resident_weight_bytes":1234}')
        validate_session_packet(bye, 'stop', bye=True)
        suite = parse_session_packet('{"id":"suite","ok":true,"cases_passed":18,"ownership_checks":5,"scope":"common_weight_residency","model_load_count":1,"resident_weight_bytes":1234}')
        validate_session_packet(suite, 'suite', suite=True)
        suite['cases_passed'] = 17
        with self.assertRaisesRegex(ValueError, 'all eighteen'):
            validate_session_packet(suite, 'suite', suite=True)

    def test_missing_owner_or_teardown_checks_cannot_complete_campaign(self):
        suite = parse_session_packet('{"id":"suite","ok":true,"cases_passed":18,"ownership_checks":4,"scope":"common_weight_residency","model_load_count":1,"resident_weight_bytes":1234}')
        with self.assertRaisesRegex(ValueError, 'five ownership'):
            validate_session_packet(suite, 'suite', suite=True)
        bye = parse_session_packet('{"id":"stop","ok":true,"event":"bye","teardown_checks":0,"model_load_count":1,"resident_weight_bytes":1234}')
        with self.assertRaisesRegex(ValueError, 'teardown check'):
            validate_session_packet(bye, 'stop', bye=True)

    def test_missing_load_evidence_cannot_complete_a_matrix_job(self):
        response = parse_session_packet('{"id":"C1-K7","ok":true,"resident_weight_bytes":1234}')
        with self.assertRaisesRegex(ValueError, 'protocol fields'):
            validate_session_packet(response, 'C1-K7')

    def test_out_of_order_packet_cannot_be_attributed_to_current_job(self):
        response = parse_session_packet('{"id":"C1-K11","ok":true,"model_load_count":1,"resident_weight_bytes":1234}')
        with self.assertRaisesRegex(ValueError, 'out-of-order'):
            validate_session_packet(response, 'C1-K7')

    def test_error_packet_fails_even_if_it_contains_load_evidence(self):
        response = parse_session_packet('{"id":"C1-K7","ok":false,"error":"invalid profile identity","model_load_count":1,"resident_weight_bytes":1234}')
        with self.assertRaisesRegex(RuntimeError, 'invalid profile identity'):
            validate_session_packet(response, 'C1-K7')


if __name__ == '__main__':
    unittest.main()
