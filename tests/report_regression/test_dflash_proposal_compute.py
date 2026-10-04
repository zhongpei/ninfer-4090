"""CPU-only contract tests; native GPU qualification is a separate explicit local run."""
import copy
import unittest

from tools.bench.calibrated_router_profile import (
    CACHE_FIELDS, EXECUTION_FLAGS, analyse_record, build_profile, physical_proposal_width,
    proposal_compute_mode, validate_identity,
)
from tools.bench.run_resident_spec_router_calibration import constant_profile
from tools.bench.run_auto_spec_router_comparison import (
    COUNTERS, profile_table, qualify_both_fixed, validate_measurement,
)
from tools.bench.run_dflash_proposal_compute_ab import qualify_compute_pairs


def identity():
    cache = {key: 1 for key in CACHE_FIELDS}
    cache['enabled'] = True
    execution = {key: False for key in EXECUTION_FLAGS}
    execution.update(vision_residency='resident', vision_max_merged_tokens=256,
                     rope_scaling_factor=1.0, rope_scaling_original_context=262144)
    return dict(artifact_id='0123456789abcdef0123456789abcdef', prefill_signature='fixture',
                hardware_class='RTX4090-sm89', backend='dflash2', kv_storage='int8',
                startup_draft_tokens=15, proposal_head='full', use_cuda_graph=True,
                max_concurrency=8, max_context=32768, prefill_chunk=1024,
                resolved_kv_capacity=65536, context_cache=cache, execution_options=execution)


def record(action, mode='full', batch=2, wall=2000000):
    requests, rounds = [], []
    for repeat in range(2):
        for slot in range(batch):
            requests.append(dict(repeat=repeat, slot=slot, latency_ns=wall - 100,
                                 prompt_tokens=64, finish_reason=1, prefix_reuse_path=0,
                                 reused_prompt_tokens=0, content='fixture output', reasoning='',
                                 matched_stop_string=None, tool_calls=[], generated_token_ids=list(range(16))))
        remaining, frontier = 15, 64
        while remaining:
            count = min(action + 1, remaining)
            rounds.append(dict(round_index=len(rounds), active_batch=batch,
                               max_execution_frontier=frontier, verify_width=action + 1,
                               draft_tokens=action, proposal_width=physical_proposal_width(action, mode),
                               backend=3, neural_drafter_executed=bool(action),
                               committed_tokens=count * batch, elapsed_ns=1000))
            remaining -= count
            frontier += count
    return dict(identity=identity(), requested_action=action,
                configuration=dict(prompt='fixture prompt', max_tokens=16, client_concurrency=batch,
                                   repeats=2, sampling=dict(temperature=0, presence_penalty=0, frequency_penalty=0)),
                requests=requests, rounds=rounds, wave_wall_ns=[wall, wall])


def auto_wrapper(mode='selected', action=7):
    item = record(action, mode)
    item.pop('requested_action')
    policy = constant_profile(identity(), action, mode)
    memory = dict(workspace_logical_peak_bytes=1024, workspace_allocator_peak_bytes=2048,
                  runtime_reservation_bytes=4096, cuda_graph_allowance_bytes=512,
                  cuda_graph_definition_count=4, cuda_graph_executable_count=4,
                  cuda_graph_prepare_peak_device_delta_bytes=512, cuda_graph_prepare_device_delta_bytes=256)
    return policy, dict(schema_version=1, artifact_type='ninfer_auto_router_measurement', record=item,
                        priming={'enabled': False}, routing_profile={'path': '/fixture/profile.json', 'cells': policy['cells']},
                        resources=dict(memory=memory, routing_counters_scope='published_engine_snapshot_including_priming',
                                       runtime_stats=dict.fromkeys(COUNTERS, 0)))


class ProposalComputeContracts(unittest.TestCase):
    def test_legacy_defaults_and_selected_widths(self):
        validate_identity(identity())
        for action in (0, 7, 11, 15):
            legacy = constant_profile(identity(), action)
            compact = constant_profile(identity(), action, 'selected')
            self.assertEqual(legacy['schema_version'], 1)
            self.assertNotIn('proposal_compute', legacy)
            self.assertEqual(proposal_compute_mode(legacy), 'full')
            self.assertEqual(proposal_compute_mode(compact), 'selected')
            self.assertEqual(physical_proposal_width(action, 'selected'), action + 1 if action else 0)
            self.assertEqual(profile_table(legacy), profile_table(compact))

    def test_strict_mode_headers(self):
        invalid = [{'schema_version': 1, 'proposal_compute': 'full'}, {'schema_version': 2},
                   {'schema_version': True}, {'schema_version': 2.0, 'proposal_compute': 'selected'},
                   {'schema_version': 3, 'proposal_compute': 'selected'}]
        invalid += [{'schema_version': 2, 'proposal_compute': mode} for mode in ('auto', '', None, True, 8)]
        for document in invalid:
            with self.subTest(document=document), self.assertRaises(ValueError):
                proposal_compute_mode(document)
        for action in (True, -1, 1, 8, 16):
            with self.assertRaises(ValueError):
                physical_proposal_width(action, 'selected')

    def test_each_actual_batch_and_action(self):
        for batch in (1, 2, 4, 8):
            for action in (0, 7, 11, 15):
                for mode in ('full', 'selected'):
                    item = record(action, mode, batch)
                    analysis = analyse_record(item, identity(), action, mode)
                    self.assertEqual(analysis['cell'], (batch, 1024))
                    self.assertTrue(analysis['stable_response'])
        for action in (7, 11):
            for measured, claimed in (('full', 'selected'), ('selected', 'full')):
                with self.assertRaises(ValueError):
                    analyse_record(record(action, measured), identity(), action, claimed)

    def test_zero_action_cannot_claim_drafter_work(self):
        for field, bad in (('proposal_width', 1), ('neural_drafter_executed', True)):
            item = record(0, 'selected')
            item['rounds'][0][field] = bad
            with self.assertRaises(ValueError):
                analyse_record(item, identity(), 0, 'selected')

    def test_matched_stop_string_is_strict(self):
        item = record(7, 'selected')
        item['requests'][0]['matched_stop_string'] = 17
        with self.assertRaises(ValueError):
            analyse_record(item, identity(), 7, 'selected')

    def test_profile_uses_only_matching_measured_compute(self):
        pairs = [dict(pair_id=str(i), order=[0, 7] if i == 0 else [7, 0],
                      baseline=record(0, 'selected'), candidate=record(7, 'selected', wall=1000000))
                 for i in range(2)]
        document = dict(schema_version=2, proposal_compute='selected',
                        artifact_type='ninfer_resident_router_measurements', identity=identity(),
                        comparisons=[dict(workload='fixture', candidate_action=7, pairs=pairs)])
        policy = build_profile(document)
        self.assertEqual(policy['proposal_compute'], 'selected')
        self.assertTrue(any(c['draft_tokens'] == 7 for c in policy['cells']))
        bad = copy.deepcopy(document)
        bad['proposal_compute'] = 'full'
        rejected = build_profile(bad)
        self.assertTrue(all(c['draft_tokens'] == 0 for c in rejected['cells']))
        self.assertFalse(rejected['provenance']['comparisons'][0]['qualified'])

    def test_auto_validates_mode_not_only_target_width(self):
        policy, raw = auto_wrapper()
        result = validate_measurement(raw, 'auto', policy, raw['record']['configuration'], '/model', 0, 7)
        self.assertGreater(result['raw_round_action_counts']['7'], 0)
        raw['record']['rounds'][0]['proposal_width'] = 16
        with self.assertRaises(ValueError):
            validate_measurement(raw, 'auto', policy, raw['record']['configuration'], '/model', 0, 7)

    def test_compute_pairs_require_both_orders_and_complete_output(self):
        def pairs(action):
            return [dict(pair_id=str(i), order=['full', 'selected'] if i == 0 else ['selected', 'full'],
                         full=analyse_record(record(action), identity(), action),
                         selected=analyse_record(record(action, 'selected', wall=1000000), identity(), action, 'selected'))
                    for i in range(2)]
        good = pairs(7)
        self.assertTrue(qualify_compute_pairs(good, 7)['qualified_compute_speedup'])
        for action in (0, 15):
            control = qualify_compute_pairs(pairs(action), action)
            self.assertTrue(control['is_aa_control'])
            self.assertFalse(control['qualified_compute_speedup'])
        bad = copy.deepcopy(good)
        bad[1]['selected']['response_key'] = 'different complete output'
        self.assertFalse(qualify_compute_pairs(bad, 7)['correct'])
        bad = copy.deepcopy(good)
        bad[1]['order'] = ['full', 'selected']
        with self.assertRaises(ValueError):
            qualify_compute_pairs(bad, 7)
        bad = copy.deepcopy(good)
        bad[1]['selected']['cell'] = None
        self.assertFalse(qualify_compute_pairs(bad, 7)['qualified_compute_speedup'])

    def test_auto_must_beat_both_fixed_controls(self):
        def comparison(k, passed=True, response='same'):
            return dict(workload='fixture', requested_clients=2, standalone_fixed_k=k,
                        qualification=dict(correct=True, vs_standalone_fixed={'qualified_performance': passed}),
                        pairs=[{'runs': {'none': {'analysis': {'responses': response}}}}])
        self.assertFalse(qualify_both_fixed([comparison(7)])['qualified_incremental_value'])
        self.assertFalse(qualify_both_fixed([comparison(7), comparison(15, False)])['qualified_incremental_value'])
        self.assertFalse(qualify_both_fixed([comparison(7), comparison(15, response='different')])['qualified_incremental_value'])
        self.assertTrue(qualify_both_fixed([comparison(7), comparison(15)])['qualified_incremental_value'])
        with self.assertRaises(ValueError):
            qualify_both_fixed([comparison(7), comparison(7)])


if __name__ == '__main__':
    unittest.main()
