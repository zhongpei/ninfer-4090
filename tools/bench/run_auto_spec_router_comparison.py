"""Fresh-Engine Auto versus true None and standalone fixed through native Engine.

Generation-wave timing excludes model startup and prompt preparation. All arms use
cache/graphs and natural repeat reuse; no explicit primer is run in this campaign.
--compare-both-fixed requires Auto to beat both fixed K7 and K15, not only None.
"""
from __future__ import annotations
import argparse
import hashlib
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tools.bench.calibrated_router_profile import (
    UPPERS, REQUEST_FIELDS, ROUND_FIELDS, boolean, canonical, fields, integer, text, validate_identity,
    physical_proposal_width, proposal_compute_mode,
)
from tools.bench.run_resident_spec_router_calibration import execute_native, read_json, selections, write_json
from tools.bench.run_spec_router_calibration import output_identity, percentile
from tools.dflash2_training.ab_suite import select_workloads

COUNTERS = ('calibrated_target_only_rounds', 'calibrated_k7_rounds', 'calibrated_k11_rounds',
            'calibrated_k15_rounds', 'calibrated_route_switches', 'calibrated_fixed_fallback_rounds')
ARMS = ('none', 'fixed', 'auto')


def profile_table(profile):
    proposal_compute_mode(profile)
    version = profile.get('schema_version')
    required = {'schema_version', 'artifact_type', 'identity', 'cells'}
    if version >= 2:
        required.add('proposal_compute')
    if version == 3:
        required.add('default_action')
    if set(profile) not in (required, required | {'provenance'}):
        raise ValueError('invalid profile fields')
    if profile['artifact_type'] != 'ninfer_spec_router_profile':
        raise ValueError('unsupported profile schema')
    if 'provenance' in profile and not isinstance(profile['provenance'], dict):
        raise ValueError('invalid profile provenance')
    validate_identity(profile['identity'])
    default_action = 0
    if version == 3:
        default_action = integer(profile['default_action'], 'profile default action')
        if default_action != 7:
            raise ValueError('schema3 profile default action must be K7')
    if not isinstance(profile['cells'], list):
        raise ValueError('profile cells must be an array')
    actions = {}
    for cell in profile['cells']:
        fields(cell, ('active_batch', 'frontier_upper', 'draft_tokens'), 'profile cell')
        batch = integer(cell['active_batch'], 'profile batch', 1, 8)
        upper = integer(cell['frontier_upper'], 'profile upper')
        action = integer(cell['draft_tokens'], 'profile action')
        allowed = (0, 11) if version == 3 else (0, 7, 11, 15)
        if upper not in UPPERS or action not in allowed or (batch, upper) in actions:
            raise ValueError('unsupported or duplicate profile cell')
        actions[batch, upper] = action
    return [{'active_batch': b, 'frontier_upper': upper,
             'draft_tokens': actions.get((b, upper), default_action)}
            for b in range(1, 9) for upper in UPPERS]


def validate_memory(memory):
    if not isinstance(memory, dict):
        raise ValueError('missing Engine memory evidence')
    for key in ('workspace_logical_peak_bytes', 'workspace_allocator_peak_bytes', 'runtime_reservation_bytes'):
        integer(memory.get(key), key, 1, (1 << 64) - 1)


def validate_measurement(raw, arm, profile, configuration, model, device, fixed_k):
    """Validate real arm-specific physics before forming paired E2E statistics."""
    table = profile_table(profile)
    compute = proposal_compute_mode(profile)
    identity = profile['identity']
    fields(configuration, ('prompt', 'max_tokens', 'client_concurrency', 'repeats', 'sampling'), 'configuration')
    text(configuration['prompt'], 'prompt', True)
    clients = integer(configuration['client_concurrency'], 'clients', 1, identity['max_concurrency'])
    repeats = integer(configuration['repeats'], 'repeats', 2)
    budget = integer(configuration['max_tokens'], 'budget', 2, identity['max_context'])
    fields(configuration['sampling'], ('temperature', 'presence_penalty', 'frequency_penalty'), 'sampling')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v != 0 for v in configuration['sampling'].values()):
        raise ValueError('only greedy zero-penalty comparison is supported')
    if arm == 'auto':
        fields(raw, ('artifact_type', 'schema_version', 'record', 'resources', 'priming', 'routing_profile'), 'auto wrapper')
        if raw['artifact_type'] != 'ninfer_auto_router_measurement':
            raise ValueError('auto must publish its independent mixed-action schema')
        record = raw['record']
        fields(record, ('identity', 'configuration', 'requests', 'rounds', 'wave_wall_ns'), 'auto record')
        if validate_identity(record['identity']) != validate_identity(identity) or record['configuration'] != configuration:
            raise ValueError('auto startup identity/configuration mismatch')
        fields(raw['routing_profile'], ('path', 'cells'), 'loaded profile audit')
        text(raw['routing_profile']['path'], 'profile path', True)
        loaded_cells = raw['routing_profile']['cells']
        validated_loaded = profile_table({'schema_version': 1, 'artifact_type': 'ninfer_spec_router_profile',
                                          'identity': identity, 'cells': loaded_cells})
        if len(loaded_cells) != 24 or validated_loaded != table or loaded_cells != table:
            raise ValueError('native loader-normalized routing table mismatch')
        fields(raw['priming'], ('enabled',), 'priming')
        if raw['priming']['enabled'] is not False:
            raise ValueError('all comparison arms must use fresh unprimed startup')
        resources = raw['resources']
        fields(resources, ('memory', 'routing_counters_scope', 'runtime_stats'), 'resources')
        if resources['routing_counters_scope'] != 'published_engine_snapshot_including_priming':
            raise ValueError('published counter scope must be explicit')
        fields(resources['runtime_stats'], COUNTERS, 'published routing counters')
        for key, value in resources['runtime_stats'].items():
            integer(value, key, 0, (1 << 64) - 1)
        if resources['runtime_stats']['calibrated_fixed_fallback_rounds']:
            raise ValueError('greedy zero-penalty auto unexpectedly used fixed fallback')
        for key in ('cuda_graph_allowance_bytes', 'cuda_graph_definition_count', 'cuda_graph_executable_count',
                    'cuda_graph_prepare_peak_device_delta_bytes', 'cuda_graph_prepare_device_delta_bytes'):
            integer(resources['memory'].get(key), key, 0, (1 << 64) - 1)
        memory = resources['memory']
        events = record['rounds']
        walls, requests = record['wave_wall_ns'], record['requests']
    elif arm in ('none', 'fixed'):
        if raw.get('artifact_type') != 'ninfer_calibration_measurement':
            raise ValueError('true None/fixed require standalone fixed schema')
        action = 0 if arm == 'none' else fixed_k
        expected = {'model': model, 'device': device, 'client_concurrency': clients,
                    'engine_concurrency': identity['max_concurrency'], 'max_context': identity['max_context'],
                    'kv_capacity': identity['resolved_kv_capacity'], 'draft_tokens': action,
                    'max_tokens': budget, 'repeats': repeats, 'proposal_head': 'full',
                    'kv': 'int8', 'cache': True, 'graph': True, 'greedy': True,
                    'presence_penalty': 0, 'frequency_penalty': 0}
        for key, value in expected.items():
            if (raw.get(key) != value or type(value) is bool and type(raw.get(key)) is not bool
                    or type(value) is int and type(raw.get(key)) is not int):
                raise ValueError('unpaired standalone configuration: ' + key)
        memory = raw['memory']
        events, walls, requests = raw['rounds'], raw['wave_wall_ns'], raw['requests']
        resources = None
    else:
        raise ValueError('unsupported comparison arm')
    if type(raw.get('schema_version')) is not int or raw['schema_version'] != 1:
        raise ValueError('unsupported measurement schema')
    validate_memory(memory)
    if not isinstance(walls, list) or len(walls) != repeats:
        raise ValueError('missing repeat wave timing')
    for wall in walls:
        integer(wall, 'wave_wall_ns', 1, (1 << 64) - 1)
    if not isinstance(requests, list) or len(requests) != clients * repeats:
        raise ValueError('missing complete requests')
    responses, latencies, request_audit = {}, [], []
    generated = 0
    for request in requests:
        fields(request, REQUEST_FIELDS, 'request')
        rep = integer(request['repeat'], 'repeat', 0, repeats - 1)
        slot = integer(request['slot'], 'slot', 0, clients - 1)
        if (rep, slot) in responses:
            raise ValueError('duplicate response')
        latency = integer(request['latency_ns'], 'latency_ns', 1, (1 << 64) - 1)
        if latency > walls[rep]:
            raise ValueError('request latency exceeds enclosing wave')
        prompt_tokens = integer(request['prompt_tokens'], 'prompt_tokens', 1, identity['max_context'])
        integer(request['prefix_reuse_path'], 'prefix source')
        integer(request['reused_prompt_tokens'], 'reused prompt', 0, prompt_tokens)
        finish = integer(request['finish_reason'], 'finish')
        tokens = request['generated_token_ids']
        if finish not in (1, 3, 4) or not isinstance(tokens, list) or not 1 <= len(tokens) <= budget or finish == 1 and len(tokens) != budget:
            raise ValueError('incomplete terminal response/output budget')
        for token in tokens:
            integer(token, 'token', 0, (1 << 31) - 1)
        for key in ('content', 'reasoning'):
            text(request[key], key)
        if request['matched_stop_string'] is not None:
            text(request['matched_stop_string'], 'stop')
        if not isinstance(request['tool_calls'], list):
            raise ValueError('tool calls must be an array')
        for call in request['tool_calls']:
            fields(call, ('name', 'arguments_json'), 'tool call')
            text(call['name'], 'tool name'); text(call['arguments_json'], 'tool arguments')
        responses[rep, slot] = canonical({**output_identity(request), 'prompt_tokens': prompt_tokens})
        generated += len(tokens)
        latencies.append(latency)
        request_audit.append({key: request[key] for key in ('repeat', 'slot', 'latency_ns', 'prompt_tokens',
                                                          'prefix_reuse_path', 'reused_prompt_tokens', 'finish_reason')})
    if not isinstance(events, list) or not events:
        raise ValueError('missing settled raw rounds')
    actions = {(c['active_batch'], c['frontier_upper']): c['draft_tokens'] for c in table}
    counts = dict.fromkeys(('0', '7', '11', '15'), 0)
    cells = {}; commits = 0; switches = 0; previous = None
    for index, event in enumerate(events):
        fields(event, ROUND_FIELDS if arm == 'auto' else tuple(k for k in ROUND_FIELDS if k != 'round_index'), 'round')
        if arm == 'auto' and integer(event['round_index'], 'round_index') != index:
            raise ValueError('round indices must be continuous in collector order')
        batch = integer(event['active_batch'], 'actual batch', 1, clients)
        frontier = integer(event['max_execution_frontier'], 'frontier', 1, min(32768, identity['max_context']))
        upper = next(upper for upper in UPPERS if frontier <= upper)
        expected_action = actions[batch, upper] if arm == 'auto' else 0 if arm == 'none' else fixed_k
        for key in ('draft_tokens', 'verify_width', 'proposal_width', 'backend'):
            integer(event[key], key)
        boolean(event['neural_drafter_executed'], 'neural drafter')
        expected_proposal = (physical_proposal_width(expected_action, compute) if arm == 'auto'
                             else expected_action + 1 if expected_action else 0)
        if (event['draft_tokens'] != expected_action or event['verify_width'] != expected_action + 1
                or event['proposal_width'] != expected_proposal or event['backend'] != (0 if arm == 'none' else 3)
                or event['neural_drafter_executed'] != bool(expected_action)):
            raise ValueError('action/table/physical width/proposal/skip mismatch')
        commit = integer(event['committed_tokens'], 'commits', 0, batch * (expected_action + 1))
        elapsed = integer(event['elapsed_ns'], 'round elapsed', 1, (1 << 64) - 1)
        commits += commit
        counts[str(expected_action)] += 1
        switches += int(previous is not None and previous != expected_action)
        previous = expected_action
        stats = cells.setdefault((batch, upper, expected_action), {'rounds': 0, 'committed_tokens': 0, 'elapsed_ns': 0, 'times': []})
        stats['rounds'] += 1; stats['committed_tokens'] += commit; stats['elapsed_ns'] += elapsed; stats['times'].append(elapsed)
    if commits == 0 or all(r['finish_reason'] == 1 for r in requests) and commits != generated - len(requests):
        raise ValueError('settled commits/first generated token accounting mismatch')
    cell_audit = []
    for (batch, upper, action), stats in sorted(cells.items()):
        times = stats.pop('times')
        cell_audit.append({'active_batch': batch, 'frontier_upper': upper, 'draft_tokens': action, **stats,
                           'round_p50_ns': percentile(times, 50), 'round_p95_ns': percentile(times, 95)})
    return {'execution_kind': 'calibrated_auto_resident_k15' if arm == 'auto' else 'true_none' if arm == 'none' else f'standalone_fixed_k{fixed_k}',
            'configuration_key': canonical(configuration), 'responses': [[rep, slot, value] for (rep, slot), value in sorted(responses.items())],
            'stable_response': len(set(responses.values())) == 1,
            'generated_tokens': generated, 'committed_tokens': commits, 'wave_wall_ns': sum(walls),
            'full_round_elapsed_ns': sum(e['elapsed_ns'] for e in events),
            'end_to_end_tokens_per_second': generated * 1e9 / sum(walls),
            'request_p50_ns': percentile(latencies, 50), 'request_p95_ns': percentile(latencies, 95),
            'raw_round_action_counts': counts, 'observed_route_switches': switches,
            'actual_cells': cell_audit, 'requests': request_audit, 'memory': memory,
            'published_routing_counters': resources['runtime_stats'] if resources else None,
            'counter_scope': 'published snapshot may lag final raw rounds; independent round counts above'}


def qualify_triples(pairs):
    if len(pairs) < 2:
        raise ValueError('at least two AB/BA triples required')
    canonical_response = pairs[0]['none']['responses']
    configuration_key = pairs[0]['none']['configuration_key']
    failures = []
    for index, pair in enumerate(pairs):
        if set(pair) != set(ARMS):
            raise ValueError('missing independent comparison arms')
        for arm, result in pair.items():
            if result['configuration_key'] != configuration_key:
                raise ValueError('unpaired prompt/configuration')
            if result['responses'] != canonical_response or not result['stable_response']:
                failures.append({'pair': index, 'arm': arm, 'reason': 'full response/repeat/slot instability'})
    correct = not failures
    summary = {'correct': correct, 'failures': failures}
    for baseline, label in (('none', 'vs_true_none'), ('fixed', 'vs_standalone_fixed')):
        ratios = [p['auto']['end_to_end_tokens_per_second'] / p[baseline]['end_to_end_tokens_per_second'] for p in pairs]
        latency = [p['auto']['request_p95_ns'] / p[baseline]['request_p95_ns'] for p in pairs]
        summary[label] = {'paired_speedups': ratios, 'paired_p95_ratios': latency,
                          'median_speedup': statistics.median(ratios) if correct else None,
                          'qualified_performance': correct and all(r > 1 for r in ratios)
                          and statistics.median(ratios) >= 1.02 and all(r <= 1.05 for r in latency)}
    return summary


def qualify_both_fixed(comparisons):
    """A None speedup or a single fixed comparator never establishes incremental value."""
    groups = {}
    for comparison in comparisons:
        key = (comparison['workload'], comparison['requested_clients'])
        fixed = comparison['standalone_fixed_k']
        group = groups.setdefault(key, {})
        if fixed in group:
            raise ValueError('duplicate workload/fixed comparison')
        group[fixed] = comparison
    checks = []
    for (workload, clients), group in sorted(groups.items()):
        complete = 7 in group and 15 in group
        correct = complete and all(group[k]['qualification']['correct'] for k in (7, 15))
        # Independently repeated controls must also agree across the two fixed campaigns.
        consistent = complete and (group[7]['pairs'][0]['runs']['none']['analysis']['responses'] ==
                                   group[15]['pairs'][0]['runs']['none']['analysis']['responses'])
        qualified = correct and consistent and all(
            group[k]['qualification']['vs_standalone_fixed']['qualified_performance']
            for k in (7, 15))
        checks.append({'workload': workload, 'requested_clients': clients,
                       'complete_fixed_k7_and_k15': complete, 'cross_campaign_output_consistent': consistent,
                       'qualified_incremental_value': bool(qualified)})
    return {'qualified_incremental_value': bool(checks) and all(c['qualified_incremental_value'] for c in checks),
            'workloads': checks,
            'scope': 'Auto must independently beat both standalone DFlash K7 and K15 on every requested workload; None is only a control'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('exe', 'model', 'profile', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--workloads', default='all')
    parser.add_argument('--prompt-dir', type=Path)
    parser.add_argument('--concurrency', type=lambda v: selections(v, range(1, 9)), default=[1, 2, 4, 8])
    fixed = parser.add_mutually_exclusive_group(required=True)
    fixed.add_argument('--fixed-draft-tokens', type=int, choices=(7, 11, 15))
    fixed.add_argument('--compare-both-fixed', action='store_true')
    parser.add_argument('--pairs', type=int, default=2); parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-context', type=int, default=32768); parser.add_argument('--kv-capacity', type=int, default=131072)
    parser.add_argument('--engine-concurrency', type=int, default=8); parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--nvml-device', required=True)
    parser.add_argument('--memory-sample-seconds', type=float, default=.05); parser.add_argument('--cooldown', type=float, default=2)
    args = parser.parse_args(argv)
    if (args.pairs < 2 or args.repeats < 2 or not 2 <= args.max_tokens <= args.max_context <= 32768
            or not 2 <= args.kv_capacity <= 262144 or not max(args.concurrency) <= args.engine_concurrency <= 8
            or not math.isfinite(args.cooldown) or args.cooldown < 0 or not math.isfinite(args.memory_sample_seconds) or args.memory_sample_seconds <= 0):
        parser.error('invalid counts, capacity/concurrency or timing options')
    if not all(p.is_file() for p in (args.exe, args.model, args.profile)):
        parser.error('explicit existing native executable/artifact/profile required')
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {'artifact_type': 'ninfer_auto_router_comparison', 'schema_version': 1,
               'timing_scope': __doc__.strip(), 'comparisons': []}
    try:
        raw_profile = args.profile.read_bytes()
        (args.out/'profile-input.json').write_bytes(raw_profile)
        profile = read_json(args.out/'profile-input.json')
        table = profile_table(profile)
        identity = profile['identity']
        if (identity['max_concurrency'] != args.engine_concurrency or identity['max_context'] != args.max_context
                or identity['resolved_kv_capacity'] != args.kv_capacity or identity['proposal_head'] != 'full'):
            raise ValueError('profile identity differs from requested startup configuration')
        summary['profile'] = {'input_path': str(args.profile.resolve()), 'sha256': hashlib.sha256(raw_profile).hexdigest(),
                              'identity': identity, 'normalized_expected_cells': table,
                              'proposal_compute': proposal_compute_mode(profile)}
        environment = {'cuda_visible_device_ordinal': args.device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'nvml_physical_device': args.nvml_device, 'explicit_priming': False, 'startup': 'fresh Engine per native process'}
        try:
            hardware = subprocess.run(['nvidia-smi', '-i', args.nvml_device, '--query-gpu=name,uuid,driver_version,memory.total',
                                       '--format=csv'], capture_output=True, text=True, timeout=5, check=True)
            environment['hardware'] = hardware.stdout
        except (OSError, subprocess.SubprocessError) as error:
            environment['hardware_query_error'] = str(error)
        write_json(args.out/'environment.json', environment)
        base = [str(args.exe.resolve()), '--model', str(args.model.resolve()), '--engine-concurrency', str(args.engine_concurrency),
                '--max-context', str(args.max_context), '--kv-capacity', str(args.kv_capacity), '--device', str(args.device)]
        (args.out/'prompts').mkdir()
        fixed_actions = (7, 15) if args.compare_both_fixed else (args.fixed_draft_tokens,)
        for workload in select_workloads(args.workloads):
            prompt = ((args.prompt_dir/(workload.name+'.txt')).read_bytes().decode('utf-8')
                      if args.prompt_dir else workload.prompt)
            text(prompt, 'prompt', True)
            prompt_path = args.out/'prompts'/(workload.name+'.txt'); prompt_path.write_bytes(prompt.encode('utf-8'))
            for clients in args.concurrency:
                for fixed_k in fixed_actions:
                    configuration = {'prompt': prompt, 'max_tokens': args.max_tokens, 'client_concurrency': clients,
                                     'repeats': args.repeats, 'sampling': {'temperature':0, 'presence_penalty':0, 'frequency_penalty':0}}
                    comparison = {'workload': workload.name, 'requested_clients': clients, 'standalone_fixed_k': fixed_k, 'pairs': []}
                    summary['comparisons'].append(comparison)
                    analyses = []
                    for pair in range(args.pairs):
                        order = list(ARMS if pair % 2 == 0 else reversed(ARMS))
                        results = {}; run_audit = {}
                        for arm in order:
                            tag = f'K{fixed_k}-' if args.compare_both_fixed else ''
                            directory = args.out/'runs'/f'{workload.name}-C{clients}-{tag}p{pair}-{arm}'
                            report_path = directory/'native.json'
                            command = [*base, '--prompt-file', str(prompt_path.resolve()), '--output', str(report_path.resolve()),
                                       '--client-concurrency', str(clients), '--repeats', str(args.repeats), '--max-tokens', str(args.max_tokens)]
                            if arm == 'auto':
                                command += ['--spec-router-profile', str((args.out/'profile-input.json').resolve()), '--allow-route-switching']
                            else:
                                command += ['--draft-tokens', str(fixed_k if arm == 'fixed' else 0)]
                            execute_native(command, directory, args.nvml_device, args.memory_sample_seconds)
                            raw = read_json(report_path)
                            if arm == 'auto' and raw.get('routing_profile', {}).get('path') != str((args.out/'profile-input.json').resolve()):
                                raise ValueError('native audited a different profile path')
                            results[arm] = validate_measurement(raw, arm, profile, configuration, str(args.model.resolve()), args.device, fixed_k)
                            write_json(directory/'analysis.json', results[arm])
                            run_audit[arm] = {'native_report': str(report_path.resolve()), 'analysis': results[arm]}
                            if args.cooldown: time.sleep(args.cooldown)
                        analyses.append(results)
                        comparison['pairs'].append({'pair_id': str(pair), 'order': order, 'runs': run_audit})
                        write_json(args.out/'comparison.json', summary)
                    comparison['qualification'] = qualify_triples(analyses)
                    write_json(args.out/'comparison.json', summary)
        summary['correct'] = all(c['qualification']['correct'] for c in summary['comparisons'])
        summary['incremental_value'] = qualify_both_fixed(summary['comparisons'])
        write_json(args.out/'comparison.json', summary)
        return 0 if summary['correct'] else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        write_json(args.out/'failure.json', {'error': str(error)})
        write_json(args.out/'comparison.json', summary)
        print('auto comparison: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
