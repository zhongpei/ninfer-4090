"""One-shot build, correctness gates, held-out DFlash GPU calibration and comparison.

One native --session process per GPU retains common weights for the entire
parameter matrix. Setup/upload is measured separately from generation waves.
Run with Python 3.11: python -m tools.bench.run_dflash_gpu_campaign --model
/path/model.ninfer --build-dir build --out /path/fresh-output. Each physical GPU
runs one process at a time. Output contains commands, raw reports, timing sidecars,
logs, memory samples, profiles and partial/final summary.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import queue
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

from tools.bench.calibrated_router_profile import build_profile, unique_object, validate_identity
from tools.bench.run_auto_spec_router_comparison import validate_measurement
from tools.bench.run_resident_spec_router_calibration import (
    constant_profile, read_json, selections, validate_wrapper, write_json,
)
from tools.bench.run_spec_router_calibration import output_identity, percentile
from tools.dflash2_training.ab_suite import select_workloads

CANDIDATES = ('none', 'k7', 'k11', 'k15', 'auto_full', 'auto_selected')
ROOT = Path(__file__).resolve().parents[2]


def requested_kv_capacity(args, clients):
    capacity = 1 if clients == 1 else 8
    return min(args.kv_capacity, args.max_context * capacity)


def validate_correctness_evidence(document, args):
    if (document.get('artifact_type') != 'ninfer_dflash_gpu_campaign'
            or document.get('schema_version') != 1):
        raise ValueError('unsupported correctness evidence summary')
    configuration = document.get('configuration', {})
    for name in ('model', 'build_dir'):
        source = configuration.get(name)
        if not isinstance(source, str) or Path(source).resolve() != getattr(args, name).resolve():
            raise ValueError('correctness evidence model/build directory mismatch: ' + name)
    suite = document.get('correctness_suite')
    if not isinstance(suite, dict):
        raise ValueError('missing completed correctness suite evidence')
    validate_session_packet(suite, 'correctness-suite', suite=True)
    return suite


def prompts():
    """Separate prompt bytes for training and held-out validation; native counts are authoritative."""
    workloads = {w.name: w.prompt for w in select_workloads('all')}
    result = {}
    for phase, names in (('train', ('prose', 'reasoning', 'structured')),
                         ('validation', ('chat', 'code', 'long-context'))):
        phase_prompts = []
        counts = (0, 125, 330) if phase == 'train' else (0, 125, 180)
        for tier, name, count in zip(('short', 'medium', 'long'), names, counts):
            paragraphs = ''.join(
                f'Paragraph {i}: The {phase} service receives a request, records its owner and trace id, '
                'checks the checksum, selects a route, verifies the response, and saves accounting.\n'
                for i in range(count))
            phase_prompts.append({'name': f'{phase}-{tier}-{name}', 'tier': tier,
                                  'prompt': paragraphs + '\n' + workloads[name]})
        result[phase] = phase_prompts
    validate_prompt_split(result)
    return result


def validate_prompt_split(workloads):
    train = {w['prompt'] for w in workloads['train']}
    validation = {w['prompt'] for w in workloads['validation']}
    if not train or not validation or train & validation:
        raise ValueError('calibration and held-out prompt bytes must be disjoint')


def record_of(raw):
    return raw.get('record', raw)


def response_signature(raw):
    record = record_of(raw)
    requests = record['requests']
    values = [json.dumps(output_identity(r), sort_keys=True, ensure_ascii=False) for r in requests]
    if not values or len(set(values)) != 1:
        raise ValueError('complete outputs differ across repeats/slots')
    return values[0]


def timing_metrics(raw, sidecar):
    """Join public timings to raw responses, never estimate TTFT from wave duration."""
    if sidecar.get('artifact_type') != 'ninfer_generation_timings' or sidecar.get('schema_version') != 1:
        raise ValueError('missing or unsupported public generation timings')
    record = record_of(raw)
    requests = {(r['repeat'], r['slot']): r for r in record['requests']}
    observed = {}
    numeric = ('first_token_seconds', 'generation_wall_seconds', 'decode_seconds',
               'prompt_wall_seconds', 'queue_wait_seconds')
    for timing in sidecar.get('requests', []):
        key = (timing['repeat'], timing['slot'])
        if key not in requests or key in observed:
            raise ValueError('duplicate or unpaired timing request')
        r = requests[key]
        for field, expected in (('generated_tokens', len(r['generated_token_ids'])),
                                ('latency_ns', r['latency_ns']), ('prompt_tokens', r['prompt_tokens']),
                                ('reused_prompt_tokens', r['reused_prompt_tokens'])):
            if type(timing.get(field)) is not int or timing[field] != expected:
                raise ValueError('timing/raw response mismatch: ' + field)
        for field in numeric:
            value = timing.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('missing or invalid timing: ' + field)
        if timing['generated_tokens'] > 1 and timing['generation_wall_seconds'] <= 0:
            raise ValueError('missing generation interval')
        observed[key] = timing
    if set(observed) != set(requests):
        raise ValueError('missing timing requests')
    result = {}
    for label, keys in (('cold', [k for k in observed if k[0] == 0]),
                        ('warm', [k for k in observed if k[0] > 0])):
        rows = [observed[k] for k in keys]
        if not rows:
            raise ValueError('cold and natural prefix-repeat measurements required')
        repeat_ids = sorted({k[0] for k in keys})
        wall_ns = sum(record['wave_wall_ns'][rep] for rep in repeat_ids)
        decode = [(r['generated_tokens'] - 1) / r['generation_wall_seconds']
                  if r['generated_tokens'] > 1 else 0 for r in rows]
        result[label] = {
            'sample_count': len(rows), 'generated_tokens': sum(r['generated_tokens'] for r in rows),
            'throughput': sum(r['generated_tokens'] for r in rows) * 1e9 / wall_ns,
            'first_token_p50_seconds': percentile([r['first_token_seconds'] for r in rows], 50),
            'first_token_p95_seconds': percentile([r['first_token_seconds'] for r in rows], 95),
            'request_p95_ns': percentile([r['latency_ns'] for r in rows], 95),
            'request_decode_p50_tokens_per_second': percentile(decode, 50),
            'cache_observations': [{'repeat': k[0], 'slot': k[1], 'reused_prompt_tokens': observed[k]['reused_prompt_tokens'],
                                    'prefix_reuse_path': requests[k]['prefix_reuse_path']} for k in keys],
            'wave_scope': 'unprimed first wave' if label == 'cold' else 'later natural reuse waves',
            'reused_prompt_tokens': [r['reused_prompt_tokens'] for r in rows],
            'prompt_tokens': sorted({r['prompt_tokens'] for r in rows}),
        }
    return result


def rank_pairs(pairs, scope):
    """Conservative paired comparison; a fast isolated run cannot promote a candidate."""
    if len(pairs) < 2 or {tuple(p['order']) for p in pairs} != {CANDIDATES, tuple(reversed(CANDIDATES))}:
        raise ValueError('complete forward and reverse candidate orders required')
    signatures = set()
    coverage = set()
    for pair in pairs:
        if set(pair['runs']) != set(CANDIDATES):
            raise ValueError('missing comparison candidate')
        for run in pair['runs'].values():
            signatures.add((run['response_signature'], run['configuration_key']))
            coverage.add(tuple(run['metrics'][scope]['prompt_tokens']))
    if len(signatures) != 1:
        raise ValueError('complete output/configuration mismatch across candidates/pairs')
    if len(coverage) != 1:
        raise ValueError('mixed prompt-token coverage across candidates/pairs')
    rates = {name: statistics.median(p['runs'][name]['metrics'][scope]['throughput'] for p in pairs)
             for name in CANDIDATES}
    fixed = max(CANDIDATES[:4], key=lambda name: rates[name])
    comparisons = {}
    for name in CANDIDATES:
        ratios = [p['runs'][name]['metrics'][scope]['throughput'] / p['runs'][fixed]['metrics'][scope]['throughput'] for p in pairs]
        p95 = [p['runs'][name]['metrics'][scope]['request_p95_ns'] / p['runs'][fixed]['metrics'][scope]['request_p95_ns'] for p in pairs]
        none_ratios = [p['runs'][name]['metrics'][scope]['throughput'] / p['runs']['none']['metrics'][scope]['throughput'] for p in pairs]
        none_p95 = [p['runs'][name]['metrics'][scope]['request_p95_ns'] / p['runs']['none']['metrics'][scope]['request_p95_ns'] for p in pairs]
        comparisons[name] = {'paired_speedups_vs_best_fixed': ratios, 'paired_p95_ratios': p95,
                             'paired_speedups_vs_none': none_ratios, 'paired_p95_ratios_vs_none': none_p95,
                             'qualified_vs_best_fixed': all(v > 1 for v in ratios)
                             and statistics.median(ratios) >= 1.02 and all(v <= 1.05 for v in p95)
                             and all(v > 1 for v in none_ratios) and statistics.median(none_ratios) >= 1.02
                             and all(v <= 1.05 for v in none_p95)}
    observed = max(CANDIDATES, key=lambda name: rates[name])
    # Resolve near-ties toward simpler None/fixed, without claiming they won.
    fixed_qualified = []
    for name in CANDIDATES[1:4]:
        ratios = [p['runs'][name]['metrics'][scope]['throughput'] / p['runs']['none']['metrics'][scope]['throughput'] for p in pairs]
        tails = [p['runs'][name]['metrics'][scope]['request_p95_ns'] / p['runs']['none']['metrics'][scope]['request_p95_ns'] for p in pairs]
        if all(r > 1 for r in ratios) and statistics.median(ratios) >= 1.02 and all(r <= 1.05 for r in tails):
            fixed_qualified.append(name)
    recommendation = 'none'
    if fixed_qualified:
        best_rate = max(rates[name] for name in fixed_qualified)
        recommendation = next(name for name in fixed_qualified if rates[name] >= best_rate / 1.02)
    qualified_auto = [name for name in CANDIDATES[4:] if comparisons[name]['qualified_vs_best_fixed']]
    if qualified_auto:
        recommendation = max(qualified_auto, key=lambda name: rates[name])
    tradeoffs = {}
    for name in CANDIDATES:
        tradeoffs[name] = {
            'first_token_p50_seconds': statistics.median(p['runs'][name]['metrics'][scope]['first_token_p50_seconds'] for p in pairs),
            'decode_tokens_per_second': statistics.median(p['runs'][name]['metrics'][scope]['request_decode_p50_tokens_per_second'] for p in pairs),
        }
    pareto = [name for name in CANDIDATES if not any(
        tradeoffs[other]['first_token_p50_seconds'] <= tradeoffs[name]['first_token_p50_seconds']
        and tradeoffs[other]['decode_tokens_per_second'] >= tradeoffs[name]['decode_tokens_per_second']
        and tradeoffs[other] != tradeoffs[name] for other in CANDIDATES)]
    return {'observed_fastest': observed, 'recommended': recommendation, 'best_fixed': fixed,
            'throughput_medians': rates, 'comparisons': comparisons,
            'round_metric_scope': 'Committed tokens include target bonus tokens; this is not proposal acceptance rate.',
            'committed_tokens_per_round': {name: statistics.median(p['runs'][name]['committed_tokens_per_round'] for p in pairs)
                                            for name in CANDIDATES},
            'interactive_metrics': tradeoffs, 'interactive_pareto': pareto,
            'sample_warning': 'Two paired orientations and few requests; p95 is descriptive.',
            'scope': scope, 'prompt_tokens': list(next(iter(coverage)))}


def bounded_process(command, directory, env, timeout, cwd=ROOT):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'command.json', command)
    write_json(directory / 'environment.json', {k: env.get(k) for k in
               ('CUDA_VISIBLE_DEVICES', 'NINFER_TEST_ARTIFACT', 'NINFER_CALIBRATION_NATIVE_EXE')})
    print('START ' + str(directory), flush=True)
    started = time.monotonic()
    with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
        process = subprocess.Popen(command, env=env, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            write_json(directory / 'result.json', {'timeout_seconds': timeout, 'returncode': process.returncode})
            raise RuntimeError('process timeout: ' + str(directory))
    write_json(directory / 'result.json', {'returncode': code, 'wall_seconds': time.monotonic() - started})
    print(f'END {directory} exit={code}', flush=True)
    if code:
        raise RuntimeError(f'process exit {code} (77 is not a pass): {directory}')



def parse_session_packet(line):
    packet = json.loads(line, object_pairs_hook=unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite protocol value: ' + value)))
    if not isinstance(packet, dict):
        raise ValueError('session packet must be an object')
    return packet


def validate_session_packet(packet, expected_id=None, *, ready=False, bye=False, suite=False):
    common = {'model_load_count', 'resident_weight_bytes'}
    required = common | ({'event', 'scope'} if ready else {'id', 'ok'}
                         | ({'event', 'teardown_checks'} if bye else set())
                         | ({'cases_passed', 'scope', 'ownership_checks'} if suite else set()))
    if not ready and packet.get('ok') is False:
        if packet.get('id') != expected_id:
            raise ValueError('out-of-order session response id')
        raise RuntimeError('native session job failed: ' + str(packet.get('error')))
    if set(packet) != required:
        raise ValueError('missing or unexpected session protocol fields')
    if type(packet['model_load_count']) is not int or packet['model_load_count'] != 1:
        raise ValueError('resident session must load model exactly once')
    if type(packet['resident_weight_bytes']) is not int or packet['resident_weight_bytes'] <= 0:
        raise ValueError('missing common resident weight bytes')
    if ready:
        if packet['event'] != 'ready' or packet['scope'] != 'common_weight_residency':
            raise ValueError('invalid resident session readiness/scope')
    elif packet['id'] != expected_id:
        raise ValueError('out-of-order session response id')
    elif packet['ok'] is not True or bye and packet['event'] != 'bye':
        raise ValueError('native session response did not complete')
    if suite and (type(packet['cases_passed']) is not int or packet['cases_passed'] != 18
                  or packet['scope'] != 'common_weight_residency'):
        raise ValueError('resident correctness suite must pass all eighteen cases')
    if suite and (type(packet['ownership_checks']) is not int or packet['ownership_checks'] != 5):
        raise ValueError('resident correctness suite must pass all five ownership checks')
    if bye and (type(packet['teardown_checks']) is not int or packet['teardown_checks'] != 1):
        raise ValueError('resident session teardown check must pass')
    return packet


class NativeSession:
    """One resident process; bounded protocol messages and retained per-job evidence."""
    def __init__(self, command, directory, gpu, env, timeout):
        self.directory, self.gpu, self.timeout = directory, gpu, timeout
        self.closed = False
        self.ids = set()
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'command.json', command)
        write_json(directory / 'environment.json', {'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES']})
        self.stderr = (directory / 'stderr.log').open('wb')
        self.messages = queue.Queue()
        started = time.monotonic()
        self.process = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr, start_new_session=True)

        def read_stdout():
            with (directory / 'stdout.log').open('wb') as log:
                try:
                    for line in self.process.stdout:
                        log.write(line)
                        log.flush()
                        self.messages.put(parse_session_packet(line.decode('utf-8')))
                    self.messages.put(RuntimeError('native session ended before protocol response'))
                except (OSError, ValueError, UnicodeError) as error:
                    self.messages.put(error)

        self.reader = threading.Thread(target=read_stdout, daemon=True)
        self.reader.start()
        try:
            self.ready = validate_session_packet(self.receive(), ready=True)
            write_json(directory / 'ready.json', {**self.ready, 'startup_wall_seconds': time.monotonic() - started})
        except Exception:
            self.close(abort=True)
            raise

    def receive(self, timeout=None):
        timeout = self.timeout if timeout is None else timeout
        try:
            packet = self.messages.get(timeout=timeout)
        except queue.Empty as error:
            raise RuntimeError(f'native session timeout after {timeout}s: {self.directory}') from error
        if isinstance(packet, Exception):
            raise packet
        return packet

    def send(self, packet):
        if self.closed or packet['id'] in self.ids:
            raise ValueError('closed session or duplicate job id')
        self.ids.add(packet['id'])
        self.process.stdin.write((json.dumps(packet, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
        self.process.stdin.flush()

    def run(self, command, directory, job_id, suite=False):
        directory.mkdir(parents=True, exist_ok=False)
        packet = {'id': job_id, 'suite': True} if suite else {'id': job_id, 'arguments': command[1:]}
        write_json(directory / 'command.json', command if command else {'resident_protocol_request': packet})
        write_json(directory / 'session-request.json', packet)
        write_json(directory / 'session.json', {'process_command': str(self.directory / 'command.json'),
                   'stderr': str(self.directory / 'stderr.log'), 'common_weight_residency': self.ready})
        stop = threading.Event()
        samples, errors = [], []

        def sampler():
            while True:
                try:
                    result = subprocess.run(['nvidia-smi', '-i', str(self.gpu), '--query-gpu=memory.used',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                            timeout=5, check=True)
                    samples.append({'monotonic_ns': time.monotonic_ns(), 'used_mib': int(result.stdout.strip())})
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    errors.append(str(error))
                    return
                if stop.wait(.1):
                    return

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        started = time.monotonic()
        print(f'START resident GPU{self.gpu} {job_id}', flush=True)
        success = False
        try:
            self.send(packet)
            response = validate_session_packet(self.receive(3600 if suite else self.timeout), job_id, suite=suite)
            if response['resident_weight_bytes'] != self.ready['resident_weight_bytes']:
                raise ValueError('resident weight bytes changed during session')
            write_json(directory / 'session-response.json', response)
            success = True
        except Exception as error:
            write_json(directory / 'failure.json', {'error': str(error)})
            self.close(abort=True)
            raise
        finally:
            stop.set()
            thread.join()
            write_json(directory / 'memory.json', {'kind': 'device_memory_observed_lower_bound', 'gpu': self.gpu,
                       'sample_interval_seconds': .1, 'samples': samples, 'errors': errors,
                       'peak_used_mib': max((s['used_mib'] for s in samples), default=None),
                       'scope': 'device-wide sampled use including common resident weights; not a standalone memory baseline'})
            write_json(directory / 'result.json', {'ok': success, 'job_wall_seconds': time.monotonic() - started})
            print(f'END resident GPU{self.gpu} {job_id} ok={success}', flush=True)
        return response

    def close(self, abort=False):
        if self.closed:
            return
        try:
            if not abort:
                job_id = '__shutdown__'
                self.send({'id': job_id, 'stop': True})
                packet = validate_session_packet(self.receive(), job_id, bye=True)
                if packet['resident_weight_bytes'] != self.ready['resident_weight_bytes']:
                    raise ValueError('resident weight bytes changed at shutdown')
                write_json(self.directory / 'shutdown.json', packet)
                self.shutdown = packet
                self.process.stdin.close()
                self.exit_code = self.process.wait(timeout=30)
                if self.exit_code != 0:
                    raise RuntimeError('native session exit failed')
        finally:
            self.closed = True
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.reader.join(timeout=5)
            self.process.stdout.close()
            if not self.process.stdin.closed:
                self.process.stdin.close()
            self.stderr.close()


class Campaign:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.summary = {'artifact_type': 'ninfer_dflash_gpu_campaign', 'schema_version': 1,
                        'complete': False, 'correct': False, 'stages': [], 'training': [], 'validation': [],
                        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        self.exe = args.build_dir.resolve() / 'bench/ninfer_spec_router_calibration_bench'
        self.workloads = prompts()
        (self.out / 'prompts').mkdir()
        for phase in self.workloads.values():
            for workload in phase:
                path = self.out / 'prompts' / (workload['name'] + '.txt')
                path.write_text(workload['prompt'], encoding='utf-8')
                workload['path'] = path
        self.identities = {}
        self.documents = {}
        self.profiles = {}
        self.output_signatures = {}
        self.device_total_mib = {}
        self.sessions = {}
        self.save()

    def save(self):
        with self.lock:
            write_json(self.out / 'summary.json', self.summary)

    def env(self, gpu=None):
        env = dict(os.environ)
        env.update(NINFER_TEST_ARTIFACT=str(self.args.model.resolve()),
                   NINFER_AUTO_NATIVE_EXE=str(self.exe),
                   NINFER_CALIBRATION_NATIVE_EXE=str(self.exe))
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        return env

    def process(self, command, tag, gpu=None, timeout=3600):
        bounded_process(command, self.out / tag, self.env(gpu), timeout)

    def native(self, command, tag, gpu):
        if self.stop.is_set():
            raise RuntimeError('another GPU lane failed; campaign stopped')
        directory = self.out / tag
        try:
            self.sessions[gpu].run(command, directory, tag)
        except Exception:
            self.stop.set()
            raise
        memory = read_json(directory / 'memory.json')
        if memory['errors'] or not memory['samples'] or memory['peak_used_mib'] > self.device_total_mib[gpu]:
            self.stop.set()
            raise ValueError('missing or invalid device memory evidence: ' + str(directory))
        return read_json(directory / 'native.json')

    def preflight(self):
        if sys.version_info[:2] != (3, 11):
            raise ValueError('selected Python 3.11 interpreter required')
        if not self.args.model.is_file() or not (self.args.build_dir / 'CMakeCache.txt').is_file():
            raise ValueError('explicit existing model and configured build directory required')
        hardware = []
        for gpu in self.args.gpus:
            query = ['nvidia-smi', '-i', str(gpu), '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used',
                     '--format=csv,noheader,nounits']
            result = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10)
            row = result.stdout.strip().split(',')
            if len(row) != 6 or int(row[-1]) > 128:
                raise RuntimeError(f'GPU {gpu} is occupied; no services will be stopped: {result.stdout.strip()}')
            self.device_total_mib[gpu] = int(row[-2])
            hardware.append(result.stdout.strip())
        write_json(self.out / 'environment.json', {'python': sys.version, 'python_executable': sys.executable, 'hardware': hardware,
                   'cuda_mapping': {str(g): 'CUDA_VISIBLE_DEVICES=' + str(g) + '; native --device 0' for g in self.args.gpus}})
        targets = ['ninfer_spec_router_calibration_bench', 'ninfer_qwen3_5_dflash_proposal_view_test',
                   'ninfer_speculative_routing_profile_test', 'ninfer_qwen3_5_calibrated_routing_real_test',
                   'ninfer_qwen3_5_resident_model_options_test']
        self.process(['cmake', '--build', str(self.args.build_dir.resolve()), '-j', '--target', *targets], 'build')
        self.process([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/report_regression'], 'cpu-report')
        self.process([sys.executable, '-m', 'unittest', 'discover', '-s', 'tools/bench'], 'cpu-bench')
        self.process([sys.executable, '-m', 'py_compile', 'tools/bench/run_dflash_gpu_campaign.py',
                      'tools/bench/test_dflash_gpu_campaign.py'], 'python-compile')
        self.process(['ctest', '--test-dir', str(self.args.build_dir.resolve()), '--output-on-failure', '-R',
                      '^(ninfer_qwen3_5_dflash_proposal_view_test|ninfer_speculative_routing_profile_test|ninfer_qwen3_5_resident_model_options_test)$'], 'cpu-native')
        self.summary['stages'].append('build and CPU gates passed')
        self.save()

    def run_correctness(self):
        if self.args.reuse_correctness is not None:
            origin = self.args.reuse_correctness.resolve()
            summary = validate_correctness_evidence(read_json(origin), self.args)
            self.summary['correctness_suite'] = summary
            self.summary['correctness_evidence_origin'] = str(origin)
            self.summary['stages'].append('Explicitly reused prior 18-case and five-ownership-check correctness evidence; suite not executed in this session')
            self.save()
            print('REUSE correctness evidence ' + str(origin), flush=True)
            return
        gpu = self.args.gpus[0]
        summary = self.sessions[gpu].run(None, self.out / 'correctness/resident-suite', 'correctness-suite', suite=True)
        self.summary['correctness_suite'] = summary
        self.summary['stages'].append('18 real correctness cases passed under the existing GPU0 resident weight load')
        self.save()

    def start_sessions(self):
        for gpu in self.args.gpus:
            command = [str(self.exe), '--session', '--model', str(self.args.model.resolve()), '--device', '0',
                       '--max-context', str(self.args.max_context), '--kv-capacity', str(self.args.kv_capacity)]
            session = NativeSession(command, self.out / f'sessions/GPU{gpu}', gpu, self.env(gpu), self.args.native_timeout)
            self.sessions[gpu] = session
        self.summary['resident_sessions'] = {str(gpu): session.ready for gpu, session in self.sessions.items()}
        self.summary['measurement_scope'] = 'Independent per-job execution state with common resident weights; setup/upload excluded from generation-wave rankings.'
        self.save()

    def close_sessions(self, abort=False):
        failures = []
        for session in self.sessions.values():
            try:
                session.close(abort=abort)
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('; '.join(failures))
        if not abort:
            self.summary['resident_sessions'] = {str(gpu): {**session.ready, 'shutdown': session.shutdown,
                                                           'exit_code': session.exit_code}
                                                  for gpu, session in self.sessions.items()}
            self.save()

    def base(self, clients):
        capacity = 1 if clients == 1 else 8
        return [str(self.exe), '--model', str(self.args.model.resolve()), '--device', '0',
                '--engine-concurrency', str(capacity), '--max-context', str(self.args.max_context),
                '--kv-capacity', str(requested_kv_capacity(self.args, clients))]

    def check_identity_capacity(self, identity, clients):
        if (identity['max_concurrency'] != (1 if clients == 1 else 8)
                or identity['max_context'] != self.args.max_context
                or identity['resolved_kv_capacity'] < requested_kv_capacity(self.args, clients)
                or identity['proposal_head'] != 'full'):
            raise ValueError('startup identity does not match requested capacity')

    def configuration(self, workload, clients):
        return {'prompt': workload['prompt'], 'max_tokens': self.args.max_tokens,
                'client_concurrency': clients, 'repeats': self.args.repeats,
                'sampling': {'temperature': 0, 'presence_penalty': 0, 'frequency_penalty': 0}}

    def command(self, workload, clients, tag):
        directory = self.out / tag
        return [*self.base(clients), '--prompt-file', str(workload['path']), '--output', str(directory / 'native.json'),
                '--timing-output', str(directory / 'timings.json'), '--client-concurrency', str(clients),
                '--repeats', str(self.args.repeats), '--max-tokens', str(self.args.max_tokens)]

    def check_prompt(self, metrics, workload, clients):
        counts = set(metrics['cold']['prompt_tokens'] + metrics['warm']['prompt_tokens'])
        if len(counts) != 1:
            raise ValueError('native prompt token counts changed across requests')
        count = next(iter(counts))
        tier = workload['tier']
        if (tier == 'short' and count + self.args.max_tokens > 1024
                or tier == 'medium' and not 1024 < count <= 8192
                or tier == 'long' and count <= 8192
                or count + self.args.max_tokens > min(self.args.max_context, requested_kv_capacity(self.args, clients) // clients)):
            raise ValueError(f'native measured prompt tokens {count} outside {tier} context/capacity contract')

    def check_output(self, raw, key):
        signature = response_signature(raw)
        with self.lock:
            previous = self.output_signatures.setdefault(key, signature)
        if signature != previous:
            self.stop.set()
            raise ValueError('complete output changed across candidates/modes/pairs: ' + str(key))
        return signature

    def train_group(self, clients, gpu):
        tag = f'identity/C{clients}'
        raw = self.native([*self.base(clients), '--identity-only', '--draft-tokens', '15',
                           '--output', str(self.out / tag / 'native.json')], tag, gpu)
        if raw.get('artifact_type') != 'ninfer_routing_identity' or raw.get('schema_version') != 1:
            raise ValueError('unsupported startup identity')
        identity = raw['identity']
        validate_identity(identity)
        self.check_identity_capacity(identity, clients)
        self.identities[clients] = identity
        for mode in ('full', 'selected'):
            document = {'artifact_type': 'ninfer_resident_router_measurements', 'schema_version': 1,
                        'identity': identity, 'comparisons': []}
            if mode == 'selected':
                document.update(schema_version=2, proposal_compute=mode)
            controls = {}
            for action in (0, 7, 11, 15):
                path = self.out / f'control-C{clients}-{mode}-K{action}.json'
                write_json(path, constant_profile(identity, action, mode))
                controls[action] = path
            for workload in self.workloads['train']:
                comparisons = {k: {'workload': f"{workload['name']}/C{clients}", 'candidate_action': k, 'pairs': []}
                               for k in (7, 11, 15)}
                signatures = set()
                for pair in range(self.args.pairs):
                    order = (0, 7, 11, 15) if pair % 2 == 0 else (15, 11, 7, 0)
                    records = {}
                    for action in order:
                        tag = f"train/C{clients}/{workload['name']}/{mode}/p{pair}-K{action}"
                        command = [*self.command(workload, clients, tag), '--draft-tokens', str(action),
                                   '--spec-router-profile', str(controls[action])]
                        raw = self.native(command, tag, gpu)
                        analysis = validate_wrapper(raw, identity, action, self.configuration(workload, clients), False, mode)
                        signatures.add(self.check_output(raw, (clients, workload['name'])))
                        metrics = timing_metrics(raw, read_json(self.out / tag / 'timings.json'))
                        self.check_prompt(metrics, workload, clients)
                        write_json(self.out / tag / 'analysis.json', analysis)
                        records[action] = raw['record']
                    if len(signatures) != 1:
                        raise ValueError('calibration complete outputs differ across actions/pairs')
                    for action, comparison in comparisons.items():
                        comparison['pairs'].append({'pair_id': str(pair), 'order': [0, action] if pair % 2 == 0 else [action, 0],
                                                    'baseline': records[0], 'candidate': records[action]})
                    self.save()
                document['comparisons'].extend(comparisons.values())
                with self.lock:
                    self.summary['training'].append({'clients': clients, 'gpu': gpu, 'mode': mode,
                                                     'workload': workload['name'], 'complete': True})
                self.save()
            self.documents[clients, mode] = document
            write_json(self.out / f'train-C{clients}-{mode}.json', document)

    def build_profiles(self):
        for capacity in sorted({1 if c == 1 else 8 for c in self.args.concurrency}):
            members = [c for c in self.args.concurrency if (1 if c == 1 else 8) == capacity]
            for mode in ('full', 'selected'):
                document = dict(self.documents[members[0], mode])
                document['comparisons'] = []
                for clients in members:
                    source = self.documents[clients, mode]
                    if validate_identity(source['identity']) != validate_identity(document['identity']):
                        raise ValueError('cannot aggregate different startup identities')
                    document['comparisons'].extend(source['comparisons'])
                profile = build_profile(document)
                path = self.out / f'auto-E{capacity}-{mode}.json'
                write_json(path, profile)
                for clients in members:
                    self.profiles[clients, mode] = (profile, path)
        self.summary['stages'].append('all calibration completed; profiles built before held-out validation')
        self.save()

    def validate_group(self, clients, gpu):
        for workload in self.workloads['validation']:
            comparison = {'clients': clients, 'gpu': gpu, 'workload': workload['name'], 'tier': workload['tier'], 'pairs': []}
            with self.lock:
                self.summary['validation'].append(comparison)
            for pair in range(self.args.pairs):
                order = CANDIDATES if pair % 2 == 0 else tuple(reversed(CANDIDATES))
                runs = {}
                with self.lock:
                    comparison['pairs'].append({'pair_id': pair, 'order': list(order), 'runs': runs})
                for name in order:
                    tag = f"validation/C{clients}/{workload['name']}/p{pair}-{name}"
                    mode = 'selected' if name == 'auto_selected' else 'full'
                    profile, profile_path = self.profiles[clients, mode]
                    command = self.command(workload, clients, tag)
                    action = int(name[1:]) if name.startswith('k') else 0
                    arm = 'auto' if name.startswith('auto') else 'none' if name == 'none' else 'fixed'
                    command += (['--spec-router-profile', str(profile_path), '--allow-route-switching'] if arm == 'auto'
                                else ['--draft-tokens', str(action)])
                    raw = self.native(command, tag, gpu)
                    if arm == 'auto' and raw.get('routing_profile', {}).get('path') != str(profile_path):
                        raise ValueError('native audited a different profile path')
                    analysis = validate_measurement(raw, arm, profile, self.configuration(workload, clients),
                                                    str(self.args.model.resolve()), 0, action)
                    if arm == 'none':
                        analysis['execution_kind'] = 'backend_none_shared_resident_weights'
                    elif arm == 'fixed':
                        analysis['execution_kind'] = f'fixed_k{action}_shared_resident_weights'
                    metrics = timing_metrics(raw, read_json(self.out / tag / 'timings.json'))
                    self.check_prompt(metrics, workload, clients)
                    signature = self.check_output(raw, (clients, workload['name']))
                    run = {'response_signature': signature, 'configuration_key': analysis['configuration_key'],
                           'metrics': metrics, 'committed_tokens_per_round': analysis['committed_tokens'] / len(record_of(raw)['rounds']),
                           'round_count': len(record_of(raw)['rounds']), 'native_report': str(self.out / tag / 'native.json'), 'analysis': analysis}
                    with self.lock:
                        runs[name] = run
                    write_json(self.out / tag / 'analysis.json', run)
                    self.save()
                self.save()
            rankings = {scope: rank_pairs(comparison['pairs'], scope) for scope in ('cold', 'warm')}
            with self.lock:
                comparison['rankings'] = rankings
            self.save()

    def parallel_groups(self, method):
        lanes = [[] for _ in self.args.gpus]
        for index, clients in enumerate(self.args.concurrency):
            lanes[index % len(lanes)].append(clients)
        def lane(groups, gpu):
            try:
                for clients in groups:
                    method(clients, gpu)
            except Exception:
                self.stop.set()
                raise
        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            jobs = [pool.submit(lane, groups, gpu) for groups, gpu in zip(lanes, self.args.gpus)]
            for job in jobs:
                job.result()

    def finish(self):
        aggregate = {}
        for group, cells in (('single', [c for c in self.summary['validation'] if c['clients'] == 1]),
                             ('multi', [c for c in self.summary['validation'] if c['clients'] > 1])):
            if not cells:
                continue
            aggregate[group] = {}
            for scope in ('cold', 'warm'):
                scores = {}
                for name in CANDIDATES:
                    ratios = [c['rankings'][scope]['throughput_medians'][name] /
                              c['rankings'][scope]['throughput_medians']['none'] for c in cells]
                    scores[name] = {'geometric_mean_speedup_vs_none': math.exp(statistics.mean(math.log(r) for r in ratios)),
                                    'worst_cell_speedup_vs_none': min(ratios)}
                winners = {c['rankings'][scope]['recommended'] for c in cells}
                aggregate[group][scope] = {'scores': scores, 'recommendation': next(iter(winners)) if len(winners) == 1 else 'per-condition table',
                                          'same_recommendation_across_measured_conditions': len(winners) == 1}
        self.summary.update(complete=True, correct=True, aggregate=aggregate)
        self.save()
        print(json.dumps({'complete': True, 'aggregate': aggregate}, indent=2), flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'build-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpus', type=lambda v: selections(v, range(256)), default=[0, 1])
    parser.add_argument('--concurrency', type=lambda v: selections(v, range(1, 9)), default=[1, 2, 4, 8])
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-context', type=int, default=32768)
    parser.add_argument('--kv-capacity', type=int, default=131072)
    parser.add_argument('--native-timeout', type=int, default=600)
    parser.add_argument('--reuse-correctness', type=Path,
                        help='Explicit prior summary.json with completed same-model/same-build resident correctness suite; native binary must be unchanged')
    args = parser.parse_args(argv)
    if not (args.pairs >= 2 and args.repeats >= 2 and 2 <= args.max_tokens <= args.max_context <= 32768
            and args.max_context <= args.kv_capacity <= 262144 and args.native_timeout > 0):
        parser.error('invalid pair/repeat counts, context/KV capacity or timeout')
    campaign = None
    try:
        campaign = Campaign(args)
        campaign.preflight()
        campaign.start_sessions()
        campaign.run_correctness()
        campaign.parallel_groups(campaign.train_group)
        campaign.build_profiles()
        campaign.parallel_groups(campaign.validate_group)
        campaign.close_sessions()
        campaign.finish()
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyError, TypeError) as error:
        if campaign:
            try:
                campaign.close_sessions(abort=True)
            except Exception as cleanup_error:
                campaign.summary['cleanup_failure'] = str(cleanup_error)
            campaign.summary['failure'] = str(error)
            campaign.save()
        print('campaign failed: ' + str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
