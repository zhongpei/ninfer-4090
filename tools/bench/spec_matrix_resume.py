"""Complete-comparison resume with raw revalidation; no inference implementation."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from tools.bench.calibrated_router_profile import analyse_comparison, fields, unique_object
from tools.bench import run_spec_router_calibration as fixed
from tools.bench import run_resident_spec_router_calibration as resident
from tools.bench import run_auto_spec_router_comparison as auto
from tools.dflash2_training import ab_suite

ROOT = Path(__file__).resolve().parents[2]
DRIVERS = {'fixed': fixed, 'resident': resident, 'auto': auto}


class EvidenceMismatch(ValueError): pass
class MatrixLocked(RuntimeError): pass


def read_json(path):
    def reject(value): raise ValueError('nonfinite JSON: '+value)
    def number(value):
        result=float(value)
        if not math.isfinite(result): reject(value)
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object,
                      parse_constant=reject, parse_float=number)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


@contextlib.contextmanager
def file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise MatrixLocked('harness lock held: '+str(path)) from error
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def absolute(value):
    if not isinstance(value,str) or not Path(value).is_absolute():raise ValueError('absolute path required: '+repr(value))
    return Path(value)


def int_range(value,low,high,name):
    if type(value)is not int or not low<=value<=high:raise ValueError('invalid '+name)


def argv_options(command):
    if not isinstance(command,list) or not command or any(not isinstance(v,str) for v in command):raise EvidenceMismatch('invalid raw command')
    values={};flags=set();index=1
    while index<len(command):
        key=command[index]
        if not key.startswith('--') or key in values or key in flags:raise EvidenceMismatch('invalid/duplicate native option: '+key)
        if index+1==len(command) or command[index+1].startswith('--'):
            flags.add(key);index+=1
        else:values[key]=command[index+1];index+=2
    return values,flags


class ResumeMatrix:
    def __init__(self,plan):
        fields(plan,('schema_version','artifact_type','suites'),'matrix plan')
        if type(plan['schema_version'])is not int or plan['schema_version']!=1 or plan['artifact_type']!='ninfer_spec_router_matrix_plan':raise ValueError('unsupported matrix schema')
        if not isinstance(plan['suites'],list) or not plan['suites']:raise ValueError('explicit suites required')
        self.plan=plan;self.units=[];self.hashes={};self.imports={};ids=set();roots=set()
        for suite in plan['suites']:
            fields(suite,('id','kind','state','depends_on','exe','model','gpu','options','output_root','existing_evidence'),'suite')
            sid=suite['id']
            if not isinstance(sid,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',sid) or sid in ids:raise ValueError('invalid/duplicate suite id')
            ids.add(sid)
            if suite['kind'] not in DRIVERS or suite['state'] not in ('ready','deferred'):raise ValueError('unsupported suite kind/state')
            for key in ('exe','model','output_root'):absolute(suite[key])
            if suite['output_root'] in roots:raise ValueError('duplicate output root')
            roots.add(suite['output_root'])
            fields(suite['gpu'],('physical','cuda_visible_devices','device'),'GPU')
            int_range(suite['gpu']['physical'],0,1,'physical GPU');int_range(suite['gpu']['device'],0,0,'visible device')
            if suite['gpu']['cuda_visible_devices']!=str(suite['gpu']['physical']):raise ValueError('explicit physical GPU/ordinal mapping required')
            options=suite['options'];common=('workloads','concurrency','pairs','repeats','max_tokens','max_context','kv_capacity','engine_concurrency','cooldown','memory_sample_seconds')
            extra=('fixed_draft_tokens','profile') if suite['kind']=='auto' else ('draft_tokens',)
            if suite['kind']=='resident':extra+=('prime','workload_label_prefix','prompt_dir')
            fields(options,common+extra,'suite options')
            for key,low,high in (('pairs',2,1000),('repeats',2,1000),('max_tokens',2,32768),('max_context',2,32768),('kv_capacity',2,262144),('engine_concurrency',1,8)):
                int_range(options[key],low,high,key)
            if options['max_tokens']>options['max_context']:raise ValueError('budget exceeds context')
            for key in ('cooldown','memory_sample_seconds'):
                value=options[key]
                if type(value) not in (int,float) or not math.isfinite(value) or value<0 or key=='memory_sample_seconds' and value==0:raise ValueError('invalid timing option')
            for key in ('workloads','concurrency'):
                if not isinstance(options[key],list) or not options[key] or len(set(options[key]))!=len(options[key]):raise ValueError('invalid/duplicate '+key)
            if any(w not in {w.name for w in ab_suite.WORKLOADS} for w in options['workloads']):raise ValueError('unknown original workload')
            for concurrency in options['concurrency']:int_range(concurrency,1,options['engine_concurrency'],'client concurrency')
            actions=[options['fixed_draft_tokens']] if suite['kind']=='auto' else options['draft_tokens']
            if not isinstance(actions,list) or not actions or len(set(actions))!=len(actions) or any(type(k)is not int or k not in (7,11,15) for k in actions):raise ValueError('invalid candidate widths')
            if suite['kind']=='resident':
                if type(options['prime'])is not bool:raise ValueError('explicit priming bool required')
                resident.suite_label(options['workload_label_prefix'])
                if options['prompt_dir'] is not None:absolute(options['prompt_dir'])
            if suite['kind']=='auto' and options['profile'] is not None:absolute(options['profile'])
            evidence=suite['existing_evidence']
            if evidence is not None:
                fields(evidence,('campaign_root','audit_path','attestation_path'),'existing evidence')
                for value in evidence.values():absolute(value)
                if Path(suite['output_root']).is_relative_to(evidence['campaign_root']):raise ValueError('managed outputs must not touch existing campaign')
            if not isinstance(suite['depends_on'],list) or len(set(suite['depends_on']))!=len(suite['depends_on']):raise ValueError('invalid dependencies')
            for workload in options['workloads']:
                for clients in options['concurrency']:
                    for action in actions:self.units.append({'suite':suite,'workload':workload,'clients':clients,'action':action,
                        'id':f'{sid}/{workload}/C{clients}/K{action}'})
        if any(d not in ids or d==s['id'] for s in plan['suites'] for d in s['depends_on']):raise ValueError('unknown/self dependency')
        visiting=set();visited=set();by_id={s['id']:s for s in plan['suites']}
        def visit(sid):
            if sid in visiting:raise ValueError('cyclic dependencies')
            if sid in visited:return
            visiting.add(sid)
            for other in by_id[sid]['depends_on']:visit(other)
            visiting.remove(sid);visited.add(sid)
        for sid in ids:visit(sid)

    def digest(self,path,cache=False):
        path=str(Path(path).resolve())
        if not cache or path not in self.hashes:
            h=hashlib.sha256()
            with open(path,'rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            self.hashes[path]=h.hexdigest()
        return self.hashes[path]

    def prompt_bytes(self,unit):
        suite=unit['suite'];options=suite['options']
        if suite['kind']=='resident' and options['prompt_dir'] is not None:
            return (Path(options['prompt_dir'])/(unit['workload']+'.txt')).read_bytes()
        return ab_suite.select_workloads(unit['workload'])[0].prompt.encode('utf-8')

    def version_fingerprint(self,unit):
        names={Path(__file__).resolve(),ROOT/'tools/bench/run_spec_router_matrix.py',Path(DRIVERS[unit['suite']['kind']].__file__).resolve(),Path(ab_suite.__file__).resolve()}
        if unit['suite']['kind'] in ('resident','auto'):
            names.add(ROOT/'tools/bench/calibrated_router_profile.py')
        if unit['suite']['kind'] in ('resident','auto'):names.add(Path(fixed.__file__).resolve())
        if unit['suite']['kind']=='auto':names.add(Path(resident.__file__).resolve())
        return {str(p):self.digest(p,cache=True) for p in sorted(names)}

    def fingerprint(self,unit):
        suite=unit['suite'];options={k:v for k,v in suite['options'].items() if k not in ('workloads','concurrency','draft_tokens')}
        prompt=self.prompt_bytes(unit);prompt.decode('utf-8')
        profile=None
        if suite['kind']=='auto':
            path=options['profile']
            if path is None:raise FileNotFoundError('auto requires an explicitly released valid profile')
            data=read_json(path);auto.profile_table(data);identity=data['identity']
            if (identity['max_concurrency']!=options['engine_concurrency'] or identity['max_context']!=options['max_context'] or identity['resolved_kv_capacity']!=options['kv_capacity'] or identity['proposal_head']!='full'):raise ValueError('auto profile/startup configuration mismatch')
            profile={'path':path,'sha256':self.digest(path)}
        value={'kind':suite['kind'],'workload':unit['workload'],'clients':unit['clients'],'action':unit['action'],
          'exe_sha256':self.digest(suite['exe'],cache=True),'model_path':suite['model'],'model_sha256':self.digest(suite['model'],cache=True),
          'versions':self.version_fingerprint(unit),'python_version':list(sys.version_info[:3]),'prompt_sha256':hashlib.sha256(prompt).hexdigest(),
          'options':options,'gpu':suite['gpu'],'profile':profile,
          'native_contract':{'kv':'int8','cache':True,'graphs':True,'temperature':0,'presence_penalty':0,'frequency_penalty':0,'proposal_head':'full','thinking':False}}
        return {'sha256':hashlib.sha256(canonical(value).encode()).hexdigest(),'identity':value}

    def unit_dir(self,unit):
        return Path(unit['suite']['output_root'])/unit['workload']/f'C{unit["clients"]}-K{unit["action"]}'

    def driver_command(self,unit,out):
        suite=unit['suite'];o=suite['options'];kind=suite['kind']
        command=[sys.executable,'-m',DRIVERS[kind].__name__,'--exe',suite['exe'],'--model',suite['model'],'--out',str(out),
          '--workloads',unit['workload'],'--concurrency',str(unit['clients'])]
        if kind=='auto':command+=['--fixed-draft-tokens',str(unit['action']),'--profile',o['profile']]
        else:command+=['--draft-tokens',str(unit['action'])]
        for key in ('pairs','repeats','max_tokens','max_context','kv_capacity','engine_concurrency','memory_sample_seconds','cooldown'):
            command+=['--'+key.replace('_','-'),str(o[key])]
        command+=['--device',str(suite['gpu']['device']),'--nvml-device',str(suite['gpu']['physical'])]
        if kind=='resident':
            command+=['--workload-label-prefix',o['workload_label_prefix'],'--prime-prefix' if o['prime'] else '--no-prime-prefix']
            if o['prompt_dir'] is not None:command+=['--prompt-dir',o['prompt_dir']]
        return command

    def _legacy(self,unit,fp):
        suite=unit['suite'];e=suite['existing_evidence'];cache_key=(suite['id'],fp['identity']['exe_sha256'],fp['identity']['model_sha256'])
        if cache_key in self.imports:return self.imports[cache_key]
        try:a=read_json(e['attestation_path']);audit=read_json(e['audit_path'])
        except FileNotFoundError as error:raise EvidenceMismatch('explicit historical attestation/audit absent') from error
        fields(a,('schema_version','artifact_type','source','audit_path','audit_sha256','binary_path','binary_sha256','model_path','model_sha256','limitations'),'operator attestation')
        if type(a['schema_version'])is not int or a['schema_version']!=1 or a['artifact_type']!='ninfer_benchmark_operator_attestation' or a['source']!='operator_attestation':raise EvidenceMismatch('unsupported operator attestation')
        for key in ('audit_path','binary_path','model_path'):absolute(a[key])
        if (a['audit_path']!=e['audit_path'] or a['audit_sha256']!=self.digest(e['audit_path']) or
            a['binary_sha256']!=audit.get('binary_sha256') or a['binary_sha256']!=fp['identity']['exe_sha256'] or
            a['model_path']!=suite['model'] or a['model_sha256']!=fp['identity']['model_sha256']):raise EvidenceMismatch('historical operator attestation/model/binary hash mismatch')
        if not isinstance(a['limitations'],list) or not a['limitations'] or any(not isinstance(x,str) or not x for x in a['limitations']):raise EvidenceMismatch('operator attestation must retain legacy limitations')
        words=shlex.split(audit.get('command','').replace('\\\n',' '));flags={};values={}
        for i,word in enumerate(words):
            if word.startswith('--'):
                if i+1<len(words) and not words[i+1].startswith('--'):values[word]=words[i+1]
                else:flags[word]=True
        options=suite['options']
        for key in ('pairs','repeats','max_tokens','max_context','kv_capacity','engine_concurrency','cooldown','memory_sample_seconds'):
            actual=values.get('--'+key.replace('_','-'))
            if actual is None or float(actual)!=options[key]:raise EvidenceMismatch('historical driver option mismatch: '+key)
        if (values.get('--device')!=str(suite['gpu']['device']) or values.get('--nvml-device')!=str(suite['gpu']['physical']) or
            'CUDA_VISIBLE_DEVICES='+suite['gpu']['cuda_visible_devices'] not in words):raise EvidenceMismatch('historical physical GPU/visible mapping mismatch')
        if suite['kind']=='resident' and (('--prime-prefix' in flags)!=options['prime'] or values.get('--workload-label-prefix')!=options['workload_label_prefix']):raise EvidenceMismatch('historical priming/suite label mismatch')
        metadata={'source_mode':'attested_legacy_operator','limitations':a['limitations'],
          'historical_versions':None,'current_import_validation_versions':fp['identity']['versions'],
          'audit_path':e['audit_path'],'audit_sha256':a['audit_sha256'],'attestation_path':e['attestation_path'],
          'attestation_sha256':self.digest(e['attestation_path']),'native_path':a['binary_path']}
        self.imports[cache_key]=metadata;return metadata

    def _arm_specs(self,unit,root):
        kind=unit['suite']['kind'];work=unit['workload'];c=unit['clients'];k=unit['action']
        for pair in range(unit['suite']['options']['pairs']):
            order=(0,k) if pair%2==0 else (k,0)
            if kind=='auto':order=('none','fixed','auto') if pair%2==0 else ('auto','fixed','none')
            for pos,arm in enumerate(order):
                name=f'{work}-C{c}-p{pair}-{arm}' if kind=='auto' else f'{work}-C{c}-K{k}-p{pair}-o{pos}-a{arm}'
                directory=root/name if kind=='fixed' else root/'runs'/name
                yield pair,arm,directory/('measurement.json' if kind=='fixed' else 'native.json')

    def validate_raw(self,unit,root,metadata):
        suite=unit['suite'];o=suite['options'];kind=suite['kind'];root=Path(root);raw_hashes={}
        environment=read_json(root/'environment.json')
        for key,expected in (('cuda_visible_device_ordinal',suite['gpu']['device']),('cuda_visible_devices',suite['gpu']['cuda_visible_devices']),('nvml_physical_device',str(suite['gpu']['physical']))):
            if str(environment.get(key))!=str(expected):raise EvidenceMismatch('raw environment mismatch: '+key)
        def retain(path):raw_hashes[str(path.resolve())]=self.digest(path)
        retain(root/'environment.json')
        reports={};resident_identity=None
        prompt=self.prompt_bytes(unit)
        configuration={'prompt':prompt.decode('utf-8'),'client_concurrency':unit['clients'],'repeats':o['repeats'],'max_tokens':o['max_tokens'],
          'sampling':{'temperature':0,'presence_penalty':0,'frequency_penalty':0}}
        profile=read_json(o['profile']) if kind=='auto' else None
        for pair,arm,path in self._arm_specs(unit,root):
            raw=read_json(path);command_path=path.parent/'command.json';memory_path=path.parent/'memory.json'
            command=read_json(command_path);values,flags=argv_options(command);memory=read_json(memory_path)
            expected={'--model':suite['model'],'--output':str(path.resolve()),'--client-concurrency':str(unit['clients']),
              '--engine-concurrency':str(o['engine_concurrency']),'--repeats':str(o['repeats']),'--max-tokens':str(o['max_tokens']),
              '--max-context':str(o['max_context']),'--kv-capacity':str(o['kv_capacity']),'--device':str(suite['gpu']['device'])}
            if command[0]!=metadata['native_path'] or any(values.get(key)!=value for key,value in expected.items()):raise EvidenceMismatch('raw native command configuration mismatch')
            if Path(values.get('--prompt-file','')).read_bytes()!=prompt:raise EvidenceMismatch('raw prompt bytes mismatch')
            allowed=set(expected)|{'--prompt-file','--draft-tokens','--spec-router-profile'}
            if set(values)-allowed:raise EvidenceMismatch('unexpected native execution option')
            expected_flags={'--prime-prefix'} if kind=='resident' and o['prime'] else {'--allow-route-switching'} if kind=='auto' and arm=='auto' else set()
            if flags!=expected_flags:raise EvidenceMismatch('raw priming/mode flags mismatch')
            if kind=='auto' and arm=='auto':
                if '--draft-tokens' in values or '--spec-router-profile' not in values:raise EvidenceMismatch('auto is not a mixed-route execution')
                if self.digest(values['--spec-router-profile'])!=self.digest(o['profile']):raise EvidenceMismatch('raw auto profile SHA mismatch')
            else:
                action=0 if arm=='none' else unit['action'] if arm=='fixed' else arm
                if values.get('--draft-tokens')!=str(action):raise EvidenceMismatch('raw arm action mismatch')
                if (kind=='resident')!=('--spec-router-profile' in values):raise EvidenceMismatch('bare None/standalone fixed must not use resident profile')
            if memory.get('kind')!='device_memory_observed_lower_bound' or str(memory.get('gpu'))!=str(suite['gpu']['physical']) or memory.get('sample_interval_seconds')!=o['memory_sample_seconds']:
                raise EvidenceMismatch('memory sampler scope/GPU/period mismatch')
            if not isinstance(memory.get('samples'),list) or not isinstance(memory.get('errors'),list):raise ValueError('missing memory sampler audit')
            samples=memory['samples']
            if any(type(s.get('used_mib'))is not int or s['used_mib']<0 or type(s.get('monotonic_ns'))is not int or s['monotonic_ns']<=0 for s in samples):raise ValueError('invalid memory samples')
            if memory.get('peak_used_mib')!=max((s['used_mib'] for s in samples),default=None):raise ValueError('memory peak does not match samples')
            for sidecar in (path,command_path,memory_path,path.parent/'stdout.log',path.parent/'stderr.log'):retain(sidecar)
            retain(Path(values['--prompt-file']))
            if '--spec-router-profile' in values:retain(Path(values['--spec-router-profile']))
            if kind=='fixed':
                fixed.validate_report(raw)
                for key,expected_value in (('model',suite['model']),('device',suite['gpu']['device']),('client_concurrency',unit['clients']),('engine_concurrency',o['engine_concurrency']),('repeats',o['repeats']),('max_tokens',o['max_tokens']),('max_context',o['max_context']),('kv_capacity',o['kv_capacity']),('draft_tokens',arm)):
                    if raw.get(key)!=expected_value:raise EvidenceMismatch('raw fixed report configuration mismatch: '+key)
                for key in ('workspace_logical_peak_bytes','workspace_allocator_peak_bytes','runtime_reservation_bytes'):
                    if type(raw.get('memory',{}).get(key))is not int or raw['memory'][key]<=0:raise ValueError('missing Engine logical memory')
                if all(r['finish_reason']==1 for r in raw['requests']) and sum(e['committed_tokens'] for e in raw['rounds'])!=sum(len(r['generated_token_ids']) for r in raw['requests'])-len(raw['requests']):raise ValueError('fixed commit/token conservation failure')
            elif kind=='resident':
                if resident_identity is None:resident_identity=raw['record']['identity']
                if (resident_identity['max_context']!=o['max_context'] or resident_identity['resolved_kv_capacity']!=o['kv_capacity'] or resident_identity['max_concurrency']!=o['engine_concurrency'] or resident_identity['proposal_head']!='full'):raise EvidenceMismatch('resident startup identity/config mismatch')
                resident.validate_wrapper(raw,resident_identity,arm,configuration,o['prime'])
                control=read_json(values['--spec-router-profile'])
                if control.get('identity')!=resident_identity or auto.profile_table(control)!=resident.constant_profile(resident_identity,arm)['cells']:raise ValueError('constant control profile does not match actual resident action')
            else:raw=auto.validate_measurement(raw,arm,profile,configuration,suite['model'],suite['gpu']['device'],unit['action'])
            reports[pair,arm]=raw
        if kind=='fixed':qualification=fixed.qualify_pairs([(reports[p,0],reports[p,unit['action']]) for p in range(o['pairs'])])
        elif kind=='resident':
            comparison={'workload':f'{o["workload_label_prefix"]}/{unit["workload"]}/C{unit["clients"]}','candidate_action':unit['action'],
              'pairs':[{'pair_id':str(p),'order':[0,unit['action']] if p%2==0 else [unit['action'],0],
                        'baseline':reports[p,0]['record'],'candidate':reports[p,unit['action']]['record']} for p in range(o['pairs'])]}
            audit=analyse_comparison(comparison,resident_identity)['audit']
            bad=[reason for reason in audit['reasons'] if any(word in reason for word in ('output','instability','mismatch','duplicate','error'))]
            qualification={'correct':not bad,'failures':bad,'qualified_performance':audit['qualified'],'audit':audit}
        else:qualification=auto.qualify_triples([{arm:reports[p,arm] for arm in ('none','fixed','auto')} for p in range(o['pairs'])])
        for key in ('audit_path','attestation_path'):
            if key in metadata:retain(Path(metadata[key]))
        return {'qualification':qualification,'raw_sha256':raw_hashes,'evidence_root':str(root.resolve())}

    def assess(self,unit,retry_failed=False):
        suite=unit['suite'];arms=suite['options']['pairs']*(3 if suite['kind']=='auto' else 2)
        result={'unit':unit['id'],'suite':suite['id'],'required_native_arms':arms,'status':'run','operation':'run_complete_comparison','eligible_to_execute':True,'reasons':[]}
        if suite['state']=='deferred':return {**result,'status':'deferred','eligible_to_execute':False,'reasons':['suite explicitly deferred; no campaign files read']}
        try:fp=self.fingerprint(unit)
        except (OSError,ValueError,TypeError,KeyError) as error:return {**result,'status':'blocked','eligible_to_execute':False,'reasons':[str(error)]}
        result['fingerprint']=fp
        directory=self.unit_dir(unit);complete=directory/'complete.json';last=directory/'last-attempt.json'
        root=None;metadata=None;checkpoint=None
        try:
            if complete.exists():
                checkpoint=read_json(complete)
                if checkpoint.get('fingerprint')!=fp:
                    prior=checkpoint.get('fingerprint',{}).get('identity',{})
                    execution=lambda identity:{k:v for k,v in identity.items() if k not in ('versions','python_version')}
                    if execution(prior)!=execution(fp['identity']):
                        return {**result,'status':'stale','reasons':['completed comparison execution identity/config/prompt/profile changed']}
                    result['validation_requalification']={'previous_versions':prior.get('versions'),'previous_python_version':prior.get('python_version'),'current_versions':fp['identity']['versions'],'current_python_version':fp['identity']['python_version']}
                root=checkpoint['evidence_root'];metadata={**checkpoint['metadata'],'current_validation_versions':fp['identity']['versions']}
                for path,digest in checkpoint['raw_sha256'].items():
                    if self.digest(path)!=digest:return {**result,'status':'stale','reasons':['completed raw evidence hash changed: '+path]}
            elif last.exists():
                previous=read_json(last)
                if previous['fingerprint']!=fp:return {**result,'status':'stale','reasons':['prior attempt configuration changed']}
                root=previous['evidence_root'];metadata=previous['metadata']
                if previous.get('driver_returncode') is None:
                    return {**result,'status':'incomplete','operation':'run_missing_comparison','reasons':['previous attempt has no completed driver exit; never splice its pairs']}
                if previous['driver_returncode']!=0:
                    return {**result,'status':'failed','eligible_to_execute':retry_failed,'reasons':['previous driver failed; explicit retry required']}
            elif suite['existing_evidence'] is not None:
                metadata=self._legacy(unit,fp);root=suite['existing_evidence']['campaign_root']
            else:return result
            validated=self.validate_raw(unit,root,metadata)
            result.update(validated,metadata=metadata,source_mode=metadata['source_mode'],limitations=metadata.get('limitations',[]))
            if not validated['qualification']['correct']:
                return {**result,'status':'failed','eligible_to_execute':retry_failed,'reasons':['complete comparison correctness failed; explicit retry required']}
            return {**result,'status':'skip','operation':'reuse_complete_comparison','eligible_to_execute':False}
        except EvidenceMismatch as error:return {**result,'status':'stale','reasons':[str(error)]}
        except FileNotFoundError as error:return {**result,'status':'incomplete','operation':'run_missing_comparison','reasons':[str(error)]}
        except (ValueError,TypeError,KeyError,OverflowError) as error:return {**result,'status':'failed','eligible_to_execute':retry_failed,'reasons':['raw evidence invalid: '+str(error)]}

    def write_completion(self,unit,result):
        if result['status']!='skip' or not result['qualification']['correct']:raise ValueError('only fully revalidated correct comparisons can complete')
        directory=self.unit_dir(unit);path=directory/'complete.json'
        if path.exists():
            old=path.read_bytes();history=directory/'checkpoint-history'/(hashlib.sha256(old).hexdigest()+'.json')
            if not history.exists():atomic_json(history,read_json(path))
        atomic_json(path,{'schema_version':1,'artifact_type':'ninfer_matrix_complete_comparison',
          'fingerprint':result['fingerprint'],'evidence_root':result['evidence_root'],'metadata':result['metadata'],
          'raw_sha256':result['raw_sha256'],'qualification':result['qualification']})

    def describe(self,retry_failed=False):
        results={s['id']:[] for s in self.plan['suites']};by_id={s['id']:s for s in self.plan['suites']};visiting=set()
        def inspect(sid):
            if sid in visiting:return
            visiting.add(sid);suite=by_id[sid]
            for dependency in suite['depends_on']:inspect(dependency)
            blocked=any(any(r['status']!='skip' for r in results[d]) for d in suite['depends_on'])
            for unit in (u for u in self.units if u['suite']['id']==sid):
                if blocked and suite['state']!='deferred':
                    result={'unit':unit['id'],'suite':sid,'status':'blocked','eligible_to_execute':False,'reasons':['dependency suite is not completely correct/reusable'],
                            'required_native_arms':suite['options']['pairs']*(3 if suite['kind']=='auto' else 2)}
                else:result=self.assess(unit,retry_failed)
                results[sid].append(result)
        for suite in self.plan['suites']:inspect(suite['id'])
        units=[r for suite in self.plan['suites'] for r in results[suite['id']]]
        return {'artifact_type':'ninfer_spec_router_matrix_inventory','schema_version':1,'unit_count':len(units),
          'status_counts':dict(Counter(r['status'] for r in units)),'units':units,
          'locking_scope':'harness-managed jobs only; cannot detect already running legacy campaigns'}

    def execution_units(self):
        """Dependencies first, retaining plan order within independent suites."""
        suites={s['id']:s for s in self.plan['suites']};seen=set();ordered=[]
        def visit(sid):
            if sid in seen:return
            seen.add(sid)
            for dependency in suites[sid]['depends_on']:visit(dependency)
            ordered.extend(u for u in self.units if u['suite']['id']==sid)
        for suite in self.plan['suites']:visit(suite['id'])
        return ordered

    def run_unit(self,unit,retry_failed=False,runner=None):
        # Reinspect under both locks; never reuse inventory decisions made before locking.
        directory=self.unit_dir(unit);physical=unit['suite']['gpu']['physical']
        gpu_lock=Path(tempfile.gettempdir())/f'ninfer-spec-matrix-{os.getuid()}'/f'gpu-{physical}.lock'
        with file_lock(directory/'job.lock'),file_lock(gpu_lock):
            result=self.assess(unit,retry_failed)
            if result['status']=='skip':self.write_completion(unit,result);return result
            if not result['eligible_to_execute']:return result
            attempt=directory/'attempts'/('attempt-'+secrets.token_hex(8));attempt.parent.mkdir(parents=True,exist_ok=True)
            metadata={'source_mode':'native_matrix_checkpoint','native_path':unit['suite']['exe'],'limitations':[]}
            marker={'fingerprint':result['fingerprint'],'evidence_root':str(attempt),'metadata':metadata,'driver_returncode':None}
            atomic_json(directory/'last-attempt.json',marker)
            complete=directory/'complete.json'
            if complete.exists():
                history=directory/'checkpoint-history'/(hashlib.sha256(complete.read_bytes()).hexdigest()+'.json')
                history.parent.mkdir(parents=True,exist_ok=True)
                os.replace(complete,history)
            command=self.driver_command(unit,attempt)
            atomic_json(directory/'attempt-commands'/(attempt.name+'.json'),{'argv':command,'cuda_visible_devices':unit['suite']['gpu']['cuda_visible_devices']})
            execute=runner or subprocess.run
            completed=execute(command,cwd=str(ROOT),env={**os.environ,'CUDA_VISIBLE_DEVICES':unit['suite']['gpu']['cuda_visible_devices']},check=False)
            marker['driver_returncode']=completed.returncode
            atomic_json(directory/'last-attempt.json',marker)
            result=self.assess(unit,retry_failed)
            if completed.returncode!=0 and result['status']=='skip':
                return {**result,'status':'failed','eligible_to_execute':False,'reasons':['driver returned nonzero despite raw reports']}
            if result['status']=='skip':self.write_completion(unit,result)
            return result
