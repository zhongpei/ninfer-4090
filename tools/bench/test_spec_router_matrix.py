"""Synthetic complete-comparison fixtures; no inference/GPU execution."""
import copy
import hashlib
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.bench import spec_matrix_resume as resume
from tools.bench.test_auto_spec_router_comparison import native_report
from tools.dflash2_training.ab_suite import select_workloads


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.exe=self.root/'native';self.exe.write_bytes(b'fixed native')
        self.model=self.root/'model';self.model.write_bytes(b'frozen model')
        self.suite={'id':'fixed','kind':'fixed','state':'ready','depends_on':[],
          'exe':str(self.exe),'model':str(self.model),'gpu':{'physical':1,'cuda_visible_devices':'1','device':0},
          'options':{'workloads':['chat'],'concurrency':[2],'draft_tokens':[7],'pairs':2,'repeats':2,
            'max_tokens':8,'max_context':32768,'kv_capacity':262144,'engine_concurrency':8,'cooldown':2,'memory_sample_seconds':.05},
          'output_root':str(self.root/'managed'),'existing_evidence':None}
        self.plan={'schema_version':1,'artifact_type':'ninfer_spec_router_matrix_plan','suites':[self.suite]}

    def tearDown(self):self.temp.cleanup()
    def matrix(self):return resume.ResumeMatrix(copy.deepcopy(self.plan))

    def populate(self, root, correct=True, gain=True):
        suite=self.suite;prompt=select_workloads('chat')[0].prompt
        (root/'chat.prompt').parent.mkdir(parents=True,exist_ok=True)
        (root/'chat.prompt').write_bytes(prompt.encode())
        write(root/'environment.json',{'cuda_visible_device_ordinal':0,'cuda_visible_devices':'1','nvml_physical_device':'1'})
        for p in range(2):
            order=(0,7) if p==0 else (7,0)
            for pos,action in enumerate(order):
                directory=root/f'chat-C2-K7-p{p}-o{pos}-a{action}'
                directory.mkdir()
                raw=native_report('fixed' if action else 'none',wall=9000 if action and gain else 10000)
                raw['model']=str(self.model)
                if action and not correct:raw['requests'][0]['content']='wrong'
                write(directory/'measurement.json',raw)
                cmd=[str(self.exe),'--model',str(self.model),'--prompt-file',str(root/'chat.prompt'),
                  '--output',str(directory/'measurement.json'),'--draft-tokens',str(action),'--client-concurrency','2',
                  '--engine-concurrency','8','--repeats','2','--max-tokens','8','--max-context','32768',
                  '--kv-capacity','262144','--device','0']
                write(directory/'command.json',cmd)
                write(directory/'memory.json',{'kind':'device_memory_observed_lower_bound','gpu':'1',
                  'sample_interval_seconds':.05,'samples':[{'monotonic_ns':1,'used_mib':100}], 'peak_used_mib':100,'errors':[]})
                (directory/'stdout.log').write_text('');(directory/'stderr.log').write_text('')

    def legacy(self, correct=True, gain=True):
        root=self.root/'legacy';self.populate(root,correct,gain)
        matrix=self.matrix();unit=matrix.units[0]
        audit={'binary_sha256':hashlib.sha256(self.exe.read_bytes()).hexdigest(),
          'command':'CUDA_VISIBLE_DEVICES=1 '+shlex.join(matrix.driver_command(unit,root))}
        write(root/'audit.json',audit)
        attestation={'schema_version':1,'artifact_type':'ninfer_benchmark_operator_attestation','source':'operator_attestation',
          'audit_path':str(root/'audit.json'),'audit_sha256':hashlib.sha256((root/'audit.json').read_bytes()).hexdigest(),
          'binary_path':str(self.exe),'binary_sha256':audit['binary_sha256'],'model_path':str(self.model),
          'model_sha256':hashlib.sha256(self.model.read_bytes()).hexdigest(),
          'limitations':['model digest is operator asserted','historical driver fingerprint absent']}
        write(self.root/'attestation.json',attestation)
        self.suite['existing_evidence']={'campaign_root':str(root),'audit_path':str(root/'audit.json'),
                                         'attestation_path':str(self.root/'attestation.json')}
        return root

    def test_valid_legacy_skip_and_nongain_skip(self):
        self.legacy(gain=False)
        matrix=self.matrix();result=matrix.assess(matrix.units[0])
        self.assertEqual(result['status'],'skip')
        self.assertEqual(result['source_mode'],'attested_legacy_operator')
        self.assertTrue(result['qualification']['correct'])
        self.assertFalse(result['qualification']['qualified_performance'])
        self.assertTrue(result['limitations'])

    def test_new_atomic_checkpoint_skip_and_raw_revalidation(self):
        self.legacy();matrix=self.matrix();unit=matrix.units[0]
        result=matrix.assess(unit);matrix.write_completion(unit,result)
        self.suite['existing_evidence']=None
        matrix=self.matrix();self.assertEqual(matrix.assess(matrix.units[0])['status'],'skip')
        raw_path=self.root/'legacy'/'chat-C2-K7-p0-o1-a7'/'measurement.json'
        raw=json.loads(raw_path.read_text());raw['requests'][0]['content']='changed';write(raw_path,raw)
        self.assertNotEqual(matrix.assess(matrix.units[0])['status'],'skip')

    def test_config_binary_model_prompt_and_version_mismatch_stale(self):
        self.legacy();matrix=self.matrix();matrix.write_completion(matrix.units[0],matrix.assess(matrix.units[0]))
        baseline=copy.deepcopy(self.plan)
        for mutate in (lambda: self.suite['options'].update(max_tokens=9),
                       lambda: self.exe.write_bytes(b'changed binary'),
                       lambda: self.model.write_bytes(b'changed model')):
            with self.subTest(mutate=mutate):
                mutate();m=self.matrix();self.assertEqual(m.assess(m.units[0])['status'],'stale')
                self.plan=copy.deepcopy(baseline);self.suite=self.plan['suites'][0]
                self.exe.write_bytes(b'fixed native');self.model.write_bytes(b'frozen model')
        m=self.matrix()
        with patch.object(m,'version_fingerprint',return_value={'changed':'version'}):
            self.assertEqual(m.assess(m.units[0])['status'],'skip')
        with patch.object(m,'prompt_bytes',return_value=b'new prompt'):
            self.assertEqual(m.assess(m.units[0])['status'],'stale')

    def test_partial_pair_incomplete_never_spliced(self):
        root=self.legacy();(root/'chat-C2-K7-p1-o1-a0'/'measurement.json').unlink()
        matrix=self.matrix();result=matrix.assess(matrix.units[0])
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['operation'],'run_missing_comparison')
        self.assertEqual(result['required_native_arms'],4)

    def test_correctness_fail_blocks_not_reused_or_automatically_retried(self):
        self.legacy(correct=False);matrix=self.matrix();result=matrix.assess(matrix.units[0])
        self.assertEqual(result['status'],'failed')
        self.assertFalse(result['eligible_to_execute'])
        result=matrix.assess(matrix.units[0],retry_failed=True)
        self.assertTrue(result['eligible_to_execute'])

    def test_missing_attestation_not_path_inferred(self):
        self.legacy();(self.root/'attestation.json').unlink()
        matrix=self.matrix();result=matrix.assess(matrix.units[0])
        self.assertEqual(result['status'],'stale')
        self.assertIn('attestation',result['reasons'][0])

    def test_sidecar_or_command_mismatch_not_reused(self):
        root=self.legacy();path=root/'chat-C2-K7-p0-o1-a7'/'command.json'
        cmd=json.loads(path.read_text());cmd[cmd.index('--max-tokens')+1]='9';write(path,cmd)
        matrix=self.matrix();self.assertNotEqual(matrix.assess(matrix.units[0])['status'],'skip')

    def test_deferred_and_dependency_block_do_not_read_running_root(self):
        self.suite['state']='deferred';self.suite['existing_evidence']={'campaign_root':'/absent/running','audit_path':'/absent/audit','attestation_path':'/absent/attestation'}
        matrix=self.matrix()
        with patch.object(matrix,'fingerprint',side_effect=AssertionError('deferred must not read artifacts')):
            self.assertEqual(matrix.assess(matrix.units[0])['status'],'deferred')
        second=copy.deepcopy(self.suite);second.update(id='dependent',state='ready',depends_on=['fixed'],output_root=str(self.root/'dependent'),existing_evidence=None)
        self.plan['suites'].append(second)
        matrix=self.matrix();report=matrix.describe()
        self.assertEqual(report['units'][1]['status'],'blocked')

    def test_no_gpu_default_cli_and_strict_plan(self):
        with self.assertRaises(ValueError):resume.ResumeMatrix({**self.plan,'unexpected':1})
        self.suite['exe']='relative'
        with self.assertRaises(ValueError):self.matrix()

    def test_job_and_gpu_locks_and_atomic_failure_preserve_checkpoint(self):
        self.legacy();matrix=self.matrix();unit=matrix.units[0];result=matrix.assess(unit)
        matrix.write_completion(unit,result)
        path=matrix.unit_dir(unit)/'complete.json';before=path.read_bytes()
        with patch('tools.bench.spec_matrix_resume.os.replace',side_effect=OSError('interrupted rename')):
            with self.assertRaises(OSError):matrix.write_completion(unit,result)
        self.assertEqual(path.read_bytes(),before)
        with resume.file_lock(self.root/'lock'):
            with self.assertRaises(resume.MatrixLocked):
                with resume.file_lock(self.root/'lock'):pass

    def test_stale_complete_new_attempt_then_skip(self):
        old=self.legacy();m=self.matrix();u=m.units[0]
        m.write_completion(u,m.assess(u))
        self.suite['options']['cooldown']=3
        m=self.matrix();u=m.units[0]
        def runner(command,**kwargs):
            self.populate(Path(command[command.index('--out')+1]))
            return type('Completed',(),{'returncode':0})()
        self.assertEqual(m.run_unit(u,runner=runner)['status'],'skip')
        self.assertEqual(self.matrix().assess(self.matrix().units[0])['status'],'skip')
        self.assertTrue(old.exists())
        self.assertTrue(list((m.unit_dir(u)/'checkpoint-history').iterdir()))

    def test_validation_version_change_revalidates_without_gpu(self):
        self.legacy();m=self.matrix();u=m.units[0]
        m.write_completion(u,m.assess(u))
        with patch.object(m,'version_fingerprint',return_value={'new':'validator'}), patch.object(m,'validate_raw',wraps=m.validate_raw) as validate:
            result=m.assess(u)
            self.assertEqual(result['status'],'skip')
            validate.assert_called_once()
            m.write_completion(u,result)
            self.assertEqual(resume.read_json(m.unit_dir(u)/'complete.json')['fingerprint']['identity']['versions'],{'new':'validator'})

    def test_resident_transitive_fixed_validator_version(self):
        m=self.matrix();u=m.units[0];u['suite']['kind']='resident'
        self.assertIn(str(Path(resume.fixed.__file__).resolve()),m.version_fingerprint(u))

    def test_auto_missing_profile_block_and_profile_hash_mismatch(self):
        from tools.bench.test_auto_spec_router_comparison import profile
        self.suite['kind']='auto'
        o=self.suite['options'];o.pop('draft_tokens');o.update(fixed_draft_tokens=7,profile=None)
        m=self.matrix();self.assertEqual(m.assess(m.units[0])['status'],'blocked')
        path=self.root/'profile.json';data=profile();write(path,data);o['profile']=str(path)
        m=self.matrix();u=m.units[0];old=m.fingerprint(u)
        # Metadata-only bytes changes still change explicit profile identity.
        data['provenance']={'audit':'different'};write(path,data)
        self.assertNotEqual(old,m.fingerprint(u))

    def test_default_cli_never_executes(self):
        from tools.bench import run_spec_router_matrix as entry
        import contextlib,io
        self.legacy(gain=False);path=self.root/'plan.json';write(path,self.plan)
        with patch('tools.bench.spec_matrix_resume.subprocess.run',side_effect=AssertionError('dry run launched process')),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry.main(['--plan',str(path)]),0)
        report=json.loads(output.getvalue())
        self.assertEqual(report['mode'],'dry_run')
        self.assertEqual(report['status_counts'],{'skip':1})

    def test_execute_topological_and_only_two_inventory_scans(self):
        from tools.bench import run_spec_router_matrix as entry
        import contextlib,io
        child=copy.deepcopy(self.suite);child.update(id='child',depends_on=['fixed'],output_root=str(self.root/'child'))
        self.plan['suites'].insert(0,child)
        m=self.matrix();calls=[];scans=[]
        def describe(*args):
            scans.append(1)
            return {'units':[{'unit':u['id'],'suite':u['suite']['id'],'status':'run' if not u['suite']['depends_on'] else 'blocked','eligible_to_execute':not u['suite']['depends_on']} for u in m.units]}
        def run(unit,*args):
            calls.append(unit['suite']['id'])
            return {'unit':unit['id'],'suite':unit['suite']['id'],'status':'skip','eligible_to_execute':False}
        path=self.root/'plan.json';write(path,self.plan)
        with patch.object(entry,'ResumeMatrix',return_value=m),patch.object(m,'describe',side_effect=describe),patch.object(m,'run_unit',side_effect=run),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(['--plan',str(path),'--execute']),0)
        self.assertEqual(calls,['fixed','child'])
        self.assertEqual(len(scans),2)


if __name__=='__main__':unittest.main()
