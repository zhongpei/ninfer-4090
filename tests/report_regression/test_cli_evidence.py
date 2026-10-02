"""CLI evidence tests. Subprocesses are CPU fixtures, not model inference."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tools.dflash2_training import ab_suite as a, cli_validation as v


def metrics():
    result = dict.fromkeys(v.COUNTERS, 0)
    result.update(schema=1, token_ids=[10, 20, 30], generated=3, decoded=2,
                  finish_reason=1, decode_seconds=0.00123456789,
                  decode_tok_s=2/0.00123456789, prefill_seconds=0.01,
                  temperature=0, presence_penalty=0, frequency_penalty=0,
                  acceptance_pct=None, tok_per_round=None)
    return result


def row(arm, pair=0, speed=100):
    return dict(arm=arm, pair=pair, comparison="candidate", workload="prose",
                returncode=0, stdout_bytes=7, stdout_sha256="a"*64,
                token_ids=[10,20,30], finish_reason=1, decode_tok_s=speed,
                metrics_source="machine-v1", metric_errors=[])


def rows():
    return [row(arm, pair, 100 if arm=="baseline" else 130)
            for pair in range(4) for arm in ("baseline","candidate")]


class MetricsTests(unittest.TestCase):
    def test_machine_full_precision_and_tokens(self):
        m=metrics(); parsed, source, errors=v.metrics_from_log(v.PREFIX+json.dumps(m), a.METRICS)
        self.assertEqual(parsed,m); self.assertEqual(source,"machine-v1"); self.assertEqual(errors,[])

    def test_pretty_si_and_exponent_are_diagnostic(self):
        for text,want in (("1.15k",1150),("1.5M",1500000),("1.2e3",1200),("999.7",999.7)):
            got,source,errors=v.metrics_from_log("summary decode speed "+text+" tok/s",a.METRICS)
            self.assertEqual(got["decode_tok_s"],want); self.assertEqual(source,"pretty-rounded")
            self.assertTrue(errors)

    def test_machine_is_authoritative_over_pretty(self):
        m=metrics(); text="decode speed 1.15k tok/s\n"+v.PREFIX+json.dumps(m)
        self.assertEqual(a.parse_metric(text,"decode_tok_s"), m["decode_tok_s"])

    def test_duplicate_or_malformed_machine_never_falls_back(self):
        for record in ("{}", "{bad", json.dumps(metrics())+"\n"+v.PREFIX+json.dumps(metrics())):
            data, source, errors=v.metrics_from_log("decode speed 1.15k tok/s\n"+v.PREFIX+record,a.METRICS)
            self.assertEqual(data,{}); self.assertEqual(source,"invalid-machine"); self.assertTrue(errors)

    def test_invalid_field_types_and_values(self):
        for key,value in (("generated",True),("token_ids",[10]),("finish_reason",None),
                          ("decode_seconds",0),("decode_tok_s",float("nan")),
                          ("temperature",1),("frequency_penalty",1),("rounds",-1)):
            m=metrics(); m[key]=value
            self.assertTrue(v.metrics_from_log(v.PREFIX+json.dumps(m),a.METRICS)[2],key)

    def test_token_prefix_localizes_divergence(self):
        diff=v.token_difference({"token_ids":[1,2,3]}, {"token_ids":[1,2,5]})
        self.assertEqual(diff["token_index"],2); self.assertEqual(diff["common_generated_prefix"],[1,2])

    def test_file_identity_streamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"artifact"; path.write_bytes(b"fixture")
            self.assertEqual(v.file_identity(path)["sha256"],hashlib.sha256(b"fixture").hexdigest())


class CLIGateTests(unittest.TestCase):
    def judge(self, rs):
        return a.judge(rs,(a.WORKLOADS[0],),(a.ARMS[0],a.Arm("candidate",(),"baseline")),1,4)["candidate"]

    def test_valid_rows_qualify(self):
        result=self.judge(rows()); self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["pooled"]["weighted_geomean_speedup"],1.3)

    def test_discarded_first_pair_still_blocks(self):
        rs=rows(); rs[1]["token_ids"]=[10,21,30]
        result=self.judge(rs); self.assertFalse(result["correct"])
        self.assertIsNone(result["pooled"]["weighted_geomean_speedup"])
        self.assertAlmostEqual(result["pooled"]["diagnostic_geomean_speedup"],1.3)

    def test_both_sides_drift_together_blocks_self_repeat(self):
        rs=rows(); rs[-1]["token_ids"]=[10,22,30]; rs[-2]["token_ids"]=[10,22,30]
        result=self.judge(rs); self.assertFalse(result["correct"])
        self.assertTrue(any(x["reason"]=="self-repeat-unstable" for x in result["workloads"]["prose"]["output"]["mismatches"]))

    def test_duplicates_missing_or_failed_runs_block(self):
        for rs in (rows()[:-1], rows()+[rows()[0]], rows()[1:]+[rows()[1]]):
            self.assertFalse(self.judge(rs)["correct"])
        rs=rows(); rs[1]["returncode"]=1; self.assertFalse(self.judge(rs)["correct"])

    def test_identical_text_different_finish_or_tokens_blocks(self):
        for key,value in (("finish_reason",2),("token_ids",[10,21,30])):
            rs=rows(); rs[1][key]=value; self.assertFalse(self.judge(rs)["correct"])

    def test_rounded_metrics_do_not_qualify(self):
        rs=rows(); rs[0]["metrics_source"]="pretty-rounded"
        self.assertFalse(self.judge(rs)["correct"])

    def test_missing_scenario_does_not_change_pool(self):
        result=self.judge(rows()); per=result["workloads"]
        pooled=a.pooled(per,(a.WORKLOADS[0],a.WORKLOADS[1]))
        self.assertIsNone(pooled["diagnostic_geomean_speedup"])
        self.assertEqual(pooled["missing_workloads"],["chat"])

    def test_comparator_closure(self):
        self.assertEqual({x.name for x in a.select_arms("tree15-stair")},
                         {"baseline","dflash2-k15","tree15","tree15-stair"})

    def test_comparison_names_isolate_same_control_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out=tmp,exe="fake",model="fake",kv_dtype="int8")
            proc=SimpleNamespace(stdout=b"answer",stderr=(v.PREFIX+json.dumps(metrics())).encode(),returncode=0)
            with patch.object(a.subprocess,"run",return_value=proc):
                left=a.run_once(args,a.ARMS[0],a.WORKLOADS[0],0,0,"comparison-a")
                right=a.run_once(args,a.ARMS[0],a.WORKLOADS[0],0,0,"comparison-b")
                self.assertNotEqual(left["stdout_path"],right["stdout_path"])
                with self.assertRaises(FileExistsError):
                    a.run_once(args,a.ARMS[0],a.WORKLOADS[0],0,0,"comparison-a")

    def test_timeout_keeps_partial_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out=tmp,exe="fake",model="fake",kv_dtype="int8")
            with patch.object(a.subprocess,"run",side_effect=subprocess.TimeoutExpired("fake",1,b"partial",b"failure")):
                r=a.run_once(args,a.ARMS[0],a.WORKLOADS[0],0,0,"timeout")
            self.assertEqual(Path(r["stdout_path"]).read_bytes(),b"partial")
            self.assertNotEqual(r["returncode"],0)

    def test_real_subprocess_cli_journal(self):
        with tempfile.TemporaryDirectory(prefix="ab space ") as tmp:
            root=Path(tmp); exe=root/"fake-cli"; model=root/"model.ninfer"; model.write_bytes(b"not a model")
            exe.write_text("#!"+sys.executable+"\nimport json,os,sys\n"
                           "assert os.environ['NINFER_AB_METRICS']=='1'\n"
                           "print('answer')\nprint('NINFER_METRICS_JSON '+json.dumps("+repr(metrics())+"),file=sys.stderr)\n")
            exe.chmod(0o755)
            command=[sys.executable,"-m","tools.dflash2_training.ab_suite","--exe",str(exe),"--model",str(model),
                     "--out",str(root/"results"),"--workloads","prose","--arms","dflash2-k15","--pairs","2",
                     "--discard","1","--cooldown","0"]
            proc=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(proc.returncode,0,proc.stderr)
            result=json.loads((root/"results/ab-results.json").read_text())
            self.assertEqual(len(result["runs"]),4)
            self.assertTrue(result["comparisons"]["dflash2-k15"]["correct"])
            second=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(len((root/"results/runs.jsonl").read_text().splitlines()),4)


class SerializerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"),"g++ required for standalone serializer fixture")
    def test_cpp_serializer_full_precision(self):
        source=Path(__file__).with_name("cli_metrics_fixture.cpp")
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/"serializer"
            subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I",str(root),str(source),"-o",str(exe)],check=True,capture_output=True)
            proc=subprocess.run([str(exe)],capture_output=True,text=True,check=True)
            data,source_tag,errors=v.metrics_from_log(proc.stdout,a.METRICS)
            self.assertEqual(errors,[]); self.assertEqual(source_tag,"machine-v1")
            self.assertEqual(data["token_ids"],[10,20,30])
            self.assertEqual(data["decode_seconds"],0.00123456789)

if __name__=="__main__": unittest.main()
