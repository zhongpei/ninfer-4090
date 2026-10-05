from copy import deepcopy

import pytest

from tools.bench.run_kv_long_context_perplexity import compare


def arm(dtype="int8"):
    return {"dtype": dtype, "depths": {8: {
        "stream_count": 1, "scored_tokens": 2, "total_nll": 4.0,
        "mean_nll": 2.0, "perplexity": 7.38905609893065,
        "coverage": [["stream-a", 0, 10, 8, 10, 8]],
    }}}


def test_compare_rejects_extra_depth_in_nonbaseline():
    other = arm("fp8")
    other["depths"][16] = deepcopy(other["depths"][8])
    with pytest.raises(RuntimeError, match="depth coverage"):
        compare([arm(), other])


def test_compare_rejects_same_count_different_scored_stream():
    other = arm("fp8")
    other["depths"][8]["coverage"][0][0] = "stream-b"
    with pytest.raises(RuntimeError, match="coverage differs"):
        compare([arm(), other])


def test_compare_matches_same_targets_and_reports_nll_change():
    other = arm("fp8")
    other["depths"][8]["mean_nll"] = 2.01
    summary = compare([arm(), other])
    assert summary["rows"][1]["delta_mean_nll_vs_baseline"] == pytest.approx(0.01)
    assert summary["rows"][1]["ppl_change_percent_vs_baseline"] == pytest.approx(1.0050167)


def test_failed_evaluator_arms_leave_summary_and_all_logs(tmp_path):
    import json
    from tools.bench.run_kv_long_context_perplexity import main

    exe = tmp_path / "failed-evaluator"
    exe.write_text("#!/bin/sh\necho allocation-failed >&2\nexit 2\n")
    exe.chmod(0o755)
    model = tmp_path / "model.ninfer"
    model.touch()
    text = tmp_path / "text.txt"
    text.write_text("sample")
    out = tmp_path / "results"
    assert main(["--exe", str(exe), "--model", str(model), "--text", str(text),
                 "--out", str(out), "--dtypes", "int8,fp8", "--depths", "8"]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert [row["dtype"] for row in summary["failures"]] == ["int8", "fp8"]
    assert (out / "fp8" / "stderr.log").read_text() == "allocation-failed\n"


@pytest.mark.parametrize("fault,expected", [
    ("valid-int8", None),
    ("valid-fp8", None),
    ("dtype", "execution differs"),
    ("missing", "without score coverage"),
    ("truncated", "invalid long-history target coverage"),
])
def test_report_rejects_wrong_execution_or_incomplete_history(tmp_path, fault, expected):
    import json
    import sys
    from argparse import Namespace
    from tools.bench.run_kv_long_context_perplexity import run_arm

    model = tmp_path / "model.ninfer"
    model.touch()
    text = tmp_path / "text.txt"
    text.write_text("sample")
    dtype = "fp8" if fault == "valid-fp8" else "int8"
    execution = {"protocol": "fixed-depth-long-history", "kv_dtype": dtype,
                 "prefix_depths": [8, 16], "tail_tokens": 2, "prefill_a8": True,
                 "prefill_cublas": False, "prefill_cublas_projections": True}
    score = {key: val for key, val in arm()["depths"][8].items() if key != "coverage"}
    depths = [{"prefix_depth": depth, **score} for depth in (8, 16)]
    windows = [{"prefix_depth": depth, "input_begin": 0, "input_end": depth + 2,
                "target_begin": depth, "target_end": depth + 2, "first_target": depth}
               for depth in (8, 16)]
    if fault == "dtype":
        execution["kv_dtype"] = "fp8"
    elif fault == "missing":
        depths.pop()
        windows.pop()
    elif fault == "truncated":
        windows[0]["input_begin"] = 4
    report = {"schema_version": 3, "artifact": {"path": str(model)},
              "execution": execution, "depths": depths,
              "streams": [{"id": "stream-a", "windows": windows}], "overall": {}}
    source = tmp_path / "report-fixture.json"
    source.write_text(json.dumps(report))
    exe = tmp_path / "report-evaluator"
    exe.write_text(f"#!{sys.executable}\n"
                   "import sys, pathlib, shutil\n"
                   "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                   "out.mkdir(parents=True)\n"
                   f"shutil.copyfile({str(source)!r}, out / 'report.json')\n")
    exe.chmod(0o755)
    args = Namespace(exe=exe, model=model, text=text, corpus=None, quick=False,
                     device=0, depths=[8, 16], tail=2, log_level="warning",
                     no_prefill_a8=False, prefill_cublas=False,
                     no_prefill_cublas_projections=False)
    if expected is None:
        result = run_arm(args, dtype, tmp_path / "arms")
        assert result["dtype"] == dtype
        assert result["execution"]["kv_dtype"] == dtype
        assert result["depths"][16]["coverage"] == [["stream-a", 0, 18, 16, 18, 16]]
    else:
        with pytest.raises(RuntimeError, match=expected):
            run_arm(args, dtype, tmp_path / "arms")
