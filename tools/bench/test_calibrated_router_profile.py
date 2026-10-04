"""Host-only qualification/coverage regressions; fixtures are synthetic measurements."""
import copy
import unittest
import tempfile
import json
import hashlib
from pathlib import Path

from tools.bench.calibrated_router_profile import build_profile, main


def identity():
    return {
        "artifact_id": "0123456789abcdef0123456789abcdef",
        "prefill_signature": "represented", "hardware_class": "RTX4090-sm89",
        "backend": "dflash2", "kv_storage": "int8", "startup_draft_tokens": 15,
        "proposal_head": "full", "use_cuda_graph": True, "max_concurrency": 8,
        "max_context": 32768, "prefill_chunk": 1024, "resolved_kv_capacity": 262144,
        "context_cache": {"enabled": True, "extra_device_state_slots": 8,
                          "host_state_slots": 8, "host_kv_capacity_bytes": 8 << 30,
                          "max_private_continuations": 16, "max_shared_prefixes": 8,
                          "max_long_anchors_per_continuation": 2, "max_cache_markers_per_request": 4},
        "execution_options": {
            **dict.fromkeys(("lm_head_q4", "lm_head_q6", "embedding_q4", "embedding_q6",
                             "gdn_state_fp16", "mlp_a8_decode", "prefill_cublas",
                             "mtp_experts_q4", "enable_vision"), False),
            "prefill_a8": True, "prefill_cublas_projections": True,
            "vision_residency": "resident", "vision_max_merged_tokens": 16384,
            "rope_scaling_factor": 1.0, "rope_scaling_original_context": 262144}}


def record(action=0, wall=10000, latency=6000, batch=2):
    return {
        "identity": identity(), "requested_action": action,
        "configuration": {"prompt": "original prompt", "max_tokens": 8,
                          "client_concurrency": 2, "repeats": 2,
                          "sampling": {"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0}},
        "wave_wall_ns": [wall, wall],
        "requests": [{"repeat": repeat, "slot": slot, "latency_ns": latency,
                      "prompt_tokens": 64, "finish_reason": 1, "prefix_reuse_path": 0,
                      "reused_prompt_tokens": 0, "content": " hello\n", "reasoning": "",
                      "matched_stop_string": None, "tool_calls": [],
                      "generated_token_ids": list(range(8))}
                     for repeat in range(2) for slot in range(2)],
        "rounds": [{"round_index": i, "active_batch": batch, "max_execution_frontier": 64 + i,
                    "draft_tokens": action, "verify_width": action + 1,
                    "proposal_width": 16 if action else 0, "backend": 3,
                    "neural_drafter_executed": bool(action), "committed_tokens": 2,
                    "elapsed_ns": wall // 10 + i} for i in range(14)]}


def comparison(action=7, speed=1.1, workload="chat"):
    pairs = []
    for i, order in enumerate(((0, action), (action, 0))):
        pairs.append({"pair_id": str(i), "order": list(order),
                      "baseline": record(), "candidate": record(action, wall=int(10000 / speed))})
    return {"workload": workload, "candidate_action": action, "pairs": pairs}


def document(*comparisons):
    return {"schema_version": 1, "artifact_type": "ninfer_resident_router_measurements",
            "identity": identity(), "comparisons": list(comparisons)}


def selected(profile, batch=2, upper=1024):
    return next(c["draft_tokens"] for c in profile["cells"]
                if (c["active_batch"], c["frontier_upper"]) == (batch, upper))


class ProfileTest(unittest.TestCase):
    def test_gain_selects_actual_cell_and_emits_full_runtime_schema(self):
        profile = build_profile(document(comparison()))
        self.assertEqual(selected(profile), 7)
        self.assertEqual(len(profile["cells"]), 24)
        self.assertEqual(selected(profile, 1), 0)
        self.assertEqual(profile["identity"], identity())
        self.assertEqual(profile["artifact_type"], "ninfer_spec_router_profile")

    def test_no_coverage_all_cells_zero(self):
        self.assertTrue(all(c["draft_tokens"] == 0 for c in build_profile(document())["cells"]))

    def test_old_resident_and_bad_physical_contract_cancel_candidate(self):
        for resident in (0, 7, 11):
            for arm_name in ("baseline", "candidate"):
                c = comparison(7)
                c["pairs"][0][arm_name]["identity"]["startup_draft_tokens"] = resident
                self.assertEqual(selected(build_profile(document(c))), 0)
        for field, value in (("proposal_width", 8), ("verify_width", 16),
                             ("backend", 0), ("neural_drafter_executed", False)):
            c = comparison()
            c["pairs"][0]["candidate"]["rounds"][0][field] = value
            self.assertEqual(selected(build_profile(document(c))), 0)
        c = comparison()
        c["pairs"][0]["baseline"]["rounds"][0]["backend"] = 0
        self.assertEqual(selected(build_profile(document(c))), 0)

    def test_root_identity_strict_no_guessing(self):
        for mutate in (lambda i: i.pop("artifact_id"), lambda i: i.update(extra=True),
                       lambda i: i["execution_options"].update(extra=False),
                       lambda i: i.update(max_context=True),
                       lambda i: i["execution_options"].update(rope_scaling_factor=float("nan"))):
            d = document()
            mutate(d["identity"])
            with self.assertRaises(ValueError):
                build_profile(d)

    def test_wrong_record_identity_or_full_output_cancel_candidate(self):
        for mutate in (lambda r: r["identity"].update(hardware_class="other"),
                       lambda r: r.pop("identity"),
                       lambda r: r["requests"][0].update(generated_token_ids=[99] * 8),
                       lambda r: r["requests"][0].update(content="hello\n"),
                       lambda r: r["requests"][0].update(reasoning="hidden")):
            c = comparison()
            mutate(c["pairs"][0]["candidate"])
            self.assertEqual(selected(build_profile(document(c))), 0)

    def test_loss_pair_p95_and_median_gate(self):
        for mutate in (lambda c: c["pairs"][0]["candidate"].update(wave_wall_ns=[10001, 10001]),
                       lambda c: c["pairs"][0]["candidate"]["requests"][0].update(latency_ns=6400),
                       lambda c: [p["candidate"].update(wave_wall_ns=[9900, 9900]) for p in c["pairs"]]):
            c = comparison()
            mutate(c)
            self.assertEqual(selected(build_profile(document(c))), 0)

    def test_every_workload_required_for_each_action(self):
        profile = build_profile(document(comparison(7, 1.10), comparison(7, 1.10, "code"),
                                         comparison(15, 1.50)))
        self.assertEqual(selected(profile), 7)

    def test_one_losing_workload_cancels_that_action(self):
        profile = build_profile(document(comparison(7, 1.2), comparison(7, 0.9, "code"),
                                         comparison(11, 1.1), comparison(11, 1.1, "code")))
        self.assertEqual(selected(profile), 11)

    def test_near_tie_selects_smaller_k_by_relative_candidate_throughput(self):
        self.assertEqual(selected(build_profile(document(comparison(7, 1.20), comparison(15, 1.21)))), 7)
        self.assertEqual(selected(build_profile(document(comparison(7, 1.20), comparison(15, 1.23)))), 15)

    def test_bad_candidate_does_not_poison_independent_qualified_action(self):
        bad = comparison(15, 1.5)
        bad["pairs"][0]["candidate"]["rounds"][0]["proposal_width"] = 8
        self.assertEqual(selected(build_profile(document(comparison(7), bad))), 7)

    def test_cross_candidate_baseline_output_instability_blocks_cell(self):
        changed = comparison(15, 1.5)
        for p in changed["pairs"]:
            for arm in ("baseline", "candidate"):
                for request in p[arm]["requests"]:
                    request["generated_token_ids"] = [99] * 8
        self.assertEqual(selected(build_profile(document(comparison(7), changed))), 0)

    def test_pairs_require_both_orders_and_unique_aligned_ids(self):
        for mutate in (lambda c: c["pairs"].pop(),
                       lambda c: c["pairs"][1].update(order=[0, 7]),
                       lambda c: c["pairs"][1].update(pair_id="0")):
            c = comparison()
            mutate(c)
            self.assertEqual(selected(build_profile(document(c))), 0)
        a, b = comparison(7), comparison(7, workload="code")
        b["pairs"][1]["pair_id"] = "different"
        self.assertEqual(selected(build_profile(document(a, b))), 0)

    def test_mixed_batch_with_no_95pct_dominance_is_not_coverage(self):
        c = comparison()
        for p in c["pairs"]:
            for arm in ("baseline", "candidate"):
                rounds = p[arm]["rounds"]
                # Half the round wall time lies in a different frontier cell.
                for event in rounds[7:]:
                    event["max_execution_frontier"] += 1024
        profile = build_profile(document(c))
        self.assertTrue(all(cell["draft_tokens"] == 0 for cell in profile["cells"]))

    def test_minority_join_drain_cell_is_audit_only(self):
        c = comparison()
        for p in c["pairs"]:
            for arm in ("baseline", "candidate"):
                event = p[arm]["rounds"][-1]
                event["max_execution_frontier"] += 1024
                event["elapsed_ns"] = 1
        profile = build_profile(document(c))
        self.assertEqual(selected(profile), 7)
        self.assertEqual(selected(profile, upper=8192), 0)
        self.assertIn("comparisons", profile["provenance"])

    def test_actual_batch_mixture_and_arm_dominance_mismatch_do_not_cover(self):
        c = comparison()
        for pair in c["pairs"]:
            for arm in ("baseline", "candidate"):
                original = pair[arm]["rounds"]
                result = []
                for i, event in enumerate(original):
                    if i < 7:
                        result.append(event)
                    else:
                        for part in range(2):
                            e = copy.deepcopy(event)
                            e.update(active_batch=1, committed_tokens=1,
                                     max_execution_frontier=event["max_execution_frontier"] + part * 100,
                                     elapsed_ns=event["elapsed_ns"] // 2 + (event["elapsed_ns"] % 2 if part else 0))
                            result.append(e)
                for round_index, event in enumerate(result):
                    event["round_index"] = round_index
                pair[arm]["rounds"] = result
        self.assertTrue(all(cell["draft_tokens"] == 0 for cell in build_profile(document(c))["cells"]))
        c = comparison()
        for pair in c["pairs"]:
            for event in pair["candidate"]["rounds"]:
                event["max_execution_frontier"] += 1024
        self.assertTrue(all(cell["draft_tokens"] == 0 for cell in build_profile(document(c))["cells"]))

    def test_weighted_pair_throughput_selection_uses_tokens_over_total_wall(self):
        choices = []
        for action, chat_wall, code_wall in ((7, 8000, 90000), (15, 9000, 85000)):
            for workload, candidate_wall, baseline_wall in (
                    ("chat", chat_wall, 10000), ("code", code_wall, 100000)):
                c = comparison(action, workload=workload)
                for pair in c["pairs"]:
                    pair["baseline"]["wave_wall_ns"] = [baseline_wall] * 2
                    pair["candidate"]["wave_wall_ns"] = [candidate_wall] * 2
                    for arm in ("baseline", "candidate"):
                        pair[arm]["configuration"]["prompt"] = workload + " original prompt"
                choices.append(c)
        profile = build_profile(document(*choices))
        self.assertEqual(selected(profile), 15)
        action7 = next(c for c in profile["provenance"]["selection"][0]["candidates"]
                       if c["draft_tokens"] == 7)
        pair = action7["paired_merged_throughput"][0]
        self.assertEqual(pair["candidate"]["tokens"], 64)
        self.assertEqual(pair["candidate"]["wave_wall_ns"], 196000)
        self.assertAlmostEqual(pair["candidate"]["throughput"], 64 * 1e9 / 196000)

    def test_95_percent_boundary_is_inclusive_and_minor_cells_are_recorded(self):
        for main_time, minor_time, expected in ((9500, 500, 7), (9499, 501, 0)):
            c = comparison()
            for pair in c["pairs"]:
                for arm in ("baseline", "candidate"):
                    rounds = pair[arm]["rounds"]
                    for i, event in enumerate(rounds[:-1]):
                        event["elapsed_ns"] = main_time // 13 + (main_time % 13 if i == 0 else 0)
                    rounds[-1].update(max_execution_frontier=1500, elapsed_ns=minor_time)
            profile = build_profile(document(c))
            self.assertEqual(selected(profile), expected)
            self.assertEqual(selected(profile, upper=8192), 0)
            self.assertEqual(len(profile["provenance"]["comparisons"][0]["pairs"][0]["baseline"]["cells"]), 2)

    def test_workload_order_patterns_and_configuration_mismatch_are_not_hidden(self):
        a, b = comparison(7), comparison(7, workload="code")
        for pair in b["pairs"]:
            pair["order"].reverse()
        self.assertEqual(selected(build_profile(document(a, b))), 0)
        a, b = comparison(7), comparison(15)
        for pair in b["pairs"]:
            for arm in ("baseline", "candidate"):
                pair[arm]["configuration"]["prompt"] = "changed original prompt"
        self.assertEqual(selected(build_profile(document(a, b))), 0)

    def test_full_round_final_stats_can_outlast_waiter_wave_wall(self):
        c = comparison()
        for pair in c["pairs"]:
            for arm in ("baseline", "candidate"):
                r = pair[arm]
                wall = sum(r["wave_wall_ns"])
                for i, event in enumerate(r["rounds"]):
                    event["elapsed_ns"] = wall // 14 + (wall % 14 + 1 if i == 0 else 0)
        profile = build_profile(document(c))
        self.assertEqual(selected(profile), 7)
        audit = profile["provenance"]["comparisons"][0]["pairs"][0]["baseline"]
        self.assertEqual(audit["full_round_elapsed_ns"], audit["wave_wall_ns"] + 1)

    def test_zero_commit_finish_round_and_identical_payload_different_indices(self):
        c = comparison()
        for pair in c["pairs"]:
            for arm in ("baseline", "candidate"):
                r = pair[arm]
                finish = copy.deepcopy(r["rounds"][-1])
                finish.update(round_index=len(r["rounds"]), committed_tokens=0)
                r["rounds"].append(finish)
                duplicate_payload = copy.deepcopy(finish)
                duplicate_payload["round_index"] += 1
                r["rounds"].append(duplicate_payload)
        profile = build_profile(document(c))
        self.assertEqual(selected(profile), 7)
        audit = profile["provenance"]["comparisons"][0]["pairs"][0]["baseline"]
        self.assertEqual(audit["cells"][0]["rounds"], 16)
        self.assertEqual(audit["cells"][0]["committed_tokens"], 28)

    def test_identical_nonzero_metadata_with_different_indices_is_valid(self):
        c = comparison()
        for pair in c["pairs"]:
            for arm in ("baseline", "candidate"):
                r = pair[arm]
                r["rounds"][1] = copy.deepcopy(r["rounds"][0])
                r["rounds"][1]["round_index"] = 1
        self.assertEqual(selected(build_profile(document(c))), 7)

    def test_whole_run_still_requires_committed_tokens(self):
        c = comparison()
        for event in c["pairs"][0]["candidate"]["rounds"]:
            event["committed_tokens"] = 0
        self.assertEqual(selected(build_profile(document(c))), 0)

    def test_round_indices_must_be_unique_continuous_and_in_record_order(self):
        for mutate in (
                lambda r: r["rounds"][1].update(round_index=0),
                lambda r: r["rounds"][1].update(round_index=2),
                lambda r: r["rounds"].reverse(),
                lambda r: r["rounds"][0].pop("round_index"),
                lambda r: r["rounds"][0].update(round_index=True)):
            c = comparison()
            mutate(c["pairs"][0]["candidate"])
            self.assertEqual(selected(build_profile(document(c))), 0)

    def test_cli_audit_hash_and_no_overwrite_or_ambiguous_json(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "measurement.json", Path(temp) / "profile.json"
            raw = json.dumps(document(), indent=2).encode()
            source.write_bytes(raw)
            self.assertEqual(main(["--input", str(source), "--output", str(output)]), 0)
            profile = json.loads(output.read_text())
            self.assertEqual(profile["provenance"]["input"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(profile["provenance"]["input"]["path"], str(source.resolve()))
            self.assertTrue(all(cell["draft_tokens"] == 0 for cell in profile["cells"]))
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                main(["--input", str(source), "--output", str(output)])
            self.assertEqual(output.read_bytes(), saved)
            source.write_text('{"identity": {}, "identity": {}}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                main(["--input", str(source), "--output", str(output)])
            source.write_text('{"schema_version": NaN}')
            with self.assertRaisesRegex(ValueError, "nonfinite JSON"):
                main(["--input", str(source), "--output", str(output)])

    def test_duplicate_requests_events_bool_or_nonfinite_time_rejected(self):
        for mutate in (
                lambda r: r["requests"].append(copy.deepcopy(r["requests"][0])),
                lambda r: r["rounds"].append(copy.deepcopy(r["rounds"][0])),
                lambda r: r["rounds"][0].update(active_batch=True),
                lambda r: r["requests"][0].update(latency_ns=float("inf")),
                lambda r: r.update(wave_wall_ns=[0, 10000])):
            c = comparison()
            mutate(c["pairs"][0]["candidate"])
            self.assertEqual(selected(build_profile(document(c))), 0)


if __name__ == "__main__":
    unittest.main()
