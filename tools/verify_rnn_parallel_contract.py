#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RNN_POLICY = ROOT / "configs/training/xiaowo.rnn-development-loop.json"
GRU_POLICY = ROOT / "configs/training/xiaowo.gru-development-loop.json"
CONFIG = ROOT / "configs/training/xiaowo.torch-domain.json"
RNN_FRESH = ROOT / "experiments/model_family/rnn_fresh_validation_registry.json"
GRU_FRESH = ROOT / "experiments/model_family/fresh_validation_registry.json"
SHADOW = ROOT / "experiments/model_family/shadow_arena_registry.json"


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    rnn = load(RNN_POLICY); gru = load(GRU_POLICY); cfg = load(CONFIG)
    rnn_fresh = load(RNN_FRESH); gru_fresh = load(GRU_FRESH); shadow = load(SHADOW)
    if rnn.get("policy") != "rnn-development-curriculum-loop-v1" or rnn.get("model_family") != "rnn":
        raise ValueError("RNN development policy identity mismatch")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if rnn.get(field) is not False:
            raise ValueError(f"RNN development requires {field}=false")
    freeze = rnn.get("candidate_freeze")
    if not isinstance(freeze, dict):
        raise ValueError("RNN candidate freeze contract missing")
    for field in ("fresh_validation_required", "shadow_required", "formal_qualification_required", "bounded_repair_only_after_freeze"):
        if freeze.get(field) is not True:
            raise ValueError(f"RNN candidate freeze requires {field}=true")
    for field in ("validation_feedback_allowed", "threshold_feedback_allowed", "training_rule_feedback_allowed"):
        if freeze.get(field) is not False:
            raise ValueError(f"RNN candidate freeze requires {field}=false")

    for field in ("max_rounds", "min_rounds", "patience", "stable_strict_pass_rounds", "epochs_per_round", "feature_cache_max_items", "lr_decay_per_round", "fixed_replay_repeat", "failure_replay_repeat_max"):
        if rnn.get(field) != gru.get(field):
            raise ValueError(f"RNN/GRU development policy drifted at {field}")
    if rnn.get("loss_controller") != gru.get("loss_controller"):
        raise ValueError("RNN/GRU loss-controller contract drifted")
    if rnn.get("failure_replay_latch_after_failure") is not True:
        raise ValueError("RNN failure replay latch policy is missing")

    spec = importlib.util.spec_from_file_location(
        "rnn_iterator_contract", ROOT / "training/iterate_rnn_development.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load RNN iterator for controller contract")
    iterator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iterator)
    initial = iterator.controller_initial(rnn)
    clean_initial = iterator.controller_next(
        rnn, initial, 0, 0, frr=0.0, far_per_hour=0.0
    )
    after_failure = iterator.controller_next(
        rnn, initial, 1, 0, frr=0.20, far_per_hour=0.0
    )
    clean_after_failure = iterator.controller_next(
        rnn, after_failure, 0, 0, frr=0.0, far_per_hour=0.0
    )
    if int(clean_initial["failure_replay_repeat"]) != 0:
        raise ValueError("RNN clean initial controller unexpectedly enables failure replay")
    if int(after_failure["failure_replay_repeat"]) != 1:
        raise ValueError("RNN controller did not enable replay after a development failure")
    if int(clean_after_failure["failure_replay_repeat"]) != 1:
        raise ValueError("RNN controller dropped replay before stability confirmation")
    if freeze.get("selection_policy") != gru.get("candidate_freeze", {}).get("selection_policy"):
        raise ValueError("RNN/GRU selection policy drifted")

    base_seed = int(cfg.get("seed", 1337))
    gru_formal = int(cfg["qualification_holdout_seed"])
    retired_formal = {int(value) for value in cfg.get("retired_qualification_holdout_seeds", [])}
    formal_evidence = {gru_formal, *retired_formal}
    rnn_formal = int(freeze.get("formal_qualification_seed", -1))
    if rnn_formal != 271844 or rnn_formal in formal_evidence:
        raise ValueError("RNN formal seed is not independently reserved")

    rnn_fresh_ns = int(freeze["fresh_validation_seed_namespace"])
    rnn_fresh_seed = base_seed + rnn_fresh_ns
    rnn_rows = [row for row in rnn_fresh.get("namespaces", []) if int(row.get("namespace", -1)) == rnn_fresh_ns]
    if len(rnn_rows) != 1 or rnn_rows[0].get("model_family") != "rnn" or rnn_rows[0].get("status") != "reserved-untouched":
        raise ValueError("RNN Fresh namespace is not uniquely reserved and untouched")
    gru_namespaces = {int(row["namespace"]) for row in gru_fresh.get("namespaces", [])}
    if rnn_fresh_ns in gru_namespaces or rnn_fresh_seed in formal_evidence or rnn_fresh_seed == rnn_formal:
        raise ValueError("RNN Fresh namespace overlaps GRU/formal evidence")

    arenas = {str(row["name"]): row for row in shadow.get("arenas", [])}
    arena_name = str(freeze["shadow_arena"])
    arena = arenas.get(arena_name)
    if not isinstance(arena, dict) or arena.get("model_family") != "rnn" or arena.get("status") != "reserved-untouched":
        raise ValueError("RNN shadow arena is not reserved and untouched")
    rnn_shadow = {int(value) for value in arena.get("seeds", [])}
    if len(rnn_shadow) != 8 or len(rnn_shadow) != len(arena.get("seeds", [])):
        raise ValueError("RNN shadow arena must contain exactly 8 unique seeds")
    gru_shadow = {int(value) for row in shadow.get("arenas", []) if row.get("model_family") == "gru" for value in row.get("seeds", [])}
    if rnn_shadow & gru_shadow or rnn_shadow & formal_evidence or rnn_formal in rnn_shadow or rnn_fresh_seed in rnn_shadow:
        raise ValueError("RNN protected evidence overlaps GRU/formal/Fresh evidence")

    model_ns = int(rnn["training_seed_namespace"])
    acoustic_ns = int(rnn["training_acoustic_seed_namespace"])
    acoustic_stride = int(rnn["training_acoustic_seed_stride"])
    gru_policy_ns = {int(gru["training_seed_namespace"]), int(gru["training_acoustic_seed_namespace"]), int(gru["candidate_freeze"]["fresh_validation_seed_namespace"])}
    if min(model_ns, acoustic_ns, acoustic_stride) <= 0 or len({model_ns, acoustic_ns, rnn_fresh_ns}) != 3:
        raise ValueError("RNN development namespaces are invalid")
    if {model_ns, acoustic_ns, rnn_fresh_ns} & gru_policy_ns:
        raise ValueError("RNN development namespaces overlap GRU namespaces")
    for round_index in range(int(rnn["max_rounds"])):
        acoustic_seed = base_seed + acoustic_ns + round_index * acoustic_stride
        if acoustic_seed in formal_evidence or acoustic_seed in rnn_shadow or acoustic_seed in {rnn_fresh_seed, rnn_formal}:
            raise ValueError("RNN acoustic seed overlaps protected evidence")

    required_files = (
        "training/model.py", "training/train_ctc.py", "training/feature_cached_trainer.py", "training/export_model.py",
        "training/iterate_rnn_development.py", "training/run_rnn_development.py",
        "tools/rnn_development_gate.py", "tools/reconcile_rnn_development_gate.py",
        "tools/verify_rnn_development_gate.py", "tools/finalize_rnn_frozen_candidate.py",
        "tools/verify_rnn_frozen_candidate.py", "tools/verify_rnn_stability_evidence.py",
        "training/validate_frozen_rnn_candidate.py", "training/qualify_frozen_rnn_formal.py",
        "tools/prepare_rnn_shadow_config.py", "tools/materialize_rnn_candidate_workspace.py",
        "tools/verify_rnn_evaluation_independence.py",
    )
    for name in required_files:
        if not (ROOT / name).is_file():
            raise ValueError(f"RNN governed pipeline member missing: {name}")
    model_source = (ROOT / "training/model.py").read_text(encoding="utf-8")
    trainer_source = (ROOT / "training/train_ctc.py").read_text(encoding="utf-8")
    exporter_source = (ROOT / "training/export_model.py").read_text(encoding="utf-8")
    iterator_source = (ROOT / "training/iterate_rnn_development.py").read_text(encoding="utf-8")
    wrapper_source = (ROOT / "training/run_rnn_development.py").read_text(encoding="utf-8")
    workflow_source = (ROOT / ".github/workflows/rnn-development-curriculum.yml").read_text(encoding="utf-8")
    qualification_workflow = (ROOT / ".github/workflows/rnn-frozen-candidate-qualification.yml").read_text(encoding="utf-8")
    if "class TinyStreamingRNN" not in model_source or "TinyStreamingRNN(" not in trainer_source:
        raise ValueError("vanilla RNN trainer/model binding is missing")
    if 'b"KWSP"' not in exporter_source:
        raise ValueError("RNN shipping model exporter ABI binding is missing")
    if "train_ctc.py" not in iterator_source or "export_model.py" not in iterator_source or "train_gru_ctc.py" in iterator_source:
        raise ValueError("RNN iterator is not bound exclusively to the vanilla RNN trainer/exporter")
    if "evaluate_development_split" not in iterator_source or "terminal_strict_streak" not in iterator_source:
        raise ValueError("RNN iterator does not enforce full robustness/stability at source")
    if "shared.loop = loop" not in wrapper_source or "shared.install_rotation(policy_path)" not in wrapper_source or "install_development_feature_cache" not in wrapper_source:
        raise ValueError("RNN wrapper does not bind shared acoustic/stress hooks to the RNN iterator")
    if "if: github.event_name == 'workflow_dispatch'" not in workflow_source or "--runner build/kws_wav" not in workflow_source or "kws_wav_gru" in workflow_source:
        raise ValueError("RNN workflow trigger/runtime boundary is invalid")
    if "if: github.event_name == 'workflow_dispatch'" not in qualification_workflow or "--runner build/kws_wav" not in qualification_workflow or "kws_wav_gru" in qualification_workflow:
        raise ValueError("RNN qualification workflow trigger/runtime boundary is invalid")

    print(json.dumps({"verified": True, "model_family": "rnn", "development_policy": rnn["policy"], "fresh_namespace": rnn_fresh_ns, "shadow_arena": arena_name, "formal_seed": rnn_formal, "gru_formal_seed": gru_formal, "parallel_with_gru": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
