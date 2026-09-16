#!/usr/bin/env python3
from __future__ import annotations

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

    # Keep model-family A/B governance comparable. Only seed namespaces and model family may differ.
    for field in ("max_rounds", "min_rounds", "patience", "stable_strict_pass_rounds", "epochs_per_round", "lr_decay_per_round", "fixed_replay_repeat", "failure_replay_repeat_max"):
        if rnn.get(field) != gru.get(field):
            raise ValueError(f"RNN/GRU development policy drifted at {field}")
    if rnn.get("loss_controller") != gru.get("loss_controller"):
        raise ValueError("RNN/GRU loss-controller contract drifted")
    if freeze.get("selection_policy") != gru.get("candidate_freeze", {}).get("selection_policy"):
        raise ValueError("RNN/GRU selection policy drifted")

    base_seed = int(cfg.get("seed", 1337))
    formal = {int(cfg["qualification_holdout_seed"])}
    formal.update(int(value) for value in cfg.get("retired_qualification_holdout_seeds", []))
    rnn_fresh_ns = int(freeze["fresh_validation_seed_namespace"])
    rnn_fresh_seed = base_seed + rnn_fresh_ns
    rnn_rows = [row for row in rnn_fresh.get("namespaces", []) if int(row.get("namespace", -1)) == rnn_fresh_ns]
    if len(rnn_rows) != 1 or rnn_rows[0].get("model_family") != "rnn" or rnn_rows[0].get("status") != "reserved-untouched":
        raise ValueError("RNN Fresh namespace is not uniquely reserved and untouched")

    gru_reserved_ns = {int(row["namespace"]) for row in gru_fresh.get("namespaces", []) if row.get("status") == "reserved-untouched"}
    if rnn_fresh_ns in gru_reserved_ns or rnn_fresh_seed in formal:
        raise ValueError("RNN Fresh namespace overlaps GRU/formal evidence")

    arenas = {str(row["name"]): row for row in shadow.get("arenas", [])}
    arena_name = str(freeze["shadow_arena"])
    arena = arenas.get(arena_name)
    if not isinstance(arena, dict) or arena.get("model_family") != "rnn" or arena.get("status") != "reserved-untouched":
        raise ValueError("RNN shadow arena is not reserved and untouched")
    rnn_shadow = {int(value) for value in arena.get("seeds", [])}
    if len(rnn_shadow) != 8:
        raise ValueError("RNN shadow arena must contain exactly 8 unique seeds")
    if len(rnn_shadow) != len(arena.get("seeds", [])):
        raise ValueError("RNN shadow arena contains duplicate seeds")
    gru_shadow = {
        int(value)
        for row in shadow.get("arenas", [])
        if row.get("model_family") == "gru" and row.get("status") == "reserved-untouched"
        for value in row.get("seeds", [])
    }
    if rnn_shadow & gru_shadow or rnn_shadow & formal or rnn_fresh_seed in rnn_shadow:
        raise ValueError("RNN protected evidence overlaps GRU/formal/Fresh evidence")

    model_ns = int(rnn["training_seed_namespace"])
    acoustic_ns = int(rnn["training_acoustic_seed_namespace"])
    acoustic_stride = int(rnn["training_acoustic_seed_stride"])
    if min(model_ns, acoustic_ns, acoustic_stride) <= 0:
        raise ValueError("RNN development namespaces must be positive")
    if len({model_ns, acoustic_ns, rnn_fresh_ns}) != 3:
        raise ValueError("RNN model/acoustic/Fresh namespaces must be distinct")
    for round_index in range(int(rnn["max_rounds"])):
        acoustic_seed = base_seed + acoustic_ns + round_index * acoustic_stride
        if acoustic_seed in formal or acoustic_seed in rnn_shadow or acoustic_seed == rnn_fresh_seed:
            raise ValueError("RNN acoustic seed overlaps protected evidence")

    required_files = (
        "training/model.py",
        "training/train_ctc.py",
        "training/export_model.py",
        "training/iterate_rnn_development.py",
        "training/run_rnn_development.py",
        "tools/rnn_development_gate.py",
        "tools/reconcile_rnn_development_gate.py",
        "tools/verify_rnn_development_gate.py",
        "tools/finalize_rnn_frozen_candidate.py",
        "tools/verify_rnn_frozen_candidate.py",
        "tools/verify_rnn_stability_evidence.py",
    )
    for name in required_files:
        if not (ROOT / name).is_file():
            raise ValueError(f"RNN governed pipeline member missing: {name}")
    model_source = (ROOT / "training/model.py").read_text(encoding="utf-8")
    trainer_source = (ROOT / "training/train_ctc.py").read_text(encoding="utf-8")
    exporter_source = (ROOT / "training/export_model.py").read_text(encoding="utf-8")
    if "class TinyStreamingRNN" not in model_source or "TinyStreamingRNN(" not in trainer_source:
        raise ValueError("vanilla RNN trainer/model binding is missing")
    if 'b"KWSP"' not in exporter_source:
        raise ValueError("RNN shipping model exporter ABI binding is missing")

    print(json.dumps({
        "verified": True,
        "model_family": "rnn",
        "development_policy": rnn["policy"],
        "fresh_namespace": rnn_fresh_ns,
        "shadow_arena": arena_name,
        "formal_seed_untouched": int(cfg["qualification_holdout_seed"]),
        "parallel_with_gru": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
