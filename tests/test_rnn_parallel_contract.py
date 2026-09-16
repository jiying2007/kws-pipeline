#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def main() -> None:
    cfg = load("configs/training/xiaowo.torch-domain.json")
    rnn = load("configs/training/xiaowo.rnn-development-loop.json")
    gru = load("configs/training/xiaowo.gru-development-loop.json")
    fresh = load("experiments/model_family/rnn_fresh_validation_registry.json")
    shadow = load("experiments/model_family/shadow_arena_registry.json")

    assert rnn["policy"] == "rnn-development-curriculum-loop-v1"
    assert rnn["model_family"] == "rnn"
    assert rnn["evidence_scope"] == "development-only"
    assert rnn["qualification_used"] is False
    assert rnn["shadow_used"] is False
    assert rnn["formal_qualification_used"] is False
    assert rnn["stable_strict_pass_rounds"] >= 2

    freeze = rnn["candidate_freeze"]
    assert freeze["validation_feedback_allowed"] is False
    assert freeze["threshold_feedback_allowed"] is False
    assert freeze["training_rule_feedback_allowed"] is False
    assert freeze["fresh_validation_required"] is True
    assert freeze["shadow_required"] is True
    assert freeze["formal_qualification_required"] is True

    fresh_ns = int(freeze["fresh_validation_seed_namespace"])
    matches = [row for row in fresh["namespaces"] if int(row["namespace"]) == fresh_ns]
    assert len(matches) == 1
    assert matches[0]["model_family"] == "rnn"
    assert matches[0]["status"] == "reserved-untouched"

    arenas = {row["name"]: row for row in shadow["arenas"]}
    arena = arenas[freeze["shadow_arena"]]
    assert arena["model_family"] == "rnn"
    assert arena["status"] == "reserved-untouched"
    shadow_seeds = {int(value) for value in arena["seeds"]}
    assert len(shadow_seeds) == 8

    formal_seed = int(freeze["formal_qualification_seed"])
    assert formal_seed == 271844
    assert formal_seed != int(cfg["qualification_holdout_seed"])
    assert formal_seed not in {int(v) for v in cfg.get("retired_qualification_holdout_seeds", [])}
    assert formal_seed not in shadow_seeds

    # RNN and GRU development/validation namespaces must be independently owned.
    rnn_namespaces = {
        int(rnn["training_seed_namespace"]),
        int(rnn["training_acoustic_seed_namespace"]),
        fresh_ns,
    }
    gru_namespaces = {
        int(gru["training_seed_namespace"]),
        int(gru["training_acoustic_seed_namespace"]),
        int(gru["candidate_freeze"]["fresh_validation_seed_namespace"]),
    }
    assert not (rnn_namespaces & gru_namespaces)
    assert len(rnn_namespaces) == 3

    model_text = (ROOT / "training/model.py").read_text(encoding="utf-8")
    iterator = (ROOT / "training/iterate_rnn_development.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "training/run_rnn_development.py").read_text(encoding="utf-8")
    gate = (ROOT / "tools/rnn_development_gate.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/rnn-development-curriculum.yml").read_text(encoding="utf-8")

    assert "class TinyStreamingRNN" in model_text
    assert "train_ctc.py" in iterator
    assert "export_model.py" in iterator
    assert 'candidate / "model.kwm"' in iterator
    assert "train_gru_ctc.py" not in iterator
    assert "export_gru_model.py" not in iterator
    assert "iterate_gru_development" not in iterator
    assert "evaluate_development_split" in iterator
    assert "terminal_strict_streak" in iterator
    assert "observed >= required" in iterator
    assert "rnn-development-full-robustness-gate-v1" in gate
    assert "rnn-development-negative-stress-support-v1" in wrapper

    assert "if: github.event_name == 'workflow_dispatch'" in workflow
    assert "--runner build/kws_wav" in workflow
    assert "kws_wav_gru" not in workflow
    assert "training/run_rnn_development.py" in workflow
    assert "tools/reconcile_rnn_development_gate.py" in workflow
    assert "tools/verify_rnn_development_gate.py" in workflow
    assert "rnn-development-curriculum" in workflow

    print("RNN parallel governed development contract: PASS")


if __name__ == "__main__":
    main()
