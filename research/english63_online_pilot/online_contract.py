"""Dependency-free bounds for one explicitly weak, non-qualified experiment."""
from __future__ import annotations

from collections import deque
import math
import re

POLICY = "english63-bounded-online-weak-mil-v1"
STATUS = "EXECUTED_WEAK_MIL_ONLINE_PROPOSALS_NOT_KWS_QUALIFICATION"
WINDOW = 200
MIN_FRAMES = 50
STEPS = 48
SEED = 20261114
LEARNING_RATE = 0.001
MIL_TEMPERATURE = 0.25
THRESHOLD = 0.80
CONFIRM_FRAMES = 3
COOLDOWN_FRAMES = 100
FRAME = 400
HOP = 160
RATE = 16000
MAX_CHUNK = 3199
KINDS = ("gru", "dsc_tcn")
TRAIN_ACTION = "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY"
DEV_ACTION = "EXPOSED_DEVELOPMENT_NO_TUNE"
QUARANTINE_ACTION = "QUARANTINE_AMBIGUOUS_NEARPHONE"
COUNTS = {TRAIN_ACTION: 30, DEV_ACTION: 12, QUARANTINE_ACTION: 21}
TRAIN_FAMILIES = {"qwen3-preset-" + name for name in
                  ("dylan", "eric", "ono_anna", "sohee", "vivian")}
DEV_FAMILIES = {"qwen3-preset-serena", "qwen3-preset-uncle_fu"}
CELL = re.compile(r"en-[0-9a-f]{24}\Z")
UNMEASURED = ("independent_english_frr", "continuous_negative_far_per_hour",
              "word_end_latency", "ssc305_board_profile")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fixed_config():
    return {"window_frames": WINDOW, "minimum_observed_frames": MIN_FRAMES,
            "steps_per_model": STEPS, "base_seed": SEED,
            "learning_rate": LEARNING_RATE, "weight_decay": 0.01,
            "gradient_clip": 1.0, "mil_temperature": MIL_TEMPERATURE,
            "proposal_threshold": THRESHOLD,
            "proposal_confirm_frames": CONFIRM_FRAMES,
            "proposal_cooldown_frames": COOLDOWN_FRAMES,
            "frame_samples": FRAME, "hop_samples": HOP,
            "sample_rate": RATE, "max_pcm_chunk": MAX_CHUNK,
            "readout": "causal-trailing-200-frame-mean-max",
            "loss": "weak-clip-smooth-max-target-background-margin-CE",
            "positive_frame_labels_created": False,
            "warm_start": "unmodified-english63-weak-pilot-checkpoints",
            "interpretation": "system-candidate-not-single-variable-causal-proof"}


def select_training_rows(rows):
    """Accept the saved split, never promote exposed or nearphone rows."""
    require(type(rows) is list and len(rows) == 63, "ONLINE_ROW_COUNT")
    counts = dict.fromkeys(COUNTS, 0)
    seen, families, output = set(), {}, []
    for row in rows:
        require(type(row) is dict, "ONLINE_ROW_TYPE")
        cell, action = row.get("cell_id"), row.get("action")
        family, role = row.get("source_family_id"), row.get("role")
        label = row.get("weak_wholeclip_class")
        require(type(cell) is str and CELL.fullmatch(cell) and cell not in seen
                and action in COUNTS and family in TRAIN_FAMILIES | DEV_FAMILIES
                and role == ("train-candidate" if family in TRAIN_FAMILIES
                             else "development-only")
                and row.get("human_gold") is False
                and row.get("training_admitted") is False
                and row.get("trusted_negative") is False
                and row.get("ctc_target_ids") is None,
                "ONLINE_ROW_ROLE_OR_AUTHORITY")
        intent = row.get("intent_requested")
        if action == QUARANTINE_ACTION:
            require(intent == "near_miss" and label is None,
                    "ONLINE_QUARANTINE_LABEL")
        else:
            require(type(label) is int and label in (0, 1, 2)
                    and intent == ("ordinary_negative" if label == 0 else "wake")
                    and action == (TRAIN_ACTION if family in TRAIN_FAMILIES
                                   else DEV_ACTION), "ONLINE_LABEL_ROLE")
        seen.add(cell)
        counts[action] += 1
        families.setdefault(family, set()).add(role)
        if action == TRAIN_ACTION:
            output.append(row)
    require(counts == COUNTS and len(families) == 7
            and all(len(value) == 1 for value in families.values())
            and len(output) == 30
            and {r["weak_wholeclip_class"] for r in output} == {0, 1, 2},
            "ONLINE_SPLIT_COUNTS")
    return sorted(output, key=lambda row: row["cell_id"])


class BoundedFrames:
    """No 12-second cap: fixed PCM carry, actual-sample clock, no EOF padding."""
    def __init__(self):
        self.tail = []
        self.samples = 0
        self.frames = 0
        self.maximum_buffer_samples = 0

    def push(self, samples):
        require(0 < len(samples) <= MAX_CHUNK, "ONLINE_PCM_CHUNK")
        values = [float(value) for value in samples]
        require(all(math.isfinite(value) and -1.0 <= value <= 1.0
                    for value in values), "ONLINE_PCM_NONFINITE")
        self.samples += len(values)
        self.tail.extend(values)
        self.maximum_buffer_samples = max(self.maximum_buffer_samples,
                                          len(self.tail))
        output = []
        while len(self.tail) >= FRAME:
            self.frames += 1
            output.append((self.frames, self.tail[:FRAME]))
            del self.tail[:HOP]
        require(len(self.tail) < FRAME, "ONLINE_PCM_CARRY_BOUND")
        return output

    def finish(self):
        expected = 0 if self.samples < FRAME else 1 + (self.samples-FRAME)//HOP
        require(self.frames == expected, "ONLINE_PCM_CLOCK")
        return self.frames  # Deliberately emits no frames and mutates no state.


class RollingPool:
    """Stdlib reference used by source tests, not the Torch numeric backend."""
    def __init__(self, width):
        require(type(width) is int and 1 <= width <= 32, "ONLINE_POOL_WIDTH")
        self.width = width
        self.rows = deque(maxlen=WINDOW)

    def push(self, row):
        require(len(row) == self.width and all(math.isfinite(float(x)) for x in row),
                "ONLINE_POOL_VALUES")
        self.rows.append(tuple(float(value) for value in row))
        return ([sum(v[i] for v in self.rows)/len(self.rows) for i in range(self.width)]
                + [max(v[i] for v in self.rows) for i in range(self.width)])


class ProposalDecoder:
    """Fixed research proposal policy; not acoustic labels or calibrated KWS."""
    def __init__(self):
        self.frame = 0
        self.candidate = 0
        self.confirmations = 0
        self.blocked_until = 0

    def consume(self, probabilities):
        require(len(probabilities) == 3
                and all(math.isfinite(float(p)) and 0 <= float(p) <= 1
                        for p in probabilities)
                and abs(sum(float(p) for p in probabilities)-1) < 1e-5,
                "ONLINE_PROBABILITIES")
        self.frame += 1
        selected = max(range(3), key=lambda i: float(probabilities[i]))
        qualifies = (self.frame >= MIN_FRAMES and self.frame >= self.blocked_until
                     and selected in (1, 2) and probabilities[selected] >= THRESHOLD)
        if not qualifies:
            self.candidate, self.confirmations = 0, 0
            return None
        self.confirmations = self.confirmations + 1 if selected == self.candidate else 1
        self.candidate = selected
        if self.confirmations < CONFIRM_FRAMES:
            return None
        self.blocked_until = self.frame + COOLDOWN_FRAMES
        self.candidate, self.confirmations = 0, 0
        return {"proposal_class": selected, "frame_index": self.frame,
                "available_after_sample": FRAME + (self.frame-1)*HOP,
                "probability": round(float(probabilities[selected]), 8),
                "research_proposal_only": True, "acoustic_truth": False}


def validate_summary(value):
    require(type(value) is dict and value.get("schema") == POLICY
            and value.get("status") == STATUS and value.get("config") == fixed_config()
            and value.get("source_audio_clips") == 63
            and value.get("role_counts") == COUNTS
            and value.get("source_pcm_samples") == 2101760
            and value.get("formal_training_admitted") is False
            and value.get("human_gold") is False
            and value.get("model_promotion_allowed") is False
            and value.get("model_selection_made") is False
            and value.get("new_tts_asr_ipa_calls") == 0
            and value.get("threshold_selection") is False
            and all(value.get(key) == "NOT_RUN" for key in UNMEASURED),
            "ONLINE_SUMMARY_AUTHORITY")
    models = value.get("models")
    require(type(models) is list and len(models) == 2
            and {m.get("kind") for m in models} == set(KINDS), "ONLINE_MODEL_ROSTER")
    for model in models:
        require(model.get("optimizer_updates") == STEPS
                and model.get("train_rows") == 30
                and model.get("train_nearphone_rows") == 0
                and model.get("train_development_rows") == 0
                and model.get("weight_file") == model["kind"] + "-weights.npz"
                and model.get("weights_sha256") != model.get("warm_start_sha256")
                and re.fullmatch(r"[0-9a-f]{64}", model.get("weights_sha256", ""))
                and re.fullmatch(r"[0-9a-f]{64}", model.get("warm_start_sha256", ""))
                and model.get("model_promotion_allowed") is False,
                "ONLINE_MODEL_AUTHORITY")
    return True
