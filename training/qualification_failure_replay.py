from __future__ import annotations

import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kws_vocab import load_tokens  # noqa: E402

from acoustic_scene import render_scene, sha256_file  # noqa: E402
from development_failure_replay import _jitter_scene  # noqa: E402
from render_domains import validate_domains  # noqa: E402
from synthetic_audio import (  # noqa: E402
    augment,
    generate_background,
    load_config,
    parse_keywords,
    render_command_tts,
    render_tone_tokens,
    token_carriers,
    validate_augment_config,
    validate_tone_config,
    write_wav,
)

POLICY = "development-qualification-repair-v1"
FAILURE_REPLAY_POLICY = "development-failure-resynthesis-v1"
RECORDING_RE = re.compile(r"^domain-qualification-(\d{6})$")
SEED_NAMESPACE = 161_000_003
MAX_UNIQUE_FAILURES = 8
EXAMPLES_PER_FAILURE = 8
REPAIR_EPOCHS = 6
REPAIR_LR_SCALE = 0.25


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def _qualification_rows(dataset: pathlib.Path) -> list[dict]:
    path = dataset / "domain-index.jsonl"
    rows = [row for row in _load_jsonl(path) if str(row.get("split")) == "qualification"]
    if not rows:
        raise ValueError("qualification repair cannot resolve development qualification rows")
    return rows


def _collect_specs(dataset: pathlib.Path, eval_dir: pathlib.Path) -> list[dict]:
    rows = _qualification_rows(dataset)
    aggregated: dict[str, dict] = {}
    for kind, path in (
        ("false-accept", eval_dir / "false-positives.jsonl"),
        ("false-reject", eval_dir / "false-rejects.jsonl"),
    ):
        for failure in _load_jsonl(path):
            match = RECORDING_RE.match(str(failure.get("recording", "")))
            if match is None:
                raise ValueError("qualification repair saw an unexpected recording id")
            index = int(match.group(1))
            if index >= len(rows):
                raise ValueError("qualification repair recording index exceeds domain-index split")
            source = rows[index]
            source_sha = str(source.get("wav_sha256") or "")
            if len(source_sha) != 64:
                raise ValueError("qualification repair source is missing development WAV SHA")
            scene = source.get("scene")
            if not isinstance(scene, dict):
                raise ValueError("qualification repair source is missing scene metadata")
            source_keyword = source.get("keyword_id")
            focus_keyword = int(
                failure.get("keyword_id")
                if failure.get("keyword_id") is not None
                else source_keyword or 0
            )
            item = aggregated.setdefault(
                source_sha,
                {
                    "source_wav_sha256": source_sha,
                    "source_base_wav_sha256": str(source.get("source_wav_sha256") or ""),
                    "source_kind": str(source.get("kind") or ""),
                    "tokens": [str(value) for value in source.get("tokens", [])],
                    "target_ids": [int(value) for value in source.get("target_ids", [])],
                    "source_keyword_id": int(source_keyword) if source_keyword is not None else None,
                    "focus_keyword_ids": set(),
                    "failure_kinds": set(),
                    "failure_hits": 0,
                    "max_false_accept_confidence": 0.0,
                    "scene": dict(scene),
                },
            )
            item["failure_hits"] += 1
            item["failure_kinds"].add(kind)
            if focus_keyword > 0:
                item["focus_keyword_ids"].add(focus_keyword)
            if kind == "false-accept":
                item["max_false_accept_confidence"] = max(
                    float(item["max_false_accept_confidence"]),
                    float(failure.get("confidence", 0.0)),
                )
    normalized: list[dict] = []
    for item in aggregated.values():
        normalized.append(
            {
                **item,
                "focus_keyword_ids": sorted(item["focus_keyword_ids"]),
                "failure_kinds": sorted(item["failure_kinds"]),
            }
        )
    normalized.sort(
        key=lambda item: (
            -int(item["failure_hits"]),
            -float(item["max_false_accept_confidence"]),
            str(item["source_wav_sha256"]),
        )
    )
    return normalized[:MAX_UNIQUE_FAILURES]


def _render_repair_rows(config_path: pathlib.Path, specs: list[dict], output: pathlib.Path) -> list[dict]:
    cfg = load_config(config_path)
    token_path = pathlib.Path(str(cfg["tokens"]))
    keyword_path = pathlib.Path(str(cfg["keywords"]))
    if not token_path.is_absolute():
        token_path = (ROOT / token_path).resolve()
    if not keyword_path.is_absolute():
        keyword_path = (ROOT / keyword_path).resolve()
    token_map = load_tokens(token_path)
    keywords = parse_keywords(keyword_path, token_map)
    keyword_text = {int(item["id"]): str(item["text"]) for item in keywords}
    carriers = token_carriers(keywords, int(cfg.get("model", {}).get("feature_dim", 32)))
    generator = cfg.get("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("generator must be an object")
    tts = generator.get("tts", {"backend": "tone"})
    augment_config = generator.get("augment", {})
    if not isinstance(tts, dict) or not isinstance(augment_config, dict):
        raise ValueError("qualification repair generator config is invalid")
    validate_tone_config(tts)
    validate_augment_config(augment_config)
    domains = validate_domains(cfg)
    seed = int(cfg.get("seed", 1337)) + SEED_NAMESPACE

    rendered_rows: list[dict] = []
    for spec_index, spec in enumerate(specs):
        for example_index in range(EXAMPLES_PER_FAILURE):
            example_seed = (
                seed
                + spec_index * 65_537
                + example_index * 4099
                + int(str(spec["source_wav_sha256"])[:8], 16)
            ) & 0x7FFFFFFF
            rng = random.Random(example_seed)
            tokens = [str(value) for value in spec["tokens"]]
            path = output / "wav" / f"q{spec_index:03d}-e{example_index:02d}.wav"
            clean_path = output / "clean" / f"q{spec_index:03d}-e{example_index:02d}.wav"
            if tokens:
                if str(tts.get("backend", "tone")) == "tone":
                    clean = render_tone_tokens(tokens, carriers, rng, tts)
                else:
                    source_keyword = spec.get("source_keyword_id")
                    text = (
                        keyword_text.get(int(source_keyword), " ".join(tokens))
                        if source_keyword
                        else " ".join(tokens)
                    )
                    clean = render_command_tts(text, tokens, str(spec["source_kind"]), clean_path, tts)
            else:
                clean = generate_background(
                    str(spec["scene"].get("noise_profile", "white")),
                    rng.uniform(1.2, 2.0),
                    rng,
                )
            augmented = augment(clean, rng, augment_config)
            scene = _jitter_scene(spec["scene"], domains, rng, example_index)
            rendered, scene_meta = render_scene(
                augmented,
                scene,
                seed=example_seed + 31_337,
                afe=domains["afe"],
            )
            write_wav(path, rendered)
            replay_sha = sha256_file(path)
            if replay_sha == str(spec["source_wav_sha256"]):
                raise ValueError("qualification repair accidentally copied a development evaluation WAV")
            rendered_rows.append(
                {
                    "path": str(path.resolve()),
                    "target_ids": [int(value) for value in spec["target_ids"]],
                    "wav_sha256": replay_sha,
                    "source_wav_sha256": str(spec["source_wav_sha256"]),
                    "source_kind": str(spec["source_kind"]),
                    "source_keyword_id": spec.get("source_keyword_id"),
                    "focus_keyword_ids": list(spec["focus_keyword_ids"]),
                    "failure_kinds": list(spec["failure_kinds"]),
                    "scene": scene_meta,
                    "seed": example_seed,
                }
            )
    return rendered_rows


def render_qualification_failure_replay(
    config_path: pathlib.Path,
    qualification_dataset: pathlib.Path,
    qualification_eval: pathlib.Path,
    previous_failure_evidence: pathlib.Path,
    output: pathlib.Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads(previous_failure_evidence.read_text(encoding="utf-8"))
    if not isinstance(previous, dict):
        raise ValueError("previous development failure evidence must be an object")
    if str(previous.get("policy")) != FAILURE_REPLAY_POLICY:
        raise ValueError("qualification repair requires development failure replay v1 evidence")
    if bool(previous.get("formal_qualification_used", True)):
        raise ValueError("qualification repair refuses formal qualification-derived training")
    if bool(previous.get("development_source_wav_bytes_copied", True)):
        raise ValueError("previous failure replay copied development WAV bytes")
    previous_manifest = pathlib.Path(str(previous.get("manifest") or ""))
    if not previous_manifest.is_file():
        raise ValueError("previous development failure replay manifest is missing")

    specs = _collect_specs(qualification_dataset, qualification_eval)
    if not specs:
        raise ValueError("qualification repair requested without development qualification failures")
    rows = _render_repair_rows(config_path, specs, output)
    repair_manifest = output / "qualification-repair-only.tsv"
    repair_manifest.write_text(
        "".join(
            f"{row['path']}\t{' '.join(str(value) for value in row['target_ids'])}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    combined_manifest = output / "development-failure-replay.tsv"
    combined_manifest.write_text(
        previous_manifest.read_text(encoding="utf-8") + repair_manifest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    previous_selected = previous.get("selected", [])
    if not isinstance(previous_selected, list):
        raise ValueError("previous development failure selected evidence is invalid")
    selected = list(previous_selected) + [
        {
            **spec,
            "source_splits": ["qualification"],
            "source_rounds": ["post-adversarial-refinement"],
        }
        for spec in specs
    ]
    base_source_splits = [str(value) for value in previous.get("source_splits", [])]
    if base_source_splits != ["calibration", "test"]:
        raise ValueError("qualification repair requires calibration/test base failure provenance")
    evidence = {
        **previous,
        "schema_version": 2,
        "policy": FAILURE_REPLAY_POLICY,
        "qualification_repair_policy": POLICY,
        "qualification_repair_used": True,
        "qualification_repair_source_splits": ["qualification"],
        "qualification_repair_examples_per_failure": EXAMPLES_PER_FAILURE,
        "qualification_repair_selected_unique_failures": len(specs),
        "qualification_repair_examples": len(rows),
        "qualification_repair_manifest_sha256": sha256_file(repair_manifest),
        "qualification_repair_seed_namespace": SEED_NAMESPACE,
        "examples": int(previous.get("examples", 0)) + len(rows),
        "selected_unique_failures": int(previous.get("selected_unique_failures", 0)) + len(specs),
        "observed_unique_failures": int(previous.get("observed_unique_failures", 0)) + len(specs),
        "manifest": str(combined_manifest),
        "manifest_sha256": sha256_file(combined_manifest),
        "formal_qualification_used": False,
        "development_source_wav_bytes_copied": False,
        "source_splits": base_source_splits,
        "selected": selected,
    }
    evidence_path = output / "development-failure-replay.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **evidence,
        "evidence": str(evidence_path),
        "repair_manifest": str(repair_manifest),
    }
