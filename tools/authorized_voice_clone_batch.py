#!/usr/bin/env python3
"""Bounded, fail-closed, private offline voice-clone batch generator.

Generation intent is never an accepted training label or product qualification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import canonical_hash, inspect_pcm16_wav, sha256_file  # noqa: E402

POLICY = "authorized-offline-voice-clone-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
IDENT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
KINDS = ("wake", "near_miss", "ordinary_negative")
WAKE_TEXTS = {"你好小窝", "小窝小窝"}
MAX_SPEAKERS = 512
MAX_RECORDINGS = 32768
MAX_PER_SPEAKER = 128
MAX_WORKERS = 4
MAX_INPUT_JSON_BYTES = 8 * 1024 * 1024


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identifier(value: object, field: str) -> str:
    need(isinstance(value, str) and IDENT.fullmatch(value) is not None, f"{field}: invalid identifier")
    return value


def digest(value: object, field: str) -> str:
    need(isinstance(value, str) and SHA.fullmatch(value) is not None, f"{field}: expected lowercase SHA-256")
    return value


def strict_json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        need(key not in result, "input JSON duplicate key: " + key)
        result[key] = value
    return result


def finite_json_float(token: str) -> float:
    value = float(token)
    need(math.isfinite(value), "input JSON non-finite number")
    return value


def reject_json_constant(token: str) -> None:
    raise ValueError("input JSON non-finite constant: " + token)


def read_object(path: Path) -> dict:
    # Campaigns, rights assertions and replay receipts are untrusted even when
    # byte-hash pinned. Bound the read itself to avoid a growing file defeating
    # a preliminary stat check and silently ambiguous JSON authorizations.
    need(path.is_file(), f"{path.name}: expected JSON file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_JSON_BYTES + 1)
    need(len(raw) <= MAX_INPUT_JSON_BYTES, f"{path.name}: input JSON exceeds byte limit")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_pairs,
                       parse_float=finite_json_float, parse_constant=reject_json_constant)
    need(type(value) is dict, f"{path.name}: expected JSON object")
    return value


def pinned_file(root: Path, raw: dict, field: str) -> tuple[Path, str]:
    need(isinstance(raw, dict), f"{field}: expected path/hash object")
    name = raw.get("path")
    need(isinstance(name, str) and bool(name.strip()), f"{field}: missing path")
    path = Path(name)
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    need(path.is_file(), f"{field}: not a regular file")
    declared = digest(raw.get("sha256"), f"{field}.sha256")
    need(sha256_file(path) == declared, f"{field}: file hash mismatch")
    return path, declared


def provider_config(root: Path, raw: object) -> dict:
    need(isinstance(raw, dict) and raw.get("kind") == "offline-voice-clone", "provider: expected offline-voice-clone")
    exe, exe_sha = pinned_file(root, raw.get("executable"), "provider.executable")
    assets = {}
    declared_assets = raw.get("assets")
    need(isinstance(declared_assets, list) and bool(declared_assets), "provider.assets: at least one pinned file required")
    for i, item in enumerate(declared_assets):
        need(isinstance(item, dict), f"provider.assets[{i}]: invalid item")
        role = identifier(item.get("role"), f"provider.assets[{i}].role")
        need(role not in assets, "provider.assets: duplicate role")
        asset, asset_sha = pinned_file(root, item, f"provider.assets[{i}]")
        assets[role] = {"path": asset, "sha256": asset_sha}
    tokens = raw.get("argv_template")
    need(isinstance(tokens, list) and bool(tokens) and all(isinstance(x, str) and x for x in tokens),
         "provider.argv_template: expected string list")
    need(tokens[0] == "{executable}", "provider.argv_template: first token must be {executable}")
    allowed = {"executable", "reference", "reference_text", "text", "output", "prompt_cache", "seed"}
    allowed |= {"asset:" + role for role in assets}
    used = {x for t in tokens for x in PLACEHOLDER.findall(t)}
    need({"executable", "reference", "text", "output", "prompt_cache", "seed"} <= used,
         "provider.argv_template: missing required placeholder")
    need(used <= allowed, "provider.argv_template: unrecognized placeholder")
    for t in tokens:
        need("{" not in PLACEHOLDER.sub("", t) and "}" not in PLACEHOLDER.sub("", t),
             "provider.argv_template: invalid brace")
    timeout = raw.get("timeout_seconds")
    need(type(timeout) is int and 1 <= timeout <= 1800, "provider.timeout_seconds: outside [1,1800]")
    identity = {"kind": "offline-voice-clone", "executable_sha256": exe_sha,
                "assets": [{"role": k, "sha256": assets[k]["sha256"]} for k in sorted(assets)],
                "argv_template": tokens, "timeout_seconds": timeout}
    return {"executable": exe, "assets": assets, "argv_template": tokens,
            "timeout_seconds": timeout, "identity": identity}


def load_campaign(plan_path: Path) -> dict:
    raw = read_object(plan_path)
    need(raw.get("schema_version") == 1 and raw.get("policy") == POLICY, "campaign policy/schema mismatch")
    campaign_id = identifier(raw.get("campaign_id"), "campaign_id")
    root = plan_path.parent
    provider = provider_config(root, raw.get("provider"))
    phrases = raw.get("phrases")
    need(isinstance(phrases, list) and 3 <= len(phrases) <= 64, "phrases: expected 3-64 entries")
    seen_phrase_ids, seen_text, kinds = set(), set(), set()
    parsed_phrases = []
    for i, row in enumerate(phrases):
        need(isinstance(row, dict), f"phrases[{i}]: expected object")
        phrase_id = identifier(row.get("id"), f"phrases[{i}].id")
        text = row.get("text")
        kind = row.get("intent")
        need(isinstance(text, str) and bool(text.strip()) and len(text) <= 120,
             f"phrases[{i}].text: invalid text")
        need(kind in KINDS, f"phrases[{i}].intent: invalid kind")
        normalized = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
        need(phrase_id not in seen_phrase_ids and normalized not in seen_text, "phrases: duplicate id/text")
        if kind == "wake":
            need(normalized in WAKE_TEXTS, "wake text is not a configured keyword")
        else:
            need(all(wake not in normalized for wake in WAKE_TEXTS),
                 "negative intent contains a full wake phrase")
        seen_phrase_ids.add(phrase_id)
        seen_text.add(normalized)
        kinds.add(kind)
        parsed_phrases.append({"id": phrase_id, "text": text.strip(), "intent": kind})
    need(kinds == set(KINDS), "phrases: require wake, near_miss and ordinary_negative")
    seeds = raw.get("seeds")
    need(isinstance(seeds, list) and 1 <= len(seeds) <= 8 and
         all(type(seed) is int and 0 <= seed <= 2**31 - 1 for seed in seeds) and
         len(set(seeds)) == len(seeds), "seeds: 1-8 unique nonnegative integers required")
    speakers = raw.get("speakers")
    need(isinstance(speakers, list) and 1 <= len(speakers) <= MAX_SPEAKERS,
         "speakers: expected 1-512 authorized sources")
    need(len(seeds) * len(phrases) <= MAX_PER_SPEAKER, "per-speaker recording cap exceeded")
    seen_speakers, seen_refs, seen_ref_pcm = set(), set(), set()
    parsed_speakers = []
    for i, row in enumerate(speakers):
        need(isinstance(row, dict), f"speakers[{i}]: expected object")
        speaker_id = identifier(row.get("speaker_id"), f"speakers[{i}].speaker_id")
        family_id = identifier(row.get("source_family_id"), f"speakers[{i}].source_family_id")
        need(speaker_id not in seen_speakers, "speakers: duplicate speaker_id")
        ref, ref_sha = pinned_file(root, row.get("reference"), f"speakers[{i}].reference")
        need(ref_sha not in seen_refs, "speakers: same reference reused as another speaker")
        ref_audio = inspect_pcm16_wav(ref)
        need(ref_audio["pcm_sha256"] not in seen_ref_pcm,
             "speakers: same decoded PCM reused as another speaker")
        need(3.0 <= ref_audio["duration_s"] <= 30.0, "reference must be 3-30 seconds mono PCM16 16kHz")
        rights, rights_sha = pinned_file(root, row.get("rights_evidence"), f"speakers[{i}].rights_evidence")
        rights_data = read_object(rights)
        need(rights_data.get("schema_version") == 1 and
             rights_data.get("source_family_id") == family_id and
             rights_data.get("voice_cloning_permitted") is True and
             rights_data.get("synthetic_kws_training_permitted") is True,
             "rights evidence lacks explicit cloning and training permissions for source family")
        scope = rights_data.get("covered_speaker_ids")
        need(isinstance(scope, list) and speaker_id in scope and
             all(isinstance(x, str) for x in scope),
             "rights evidence does not explicitly cover this speaker")
        basis = rights_data.get("authorization_basis")
        need(isinstance(basis, str) and len(basis.strip()) >= 8,
             "rights evidence requires a source-specific authorization basis")
        ref_text = row.get("reference_text", "")
        need(isinstance(ref_text, str) and len(ref_text) <= 1000,
             "reference_text must be text of at most 1000 characters")
        parsed_speakers.append({"speaker_id": speaker_id, "source_family_id": family_id,
                                "reference": ref, "reference_sha256": ref_sha,
                                "reference_pcm_sha256": ref_audio["pcm_sha256"],
                                "rights_evidence": rights, "rights_sha256": rights_sha,
                                "reference_text": ref_text})
        seen_speakers.add(speaker_id)
        seen_refs.add(ref_sha)
        seen_ref_pcm.add(ref_audio["pcm_sha256"])
    planned = len(parsed_speakers) * len(parsed_phrases) * len(seeds)
    # Hyphens in IDs must not create ambiguous source_id aliases.
    all_ids = [f"{speaker['speaker_id']}-{phrase['id']}-s{seed}"
               for speaker in parsed_speakers for phrase in parsed_phrases for seed in seeds]
    need(len(set(all_ids)) == planned, "speaker/phrase IDs produce ambiguous recording IDs")
    max_attempts = raw.get("max_attempts")
    need(type(max_attempts) is int and planned <= max_attempts <= MAX_RECORDINGS,
         "max_attempts must be a finite explicit budget covering the plan (max 32768)")
    identity = {"policy": POLICY, "campaign_id": campaign_id,
                "provider": provider["identity"],
                "speakers": [{k: v for k, v in row.items() if k in (
                    "speaker_id", "source_family_id", "reference_sha256", "reference_pcm_sha256",
                    "rights_sha256", "reference_text")} for row in parsed_speakers],
                "phrases": parsed_phrases, "seeds": seeds, "max_attempts": max_attempts}
    return {"campaign_id": campaign_id, "sha256": canonical_hash(identity),
            "provider": provider, "speakers": parsed_speakers, "phrases": parsed_phrases,
            "seeds": seeds, "recordings": planned}


def create_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        need(path.read_text(encoding="utf-8") == value, f"existing output differs: {path.name}")


def claim_invocation_once(path: Path, value: str) -> None:
    """Atomically reserve one provider call; a second claimant must never run it.

    Unlike create_once for immutable metadata, the attempt marker must not
    treat an already-existing *equal* file as permission to invoke again.
    An in-flight or abruptly interrupted attempt is conservatively consumed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError(
            "attempt already claimed by another execution or unknown; no automatic retry"
        ) from exc


def render_argv(provider: dict, speaker: dict, phrase: dict, seed: int,
                output: Path, cache: Path) -> list[str]:
    mapping = {"executable": str(provider["executable"]),
               "reference": str(speaker["reference"]),
               "reference_text": speaker["reference_text"], "text": phrase["text"],
               "output": str(output), "prompt_cache": str(cache), "seed": str(seed)}
    mapping.update({"asset:" + role: str(asset["path"])
                    for role, asset in provider["assets"].items()})
    return [PLACEHOLDER.sub(lambda match: mapping[match.group(1)], token)
            for token in provider["argv_template"]]


def run_speaker(campaign: dict, speaker: dict, output_root: Path) -> list[dict]:
    provider = campaign["provider"]
    ident = speaker["speaker_id"]
    cache_dir = output_root / "private-prompt-cache" / ident / speaker["reference_sha256"][:16]
    # Recheck small private inputs once per speaker. Model assets were hash-pinned
    # during campaign preflight; rehashing multi-GB weights per utterance is unsafe.
    need(sha256_file(speaker["reference"]) == speaker["reference_sha256"],
         "reference changed after campaign preflight")
    need(sha256_file(speaker["rights_evidence"]) == speaker["rights_sha256"],
         "rights evidence changed after campaign preflight")
    records = []
    for phrase in campaign["phrases"]:
        for seed in campaign["seeds"]:
            item_id = f"{ident}-{phrase['id']}-s{seed}"
            audio = output_root / "audio" / ident / (item_id + ".wav")
            marker = output_root / "attempts" / ident / (item_id + ".intent.json")
            receipt = output_root / "attempts" / ident / (item_id + ".receipt.json")
            intent = {"schema_version": 1, "campaign_sha256": campaign["sha256"],
                      "item_id": item_id, "reference_audio_sha256": speaker["reference_sha256"],
                      "rights_evidence_sha256": speaker["rights_sha256"],
                      "requested_intent": phrase["intent"], "seed": seed}
            intent_text = json.dumps(intent, ensure_ascii=False, sort_keys=True) + "\n"
            if marker.exists():
                need(marker.read_text(encoding="utf-8") == intent_text, "attempt marker identity mismatch")
                need(receipt.is_file(), "unreceipted invocation is consumed/unknown; no automatic retry")
                row = read_object(receipt)
                expected_metadata = {
                    "schema_version": 1, "evidence_class": "unreviewed-synthetic-voice-clone-v1",
                    "campaign_sha256": campaign["sha256"], "item_id": item_id,
                    "audio": str(audio.relative_to(output_root)),
                    "speaker_id": ident, "source_family_id": speaker["source_family_id"],
                    "derivation_family_id": "clone-" + speaker["reference_sha256"],
                    "reference_audio_sha256": speaker["reference_sha256"],
                    "rights_evidence_sha256": speaker["rights_sha256"],
                    "provider_identity_sha256": canonical_hash(provider["identity"]),
                    "source_id": item_id, "text_requested": phrase["text"],
                    "requested_intent": phrase["intent"], "seed": seed,
                    "synthetic": True, "training_admitted": False,
                    "qualification_allowed": False,
                }
                need(audio.is_file() and all(row.get(k) == v for k, v in expected_metadata.items()),
                     "saved receipt metadata/output mismatch")
                checked = inspect_pcm16_wav(audio)
                need(checked["file_sha256"] == row.get("file_sha256") and
                     checked["pcm_sha256"] == row.get("pcm_sha256"),
                     "completed output WAV changed")
            else:
                need(not receipt.exists() and not audio.exists(), "orphan output must not be overwritten")
                argv = render_argv(provider, speaker, phrase, seed, audio, cache_dir)
                need(Path(argv[0]).resolve() == provider["executable"], "provider executable drifted")
                claim_invocation_once(marker, intent_text)  # exclusive durable claim BEFORE provider invocation
                audio.parent.mkdir(parents=True, exist_ok=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
                env = {key: os.environ[key] for key in
                       ("PATH", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "HOME")
                       if key in os.environ}  # do not forward auth tokens or API keys
                subprocess.run(argv, shell=False, check=True,
                               timeout=provider["timeout_seconds"], env=env)
                checked = inspect_pcm16_wav(audio)
                need(0.2 <= checked["duration_s"] <= 30.0, "generated audio duration outside 0.2-30s")
                row = {"schema_version": 1, "evidence_class": "unreviewed-synthetic-voice-clone-v1",
                       "campaign_sha256": campaign["sha256"], "item_id": item_id,
                       "audio": str(audio.relative_to(output_root)),
                       "speaker_id": ident, "source_family_id": speaker["source_family_id"],
                       "derivation_family_id": "clone-" + speaker["reference_sha256"],
                       "reference_audio_sha256": speaker["reference_sha256"],
                       "rights_evidence_sha256": speaker["rights_sha256"],
                       "provider_identity_sha256": canonical_hash(provider["identity"]),
                       "source_id": item_id, "text_requested": phrase["text"],
                       "requested_intent": phrase["intent"], "seed": seed,
                       "file_sha256": checked["file_sha256"],
                       "pcm_sha256": checked["pcm_sha256"], "frames": checked["frames"],
                       "duration_s": checked["duration_s"], "synthetic": True,
                       "training_admitted": False, "qualification_allowed": False}
                create_once(receipt, json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            records.append(row)
    return records


def execute(campaign: dict, root: Path, workers: int) -> dict:
    need(type(workers) is int and 1 <= workers <= MAX_WORKERS, "workers must be 1-4")
    root = root.resolve()
    need(root != ROOT and ROOT not in root.parents, "private output cannot be inside the source repository")
    root.mkdir(parents=True, exist_ok=True)
    create_once(root / "campaign.json", json.dumps({"policy": POLICY,
        "campaign_id": campaign["campaign_id"], "campaign_sha256": campaign["sha256"],
        "recordings": campaign["recordings"], "training_admitted": False,
        "qualification_allowed": False}, sort_keys=True) + "\n")
    # Source and executable hashes are validated by load_campaign before any
    # process is launched. The caller must keep those files immutable while running.
    all_records = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run_speaker, campaign, speaker, root): speaker["speaker_id"]
                for speaker in campaign["speakers"]}
        for future in as_completed(jobs):
            try:
                all_records.extend(future.result())
            except Exception as exc:  # retain original markers; never retry unknown invocation
                failures.append(f"{jobs[future]}: {exc}")
    if failures:
        raise RuntimeError("voice-clone campaign incomplete; do not retry consumed attempts: " + "; ".join(sorted(failures)))
    need(len(all_records) == campaign["recordings"], "incomplete voice-clone receipt set")
    all_records.sort(key=lambda row: row["item_id"])
    create_once(root / "manifest.jsonl", "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                                     for row in all_records))
    report = {"policy": POLICY, "campaign_sha256": campaign["sha256"],
              "recordings": len(all_records), "reference_roster_entries": len(campaign["speakers"]),
              "independent_real_speakers_verified": False, "training_admitted": False,
              "qualification_allowed": False, "manifest_sha256": sha256_file(root / "manifest.jsonl")}
    create_once(root / "summary.json", json.dumps(report, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan/run authorized offline KWS voice clones (no model bundled).")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="run pinned offline command provider")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    campaign = load_campaign(args.plan.resolve(strict=True))
    if args.execute:
        report = execute(campaign, args.output_root, args.workers)
    else:
        report = {"policy": POLICY, "campaign_sha256": campaign["sha256"],
                  "recordings_planned": campaign["recordings"],
                  "reference_speakers": len(campaign["speakers"]),
                  "dry_run": True, "provider_invocations": 0,
                  "training_admitted": False, "qualification_allowed": False}
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
