#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

HEX = set("0123456789abcdef")
PREFLIGHT_POLICY = "shadow-adversarial-failure-formal-preflight-v2"
DATA_POLICY = "train-only-balanced-mining-v1"
ADVERSARIAL_POLICY = "balanced-strict-prefix-anchor-topk-v2"
FAILURE_REPLAY_POLICY = "development-failure-resynthesis-v1"
EXPECTED_ADVERSARIAL_TOP_K = 64
EXPECTED_ADVERSARIAL_PROBES = 2
EXPECTED_ADVERSARIAL_REPLAY_PER_SEQUENCE = 8
EXPECTED_ADVERSARIAL_REPLAY_EXAMPLES = 512
EXPECTED_ADVERSARIAL_MIN_PER_KEYWORD = 24
EXPECTED_SAFE_LEXICAL_POOL = 1330
EXPECTED_PREFIX_ANCHORS = {
    ("ni3",),
    ("ni3", "hao3"),
    ("ni3", "hao3", "xiao3"),
    ("xiao3",),
    ("xiao3", "wo1"),
    ("xiao3", "wo1", "xiao3"),
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: pathlib.Path, label: str, *, allow_empty: bool = False) -> pathlib.Path:
    if not path.is_file() or (not allow_empty and path.stat().st_size == 0):
        raise ValueError(f"missing/empty {label}: {path}")
    return path


def load_json(path: pathlib.Path, label: str) -> dict:
    value = json.loads(require_file(path, label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_sha(value: object, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in HEX for ch in text):
        raise ValueError(f"{label} is missing/invalid")
    return text


def require_zero_error(summary: dict, *, expected: int, label: str) -> None:
    if (
        int(summary.get("expected", -1)) != expected
        or int(summary.get("matched", -1)) != expected
        or int(summary.get("false_rejects", -1)) != 0
        or int(summary.get("false_accepts", -1)) != 0
    ):
        raise ValueError(f"{label} is not strict {expected}/{expected} zero-error")


def resolve_best(root: pathlib.Path, name: str) -> pathlib.Path:
    matches = [path for path in root.rglob(name) if path.is_file() and path.parent.name == "best"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one best/{name}, got {len(matches)}")
    return require_file(matches[0], f"best/{name}")


def copy_required(root: pathlib.Path, dist: pathlib.Path) -> dict[str, pathlib.Path]:
    best = {
        "model.kwm": "xiaowo-model.kwm",
        "model.pt": "xiaowo-model.pt",
        "model-provenance.json": "xiaowo-model-provenance.json",
        "keywords.kwk": "xiaowo-keywords.kwk",
        "keywords.tsv": "xiaowo-keywords.tsv",
        "development-failure-replay.json": "development-failure-replay.json",
    }
    evidence = {
        root / "training-run-summary.json": "training-run-summary.json",
        root / "robustness-summary.json": "robustness-summary.json",
        root / "hard-negative-stream" / "summary.json": "continuous-far-summary.json",
        root / "hard-negative-stream" / "stream-contract.json": "continuous-far-stream-contract.json",
        root / "qualification-dataset" / "qualification-cohort.json": "qualification-cohort.json",
        root / "best" / "qualification" / "summary.json": "qualification-summary.json",
        root / "shadow-qualification" / "summary.json": "shadow-qualification-summary.json",
        root / "adversarial-refinement" / "summary.json": "adversarial-refinement-summary.json",
        root / "best" / "adversarial-lexicon.json": "adversarial-lexicon.json",
        root / "formal-preflight.json": "formal-preflight.json",
    }
    dist.mkdir(parents=True, exist_ok=True)
    copied: dict[str, pathlib.Path] = {}
    for source_name, target_name in best.items():
        source = resolve_best(root, source_name)
        target = dist / target_name
        shutil.copy2(source, target)
        copied[target_name] = target
    for source, target_name in evidence.items():
        target = dist / target_name
        shutil.copy2(require_file(source, target_name), target)
        copied[target_name] = target
    return copied


def git_tree(repo: str, commit: str) -> str:
    payload = subprocess.check_output(["gh", "api", f"repos/{repo}/git/commits/{commit}"], text=True)
    return str(json.loads(payload)["tree"]["sha"])


def provenance_manifest_count(provenance: dict, name: str, digest: str) -> int:
    manifests = provenance.get("training", {}).get("manifests", [])
    if not isinstance(manifests, list):
        raise ValueError("model provenance training.manifests is invalid")
    return sum(
        1
        for row in manifests
        if isinstance(row, dict)
        and str(row.get("name")) == name
        and str(row.get("sha256")) == digest
    )


def verify(args: argparse.Namespace) -> dict:
    root = args.artifact_root.resolve()
    dist = args.dist.resolve()
    copied = copy_required(root, dist)
    files = {
        name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for name, path in sorted(copied.items())
    }

    provenance = load_json(dist / "xiaowo-model-provenance.json", "model provenance")
    if provenance.get("model", {}).get("sha256") != files["xiaowo-model.kwm"]["sha256"]:
        raise ValueError("model provenance SHA does not match promoted model.kwm")
    if provenance.get("checkpoint", {}).get("sha256") != files["xiaowo-model.pt"]["sha256"]:
        raise ValueError("model provenance checkpoint SHA does not match promoted model.pt")

    training = load_json(dist / "training-run-summary.json", "training summary")
    if not bool(training.get("qualified")):
        raise ValueError("training summary is not qualified")
    artifacts = training.get("artifacts", {})
    for name, target in {
        "model.kwm": "xiaowo-model.kwm",
        "model.pt": "xiaowo-model.pt",
        "model-provenance.json": "xiaowo-model-provenance.json",
        "keywords.kwk": "xiaowo-keywords.kwk",
        "keywords.tsv": "xiaowo-keywords.tsv",
    }.items():
        if artifacts.get(name) != files[target]["sha256"]:
            raise ValueError(f"training summary artifact SHA mismatch for {name}")

    qualification = load_json(dist / "qualification-summary.json", "qualification summary")
    require_zero_error(qualification, expected=256, label="qualification evidence")
    per_keyword = qualification.get("per_keyword", {})
    for keyword_id in ("1", "2"):
        require_zero_error(
            per_keyword.get(keyword_id, {}),
            expected=128,
            label=f"qualification keyword {keyword_id}",
        )

    robustness = load_json(dist / "robustness-summary.json", "robustness summary")
    if not bool(robustness.get("qualified")) or robustness.get("failures") != []:
        raise ValueError("robustness evidence is not qualified with zero failures")

    continuous_far = load_json(dist / "continuous-far-summary.json", "continuous FAR summary")
    if (
        not bool(continuous_far.get("qualified"))
        or int(continuous_far.get("false_accepts", -1)) != 0
        or float(continuous_far.get("far_per_hour", -1.0)) != 0.0
        or not bool(continuous_far.get("full_negative_manifest_coverage"))
    ):
        raise ValueError("continuous FAR evidence is not strict zero-error/full-coverage")
    if continuous_far.get("model_sha256") != files["xiaowo-model.kwm"]["sha256"]:
        raise ValueError("continuous FAR model SHA does not match promoted model")

    stream = load_json(dist / "continuous-far-stream-contract.json", "continuous FAR stream contract")
    if stream.get("model_sha256") != files["xiaowo-model.kwm"]["sha256"]:
        raise ValueError("continuous FAR stream contract model SHA mismatch")
    if not bool(stream.get("full_negative_manifest_coverage")):
        raise ValueError("continuous FAR stream contract lacks full coverage")
    if float(stream.get("observed_min_payload_gap_seconds", 0.0)) < float(
        stream.get("decoder_retention_proof_window_seconds", 1.6)
    ):
        raise ValueError("continuous FAR payload gap does not clear decoder retention window")

    cohort = load_json(dist / "qualification-cohort.json", "qualification cohort")
    if not bool(cohort.get("generated_after_training")) or not bool(cohort.get("seed_disjoint")):
        raise ValueError("qualification cohort is not untouched/disjoint")
    if not bool(cohort.get("strict_development_candidate_required")):
        raise ValueError("qualification cohort lacks strict development prerequisite evidence")
    development_manifest_sha = require_sha(
        cohort.get("development_manifest_sha256"), "qualification cohort development manifest SHA"
    )
    if str(cohort.get("development_candidate_policy")) != "latest-strict-gate-passing-round":
        raise ValueError("qualification cohort development candidate policy is not strict/latest")
    selected_round = int(training.get("best_round", -2))
    selected_frontend = str(training.get("best_frontend") or "")
    if int(cohort.get("development_selected_round", -1)) != selected_round:
        raise ValueError("qualification cohort development round differs from finalized model")
    if str(cohort.get("development_selected_frontend") or "") != selected_frontend:
        raise ValueError("qualification cohort development frontend differs from finalized model")
    if int(cohort.get("expected_wakes", -1)) != 256:
        raise ValueError("qualification cohort expected-wake count is not 256")
    for key in (
        "overlapping_development_active_wav_sha256",
        "overlapping_exposed_active_wav_sha256",
        "overlapping_retired_active_wav_sha256",
    ):
        if int(cohort.get(key, -1)) != 0:
            raise ValueError(f"qualification cohort overlap is non-zero: {key}")
    qualification_seed = int(cohort.get("qualification_seed", -1))
    retired = [int(value) for value in cohort.get("retired_exposed_qualification_seeds", [])]
    if qualification_seed < 0 or qualification_seed in retired:
        raise ValueError("qualification seed is invalid or already retired in its own cohort evidence")

    if str(cohort.get("formal_preflight_policy")) != PREFLIGHT_POLICY:
        raise ValueError("qualification cohort lacks Data V3 guarded formal preflight policy")
    for key in (
        "shadow_qualification_required",
        "shadow_qualification_qualified",
        "adversarial_refinement_required",
        "model_provenance_adversarial_manifest_verified",
        "adversarial_overlap_guard_included",
        "failure_replay_required",
        "model_provenance_failure_replay_manifest_verified",
    ):
        if not bool(cohort.get(key)):
            raise ValueError(f"qualification cohort prerequisite is false: {key}")
    if bool(cohort.get("adversarial_formal_qualification_used", True)):
        raise ValueError("qualification cohort says adversarial training used formal qualification")
    if bool(cohort.get("failure_replay_formal_qualification_used", True)):
        raise ValueError("qualification cohort says failure replay used formal qualification")
    if bool(cohort.get("failure_replay_development_source_wav_bytes_copied", True)):
        raise ValueError("qualification cohort says failure replay copied development WAV bytes")

    shadow = load_json(dist / "shadow-qualification-summary.json", "shadow qualification summary")
    refinement = load_json(dist / "adversarial-refinement-summary.json", "adversarial refinement summary")
    adversarial = load_json(dist / "adversarial-lexicon.json", "adversarial lexicon evidence")
    failure = load_json(dist / "development-failure-replay.json", "development failure replay evidence")
    preflight = load_json(dist / "formal-preflight.json", "formal preflight")

    sha_bindings = {
        "shadow_summary_sha256": "shadow-qualification-summary.json",
        "adversarial_refinement_summary_sha256": "adversarial-refinement-summary.json",
        "adversarial_lexicon_evidence_sha256": "adversarial-lexicon.json",
        "failure_replay_evidence_sha256": "development-failure-replay.json",
        "formal_preflight_sha256": "formal-preflight.json",
    }
    for field, filename in sha_bindings.items():
        if require_sha(cohort.get(field), f"cohort {field}") != files[filename]["sha256"]:
            raise ValueError(f"qualification cohort {field} does not match promoted evidence")

    if (
        int(shadow.get("schema_version", 0)) != 1
        or str(shadow.get("evidence_class")) != "development-only-shadow-qualification"
        or not bool(shadow.get("qualified"))
        or bool(shadow.get("formal_qualification_seed_consumed", True))
    ):
        raise ValueError("shadow qualification evidence contract failed")
    if str(shadow.get("development_manifest_sha256") or "") != development_manifest_sha:
        raise ValueError("shadow qualification development manifest SHA mismatch")
    if int(shadow.get("development_selected_round", -1)) != selected_round:
        raise ValueError("shadow qualification selected round mismatch")
    if str(shadow.get("development_selected_frontend") or "") != selected_frontend:
        raise ValueError("shadow qualification selected frontend mismatch")
    shadow_results = shadow.get("results")
    if not isinstance(shadow_results, list) or len(shadow_results) != int(cohort.get("shadow_seed_count", -1)):
        raise ValueError("shadow qualification result count differs from cohort")
    if len(shadow_results) < 8:
        raise ValueError("shadow qualification arena is smaller than eight seeds")
    gap_floor = float(cohort.get("shadow_min_surrogate_separation", -1.0))
    if gap_floor <= 0.0 or abs(float(shadow.get("min_surrogate_separation", -2.0)) - gap_floor) > 1.0e-12:
        raise ValueError("shadow surrogate separation policy mismatch")
    for row in shadow_results:
        if (
            not isinstance(row, dict)
            or not bool(row.get("qualified"))
            or not bool(row.get("runtime_qualified"))
            or not bool(row.get("surrogate_separation_qualified"))
        ):
            raise ValueError("shadow qualification contains a non-qualified seed")
        require_zero_error(row.get("qualification", {}), expected=256, label="shadow seed qualification")
        if float(row.get("surrogate", {}).get("minimum_separation", -1.0)) < gap_floor:
            raise ValueError("shadow seed surrogate separation fell below the promoted floor")

    if (
        str(refinement.get("policy")) != "post-domain-adversarial-refinement-v1"
        or not bool(refinement.get("qualified"))
        or bool(refinement.get("formal_qualification_used", True))
    ):
        raise ValueError("adversarial refinement evidence contract failed")
    if str(refinement.get("output_development_manifest_sha256") or "") != development_manifest_sha:
        raise ValueError("adversarial refinement output development manifest SHA mismatch")
    if int(refinement.get("refinement_round", -1)) != selected_round:
        raise ValueError("adversarial refinement round differs from promoted model")
    if str(refinement.get("frontend") or "") != selected_frontend:
        raise ValueError("adversarial refinement frontend differs from promoted model")

    if (
        str(adversarial.get("evidence_class")) != "development-only-adversarial-lexicon"
        or bool(adversarial.get("formal_qualification_used", True))
        or str(adversarial.get("data_augmentation_policy")) != DATA_POLICY
        or str(adversarial.get("selection_policy")) != ADVERSARIAL_POLICY
        or int(adversarial.get("max_length", -1)) != 5
        or int(adversarial.get("enumerated_sequences", -1)) != EXPECTED_SAFE_LEXICAL_POOL
        or int(adversarial.get("top_k", -1)) != EXPECTED_ADVERSARIAL_TOP_K
        or int(adversarial.get("probes_per_sequence", -1)) != EXPECTED_ADVERSARIAL_PROBES
        or int(adversarial.get("replay_examples_per_sequence", -1)) != EXPECTED_ADVERSARIAL_REPLAY_PER_SEQUENCE
        or int(adversarial.get("replay_examples", -1)) != EXPECTED_ADVERSARIAL_REPLAY_EXAMPLES
        or int(adversarial.get("min_per_keyword", -1)) != EXPECTED_ADVERSARIAL_MIN_PER_KEYWORD
    ):
        raise ValueError("adversarial lexicon Data V3 policy contract failed")
    per_keyword_selected = adversarial.get("per_keyword_selected", {})
    for keyword_id in ("1", "2"):
        if int(per_keyword_selected.get(keyword_id, 0)) < EXPECTED_ADVERSARIAL_MIN_PER_KEYWORD:
            raise ValueError(f"adversarial keyword {keyword_id} quota is below Data V3 minimum")
    anchors = {
        tuple(str(token) for token in item)
        for item in adversarial.get("strict_prefix_anchors", [])
        if isinstance(item, list)
    }
    if anchors != EXPECTED_PREFIX_ANCHORS:
        raise ValueError("adversarial strict-prefix anchor set drifted")
    for field in (
        "enumerated_sequences",
        "top_k",
        "probes_per_sequence",
        "replay_examples_per_sequence",
        "replay_examples",
        "min_per_keyword",
    ):
        if int(adversarial.get(field, -1)) != int(cohort.get(f"adversarial_{field}", -2)):
            raise ValueError(f"adversarial {field} differs from formal cohort")
    if dict(per_keyword_selected) != dict(cohort.get("adversarial_per_keyword_selected", {})):
        raise ValueError("adversarial per-keyword selection differs from formal cohort")
    if list(adversarial.get("strict_prefix_anchors", [])) != list(
        cohort.get("adversarial_strict_prefix_anchors", [])
    ):
        raise ValueError("adversarial strict-prefix anchors differ from formal cohort")
    if str(adversarial.get("selection_policy")) != str(cohort.get("adversarial_selection_policy")):
        raise ValueError("adversarial selection policy differs from formal cohort")
    if str(adversarial.get("data_augmentation_policy")) != str(
        cohort.get("adversarial_data_augmentation_policy")
    ):
        raise ValueError("adversarial data policy differs from formal cohort")
    adversarial_manifest_sha = require_sha(adversarial.get("manifest_sha256"), "adversarial manifest SHA")
    if adversarial_manifest_sha != require_sha(
        cohort.get("adversarial_manifest_sha256"), "cohort adversarial manifest SHA"
    ):
        raise ValueError("adversarial manifest SHA differs from formal cohort")

    if (
        str(failure.get("evidence_class")) != "development-only-failure-resynthesis"
        or str(failure.get("policy")) != FAILURE_REPLAY_POLICY
        or not bool(failure.get("enabled"))
        or bool(failure.get("formal_qualification_used", True))
        or bool(failure.get("development_source_wav_bytes_copied", True))
        or list(failure.get("source_splits", [])) != ["calibration", "test"]
    ):
        raise ValueError("development failure replay evidence contract failed")
    failure_manifest_sha = require_sha(failure.get("manifest_sha256"), "failure replay manifest SHA")
    if failure_manifest_sha != require_sha(
        cohort.get("failure_replay_manifest_sha256"), "cohort failure replay manifest SHA"
    ):
        raise ValueError("failure replay manifest SHA differs from formal cohort")
    for field in (
        "examples",
        "observed_unique_failures",
        "selected_unique_failures",
    ):
        if int(failure.get(field, -1)) != int(cohort.get(f"failure_replay_{field}", -2)):
            raise ValueError(f"failure replay {field} differs from formal cohort")
    failure_examples = int(failure.get("examples", 0))
    if bool(cohort.get("failure_replay_overlap_guard_included")) != (failure_examples > 0):
        raise ValueError("failure replay overlap-guard state disagrees with replay support")

    if str(preflight.get("policy")) != PREFLIGHT_POLICY:
        raise ValueError("formal preflight policy drifted")
    scalar_preflight_bindings = {
        "development_manifest_sha256": development_manifest_sha,
        "development_selected_round": selected_round,
        "development_selected_frontend": selected_frontend,
        "shadow_summary_sha256": files["shadow-qualification-summary.json"]["sha256"],
        "adversarial_refinement_summary_sha256": files["adversarial-refinement-summary.json"]["sha256"],
        "adversarial_lexicon_evidence_sha256": files["adversarial-lexicon.json"]["sha256"],
        "adversarial_manifest_sha256": adversarial_manifest_sha,
        "failure_replay_evidence_sha256": files["development-failure-replay.json"]["sha256"],
        "failure_replay_manifest_sha256": failure_manifest_sha,
        "failure_replay_examples": failure_examples,
    }
    for field, expected in scalar_preflight_bindings.items():
        if preflight.get(field) != expected:
            raise ValueError(f"formal preflight {field} mismatch")
    if not bool(preflight.get("model_provenance_adversarial_manifest_verified")):
        raise ValueError("formal preflight lacks adversarial training provenance proof")
    if not bool(preflight.get("model_provenance_failure_replay_manifest_verified")):
        raise ValueError("formal preflight lacks failure-replay training provenance proof")

    if provenance_manifest_count(
        provenance, "adversarial-hard-negatives.tsv", adversarial_manifest_sha
    ) != 1:
        raise ValueError("promoted model provenance does not prove adversarial replay training")
    failure_manifest_matches = provenance_manifest_count(
        provenance, "development-failure-replay.tsv", failure_manifest_sha
    )
    if failure_examples > 0 and failure_manifest_matches != 1:
        raise ValueError("promoted model provenance does not prove development failure replay training")
    if failure_examples == 0 and failure_manifest_matches != 0:
        raise ValueError("zero-support failure replay unexpectedly appears in model training provenance")

    keyword_rows = [
        raw.split("\t")
        for raw in (dist / "xiaowo-keywords.tsv").read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]
    expected_keywords = [
        ("1", "你好小窝", 0.55, "ni3 hao3 xiao3 wo1"),
        ("2", "小窝小窝", 0.55, "xiao3 wo1 xiao3 wo1"),
    ]
    if len(keyword_rows) != len(expected_keywords):
        raise ValueError("promoted keyword TSV must contain exactly two shipping wake words")
    for row, expected in zip(keyword_rows, expected_keywords):
        if (
            len(row) < 4
            or row[0] != expected[0]
            or row[1] != expected[1]
            or abs(float(row[2]) - expected[2]) > 1.0e-9
            or row[3] != expected[3]
        ):
            raise ValueError(f"promoted keyword contract drifted: {row!r}")
    if any(row[1] == "小窝" or len(row[1]) != 4 for row in keyword_rows):
        raise ValueError("two-character 小窝 is not a shipping wake word")

    provenance_repository_sha = str(
        provenance.get("training", {}).get("environment", {}).get("repository_sha", "")
    )
    if len(provenance_repository_sha) != 40 or any(ch not in HEX for ch in provenance_repository_sha):
        raise ValueError("model provenance repository SHA is missing/invalid")
    expected_tree_sha = git_tree(args.repository, args.expected_head_sha)
    provenance_tree_sha = git_tree(args.repository, provenance_repository_sha)
    if provenance_tree_sha != expected_tree_sha:
        raise ValueError("model provenance repository tree differs from requested training HEAD tree")

    manifest = {
        "schema_version": 4,
        "source": {
            "repository": args.repository,
            "training_run_id": args.training_run_id,
            "head_sha": args.expected_head_sha,
            "head_tree_sha": expected_tree_sha,
            "provenance_repository_sha": provenance_repository_sha,
            "provenance_tree_sha": provenance_tree_sha,
            "artifact_id": args.artifact_id,
            "artifact_digest": args.artifact_digest or None,
            "artifact_size_bytes": args.artifact_size,
            "qualification_seed": qualification_seed,
            "development_manifest_sha256": development_manifest_sha,
            "formal_preflight_sha256": files["formal-preflight.json"]["sha256"],
            "shadow_summary_sha256": files["shadow-qualification-summary.json"]["sha256"],
            "adversarial_refinement_summary_sha256": files["adversarial-refinement-summary.json"]["sha256"],
            "adversarial_lexicon_evidence_sha256": files["adversarial-lexicon.json"]["sha256"],
            "failure_replay_evidence_sha256": files["development-failure-replay.json"]["sha256"],
        },
        "acceptance": {
            "qualification_expected": 256,
            "qualification_matched": 256,
            "qualification_false_rejects": 0,
            "qualification_false_accepts": 0,
            "strict_development_candidate_required": True,
            "adversarial_refinement_required": True,
            "adversarial_refinement_qualified": True,
            "adversarial_data_policy": DATA_POLICY,
            "adversarial_selection_policy": ADVERSARIAL_POLICY,
            "adversarial_top_k": EXPECTED_ADVERSARIAL_TOP_K,
            "adversarial_replay_examples": EXPECTED_ADVERSARIAL_REPLAY_EXAMPLES,
            "failure_replay_required": True,
            "failure_replay_policy": FAILURE_REPLAY_POLICY,
            "failure_replay_examples": failure_examples,
            "shadow_qualification_required": True,
            "shadow_qualification_qualified": True,
            "shadow_seed_count": len(shadow_results),
            "shadow_min_surrogate_separation": gap_floor,
            "robustness_qualified": True,
            "continuous_far_per_hour": 0.0,
            "continuous_far_full_negative_manifest_coverage": True,
        },
        "files": files,
    }
    manifest_path = dist / "model-promotion-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files[manifest_path.name] = {
        "sha256": sha256(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }

    sums = dist / "MODEL_SHA256SUMS"
    release_files = [path for path in sorted(dist.iterdir()) if path.is_file() and path.name != sums.name]
    sums.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in release_files),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=pathlib.Path)
    parser.add_argument("--dist", required=True, type=pathlib.Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--training-run-id", required=True, type=int)
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--artifact-id", required=True, type=int)
    parser.add_argument("--artifact-digest", default="")
    parser.add_argument("--artifact-size", type=int, default=0)
    args = parser.parse_args()
    if len(args.expected_head_sha) != 40 or any(ch not in HEX for ch in args.expected_head_sha):
        raise ValueError("expected head SHA must be 40 lowercase hex")
    manifest = verify(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
