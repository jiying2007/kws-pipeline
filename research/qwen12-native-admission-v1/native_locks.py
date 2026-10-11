"""Pre-audio, resolver-compatible native locks; stdlib only, no inference.

The caller first verifies setup and loads through the retained exact loader in a
read-only runtime. Its measured receipts, not a rehash of multi-GB checkpoints,
are bound here. This module retains bounded public source/tokenizer/config bytes.
The source root must be the fresh, minimal staged /code tree, never a worktree.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import stat
import zipfile

MIB = 1024 ** 2
MAX_SOURCE = 8 * MIB
MAX_RETAINED = 64 * MIB
SHA = re.compile(r"[0-9a-f]{64}\Z")
HEAD = re.compile(r"[0-9a-f]{40}\Z")
MODELS = {
    "qwen06": ("Qwen/Qwen3-ASR-0.6B", "qwen3-asr", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"),
    "sensevoice": ("FunAudioLLM/SenseVoiceSmall", "sensevoice", "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"),
}
TOKENIZER = "chn_jpn_yue_eng_ko_spectok.bpe.model"
QWEN_RETAINED_ASSETS = frozenset(("chat_template.json", "config.json", "generation_config.json",
                                   "preprocessor_config.json", "tokenizer_config.json"))
PARSER = "research/qwen6_asr/sense_parser.py"
PARSER_SHA = "878278e46ae9af349a93e37dddffa83e916f447ef82fb6dd61fa8948bb8261d7"
QWEN_EXTENSIONS = tuple("transformers/" + name for name in (
    "generation/utils.py", "generation/configuration_utils.py", "generation/logits_process.py",
    "generation/stopping_criteria.py", "models/whisper/feature_extraction_whisper.py",
    "feature_extraction_sequence_utils.py", "feature_extraction_utils.py", "processing_utils.py",
    "audio_utils.py", "tokenization_utils_base.py", "tokenization_utils_fast.py"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(name):
    require(type(name) is str and name and "\\" not in name and "\x00" not in name,
            "invalid relative path")
    parts = name.split("/")
    require(not PurePosixPath(name).is_absolute() and all(p not in ("", ".", "..") for p in parts),
            "unsafe relative path")
    return name


def regular(path, maximum=None):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink path prohibited")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "unaliased regular file required")
    require(maximum is None or info.st_size <= maximum, "bounded public file required")
    return info


def stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_nlink,
            info.st_mtime_ns, info.st_ctime_ns)


def read(path, maximum=MAX_SOURCE):
    before = regular(path, maximum)
    with Path(path).open("rb") as stream:
        body = stream.read(maximum + 1)
    require(stamp(before) == stamp(regular(path, maximum)) and len(body) == before.st_size,
            "file changed during bounded read")
    return body


def identity(body):
    return {"bytes": len(body), "sha256": digest(body)}


def measured(row, *, allow_empty=False):
    require(type(row) is dict and type(row.get("bytes")) is int and
            row["bytes"] >= (0 if allow_empty else 1) and
            type(row.get("sha256")) is str and SHA.fullmatch(row["sha256"]), "invalid measured identity")
    return {key: row[key] for key in ("bytes", "sha256")}


def matches(body, row):
    require(identity(body) == measured(row, allow_empty=True), "measured bytes/hash drift")


def retain(output, name, body):
    path = Path(output) / relative(name)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "retained path alias")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(body)
    matches(read(path, max(MAX_SOURCE, len(body))), identity(body))


def rows_by_name(rows, field):
    require(type(rows) is list and bool(rows), "nonempty measured inventory required")
    result = {}
    for row in rows:
        require(type(row) is dict, "invalid inventory row")
        name = relative(row.get(field))
        require(name not in result, "duplicate inventory path")
        result[name] = measured(row)
    return result


def verify_installed_extension_sources(runtime, runtime_lock, selected_paths=QWEN_EXTENSIONS):
    """Compare selected installed transformers bytes to the exact retained wheel.

    Call immediately after setup.verify_runtime, before model-package imports.
    runtime_lock must be the same independently pinned runtime-lock.json consumed
    by that verifier. The wheel is streamed once; only bounded .py bodies are read.
    """
    require(type(runtime_lock) is dict and type(runtime_lock.get("files")) is list,
            "verified runtime lock required")
    candidates = [r for r in runtime_lock["files"] if r.get("name") == "transformers"]
    require(len(candidates) == 1 and candidates[0].get("kind") == "wheel", "exact transformers wheel required")
    row = candidates[0]
    name = relative(row["filename"])
    require("/" not in name and name.endswith(".whl"), "wheel filename")
    expected = measured({"bytes": row["size_bytes"], "sha256": row["sha256"]})
    require(expected["bytes"] <= 128 * MIB, "bounded transformers wheel")
    runtime = Path(runtime)
    wheel = runtime / "downloads" / name
    before = regular(wheel, 128 * MIB)
    require(before.st_size == expected["bytes"], "locked wheel size drift")
    selected = tuple(selected_paths)
    require(bool(selected) and len(set(selected)) == len(selected), "unique extension sources required")
    for member in selected:
        relative(member)
        require(member.startswith("transformers/") and member.endswith(".py"), "transformers source only")
    result = {}
    with wheel.open("rb") as stream:
        require(stamp(os.fstat(stream.fileno())) == stamp(before), "wheel changed at open")
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(MIB), b""):
            h.update(block)
        require(h.hexdigest() == expected["sha256"], "locked wheel hash drift")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            names = archive.namelist()
            for member in selected:
                require(names.count(member) == 1, "missing/duplicate locked wheel source")
                info = archive.getinfo(member)
                require(0 < info.file_size <= MAX_SOURCE and not info.is_dir(), "bounded wheel source")
                source = archive.read(info)
                installed = read(runtime / "venv/lib/python3.12/site-packages" / member)
                require(installed == source, "installed source differs from locked wheel")
                result[member] = {**identity(source), "distribution": "transformers",
                                  "wheel_sha256": expected["sha256"]}
        require(stamp(os.fstat(stream.fileno())) == stamp(before), "wheel changed during verification")
    require(stamp(regular(wheel)) == stamp(before), "wheel path replaced during verification")
    return result


def staged_sources(code_root, output, prefix):
    """Retain all staged .py bodies; empty modules have honest separate metadata.

    The existing resolver requires bytes >= 1 in files. Empty module bodies are
    still copied exactly, under empty_source_files; they are never padded/faked.
    """
    root = Path(code_root)
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), "staged code root")
    files, empty, total = [], [], 0
    for base, directories, names in os.walk(root, followlinks=False):
        require(not any((Path(base) / n).is_symlink() for n in directories + names), "staged code alias")
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = Path(base) / name
            rel = relative(path.relative_to(root).as_posix())
            body = read(path)
            total += len(body)
            require(total <= MAX_RETAINED, "staged source output budget")
            target = "retained-source/code/" + rel
            retain(output, target, body)
            row = {"path": prefix + "/" + target, **identity(body), "role": "staged-public-code"}
            (files if body else empty).append(row)
    require(bool(files), "nonempty staged Python source inventory required")
    return files, empty, total


def validate_run(executed_head_sha, run_id):
    require(type(executed_head_sha) is str and HEAD.fullmatch(executed_head_sha), "actual executed head SHA required")
    require(type(run_id) is str and re.fullmatch(r"[1-9][0-9]*", run_id), "actual workflow run ID required")


def build_native_locks(runtime, lock, output, *, family, code_root, loaded_receipt,
                       verified_runtime, executed_head_sha, run_id,
                       installed_source_inventory=None, frontend_config_bytes=None):
    """Retain source/model locks before NativeCaptureAdapter freezes config.

    Returns (source_lock_bytes, model_lock_bytes), also writes native-source-lock
    and native-asset-lock.json once. output is the worker's /output directory;
    archive member names include output/<family>/ because the host archive does.
    Sense caller must first retain canonical native-frontend-config.json from
    sense_effective, then pass those exact bytes to this helper AND the adapter.
    """
    validate_run(executed_head_sha, run_id)
    require(family in MODELS, "fixed Qwen/Sense family")
    model_id, model_family, revision = MODELS[family]
    require(lock.get("model_id") == model_id, "model lock family drift")
    assets = lock["asset_lock"]
    require(assets.get("model_id") == model_id and assets.get("revision") == revision and
            digest(canonical(assets)) == lock["asset_lock_sha256"], "original asset lock identity drift")
    original = rows_by_name(assets["files"], "filename")
    receipt = loaded_receipt["assets"]
    require(receipt.get("model_id") == model_id and receipt.get("revision") == revision and
            receipt.get("asset_lock_sha256") == lock["asset_lock_sha256"], "loaded asset receipt drift")
    require(rows_by_name(receipt["verified_files"], "filename") == original and
            rows_by_name(verified_runtime["model_assets"][family], "filename") == original and
            receipt.get("verified_bytes") == sum(r["bytes"] for r in original.values()),
            "measured setup/load asset inventories disagree")
    pins = lock["source_lock"]
    require(type(pins) is list and bool(pins) and digest(canonical(pins)) == lock["source_lock_sha256"],
            "original source pin inventory drift")
    verified = {}
    for row in verified_runtime["source_members"]:
        name = relative(row["relative_path"])
        observed = measured(row)
        require(name not in verified or verified[name] == observed, "conflicting verified runtime source")
        verified[name] = observed
    runtime, output, code_root = Path(runtime), Path(output), Path(code_root)
    prefix = "output/" + family
    files, empty, total = staged_sources(code_root, output, prefix)
    names = set()
    selected = []
    for pin in pins:
        name = relative(pin["relative_path"])
        require(name.endswith(".py") and name not in names and name in verified,
                "missing/duplicate pinned runtime source")
        require(verified[name]["sha256"] == pin["sha256"], "setup source differs from original pin")
        names.add(name)
        selected.append((name, verified[name], pin["distribution"]))
    if family == "qwen06":
        require(frontend_config_bytes is None, "Qwen does not consume SenseVoice frontend snapshot")
        require(type(installed_source_inventory) is dict and set(installed_source_inventory) == set(QWEN_EXTENSIONS),
                "verified Qwen generation/processing source extensions required")
        for name in QWEN_EXTENSIONS:
            require(name not in names, "duplicate extended runtime source")
            names.add(name)
            selected.append((name, measured(installed_source_inventory[name]), "transformers"))
    for name, expected, distribution in selected:
        body = read(runtime / "venv/lib/python3.12/site-packages" / name)
        matches(body, expected)
        total += len(body)
        require(total <= MAX_RETAINED, "retained source output budget")
        target = "retained-source/site-packages/" + name
        retain(output, target, body)
        files.append({"path": prefix + "/" + target, **identity(body),
                      "role": "installed-runtime-source", "distribution": distribution})
    model_rows = []
    model_root = runtime / "models" / family
    for name, observed in sorted(original.items()):
        path = model_root / name
        require(regular(path).st_size == observed["bytes"], "measured model asset size/type drift")
        weights = name in ("model.safetensors", "model.pt")
        target = "retained-assets/" + name
        keep_body = not weights and (family == "sensevoice" or name in QWEN_RETAINED_ASSETS)
        if keep_body:
            body = read(path)
            matches(body, observed)
            total += len(body)
            require(total <= MAX_RETAINED, "retained public dependency output budget")
            retain(output, target, body)
        model_rows.append({"path": prefix + "/" + target, **observed,
                           "kind": "weights" if weights else ("model-tokenizer-asset" if name in ("merges.txt", "vocab.json", TOKENIZER) else "model-asset"),
                           "body_retained": keep_body})
    if family == "sensevoice":
        require(type(frontend_config_bytes) is bytes and frontend_config_bytes,
                "actual pre-audio frontend config bytes required")
        require(canonical(json.loads(frontend_config_bytes)) == frontend_config_bytes,
                "canonical frontend config bytes required")
        require(read(output / "native-frontend-config.json") == frontend_config_bytes,
                "frontend retained bytes differ from adapter input")
        parser_member = prefix + "/retained-source/code/" + PARSER
        matches_parser = [r for r in files if r["path"] == parser_member and r["sha256"] == PARSER_SHA]
        require(len(matches_parser) == 1, "actual retained SenseVoice parser source required")
        model_rows.append({"path": prefix + "/native-frontend-config.json", **identity(frontend_config_bytes),
            "kind": "runtime-validation-config", "role": "sensevoice-live-frontend-validation",
            "body_retained": True})
    model_rows.sort(key=lambda row: row["path"])
    source = {"schema": "retained-asr-source-lock-v1", "executed_head_sha": executed_head_sha,
              "workflow_run_id": run_id, "model_id": model_id, "files": sorted(files, key=lambda row: row["path"]),
              "empty_source_files": sorted(empty, key=lambda row: row["path"]),
              "original_source_lock_sha256": lock["source_lock_sha256"]}
    model = {"schema": "retained-asr-model-lock-v1", "provider_id": "huggingface", "model_id": model_id,
             "model_revision": revision, "model_family": model_family, "assets": model_rows,
             "model_sha256": digest(canonical(model_rows)),
             "original_asset_lock_sha256": lock["asset_lock_sha256"],
             "asset_verification": "retained-loader-and-setup-measured-receipts; readonly-runtime; weights-not-rehashed-or-retained"}
    source_raw, model_raw = canonical(source), canonical(model)
    retain(output, "native-source-lock.json", source_raw)
    retain(output, "native-asset-lock.json", model_raw)
    return source_raw, model_raw


WHISPER_REVISION = "86098128c0b4f24f0e2aa2994de830614b474227"
WHISPER_MODEL_SHA = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
WHISPER_ASSETS = frozenset(("small.pt", "whisper/assets/gpt2.tiktoken",
                            "whisper/assets/multilingual.tiktoken", "whisper/assets/mel_filters.npz"))


def build_whisper_locks(runtime, output, *, code_root, official_source_lock,
                        verified_asset_lock, executed_head_sha, run_id):
    """Fresh Whisper wrapper helper; historical FixedWhisper stays unchanged.

    Call with actual setup-measured runtime asset-lock.json after independent
    setup verification, before the fresh wrapper freezes native config. Its
    exact loader still verifies checkpoint bytes before any model load. This
    helper never reads/copies that checkpoint or claims a load has occurred.
    """
    validate_run(executed_head_sha, run_id)
    runtime, output = Path(runtime), Path(output)
    require(official_source_lock.get("schema") == "whisper-screen-official-source-lock-v1" and
            official_source_lock.get("source_revision") == WHISPER_REVISION and
            official_source_lock.get("source_repository") == "https://github.com/openai/whisper",
            "exact official Whisper source lock required")
    require(verified_asset_lock.get("schema") == "whisper-screen-loaded-assets-lock-v1" and
            verified_asset_lock.get("model_id") == "openai/whisper-small-native" and
            set(verified_asset_lock.get("files", {})) == WHISPER_ASSETS, "measured small Whisper asset inventory required")
    asset_raw = read(runtime / "asset-lock.json")
    require(canonical(json.loads(asset_raw)) == canonical(verified_asset_lock), "runtime Whisper asset lock drift")
    assets = {name: measured(row) for name, row in verified_asset_lock["files"].items()}
    require(assets["small.pt"]["sha256"] == WHISPER_MODEL_SHA, "official small checkpoint hash required")
    require(regular(runtime / "model/small.pt").st_size == assets["small.pt"]["bytes"], "small checkpoint size/type drift")
    prefix = "output/whisper"
    files, empty, total = staged_sources(code_root, output, prefix)
    official = {}
    for row in official_source_lock["files"]:
        name = relative(row["path"])
        require(name not in official, "duplicate official Whisper source path")
        expected = measured({"bytes": row["size_bytes"], "sha256": row["sha256"]})
        body = read(runtime / "source" / name)
        matches(body, expected)
        official[name] = expected
        if name.endswith(".py"):
            target = "retained-source/official/" + name
            total += len(body)
            require(total <= MAX_RETAINED, "retained Whisper source budget")
            retain(output, target, body)
            files.append({"path": prefix + "/" + target, **identity(body), "role": "official-runtime-source",
                          "source_revision": WHISPER_REVISION})
    require(any(r["path"].endswith("/whisper/decoding.py") for r in files) and
            any(r["path"].endswith("/whisper/audio.py") for r in files), "official decoder/frontend source required")
    model_rows = []
    for name, row in sorted(assets.items()):
        weights = name == "small.pt"
        target = "retained-assets/" + name
        keep_body = not weights and name != "whisper/assets/gpt2.tiktoken"
        if not weights:
            require(official.get(name) == row, "Whisper source/asset inventory disagreement")
        if keep_body:
            body = read(runtime / "source" / name)
            matches(body, row)
            total += len(body)
            require(total <= MAX_RETAINED, "retained Whisper public dependency budget")
            retain(output, target, body)
        model_rows.append({"path": prefix + "/" + target, **row,
                           "kind": "weights" if weights else "model-asset", "body_retained": keep_body})
    source = {"schema": "retained-asr-source-lock-v1", "executed_head_sha": executed_head_sha,
              "workflow_run_id": run_id, "model_id": "openai/whisper-small-native",
              "official_source_revision": WHISPER_REVISION,
              "original_source_lock_canonical_sha256": digest(canonical(official_source_lock)),
              "files": sorted(files, key=lambda row: row["path"]),
              "empty_source_files": sorted(empty, key=lambda row: row["path"])}
    model = {"schema": "retained-asr-model-lock-v1", "provider_id": "openai",
             "model_id": "openai/whisper-small-native", "model_revision": WHISPER_REVISION,
             "model_family": "whisper", "assets": model_rows, "model_sha256": digest(canonical(model_rows)),
             "original_asset_lock_sha256": digest(asset_raw),
             "asset_verification": "setup-measured-acquisition-inventory; readonly-runtime; loader-must-reverify-before-load; weights-not-rehashed-or-retained"}
    source_raw, model_raw = canonical(source), canonical(model)
    retain(output, "native-source-lock.json", source_raw)
    retain(output, "native-asset-lock.json", model_raw)
    return source_raw, model_raw


def public_body_members(source_lock_bytes, model_lock_bytes):
    """Exact full archive-member allowlist for this helper's retained bodies.

    Native capture/config/generation JSON emitted separately remains the host's
    responsibility. This list includes both lock files, source bodies (including
    true empty modules), and assets explicitly selected before inference; it
    never includes metadata-only weights, large Qwen tokenizer data or unused
    Whisper GPT-2 tokenizer bodies.
    """
    source, model = json.loads(source_lock_bytes), json.loads(model_lock_bytes)
    require(source.get("schema") == "retained-asr-source-lock-v1" and
            model.get("schema") == "retained-asr-model-lock-v1", "retained native lock schemas")
    files = source["files"] + source.get("empty_source_files", [])
    members = [relative(row["path"]) for row in files]
    members.extend(relative(row["path"]) for row in model["assets"] if row.get("body_retained") is True)
    prefixes = {"/".join(name.split("/")[:2]) for name in members}
    require(len(prefixes) == 1 and next(iter(prefixes)) in
            ("output/qwen06", "output/sensevoice", "output/whisper"), "one native archive family")
    prefix = next(iter(prefixes))
    members.extend((prefix + "/native-source-lock.json", prefix + "/native-asset-lock.json"))
    require(len(members) == len(set(members)), "duplicate retained public body")
    return tuple(sorted(members))
