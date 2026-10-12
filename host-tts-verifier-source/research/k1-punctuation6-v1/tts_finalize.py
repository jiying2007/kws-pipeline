"""Bounded post-exit reconciliation of the fixed six TTS cells, with no calls.

Only the host may invoke this after all worker processes have terminated. This
never treats WAVs, EOS records, or unfinished atomic saves as completed calls.
Original claims remain untouched. Invalid incomplete audio and unfinished JSON
saves are moved intact into a fixed private sibling, never published as bytes.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import struct
import wave

from contract import canonical, cells, require, SHA, validate_ledger
from tts_common import atomic_json, fsync_directory, SAFE_EXCEPTIONS, unaliased
from tts_worker import new_outcomes

SCHEMA = "k1-punctuation6-tts-finalization-v1"
MAX_JSON = 256 * 1024
MAX_AUDIO = 16 * 1024 * 1024
TEMPORARIES = ("outcomes.json.tmp", "eos.json.tmp")
REASONS = {"invalid_waveform", "native_unavailable", "pair_geometry_mismatch"}


def _receipt(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _validate_receipt(value, maximum, *, empty=True):
    require(type(value) is dict and set(value) == {"bytes", "sha256"}, "exact byte receipt")
    require(type(value["bytes"]) is int and (0 if empty else 1) <= value["bytes"] <= maximum,
            "bounded receipt bytes")
    require(type(value["sha256"]) is str and SHA.fullmatch(value["sha256"]), "receipt hash")


def validate_finalization(value, rows):
    """Validate public hash/length-only evidence; private bodies are not inputs."""
    require(type(value) is dict and set(value) == {"schema", "original_outcomes", "original_eos",
        "reconciled_cells", "quarantined_audio", "preserved_temporaries", "calls_resumed",
        "completion_inferred"}, "exact finalization fields")
    require(value["schema"] == SCHEMA and value["calls_resumed"] is False and
            value["completion_inferred"] is False, "fixed no-resume finalization")
    require(type(rows) is list and len(rows) == 6 and all(type(row) is dict for row in rows) and
            [row.get("cell_id") for row in rows] == [cell["cell_id"] for cell in cells()],
            "finalization fixed six rows")
    by_id = {row["cell_id"]: row for row in rows}
    for key in ("original_outcomes", "original_eos"):
        if value[key] is not None:
            _validate_receipt(value[key], MAX_JSON, empty=False)
    reconciled = value["reconciled_cells"]
    require(type(reconciled) is list and len(reconciled) <= 6, "bounded reconciled cells")
    ids = []
    for row in reconciled:
        require(type(row) is dict and set(row) == {"cell_id", "previous_status"}, "reconciled fields")
        cell_id = row["cell_id"]
        require(type(cell_id) is str and cell_id in by_id and
                row["previous_status"] in ("not_run", "started"), "fixed previous cell state")
        require(by_id[cell_id]["status"] == "failed_no_retry" and
                type(by_id[cell_id]["attempts"]) is int and by_id[cell_id]["attempts"] == 1,
                "reconciled claim remains consumed and failed")
        ids.append(cell_id)
    require(ids == sorted(set(ids)), "unique ordered reconciled cells")
    if value["original_outcomes"] is None:
        require(not any(row.get("status") == "complete" for row in rows) and
                all(row["previous_status"] == "not_run" for row in reconciled),
                "missing original outcomes cannot establish prior completion/start metadata")
    if any(row.get("status") == "complete" for row in rows):
        require(value["original_eos"] is not None, "complete cells require original EOS evidence")
    audio = value["quarantined_audio"]
    require(type(audio) is list and len(audio) <= 12, "bounded audio quarantine")
    allowed = {folder + "/" + cell_id + ".wav": cell_id for cell_id in by_id for folder in ("raw", "derived")}
    paths = []
    for row in audio:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256", "reason"}, "audio receipt fields")
        path = row["path"]
        require(type(path) is str and path in allowed and type(row["reason"]) is str and
                row["reason"] in REASONS, "fixed quarantine path/reason")
        require(row["reason"] == "invalid_waveform" or path.startswith("derived/"), "pair quarantine is derivative only")
        target = by_id[allowed[path]]
        require(target["status"] == "failed_no_retry" and type(target["attempts"]) is int and
                target["attempts"] == 1, "only failed consumed cells may quarantine audio")
        _validate_receipt({key: row[key] for key in ("bytes", "sha256")}, MAX_AUDIO)
        paths.append(path)
    require(paths == sorted(set(paths)) and sum(row["bytes"] for row in audio) <= MAX_AUDIO,
            "unique bounded cumulative quarantine bytes")
    temporaries = value["preserved_temporaries"]
    require(type(temporaries) is list and len(temporaries) <= 2, "bounded temporary receipts")
    names = []
    for row in temporaries:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and
                type(row["path"]) is str and row["path"] in TEMPORARIES, "fixed temporary path")
        _validate_receipt({key: row[key] for key in ("bytes", "sha256")}, MAX_JSON)
        names.append(row["path"])
    require(names == sorted(set(names)), "unique ordered temporary receipts")
    require(len(canonical(value)) <= 16 * 1024, "bounded finalization metadata")
    return True


def _read(path, maximum):
    path = unaliased(path, file=True)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= maximum,
                "bounded unaliased evidence file")
        raw = handle.read(maximum + 1)
        require(len(raw) == info.st_size and len(raw) <= maximum, "stable bounded evidence read")
        return raw


def _signature(path):
    info = unaliased(path, file=True).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _optional(path, maximum):
    path = unaliased(path)
    return _read(path, maximum) if path.exists() else None


def _save_bytes(path, raw):
    fd = os.open(unaliased(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    fsync_directory(path.parent)


def _claim(cell):
    return canonical({"schema": "k1-punctuation6-tts-start-v1", "cell_id": cell["cell_id"],
        "seed": cell["seed"], "voice": cell["voice"], "text": cell["text"], "attempt": 1,
        "request_sha256": hashlib.sha256(canonical(cell)).hexdigest()})


def finalize_after_exit(output, runtime, *, worker_exited):
    """Reconcile once after verified process exit, or verify an unchanged repeat.

    A malformed/non-prefix claim or contradictory saved metadata fails closed.
    The caller must not run/resume a worker after this call, even on error.
    No private evidence is removed; quarantine is a same-filesystem rename.
    """
    from tts_semantics import read_json, raw_wave, pcm_wave, validate_eos

    require(worker_exited is True, "all worker processes must have exited")
    runtime = unaliased(Path(runtime).absolute())
    output = unaliased(Path(output).absolute())
    require(runtime.is_dir() and output.is_dir() and output == runtime / "publication-members",
            "fixed host publication/private sibling roots")
    private = unaliased(runtime / "finalization-private")
    original = {name: _optional(output / name, MAX_JSON) for name in ("outcomes.json", "eos.json")}
    result = read_json(original["outcomes.json"]) if original["outcomes.json"] is not None else new_outcomes()
    old_result = deepcopy(result)
    base = new_outcomes()
    require(type(result) is dict and set(base) <= set(result) <= set(base) | {"error_class", "finalization"},
            "fixed outcomes fields before reconciliation")
    for key in ("schema", "experiment", "human_gold", "training_admitted"):
        require(canonical(result[key]) == canonical(base[key]), "fixed outcome identity")
    require(result["status"] in ("not_run", "started", "stopped_no_retry", "completed_six_once"), "saved status")
    counts = validate_ledger("tts", result["ledger"])
    require(type(result["consumed_attempts"]) is int and result["consumed_attempts"] == counts["consumed"],
            "saved consumed accounting")
    require(type(result["cells"]) is list and len(result["cells"]) == 6, "saved six-cell denominator")
    if result["status"] == "not_run":
        require(counts["consumed"] == 0, "not_run cannot hide saved consumption")
    if result["status"] == "completed_six_once":
        require(counts["complete"] == 6, "saved completion requires six completed cells")
    if "error_class" in result:
        require(result["error_class"] in SAFE_EXCEPTIONS, "safe saved error")
    allowed_claims = {cell["cell_id"] + ".started.json" for cell in cells()}
    for directory, allowed in (("claims", allowed_claims), ("raw", {cell["cell_id"] + ".wav" for cell in cells()}),
                               ("derived", {cell["cell_id"] + ".wav" for cell in cells()})):
        path = unaliased(output / directory)
        if path.exists():
            require(path.is_dir() and {entry.name for entry in path.iterdir()} <= allowed,
                    "fixed claim/audio directory inventory")
    evidence = {"schema": SCHEMA, "original_outcomes": _receipt(original["outcomes.json"]) if original["outcomes.json"] is not None else None,
        "original_eos": _receipt(original["eos.json"]) if original["eos.json"] is not None else None,
        "reconciled_cells": [], "quarantined_audio": [], "preserved_temporaries": [],
        "calls_resumed": False, "completion_inferred": False}
    claims = {}
    audio = {}; audio_signatures = {}; moved = {}; audio_bytes = 0
    for index, (cell, row, ledger) in enumerate(zip(cells(), result["cells"], result["ledger"])):
        require(type(row) is dict and set(cell) | {"status", "attempts", "observation"} <= set(row) <=
                set(cell) | {"status", "attempts", "observation", "claim", "eos", "generation_seconds", "error_class"},
                "saved fixed cell fields")
        require(all(canonical(row[key]) == canonical(value) for key, value in cell.items()), "saved fixed cell identity")
        require(row["status"] == ledger["status"] and type(row["attempts"]) is int and row["attempts"] == ledger["attempts"],
                "saved cell/ledger agreement")
        if "error_class" in row:
            require(row["error_class"] in SAFE_EXCEPTIONS, "safe saved cell error")
        claim = _optional(output / "claims" / (cell["cell_id"] + ".started.json"), MAX_JSON)
        if claim is not None:
            require(claim == _claim(cell), "original canonical fixed request claim required")
            claims[cell["cell_id"]] = _receipt(claim)
        if row["attempts"]:
            require(claim is not None and row.get("claim") == _receipt(claim), "saved consumed claim byte identity")
        else:
            require(set(row) == set(cell) | {"status", "attempts", "observation"} and row["observation"] is None,
                    "unstarted cell metadata")
        if row["status"] != "complete":
            require(row["observation"] is None and "eos" not in row and "generation_seconds" not in row,
                    "incomplete metadata cannot claim completion")
        if claim is not None and row["status"] in ("not_run", "started"):
            evidence["reconciled_cells"].append({"cell_id": cell["cell_id"], "previous_status": row["status"]})
            row.update(status="failed_no_retry", attempts=1, claim=_receipt(claim))
            ledger.update(status="failed_no_retry", attempts=1)
        parsed = {}
        for folder, parser in (("raw", raw_wave), ("derived", pcm_wave)):
            name = folder + "/" + cell["cell_id"] + ".wav"
            path = unaliased(output / name)
            before = _signature(path) if path.exists() else None
            raw = _optional(path, MAX_AUDIO - audio_bytes)
            if raw is None:
                continue
            require(_signature(path) == before, "stable audio during bounded read")
            audio_signatures[name] = before
            audio_bytes += len(raw); audio[name] = raw
            require(claim is not None, "audio requires genuine consumed claim")
            try:
                parsed[folder] = parser(raw)
            except (ValueError, EOFError, struct.error, wave.Error):
                require(row["status"] == "failed_no_retry", "complete cell audio cannot be quarantined")
                moved[name] = "invalid_waveform"
        raw_name = "raw/" + cell["cell_id"] + ".wav"
        derived_name = "derived/" + cell["cell_id"] + ".wav"
        if "derived" in parsed and "raw" not in parsed:
            require(row["status"] == "failed_no_retry", "complete derivative requires native source")
            moved[derived_name] = "native_unavailable"
        elif "raw" in parsed and "derived" in parsed:
            if parsed["derived"]["frames"] != (parsed["raw"]["source_frames"] * 2 + 2) // 3:
                require(row["status"] == "failed_no_retry", "complete frame pair cannot be repaired")
                moved[derived_name] = "pair_geometry_mismatch"
        if row["status"] == "complete":
            require(set(parsed) == {"raw", "derived"} and type(row["observation"]) is dict,
                    "preserve complete cell only with both measured WAVs")
            require(all(row["observation"].get(key) == value for key, value in (parsed["raw"] | parsed["derived"]).items()),
                    "complete waveform byte/geometry binding")
            validate_eos(row.get("eos"))
            require(parsed["raw"]["source_frames"] == (row["eos"]["iterations"] - 1) * 1920,
                    "complete codec/native frame binding")
    counts = validate_ledger("tts", result["ledger"])
    require(list(claims) == [cell["cell_id"] for cell in cells()[:len(claims)]] and
            len(claims) == counts["consumed"], "claims must form the exact completed/failed prefix")
    result["consumed_attempts"] = counts["consumed"]
    if evidence["reconciled_cells"] or result["status"] == "started":
        result["status"] = "stopped_no_retry"
    eos = read_json(original["eos.json"]) if original["eos.json"] is not None else {"schema": "k1-punctuation6-eos-v1", "rows": []}
    require(set(eos) == {"schema", "rows"} and eos["schema"] == "k1-punctuation6-eos-v1" and
            type(eos["rows"]) is list and len(eos["rows"]) <= 6, "bounded saved EOS aggregate")
    eos_by = {}
    for row in eos["rows"]:
        require(type(row) is dict and type(row.get("cell_id")) is str and row["cell_id"] in claims and
                row["cell_id"] not in eos_by, "EOS requires unique original consumed claim")
        eos_by[row["cell_id"]] = row
    for row in result["cells"]:
        if row["status"] == "complete":
            require(eos_by.get(row["cell_id"]) == {"cell_id": row["cell_id"], **row["eos"]}, "preserve original complete EOS")
    temporaries = {name: raw for name in TEMPORARIES if (raw := _optional(output / name, MAX_JSON)) is not None}
    evidence["preserved_temporaries"] = [{"path": name, **_receipt(raw)} for name, raw in sorted(temporaries.items())]
    evidence["quarantined_audio"] = [{"path": name, **_receipt(audio[name]), "reason": reason} for name, reason in sorted(moved.items())]
    if "finalization" in old_result:
        require(result == old_result and not moved and not temporaries, "finalized evidence cannot acquire new state")
        validate_finalization(old_result["finalization"], result["cells"])
        require(not any((output / row["path"]).exists() for row in old_result["finalization"]["quarantined_audio"]),
                "quarantined audio cannot reappear in publication")
        _verify_private(private, old_result["finalization"], audio_remaining=MAX_AUDIO-audio_bytes)
        return result
    changed = result != old_result or bool(moved) or bool(temporaries) or any(raw is None for raw in original.values())
    if not changed:
        return result
    result["finalization"] = evidence
    validate_finalization(evidence, result["cells"])
    require(len(canonical(result)) <= MAX_JSON, "bounded finalized outcomes")
    require(not private.exists(), "private finalization already started; fail closed, never overwrite")
    private.mkdir(mode=0o700); fsync_directory(runtime)
    for name, raw in original.items():
        if raw is not None:
            _save_bytes(private / ("original-" + name), raw)
    for name in sorted(set(moved) | set(temporaries)):
        target = private / name
        if target.parent != private and not target.parent.exists():
            target.parent.mkdir(mode=0o700); fsync_directory(private)
        require(not target.exists(), "preserved evidence cannot be overwritten")
        source = unaliased(output / name, file=True)
        expected = audio[name] if name in moved else temporaries[name]
        if name in moved:
            require(_signature(source) == audio_signatures[name], "audio unchanged before preservation")
        else:
            require(_read(source, MAX_JSON) == expected, "temporary unchanged before preservation")
        os.rename(source, unaliased(target))
        if name in moved:
            require(_signature(target) == audio_signatures[name], "same audio inode preserved by rename")
            fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fsync_directory(target.parent); fsync_directory(source.parent)
    if original["eos.json"] is None:
        atomic_json(output / "eos.json", eos)
    atomic_json(output / "outcomes.json", result)
    _verify_private(private, evidence, audio_signatures=audio_signatures)
    return result


def _verify_private(private, evidence, *, audio_remaining=MAX_AUDIO, audio_signatures=None):
    require(private.is_dir(), "private original evidence retained")
    for name in ("outcomes", "eos"):
        receipt = evidence["original_" + name]
        path = private / ("original-" + name + ".json")
        if receipt is not None:
            require(_receipt(_read(path, MAX_JSON)) == receipt, "original metadata preserved exactly")
        else:
            require(not path.exists(), "absent original cannot be invented")
    for row in evidence["quarantined_audio"] + evidence["preserved_temporaries"]:
        path = private / row["path"]
        if row["path"].endswith(".wav"):
            if audio_signatures is not None:
                require(_signature(path) == audio_signatures[row["path"]], "same preserved audio inode")
                continue
            maximum = audio_remaining
            audio_remaining -= row["bytes"]
            require(audio_remaining >= 0, "bounded total private/public audio read")
        else:
            maximum = MAX_JSON
        require(_receipt(_read(path, maximum)) == {key: row[key] for key in ("bytes", "sha256")},
                "quarantined bytes preserved exactly")
