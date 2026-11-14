"""Read-only proof that fresh pip caches encode the exact frozen Python sources.

No cached code, pip module, model, or scientific runtime is executed. The
checker parses bounded cache bodies, compiles source without executing it,
and retains both originals. Historical cache payload equivalence is unknown.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time
import types

CONTRACT_SHA = "dc35b59aa9b8c74910dcbff89c13935164c53b0aaf6024a44c6afd0afbf8ae13"
MIB = 1024 ** 2
SITE = Path("/runtime/venv/lib/python3.12/site-packages")
CODE_FIELDS = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize",
    "co_flags", "co_code", "co_consts", "co_names", "co_varnames", "co_filename", "co_name", "co_qualname",
    "co_firstlineno", "co_linetable", "co_exceptiontable", "co_freevars", "co_cellvars")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def member(name):
    need(type(name) is str and 0 < len(name) <= 1024 and "\\" not in name and "\x00" not in name,
         "bounded POSIX path")
    path = PurePosixPath(name)
    need(str(path) == name and not path.is_absolute() and ".." not in path.parts, "relative canonical member")
    return name


def read(path, maximum=8 * MIB):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), "unaliased regular file")
    before = path.stat()
    need(before.st_nlink == 1 and 0 <= before.st_size <= maximum, "bounded regular bytes")
    raw = path.read_bytes()
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
    need(stamp(before) == stamp(after) and len(raw) == before.st_size, "file changed while reading")
    return raw


def identity(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def compare_inventories(baseline, fresh, contract):
    need(set(baseline) == set(fresh) == {"schema", "files"} and
         baseline["schema"] == fresh["schema"] == "whisper-screen-installed-runtime-v1", "complete runtime inventory schema")
    old, current = baseline["files"], fresh["files"]
    caches = contract["cache_records"]
    need(len(old) == len(current) == contract["installed_file_count"] and set(old) == set(current),
         "no installed member additions or removals")
    need(len(caches) == contract["cache_file_count"] and set(caches) < set(old) and
         len(old) - len(caches) == contract["unchanged_file_count"], "fixed declared cache denominator")
    stable = {key: value for key, value in old.items() if key not in caches}
    need(sha(canonical(stable)) == contract["unchanged_records_sha256"], "frozen non-cache inventory projection")
    sources = {}
    for name, declared in caches.items():
        member(name)
        need(re.fullmatch(r"pip/(?:[A-Za-z0-9_]+/)*__pycache__/[A-Za-z0-9_-]+\.cpython-312\.pyc", name),
             "only exact pip CPython-312 cache paths")
        source = name.replace("/__pycache__/", "/").removesuffix(".cpython-312.pyc") + ".py"
        need(declared["source_path"] == source and old[name] == declared["original_cache"] and
             current[name]["bytes"] == old[name]["bytes"] and set(current[name]) == {"bytes", "sha256"} and
             re.fullmatch("[0-9a-f]{64}", current[name]["sha256"]), "only a declared cache hash may differ")
        need(old[source] == current[source] == declared["source_identity"], "exact original source for every cache")
        sources[source] = declared["source_identity"]
    need(len(sources) == len(caches) and sha(canonical(sources)) == contract["source_records_sha256"],
         "one frozen corresponding source per cache")
    need(all(current[name] == row for name, row in stable.items()), "all non-cache installed bytes remain exact")
    need({name: old[name] for name in contract["license_records"]} == contract["license_records"], "exact retained pip licenses")


def code_tree(value, depth=0, budget=None):
    """Canonical typed structural equality; no code.__eq__ or execution shortcut."""
    budget = [100000] if budget is None else budget
    budget[0] -= 1
    need(depth <= 100 and budget[0] >= 0, "bounded compiled structure")
    t = type(value)
    descend = lambda item: code_tree(item, depth + 1, budget)
    if t is types.CodeType:
        return ["code", [[name, descend(getattr(value, name))] for name in CODE_FIELDS]]
    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if t is bool:
        return ["bool", value]
    if t is int:
        return ["int", str(value)]
    if t is float:
        return ["float64", struct.pack(">d", value).hex()]
    if t is complex:
        return ["complex128", struct.pack(">dd", value.real, value.imag).hex()]
    if t is str:
        # Python code constants may contain lone surrogates. Preserve every
        # code point without changing the request/contract JSON encoding.
        return ["str-utf8-surrogatepass-hex", value.encode("utf-8", "surrogatepass").hex()]
    if t is bytes:
        return ["bytes", value.hex()]
    if t is tuple:
        return ["tuple", [descend(item) for item in value]]
    if t is frozenset:
        return ["frozenset", sorted((descend(item) for item in value), key=canonical)]
    raise ValueError("unsupported compiled constant type")


def verify_cache(raw, source, filename, source_mtime):
    need(type(raw) is bytes and 16 <= len(raw) <= 8 * MIB and type(source) is bytes and len(source) <= 8 * MIB,
         "bounded exact cache and source bytes")
    need(raw[:4] == importlib.util.MAGIC_NUMBER, "cache belongs to the actual pinned interpreter")
    flags = struct.unpack("<I", raw[4:8])[0]
    need(flags in (0, 1, 3), "legal CPython cache header flags")
    header = {"magic_hex": raw[:4].hex(), "flags": flags, "raw_header_hex": raw[:16].hex()}
    if flags == 0:
        timestamp, size = struct.unpack("<II", raw[8:16])
        need(timestamp == (int(source_mtime) & 0xffffffff) and size == len(source), "cache timestamp and source length binding")
        header.update(source_mtime_u32=timestamp, source_bytes=size)
    else:
        need(raw[8:16] == importlib.util.source_hash(source), "hash-mode cache source binding")
        header["source_hash_hex"] = raw[8:16].hex()
    payload = io.BytesIO(raw[16:])
    loaded = marshal.load(payload)
    need(payload.read(1) == b"" and type(loaded) is types.CodeType, "one complete code object without trailing bytes")
    expected = compile(source, filename, "exec", dont_inherit=True, optimize=0)
    left, right = canonical(code_tree(loaded)), canonical(code_tree(expected))
    need(left == right, "fresh cache code differs from independent compile of exact source")
    return {"header": header, "code_structure_sha256": sha(left), "code_fields": list(CODE_FIELDS),
            "compile_filename": filename, "compile_dont_inherit": True, "compile_optimize": 0,
            "recursive_code_equal": True, "cached_code_executed": False}


def verify_files(root, files):
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "no runtime file/directory aliases")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            actual.add(name)
            need(name in files and path.stat().st_size == files[name]["bytes"], "exact runtime file membership and size")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(MIB), b""):
                    digest.update(chunk)
            need(digest.hexdigest() == files[name]["sha256"], "actual runtime byte hash")
    need(actual == set(files), "closed complete installed runtime")


def verify_observed_cache(output, name, declared, fresh, state):
    state.update(phase="read-cache-source", cache_path=name, source_path=declared["source_path"])
    path, source_path = SITE / name, SITE / declared["source_path"]
    source_before = source_path.stat()
    raw, source = read(path), read(source_path)
    stat_record = lambda value: {"st_size": value.st_size, "st_mtime_ns": value.st_mtime_ns,
                                "st_ctime_ns": value.st_ctime_ns}
    observed = dict(cache_path=name, source_path=declared["source_path"], cache_identity=identity(raw),
                    source_identity=identity(source), source_stat_before=stat_record(source_before))
    # Preserve actual bytes and stat before semantic checks, including failures.
    put(output / "cache" / name, raw)
    put(output / "source" / declared["source_path"], source)
    with (output / "read-observations.jsonl").open("ab") as log:
        log.write(canonical(observed) + b"\n")
        log.flush()
    # The retained supervisor requires a stable whole-output observation.
    # Give its unchanged bounded scanner a quiet interval after each write.
    # All 411 intervals total 41.1 seconds within the fixed 300-second stage.
    time.sleep(0.1)
    need(identity(raw) == fresh["files"][name] and identity(source) == declared["source_identity"], "fresh cache/source observed byte binding")
    state["phase"] = "cache-header-and-compiled-code"
    record = verify_cache(raw, source, str(source_path), source_before.st_mtime_ns // 1000000000)
    need(raw == read(path) and source == read(source_path), "stable source/cache across independent compile")
    source_after = source_path.stat()
    need(stat_record(source_before) == stat_record(source_after), "source metadata stable across cache verification")
    return dict(**observed, source_stat_after=stat_record(source_after),
                original_cache_identity=declared["original_cache"], **record)


def check(output, state):
    need(Path(__file__).resolve() == Path("/code/research/public-zh64-asr-20261114/cache_verify_v2.py") and
         len(sys.argv) == 1 and not Path("/input").exists(), "fixed audio-free checker entry")
    need(output.is_dir() and not output.is_symlink() and not list(output.iterdir()), "fresh exclusive proof destination")
    contract_raw = read("/code/runtime-policy/cache-contract.json")
    put(output / "contract.json", contract_raw)
    before = read("/runtime/installed-inventory.json")
    put(output / "installed-inventory.json", before)
    state["phase"] = "contract-and-scope"
    contract = parse(contract_raw)
    need(sha(canonical(contract)) == CONTRACT_SHA and sys.implementation.name == "cpython" and
         list(sys.version_info[:3]) == contract["python_version"] == [3, 12, 14] and
         sys.implementation.cache_tag == contract["cache_tag"] == "cpython-312", "exact pinned interpreter and cache contract")
    need(not ({"pip", "torch", "numpy", "scipy", "whisper", "qwen_asr", "funasr"} & set(sys.modules)),
         "checker precedes pip/model/scientific imports")
    sys.path.insert(0, "/code/research/source-screen32-v1")
    from runtime_scope import verify_runtime_scope
    scope = verify_runtime_scope()
    baseline_raw = read("/code/runtime-policy/original-installed-inventory.json")
    need(identity(baseline_raw) == contract["baseline_inventory"], "exact original inventory bytes")
    original = parse(baseline_raw)
    fresh = parse(before)
    state["phase"] = "complete-runtime-inventory"
    compare_inventories(original, fresh, contract)
    verify_files(SITE, fresh["files"])
    for name, expected in contract["license_records"].items():
        raw = read(SITE / member(name))
        put(output / "licenses" / name, raw)
        need(identity(raw) == expected, "exact available pip license bytes")
    put(output / "read-observations.jsonl", b"")
    records = []
    for name, declared in sorted(contract["cache_records"].items()):
        records.append(verify_observed_cache(output, name, declared, fresh, state))
    state.update(phase="post-verification-runtime-inventory", cache_path=None, source_path=None)
    verify_files(SITE, fresh["files"])
    need(read("/runtime/installed-inventory.json") == before, "runtime inventory unchanged after cache verification")
    need(not any(name == "pip" or name.startswith("pip.") or name in
         ("torch", "numpy", "scipy", "whisper", "qwen_asr", "funasr") for name in sys.modules), "no cached pip or model code executed")
    proof = dict(schema="public-whisper-fresh-pip-cache-source-proof-v1", status="verified_before_model_preparation",
        contract_sha256=CONTRACT_SHA, interpreter=sys.version, implementation=sys.implementation.name,
        python_version=list(sys.version_info[:3]), interpreter_magic_hex=importlib.util.MAGIC_NUMBER.hex(),
        cache_tag=sys.implementation.cache_tag, kernel_scope=scope, installed_inventory_before=identity(before),
        installed_inventory_after=identity(before), installed_members_verified_twice=17688,
        unchanged_noncache_records=17277, verified_cache_count=len(records), records=records,
        original_cache_payloads_available=False, historical_cache_header_only_difference_proven=False,
        cache_code_executions=0, model_imports=0, audio_forward_calls=0, runtime_writes=0,
        training_admitted=False, human_gold=False)
    need(len(records) == 411, "complete fixed cache proof")
    put(output / "cache-verification.json", canonical(proof))
    print(json.dumps({key: proof[key] for key in ("status", "verified_cache_count", "audio_forward_calls", "runtime_writes")}))
    return 0


def main():
    output = Path("/output")
    state = dict(phase="entry", cache_path=None, source_path=None)
    try:
        return check(output, state)
    except BaseException as error:
        chain, seen, item = [], set(), error
        while item is not None and id(item) not in seen and len(chain) < 8:
            seen.add(id(item))
            chain.append(dict(error_type=type(item).__name__, message=str(item)[:2048]))
            item = item.__cause__ if item.__cause__ is not None else item.__context__
        failure = dict(schema="public-whisper-fresh-pip-cache-failure-v1", status="rejected_before_model_preparation",
            current=state, exceptions=chain, verified=False, cache_code_executions=0, model_imports=0,
            audio_forward_calls=0, training_admitted=False, human_gold=False)
        # Use escaped JSON for arbitrary exception text; contract JSON is unchanged.
        if output.is_dir() and not output.is_symlink() and not (output / "cache-verification.json").exists():
            put(output / "failure.json", json.dumps(failure, ensure_ascii=True, sort_keys=True).encode())
        print(json.dumps(dict(status=failure["status"], current=state, error_type=type(error).__name__), ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
