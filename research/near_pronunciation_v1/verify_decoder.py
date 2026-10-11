#!/usr/bin/env python3
"""Compile/test the pinned research decoder using existing CC and stdlib only.

No model file, acoustic inference, replay campaign, package install or network.
All executed input frames are purpose-built synthetic unit-test logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from prepare_decoder import digest, materialize

STRICT = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
          "-Wconversion", "-Wshadow", "-Wcast-qual"]
GUARD = ["-DKWS_RESEARCH_CHILD_ADMISSION=1"]
SUPPORT = ["src/frontend.c", "src/model.c", "src/keyword_pack.c"]
TOOL_SUPPORT = ["tools/tool_io.c", "tools/sha256.c", "tools/kws_trace_io.c"]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, **kwargs)


def dependency_closure(flags: list[str], source: Path, allowed: dict[Path, str]) -> dict[str, str]:
    # Ask the same compiler/preprocessor that will build the unit. -MM includes
    # local quote/include resolution but omits installed system headers. A
    # source-local header wins over -I ordering; never trust ordering alone.
    raw = run([*flags, "-MM", "-MT", "research_dependencies", str(source)]).stdout.decode()
    target, separator, dependencies = raw.replace("\\\n", "").partition(":")
    if separator != ":" or target != "research_dependencies":
        raise ValueError("unrecognized compiler dependency output")
    found = {}
    for name in shlex.split(dependencies, comments=False, posix=True):
        path = Path(name).resolve(strict=True)
        if path not in allowed or digest(path.read_bytes()) != allowed[path]:
            raise ValueError(f"unbound or changed compiler dependency: {path}")
        found[str(path)] = allowed[path]
    if str(source.resolve(strict=True)) not in found:
        raise ValueError("compiler dependency output omitted the translation unit")
    return found


def stream_hash(executable: Path) -> str:
    hasher = hashlib.sha256()
    with subprocess.Popen([str(executable)], stdout=subprocess.PIPE) as proc:
        if proc.stdout is None:
            raise ValueError("parity probe stdout unavailable")
        while block := proc.stdout.read(1024 * 1024):
            hasher.update(block)
        if proc.wait() != 0:
            raise ValueError(f"parity probe failed: {executable.name}")
    return hasher.hexdigest()


def verify(root: Path, output: Path, compiler: str) -> dict:
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("verification output already exists; use a fresh directory")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise ValueError("requested compiler is not installed; no installation attempted")
    output.mkdir(parents=True)
    derived = output / "derived"
    source_receipt = materialize(root, derived)
    here = Path(__file__).resolve().parent
    generated = output / "generated"
    run([sys.executable, str(root / "tools/gen_parameter_limits.py"),
         str(root / "configs/parameter-contract.json"), str(generated)])
    common = [f"-I{generated}", f"-I{root / 'include'}",
              f"-I{root / 'src'}", f"-I{root / 'tools'}"]
    derived_includes = [f"-I{derived / 'src'}", f"-I{derived / 'tools'}", *common]
    commands = []
    allowed = {(root / name).resolve(): sha for name, sha in source_receipt["original_source_sha256"].items()}
    allowed.update({(derived / name).resolve(): sha for name, sha in source_receipt["derived_source_sha256"].items()})
    allowed.update({path.resolve(): digest(path.read_bytes()) for path in here.glob("test_*.c")})
    generated_header = generated / "kws_parameter_limits.h"
    allowed[generated_header.resolve()] = digest(generated_header.read_bytes())
    resolved_dependencies = {}
    def build(name: str, sources: list[Path], research=False, original=False,
              extra: list[str] | None = None) -> Path:
        path = output / name
        flags = [compiler_path, *STRICT, *(GUARD if research else []),
                 *(common if original else derived_includes), *(extra or [])]
        for source in sources:
            resolved_dependencies.update(dependency_closure(flags, source, allowed))
        command = [*flags, *map(str, sources), "-lm", "-o", str(path)]
        run(command)
        commands.append(command)
        return path
    objects = {}
    for source in ("src/decoder.c", "src/kws.c", "tools/kws_decoder_replay.c"):
        for optimization in ("-O0", "-O2", "-Os"):
            baseline = build(Path(source).name + optimization + ".original.o",
                             [root / source], original=True, extra=[optimization, "-c"])
            guarded_off = build(Path(source).name + optimization + ".guard-off.o",
                                [derived / source], extra=[optimization, "-c"])
            if baseline.read_bytes() != guarded_off.read_bytes():
                raise ValueError(f"guard-off object changed: {source} {optimization}")
            objects[source + ":" + optimization] = digest(baseline.read_bytes())
    tests = {}
    for source in ("tests/test_decoder.c", "tests/test_decoder_retention.c"):
        for mode in ("original", "guard-off", "research-default"):
            original = mode == "original"
            executable = build(Path(source).stem + "-" + mode,
                               [(root if original else derived) / "src/decoder.c", root / source],
                               research=mode == "research-default", original=original)
            tests[executable.name] = run([str(executable)]).stdout.decode()
    focused = build("test-child-admission", [derived / "src/decoder.c", here / "test_child_admission.c"],
                    research=True)
    tests[focused.name] = run([str(focused)]).stdout.decode()
    core = [derived / "src/kws.c", derived / "src/decoder.c", *[root / x for x in SUPPORT]]
    tools = [root / x for x in TOOL_SUPPORT]
    contract = build("test-replay-contract", [*core, *tools, here / "test_replay_contract.c"], research=True)
    tests[contract.name] = run([str(contract)]).stdout.decode()
    replay = build("kws_decoder_replay_research", [*core, *tools, derived / "tools/kws_decoder_replay.c"],
                   research=True)
    bad_args = [[], ["--child-admission", "wrong"],
                ["--child-admission", "strict_children", "--child-admission", "single_fuzzy_wo"],
                ["--child-admission", "single_fuzzy_wo", "--fuzzy-child-cost-log", "-16"],
                ["--state-retention", "0.94"]]
    for args in bad_args:
        result = subprocess.run([str(replay), "missing-model", "missing-pack", "missing-trace", "unit", *args],
                                capture_output=True)
        if result.returncode != 2 or b"cannot open" in result.stderr:
            raise ValueError("CLI did not reject mode/override before data access")
    tests["cli_fail_closed"] = "missing, unknown, duplicate and override modes rejected before input access"
    hashes = {}
    sizes = {}
    for mode in ("original", "guard-off", "research-default"):
        original = mode == "original"
        selected = root if original else derived
        probe = build("parity-" + mode, [selected / "src/decoder.c", here / "test_default_parity.c"],
                      research=mode == "research-default", original=original)
        hashes[mode] = stream_hash(probe)
        layout = build("layout-" + mode,
                       [selected / "src/decoder.c", *[root / x for x in SUPPORT], here / "test_resource_layout.c"],
                       research=mode == "research-default", original=original)
        sizes[mode] = json.loads(run([str(layout)]).stdout)
    if len(set(hashes.values())) != 1:
        raise ValueError("default events or original state bytes changed")
    if sizes["original"] != sizes["guard-off"]:
        raise ValueError("guard-off ABI/resource layout changed")
    for key in sizes["original"]:
        if key.startswith("public_") and sizes["original"][key] != sizes["research-default"][key]:
            raise ValueError("research build changed public ABI layout")
    # Negative preparer tests: existing outputs and source drift are fatal.
    try:
        materialize(root, derived)
    except ValueError:
        pass
    else:
        raise ValueError("preparer allowed seeded output")
    drift = output / "drift-source"
    for name in source_receipt["original_source_sha256"]:
        dest = drift / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    with (drift / "src/decoder.c").open("ab") as stream:
        stream.write(b"\n/* deliberate source-drift unit fixture */\n")
    for optimization in ([], ["-O"], ["-OO"]):
        failure_dir = output / ("refused-drift" + "".join(optimization))
        result = subprocess.run([sys.executable, *optimization, str(here / "prepare_decoder.py"),
                                 "--source-root", str(drift), "--output-dir", str(failure_dir)],
                                capture_output=True)
        if result.returncode != 2 or failure_dir.exists():
            raise ValueError("source-drift rejection failed")
    tests["preparation_fail_closed"] = "seeded output and drift rejected in Python normal/-O/-OO"
    # An untracked source-local quote include must not silently shadow the
    # freshly generated contract header, even though all pinned files match.
    shadow = output / "shadow-source"
    shadow_allowed = dict(allowed)
    for name, sha in source_receipt["original_source_sha256"].items():
        dest = shadow / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
        shadow_allowed[dest.resolve()] = sha
    (shadow / "src/kws_parameter_limits.h").write_text("/* forbidden shadow fixture */\n")
    try:
        dependency_closure([compiler_path, *STRICT, f"-I{generated}",
                            f"-I{shadow / 'include'}", f"-I{shadow / 'src'}"],
                           shadow / "src/decoder.c", shadow_allowed)
    except ValueError as exc:
        if "unbound or changed compiler dependency" not in str(exc):
            raise
    else:
        raise ValueError("dependency closure allowed an unbound shadow header")
    tests["dependency_closure_fail_closed"] = "untracked quote-header shadow rejected before compilation"
    # No unrequested sanitizer/environment installation; use the installed CC.
    sanitizer = build("test-child-admission-sanitized",
                      [derived / "src/decoder.c", here / "test_child_admission.c"], research=True,
                      extra=["-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
    tests[sanitizer.name] = run([str(sanitizer)], env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}).stdout.decode()
    final_hashes = {name: digest((root / name).read_bytes())
                    for name in source_receipt["original_source_sha256"]}
    if final_hashes != source_receipt["original_source_sha256"]:
        raise ValueError("shipping source changed during verification")
    for path, expected in allowed.items():
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"bound build input changed during verification: {path}")
    receipt = {
        **source_receipt,
        "kind": "offline_research_decoder_verification",
        "compiler_path": compiler_path,
        "compiler_version": run([compiler_path, "--version"]).stdout.decode().splitlines()[0],
        "compile_commands": commands,
        "resolved_local_dependency_sha256": resolved_dependencies,
        "guard_off_object_parity_sha256": objects,
        "default_20000_frame_state_and_event_sha256": hashes,
        "resource_layout": sizes,
        "tests": tests,
        "research_replay_executable_sha256": digest(replay.read_bytes()),
        "research_replay_executable": str(replay),
        "verification_inputs_sha256": {p.name: digest(p.read_bytes()) for p in sorted(here.glob("test_*.c"))},
        "verifier_sha256": digest(Path(__file__).read_bytes()),
        "model_file_opened": False,
        "acoustic_inference_run": False,
        "acoustic_campaign_run": False,
        "board_resource_claim": False,
        "leak_sanitizer_run": False,
        "sanitizer_scope": "AddressSanitizer and UndefinedBehaviorSanitizer; no leak claim",
    }
    (output / "verification-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args()
    try:
        receipt = verify(args.source_root, args.output_dir, args.cc)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            sys.stderr.write(exc.stderr.decode(errors="replace"))
        parser.exit(2, f"decoder verification failed: {exc}\n")
    print(json.dumps({"result": "pass", "tests": receipt["tests"],
                      "resource_layout": receipt["resource_layout"],
                      "receipt": str(args.output_dir / "verification-receipt.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
