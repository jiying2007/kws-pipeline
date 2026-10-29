#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import pathlib
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint  # noqa: E402
from validate_shipping_keywords import validate_shipping_keywords  # noqa: E402


def compile_one(tokens: pathlib.Path, keywords: pathlib.Path, root: pathlib.Path):
    header = root / "keywords.h"
    manifest = root / "keywords.json"
    pack = root / "keywords.kwk"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools" / "compile_keywords.py"),
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--out-header",
            str(header),
            "--out-json",
            str(manifest),
            "--out-pack",
            str(pack),
        ]
    )
    return header, manifest, pack


def verify_threshold_wire(root: pathlib.Path) -> None:
    """Strict compiler/loader checks; only a vocabulary descriptor, no model."""
    root.mkdir(parents=True)
    tokens = root / "tokens.txt"
    token_bytes = b"<blk> 0\ntest 1\n"
    tokens.write_bytes(token_bytes)
    generated = root / "generated"
    subprocess.run(
        [sys.executable, "-S", "-B", str(ROOT / "tools/gen_parameter_limits.py"),
         str(ROOT / "configs/parameter-contract.json"), str(generated)],
        check=True, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    harness = root / "threshold.c"
    harness.write_text(r'''#include "kws_pipeline/kws.h"
#include "kws_parameter_limits.h"
#include "keywords.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
  unsigned char blob[72];
  kws_model_t vocabulary = {0};
  kws_keyword_pack_t pack;
  uint32_t header_bits = 0u, pack_bits = 0u;
  FILE *stream;
  kws_status_t status;
  if (argc != 2 || (stream = fopen(argv[1], "rb")) == NULL) return 20;
  if (fread(blob, 1u, sizeof(blob), stream) != sizeof(blob) ||
      fgetc(stream) != EOF || ferror(stream)) { fclose(stream); return 21; }
  if (fclose(stream) != 0) return 22;
  vocabulary.vocab_size = 2u;
  vocabulary.vocab_fingerprint = kws_generated_vocab_fingerprint;
  status = kws_keyword_pack_open(blob, sizeof(blob), &vocabulary, &pack);
  memcpy(&header_bits, &kws_generated_keywords[0].threshold, sizeof(header_bits));
  if (status == KWS_OK) memcpy(&pack_bits, &pack.keywords[0].threshold, sizeof(pack_bits));
  printf("%d %" PRIu32 " %" PRIu32 " %d\n", (int)status,
         header_bits, pack_bits,
         KWS_PARAM_THRESHOLD_VALID(kws_generated_keywords[0].threshold) ? 1 : 0);
  return 0;
}
''', encoding="utf-8")
    failures: list[str] = []
    cases: list[dict] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    values = [("usual", "0.55", True), ("decimal", "0.1", True),
              ("precision", "0.12345678", True),
              ("header_high", "0.9999999", True),
              ("header_low", "1e-7", True),
              ("max_below_one", "0.999999940395355224609375", True),
              ("min_subnormal", "1.401298464324817e-45", True),
              ("rounds_to_one", "0.99999999", False),
              ("rounds_to_zero", "1e-50", False),
              ("zero", "0", False), ("one", "1", False),
              ("nan", "nan", False), ("infinity", "inf", False),
              ("preserve_outputs", "1e-50", False)]
    for name, raw, accepted in values:
        directory = root / name
        directory.mkdir()
        keyword_bytes = f"1\ttest\t{raw}\ttest\n".encode()
        keywords = directory / "keywords.tsv"
        keywords.write_bytes(keyword_bytes)
        header, pack, manifest = [directory / f"keywords.{suffix}"
                                  for suffix in ("h", "kwk", "json")]
        sentinel = b"existing output must survive rejected input\n"
        if name == "preserve_outputs":
            for output in (header, pack, manifest):
                output.write_bytes(sentinel)
        command = [sys.executable, "-S", "-B", str(ROOT / "tools/compile_keywords.py"),
                   "--tokens", str(tokens), "--keywords", str(keywords),
                   "--out-header", str(header), "--out-pack", str(pack),
                   "--out-json", str(manifest)]
        completed = subprocess.run(command, capture_output=True, timeout=30)
        record = {"case": name, "threshold": raw, "expected_accepted": accepted,
                  "compiler_command": command, "compiler_exit": completed.returncode}
        (directory / "compiler.stdout").write_bytes(completed.stdout)
        (directory / "compiler.stderr").write_bytes(completed.stderr)
        check(tokens.read_bytes() == token_bytes and keywords.read_bytes() == keyword_bytes,
              f"{name}: compiler modified input")
        if not accepted:
            check(completed.returncode != 0, f"{name}: invalid wire threshold accepted")
            check(b"threshold" in completed.stderr.lower(), f"{name}: missing diagnostic")
            for output in (header, pack, manifest):
                if name == "preserve_outputs":
                    check(output.read_bytes() == sentinel, f"{name}: existing {output.name} changed")
                else:
                    check(not output.exists(), f"{name}: invalid input wrote {output.name}")
        elif completed.returncode != 0:
            check(False, f"{name}: valid threshold rejected")
        else:
            expected_bits = struct.unpack("<I", struct.pack("<f", float(raw)))[0]
            wire = pack.read_bytes()
            check(len(wire) == 72, f"{name}: unexpected pack length")
            actual_bits = struct.unpack_from("<I", wire, 28)[0]
            check(actual_bits == expected_bits, f"{name}: pack threshold changed")
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            check(metadata["keywords"][0]["threshold"] == float(raw),
                  f"{name}: JSON input-value compatibility changed")
            compiler = shlex.split(os.environ.get("CC", "cc"))
            build = compiler + ["-std=c11", "-Wall", "-Wextra", "-Werror",
                                "-I", str(ROOT / "include"), "-I", str(generated),
                                "-I", str(directory), str(harness),
                                str(ROOT / "src/keyword_pack.c"), "-lm",
                                "-o", str(directory / "probe")]
            compiled = subprocess.run(build, capture_output=True, timeout=30)
            record.update(build_command=build, build_exit=compiled.returncode)
            (directory / "build.stdout").write_bytes(compiled.stdout)
            (directory / "build.stderr").write_bytes(compiled.stderr)
            check(compiled.returncode == 0, f"{name}: generated header failed compilation")
            if compiled.returncode == 0:
                probe = subprocess.run([str(directory / "probe"), str(pack)],
                                       capture_output=True, text=True, timeout=30)
                record.update(probe_exit=probe.returncode, probe_stdout=probe.stdout,
                              probe_stderr=probe.stderr, expected_bits=expected_bits)
                check(probe.returncode == 0, f"{name}: probe failed")
                fields = probe.stdout.strip().split()
                check(fields == ["0", str(expected_bits), str(expected_bits), "1"],
                      f"{name}: loader/header must match binary32 bits, got {fields}")
        cases.append(record)
    report = {"cases": cases, "failures": failures, "model_execution": False,
              "scope": "compiler, real pack parser and C-header validation; no engine/inference"}
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"keyword threshold wire: {len(cases)} cases, {len(failures)} failed checks")
    if failures:
        raise AssertionError("keyword threshold wire contract: " + "; ".join(failures))



def verify_output_aliases(root: pathlib.Path) -> None:
    """Real CLI fixtures: all aliases fail before any input or output changes."""
    root.mkdir()
    contract_bytes = (ROOT / "configs/parameter-contract.json").read_bytes()
    baseline_outputs = None
    checked = 0

    def fixture(name: str):
        case = root / name
        case.mkdir()
        inputs = {"tokens": case / "tokens.txt", "keywords": case / "keywords.tsv",
                  "contract": case / "contract.json"}
        inputs["tokens"].write_bytes(b"<blk> 0\ntest 1\n")
        inputs["keywords"].write_bytes(b"1\ttest\t0.55\ttest\n")
        inputs["contract"].write_bytes(contract_bytes)
        outputs = {"out-header": case / "keywords.h", "out-pack": case / "keywords.kwk",
                   "out-json": case / "keywords.json"}
        for path in outputs.values():
            path.write_bytes(b"pre-existing artifact\n")
        return case, inputs, outputs

    def alias(case: pathlib.Path, target: pathlib.Path, form: str) -> pathlib.Path:
        if form == "same":
            return target
        if form == "normalized":
            (case / "child").mkdir()
            return case / "child" / ".." / target.name
        result = case / "alias"
        if form == "symlink":
            result.symlink_to(target)
        else:
            result.hardlink_to(target)
        return result

    def run(inputs, outputs):
        command = [sys.executable, "-S", "-B", str(ROOT / "tools/compile_keywords.py")]
        for key, value in {**inputs, **outputs}.items():
            command.extend(["--" + key, str(value)])
        return subprocess.run(command, capture_output=True, timeout=30)

    def rejected(inputs, outputs, watched, label: str, diagnostic: bytes = b"alias"):
        nonlocal checked
        before = {path: path.read_bytes() for path in watched}
        completed = run(inputs, outputs)
        if completed.returncode == 0 or diagnostic not in completed.stderr.lower():
            raise AssertionError(f"{label}: alias/identity error accepted: {completed.stderr!r}")
        if completed.stdout:
            raise AssertionError(f"{label}: rejection emitted success output")
        for path, data in before.items():
            if path.read_bytes() != data:
                raise AssertionError(f"{label}: rejected compilation changed {path.name}")
        checked += 1

    for output_key in ("out-header", "out-pack", "out-json"):
        for input_key in ("tokens", "keywords", "contract"):
            for form in ("same", "normalized", "symlink", "hardlink"):
                label = f"{output_key}-{input_key}-{form}"
                case, inputs, outputs = fixture(label)
                watched = list(inputs.values()) + list(outputs.values())
                outputs[output_key] = alias(case, inputs[input_key], form)
                rejected(inputs, outputs, watched, label)

    keys = ("out-header", "out-pack", "out-json")
    for index, first in enumerate(keys):
        for second in keys[index + 1:]:
            for form in ("same", "normalized", "symlink", "hardlink"):
                label = f"{first}-{second}-{form}"
                case, inputs, outputs = fixture(label)
                watched = list(inputs.values()) + list(outputs.values())
                outputs[second] = alias(case, outputs[first], form)
                rejected(inputs, outputs, watched, label)
            label = f"{first}-{second}-absent"
            case, inputs, outputs = fixture(label)
            watched = list(inputs.values()) + list(outputs.values())
            shared = case / "new-artifact"
            outputs[first] = shared
            outputs[second] = shared
            rejected(inputs, outputs, watched, label)
            if shared.exists():
                raise AssertionError(f"{label}: rejected compile created shared output")

    for form in ("loop", "not-directory"):
        case, inputs, outputs = fixture("identity-error-" + form)
        watched = list(inputs.values()) + list(outputs.values())
        if form == "loop":
            (case / "loop").symlink_to("loop")
            outputs["out-json"] = case / "loop"
        else:
            outputs["out-json"] = inputs["tokens"] / "child"
        rejected(inputs, outputs, watched, form, diagnostic=b"error:")

    for form in ("new", "existing", "symlink", "hardlink"):
        case, inputs, outputs = fixture("valid-" + form)
        before = {path: path.read_bytes() for path in inputs.values()}
        if form == "new":
            for path in outputs.values():
                path.unlink()
        elif form in ("symlink", "hardlink"):
            target = case / "unrelated-artifact"
            target.write_bytes(b"replaceable unrelated output\n")
            outputs["out-json"] = alias(case, target, form)
        completed = run(inputs, outputs)
        if completed.returncode != 0:
            raise AssertionError(f"non-alias {form} rejected: {completed.stderr!r}")
        for path, data in before.items():
            if path.read_bytes() != data:
                raise AssertionError(f"non-alias {form} changed an input")
        actual = {key: path.read_bytes() for key, path in outputs.items()}
        if baseline_outputs is None:
            baseline_outputs = actual
        elif actual != baseline_outputs:
            raise AssertionError(f"non-alias {form} changed compiled artifact bytes")
        checked += 1
    print(f"keyword output alias protection: {checked} cases passed")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        verify_threshold_wire(root / "threshold-wire")
        verify_output_aliases(root / "output-aliases")
        token_file = ROOT / "keywords" / "tokens.example.txt"
        header, manifest, pack = compile_one(
            token_file, ROOT / "keywords" / "zh_cn_example.tsv", root
        )
        text = header.read_text(encoding="utf-8")
        assert "kws_generated_keyword_count" in text
        assert "kws_generated_vocab_fingerprint" in text
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        items = metadata["keywords"]
        expected_fingerprint = vocab_fingerprint(load_tokens(token_file))
        assert metadata["vocab_size"] == 5
        assert metadata["vocab_fingerprint"] == f"0x{expected_fingerprint:016x}"

        blob = pack.read_bytes()
        magic, version, header_bytes, count, vocab_size, total_bytes, fingerprint = struct.unpack(
            "<4sHHHHIQ", blob[:24]
        )
        assert magic == b"KWKP"
        assert version == 3
        assert header_bytes == 24
        assert count == len(items) == 2
        assert vocab_size == 5
        assert fingerprint == expected_fingerprint
        assert total_bytes == len(blob) == 24 + count * 48

        fields = struct.unpack("<IfHBBBBH16H", blob[24:72])
        kid, threshold, num_tokens, min_blanks, priority, policy, grace, reserved, *tokens = fields
        assert kid == items[0]["id"]
        assert abs(threshold - items[0]["threshold"]) < 1.0e-6
        assert num_tokens == 4
        assert (min_blanks, priority, policy, grace, reserved) == (0, 0, 0, 0, 0)
        assert tokens[:4] == [1, 2, 3, 4]
        assert tokens[4:] == [0] * 12

        # Explicit overlapping keyword policy must survive TSV -> JSON/header/pack.
        overlap = root / "overlap.tsv"
        overlap.write_text(
            "10\t小窝\t0.55\txiao3 wo1\t1\t2\tlongest\t0\n"
            "11\t小窝小窝\t0.60\txiao3 wo1 xiao3 wo1\t0\t3\tgrace\t4\n",
            encoding="utf-8",
        )
        policy_root = root / "policy"
        policy_root.mkdir()
        _, policy_json, policy_pack = compile_one(token_file, overlap, policy_root)
        policy_items = json.loads(policy_json.read_text(encoding="utf-8"))["keywords"]
        assert policy_items[0]["prefix_policy"] == "longest"
        assert policy_items[0]["min_trailing_blanks"] == 1
        assert policy_items[1]["prefix_policy"] == "grace"
        assert policy_items[1]["grace_frames"] == 4
        pblob = policy_pack.read_bytes()
        p0 = struct.unpack("<IfHBBBBH16H", pblob[24:72])
        p1 = struct.unpack("<IfHBBBBH16H", pblob[72:120])
        assert p0[3:8] == (1, 2, 1, 0, 0)
        assert p1[3:8] == (0, 3, 2, 4, 0)

        # Compile the checked-in example itself, not only synthetic overlap TSVs.
        example_root = root / "overlap-example"
        example_root.mkdir()
        _, example_json, example_pack = compile_one(
            token_file, ROOT / "keywords" / "zh_cn_overlap_example.tsv", example_root
        )
        example_items = json.loads(example_json.read_text(encoding="utf-8"))["keywords"]
        assert [item["id"] for item in example_items] == [1, 2, 3]
        assert [item["priority"] for item in example_items] == [5, 15, 10]
        assert [item["prefix_policy"] for item in example_items] == [
            "grace", "longest", "immediate"
        ]
        example_blob = example_pack.read_bytes()
        assert len(example_blob) == 24 + 3 * 48
        for index, expected in enumerate(((1, 5, 2, 3, 0),
                                           (1, 15, 1, 0, 0),
                                           (1, 10, 0, 0, 0))):
            start = 24 + index * 48
            assert struct.unpack("<IfHBBBBH16H", example_blob[start:start + 48])[3:8] == expected

        # Shipping validation must compare threshold semantics, not decimal formatting.
        shipping = root / "shipping.tsv"
        shipping.write_text(
            "1\t你好小窝\t0.550000\tni3 hao3 xiao3 wo1\n"
            "2\t小窝小窝\t0.55\txiao3 wo1 xiao3 wo1\n",
            encoding="utf-8",
        )
        canonical = validate_shipping_keywords(shipping)
        assert [item["threshold"] for item in canonical] == ["0.55", "0.55"]

        drifted_shipping = root / "shipping-drifted.tsv"
        drifted_shipping.write_text(
            shipping.read_text(encoding="utf-8").replace("0.550000", "0.56", 1),
            encoding="utf-8",
        )
        try:
            validate_shipping_keywords(drifted_shipping)
        except ValueError as exc:
            assert "threshold drifted" in str(exc)
        else:
            raise AssertionError("shipping threshold semantic drift must be rejected")

        reordered = root / "tokens-reordered.txt"
        reordered.write_text("wo1 4\nxiao3 3\nhao3 2\nni3 1\n<blk> 0\n", encoding="utf-8")
        assert vocab_fingerprint(load_tokens(reordered)) == expected_fingerprint
        changed = root / "tokens-changed.txt"
        changed.write_text("<blk> 0\nni3 1\nxiao3 2\nhao3 3\nwo1 4\n", encoding="utf-8")
        assert vocab_fingerprint(load_tokens(changed)) != expected_fingerprint
        sparse = root / "tokens-sparse.txt"
        sparse.write_text("<blk> 0\nni3 2\n", encoding="utf-8")
        try:
            load_tokens(sparse)
        except ValueError as exc:
            assert "contiguous" in str(exc)
        else:
            raise AssertionError("sparse token IDs must be rejected")

    print("test_keyword_compile: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
