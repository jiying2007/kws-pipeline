#!/usr/bin/env python3
"""Import two declared ASR outputs; no inference, audio access, or gold promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import unicodedata

SCHEMAS = {name: f"kws-asr-evidence-{name}-v1" for name in
           ("manifest", "rules", "input", "readout", "receipt")}
RULE_VERSION = "conservative-text-evidence-v1"
STATES = ("positive", "negative", "unknown")
KINDS = ("synthetic_fixture", "not_run_template", "declared_model_output")
STATUSES = ("success", "error", "timeout", "not_run")
COMPLETENESS = ("complete", "incomplete", "unknown")
FLAGS = ("missing_characters", "incomplete", "ambiguous", "non_speech", "decoding_warning")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
MAX_RECORDS = 256
MAX_TARGETS = 4
MAX_TEXT = 1024
MAX_TOTAL_TEXT = 32768  # Per model, before normalization.
MAX_TARGET_LENGTH = 16
MAX_LEXICON_ENTRIES = 100000
MAX_PHRASE_LENGTH = 128
MAX_RULE_MARKERS = 32
REQUIRED_MARKERS = ("[unk]", "<unk>", "[inaudible]", "[不清]", "[缺字]", "[不全]",
                    "听不清", "不确定", "�", "□", "…", "...")
FILE_LIMITS = {"manifest": 512 * 1024, "rules": 64 * 1024,
               "model_a": 2 * 1024 * 1024, "model_b": 2 * 1024 * 1024,
               "characters": 8 * 1024 * 1024, "phrases": 8 * 1024 * 1024}
MAX_OUTPUT_BYTES = 32 * 1024 * 1024


class ContractError(ValueError):
    """An invalid declaration; never emit a partial accepted readout."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def exact_keys(value, keys, context):
    require(type(value) is dict and set(value) == set(keys),
            f"{context}: expected exact fields {', '.join(keys)}")


def string(value, context, maximum=256, minimum=1):
    require(type(value) is str and minimum <= len(value) <= maximum,
            f"{context}: invalid string length/type")
    require(not any(unicodedata.category(c) == "Cs" or c == "\x00" for c in value),
            f"{context}: invalid Unicode scalar/NUL")
    return value


def identity(value, context):
    string(value, context)
    require(not any(c.isspace() or unicodedata.category(c).startswith("C") for c in value),
            f"{context}: whitespace/control in identity")


def sha(value, context):
    require(type(value) is str and HEX64.fullmatch(value), f"{context}: expected lowercase SHA256")


def enum(value, options, context):
    require(type(value) is str and value in options, f"{context}: unsupported value/type")


def bounded_list(value, context, minimum, maximum):
    require(type(value) is list and minimum <= len(value) <= maximum,
            f"{context}: invalid array length/type")


def decode_json(data, context):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{context}: duplicate JSON key")
            result[key] = value
        return result

    def reject_number(_):
        raise ContractError(f"{context}: floating/nonfinite numbers are unsupported")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=reject_number, parse_float=reject_number)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ContractError(f"{context}: invalid strict JSON ({type(exc).__name__})") from exc
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        require(depth <= 8, f"{context}: excessive JSON nesting")
        if type(item) is str:
            string(item, context, maximum=FILE_LIMITS.get(context, 8 * 1024 * 1024), minimum=0)
        elif type(item) is list:
            pending.extend((v, depth + 1) for v in item)
        elif type(item) is dict:
            for key, val in item.items():
                string(key, context, maximum=MAX_TEXT, minimum=0)
                pending.append((val, depth + 1))
    return value


def load_input(path, context, expected_sha=None):
    """Hash and parse exactly the same bounded bytes from a regular, non-symlink file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            require(stat.S_ISREG(info.st_mode), f"{context}: regular file required")
            limit = FILE_LIMITS[context]
            require(0 < info.st_size <= limit, f"{context}: file byte limit exceeded/empty")
            data = handle.read(limit + 1)
            require(len(data) <= limit, f"{context}: file byte limit exceeded")
    except OSError as exc:
        raise ContractError(f"{context}: input unavailable ({exc.__class__.__name__})") from exc
    actual = hashlib.sha256(data).hexdigest()
    if expected_sha is not None:
        sha(expected_sha, context + " predeclared hash")
        require(actual == expected_sha, f"{context}: predeclared SHA mismatch")
    return decode_json(data, context), actual, (info.st_dev, info.st_ino)


def validate_model(model):
    exact_keys(model, ("model_id", "revision", "run_id"), "model")
    identity(model["model_id"], "model ID")
    identity(model["run_id"], "run ID")
    require(type(model["revision"]) is str and REVISION.fullmatch(model["revision"]),
            "model revision: immutable 40/64 lowercase hex required")


def validate_manifest(manifest):
    exact_keys(manifest, ("schema_version", "probe_set_id", "execution_kind", "target_order",
                          "models", "records"), "manifest")
    require(manifest["schema_version"] == SCHEMAS["manifest"], "manifest version mismatch")
    identity(manifest["probe_set_id"], "probe set ID")
    enum(manifest["execution_kind"], KINDS, "manifest execution kind")
    targets = manifest["target_order"]
    bounded_list(targets, "targets", 1, MAX_TARGETS)
    for target in targets:
        string(target, "target", MAX_TARGET_LENGTH, 2)
        require(unicodedata.normalize("NFC", target) == target and
                all(unicodedata.category(c).startswith("L") for c in target),
                "targets must be NFC contiguous letters")
    require(len(set(targets)) == len(targets), "duplicate targets")
    bounded_list(manifest["models"], "models", 2, 2)
    for model in manifest["models"]:
        validate_model(model)
        fixture_id = model["model_id"].startswith("fixture-only-")
        require(fixture_id == (manifest["execution_kind"] == "synthetic_fixture"),
                "fixture model IDs cannot mix with external declarations/templates")
    require(manifest["models"][0]["model_id"] != manifest["models"][1]["model_id"],
            "two distinct model IDs required (not proof of statistical independence)")
    bounded_list(manifest["records"], "manifest records", 1, MAX_RECORDS)
    index = {}
    for row in manifest["records"]:
        exact_keys(row, ("recording_id", "wav_sha256", "human_target_presence"), "manifest row")
        identity(row["recording_id"], "recording ID")
        require(row["recording_id"] not in index, "duplicate manifest recording ID")
        sha(row["wav_sha256"], "WAV SHA")
        bounded_list(row["human_target_presence"], "human labels", len(targets), len(targets))
        for label in row["human_target_presence"]:
            enum(label, STATES, "human label")
        index[row["recording_id"]] = row
    return index


def normalize(raw):
    """Preserve words, punctuation and raw text; no NFKC or homophone repair."""
    return " ".join(unicodedata.normalize("NFC", raw).split())


def toneless(pinyin):
    value = unicodedata.normalize("NFD", pinyin).replace("u\u0308", "v")
    return re.sub(r"[1-5]$", "", "".join(c for c in value if not unicodedata.combining(c)).replace("u:", "v"))


def validate_readings(value):
    bounded_list(value, "lexical readings", 1, 16)
    for reading in value:
        string(reading, "lexical reading", 32)
        require(reading == unicodedata.normalize("NFC", reading) and
                all(c.isalpha() or c in "12345:" for c in reading), "invalid lexical reading")
    require(len(set(value)) == len(value), "duplicate lexical readings")


class Rules:
    def __init__(self, config, rules_sha, characters, phrases, targets):
        exact_keys(config, ("schema_version", "rule_version", "target_order", "lexicons",
                            "uncertainty_markers"), "rules")
        require(config["schema_version"] == SCHEMAS["rules"], "rules version mismatch")
        require(config["rule_version"] == RULE_VERSION, "unsupported rule algorithm")
        require(config["target_order"] == targets, "rules target order mismatch")
        exact_keys(config["lexicons"], ("characters", "phrases"), "lexicons")
        for kind, values in (("characters", characters), ("phrases", phrases)):
            spec = config["lexicons"][kind]
            exact_keys(spec, ("sha256", "entries"), "lexicon declaration")
            sha(spec["sha256"], "lexicon SHA")
            require(type(spec["entries"]) is int and 1 <= spec["entries"] <= MAX_LEXICON_ENTRIES,
                    "invalid lexicon entry count")
            require(type(values) is dict and len(values) == spec["entries"], "lexicon size mismatch")
        for key, values in characters.items():
            require(type(key) is str and key.isascii() and key.isdecimal() and
                    str(int(key)) == key and 0 <= int(key) <= 0x10ffff and
                    not 0xd800 <= int(key) <= 0xdfff, "invalid character codepoint")
            string(values, "character readings", 16 * 33)
            validate_readings(values.split(","))
        for key, values in phrases.items():
            string(key, "phrase", MAX_PHRASE_LENGTH, 2)
            require(key == normalize(key) and all(unicodedata.category(c).startswith("L") for c in key),
                    "invalid lexical phrase")
            bounded_list(values, "phrase readings", len(key), len(key))
            for readings in values:
                validate_readings(readings)
        markers = config["uncertainty_markers"]
        bounded_list(markers, "uncertainty markers", 1, MAX_RULE_MARKERS)
        for marker in markers:
            string(marker, "uncertainty marker", 64)
            require(marker == normalize(marker), "marker must be normalized")
        require(len({m.casefold() for m in markers}) == len(markers), "duplicate uncertainty markers")
        require(set(REQUIRED_MARKERS) <= set(markers), "mandatory uncertainty markers missing")
        self.config, self.sha256 = config, rules_sha
        self.characters, self.phrases = characters, phrases
        self.targets = targets
        self.phrase_lengths = sorted({len(p) for p in phrases}, reverse=True)
        self.target_tokens = [self.tokenize(t)[0] for t in targets]
        require(all(token["kind"] == "lexical" and len(token["readings"]) == 1
                    for tokens in self.target_tokens for token in tokens),
                "each target needs complete unambiguous lexicon readings")
    def tokenize(self, text):
        """Greedy frozen lexical lookup; every normalized character keeps its index."""
        tokens, units = [], []
        i = 0
        while i < len(text):
            char = text[i]
            if char.isspace() or unicodedata.category(char).startswith('P'):
                tokens.append({'index': i, 'character': char, 'readings': [], 'kind': 'boundary'})
                i += 1
                continue
            phrase = next((text[i:i+n] for n in self.phrase_lengths if n <= len(text)-i and text[i:i+n] in self.phrases), None)
            if phrase:
                readings = self.phrases[phrase]
                require(len(readings) == len(phrase), 'Malformed frozen phrase entry')
                units.append({'start': i, 'text': phrase, 'source': 'phrases_dict'})
                for offset, variants in enumerate(readings):
                    tokens.append({'index': i + offset, 'character': phrase[offset], 'readings': list(dict.fromkeys(variants)), 'kind': 'lexical'})
                i += len(phrase)
            else:
                readings = self.characters.get(str(ord(char)))
                tokens.append({'index': i, 'character': char, 'readings': list(dict.fromkeys(readings.split(','))) if readings else [], 'kind': 'lexical' if readings else 'oov'})
                units.append({'start': i, 'text': char, 'source': 'pinyin_dict' if readings else 'out_of_vocabulary'})
                i += 1
        return tokens, units

    def derive(self, row):
        raw = row['raw_text']
        text = normalize(raw) if raw is not None else ''
        tokens, units = self.tokenize(text)
        issues = []
        if any(unicodedata.category(c).startswith('C') and c not in '\t\r\n' for c in (raw or '')):
            issues.append('unsupported_control_character')
        if '<asr_text>' in text or re.search(r'<\|[^>]*\|>', text) or re.match(r'^language\s+', text, re.IGNORECASE):
            issues.append('decoder_metadata_not_stripped')
        if row['status'] != 'success':
            issues.append('non_success_status')
        if not text or not any(unicodedata.category(c)[0] in 'LN' for c in text):
            issues.append('empty_transcript')
        if row['completeness'] != 'complete':
            issues.append('transcript_not_declared_complete')
        issues.extend('quality_flag:' + flag for flag in row['quality_flags'])
        issues.extend('uncertainty_marker:' + marker for marker in self.config['uncertainty_markers'] if marker.casefold() in text.casefold())
        lexical_issues = []
        if any(t['kind'] == 'oov' for t in tokens):
            lexical_issues.append('lexicon_oov')
        if any(len(t['readings']) > 1 for t in tokens):
            lexical_issues.append('lexicon_ambiguous_reading')
        targets = []
        for target_index, target in enumerate(self.targets):
            evidence = target_evidence(text, tokens, target, self.target_tokens[target_index])
            blockers = list(issues)
            if blockers:
                state = 'unknown'
            elif evidence['literal_target_present']:
                state = 'positive'
            else:
                blockers.extend(lexical_issues)
                if evidence['confusable_or_partial_target_candidate']:
                    blockers.append('confusable_or_partial_target')
                state = 'unknown' if blockers else 'negative'
            targets.append({'target': target, 'presence': state, 'blockers': blockers, 'rule_evidence': evidence, 'evidence_scope': 'ASR text only; not audio truth'})
        return {
            'raw_text': raw, 'normalized_text': text, 'status': row['status'],
            'completeness': row['completeness'], 'quality_flags': row['quality_flags'],
            'normalization_version': 'nfc-whitespace-boundaries-v1',
            'rule_version': self.config['rule_version'],
            'pinyin_tokens': tokens, 'lexical_units': units,
            'pinyin_scope': 'Dictionary text readings only; not acoustic tone measurement',
            'targets': targets,
        }

def edit_distance(a, b):
    prior = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prior[j]+1, prior[j-1]+(x != y)))
        prior = cur
    return prior[-1]

def target_evidence(text, tokens, target, target_tokens):
    literal = target in text
    without_boundaries = ''.join(c for c in text if not (c.isspace() or unicodedata.category(c).startswith('P')))
    boundary_candidate = not literal and target in without_boundaries
    near_text = []
    phonetic = []
    n = len(target)
    target_toned = [set(t['readings']) for t in target_tokens]
    target_plain = [{toneless(v) for v in vs} for vs in target_toned]
    for start in range(len(text)):
        for size in (n-1, n, n+1):
            chunk = text[start:start+size]
            if len(chunk) != size or any(c.isspace() or unicodedata.category(c).startswith('P') for c in chunk):
                continue
            if chunk != target and edit_distance(chunk, target) <= 1:
                near_text.append({'start': start, 'text': chunk, 'kind': 'one_edit_text_candidate'})
        window = tokens[start:start+n]
        if len(window) != n or any(t['kind'] != 'lexical' for t in window):
            continue
        toned = [set(t['readings']) for t in window]
        plain = [{toneless(v) for v in vs} for vs in toned]
        tone_candidate = all(a & b for a, b in zip(toned, target_toned))
        syllable_mismatches = sum(not bool(a & b) for a, b in zip(plain, target_plain))
        if tone_candidate or syllable_mismatches <= 1:
            phonetic.append({'start': start, 'text': ''.join(t['character'] for t in window), 'dictionary_tone_sequence_candidate': tone_candidate, 'toneless_syllable_mismatches': syllable_mismatches, 'acoustic_tone_measured': False})
    return {
        'literal_target_present': literal,
        'one_edit_text_candidates': near_text,
        'boundary_separated_target_candidate': boundary_candidate,
        'dictionary_phonetic_candidates': phonetic,
        'confusable_or_partial_target_candidate': bool(near_text or phonetic or boundary_candidate),
        'pinyin_is_acoustic_measurement': False,
    }

def validate_run(run, manifest, manifest_index, manifest_sha, rules, slot):
    exact_keys(run, ("schema_version", "execution_kind", "manifest_sha256", "rules_sha256",
                     "model", "records"), "model output")
    require(run["schema_version"] == SCHEMAS["input"], "model output version mismatch")
    require(run["execution_kind"] == manifest["execution_kind"], "execution kind differs from manifest")
    require(run["manifest_sha256"] == manifest_sha, "model output manifest SHA mismatch")
    require(run["rules_sha256"] == rules.sha256, "model output rules SHA mismatch")
    validate_model(run["model"])
    require(run["model"] == manifest["models"][slot], "model ID/revision/run differs from predeclaration")
    bounded_list(run["records"], "model records", len(manifest_index), len(manifest_index))
    index, total_text = {}, 0
    for row in run["records"]:
        exact_keys(row, ("recording_id", "wav_sha256", "status", "raw_text", "completeness",
                         "quality_flags"), "model row")
        identity(row["recording_id"], "model recording ID")
        rid = row["recording_id"]
        require(rid in manifest_index and rid not in index, "unknown/duplicate model recording ID")
        sha(row["wav_sha256"], "model WAV SHA")
        require(row["wav_sha256"] == manifest_index[rid]["wav_sha256"], "model WAV SHA mismatch")
        enum(row["status"], STATUSES, "model status")
        enum(row["completeness"], COMPLETENESS, "completeness")
        raw = row["raw_text"]
        if raw is not None:
            string(raw, "raw transcript", MAX_TEXT, 0)
            total_text += len(raw)
        require(total_text <= MAX_TOTAL_TEXT, "model total transcript length limit exceeded")
        require(row["status"] != "success" or type(raw) is str, "successful transcript must be string")
        flags = row["quality_flags"]
        bounded_list(flags, "quality flags", 0, len(FLAGS))
        for flag in flags:
            enum(flag, FLAGS, "quality flag")
        require(len(set(flags)) == len(flags), "duplicate quality flags")
        if manifest["execution_kind"] == "not_run_template" or row["status"] == "not_run":
            require(row["status"] == "not_run" and raw is None and
                    row["completeness"] == "unknown" and not flags,
                    "not-run evidence cannot contain predictions or completion/quality claims")
        index[rid] = row
    require(set(index) == set(manifest_index), "missing/extra model recording IDs")
    if manifest["execution_kind"] == "declared_model_output":
        require(any(r["status"] != "not_run" for r in index.values()),
                "all-not-run output cannot claim declared model execution")
    return index


def metric(rows, slot, target_index):
    stats = {"human_known_bits": 0, "human_unknown_bits": 0,
             "human_positive_bits": 0, "human_negative_bits": 0,
             "true_positive": 0, "false_negative": 0, "true_negative": 0, "false_positive": 0,
             "unknown_on_positive": 0, "unknown_on_negative": 0, "covered_known_bits": 0,
             "error_bits": []}
    for row in rows:
        human = row["human_target_presence"][target_index]
        if human == "unknown":
            stats["human_unknown_bits"] += 1
            continue
        stats["human_known_bits"] += 1
        stats["human_" + human + "_bits"] += 1
        pred = row["model_evidence"][slot]["targets"][target_index]["presence"]
        if pred == "unknown":
            stats["unknown_on_" + human] += 1
        else:
            stats["covered_known_bits"] += 1
            category = ("true_" if pred == human else "false_") + pred
            stats[category] += 1
            if pred != human:
                stats["error_bits"].append({"recording_id": row["recording_id"],
                    "wav_sha256": row["wav_sha256"], "target_index": target_index,
                    "human": human, "prediction": pred, "error_type": category})
    known = stats["human_known_bits"]
    stats["known_bit_coverage"] = {"numerator": stats["covered_known_bits"],
        "denominator": known, "fraction": stats["covered_known_bits"] / known if known else None}
    return stats


def join(manifest, manifest_sha, rules, run_a, run_b):
    index = validate_manifest(manifest)
    models = [run_a, run_b]
    inputs = [validate_run(run, manifest, index, manifest_sha, rules, slot)
              for slot, run in enumerate(models)]
    rows = []
    for rid, reference in index.items():
        derived = [rules.derive(r[rid]) for r in inputs]
        labels = list(reference["human_target_presence"])
        final, consensus = [], []
        for target, human, t in zip(rules.targets, labels, range(len(labels))):
            states = [e["targets"][t]["presence"] for e in derived]
            agreement = states[0] == states[1] and states[0] != "unknown"
            consensus.append({"target": target, "presence": states[0] if agreement else "unknown",
                              "scope": "weak model consensus only", "promoted_to_human_gold": False})
            final.append({"target": target, "presence": human,
                "source": "predeclared_human" if human != "unknown" else "unresolved_human_unknown",
                "human_conflict_models": [i for i, p in enumerate(states)
                                          if human != "unknown" and p not in (human, "unknown")],
                "model_disagreement": states[0] != states[1], "gold_changed_by_models": False})
        rows.append({"recording_id": rid, "wav_sha256": reference["wav_sha256"],
            "human_target_presence": labels, "model_evidence": derived,
            "weak_model_consensus": consensus, "final_target_presence": final})
    per_model = [{"model": run["model"], "per_target": [
        {"target": target, **metric(rows, slot, t)} for t, target in enumerate(rules.targets)]}
        for slot, run in enumerate(models)]
    paired = []
    for t, target in enumerate(rules.targets):
        outcome = {"target": target, "human_known_bits": 0, "both_covered_known_bits": 0,
            "both_wrong_known_bits": [], "definite_disagreements_known_bits": [],
            "one_unknown_disagreements_known_bits": [], "both_unknown_known_bits": []}
        for row in rows:
            human = row["human_target_presence"][t]
            if human == "unknown":
                continue
            outcome["human_known_bits"] += 1
            states = [m["targets"][t]["presence"] for m in row["model_evidence"]]
            ident = {"recording_id": row["recording_id"], "wav_sha256": row["wav_sha256"],
                     "target_index": t, "human": human, "model_states": states}
            if "unknown" not in states:
                outcome["both_covered_known_bits"] += 1
                if all(p != human for p in states):
                    outcome["both_wrong_known_bits"].append(ident)
                if states[0] != states[1]:
                    outcome["definite_disagreements_known_bits"].append(ident)
            elif states == ["unknown", "unknown"]:
                outcome["both_unknown_known_bits"].append(ident)
            else:
                outcome["one_unknown_disagreements_known_bits"].append(ident)
        paired.append(outcome)
    return {"schema_version": SCHEMAS["readout"], "execution_kind": manifest["execution_kind"],
        "evidence_level": "synthetic" if manifest["execution_kind"] == "synthetic_fixture"
                          else "external-declarations-only",
        "inference_performed_by_this_tool": False, "wav_bytes_verified_by_this_tool": False,
        "manifest_sha256": manifest_sha, "rules_sha256": rules.sha256, "rule_version": RULE_VERSION,
        "probe_set_id": manifest["probe_set_id"], "target_order": rules.targets,
        "predeclared_models": manifest["models"], "lexicons": rules.config["lexicons"],
        "human_label_counts": {state: sum(state == v for row in rows
                                          for v in row["human_target_presence"]) for state in STATES},
        "limitations": [
            "External model outputs are declarations: ID/hash/model association is validated, not actual WAV consumption.",
            "The caller must retain raw decoder tokens and independently bound execution receipts outside this tool.",
            "Human labels are predeclared inputs, not independently certified by this tool; unknown never becomes gold.",
            "Dictionary phonetics are text evidence, not acoustic pronunciation or tone measurements.",
            "Distinct model IDs do not prove independence; agreement can be jointly wrong.",
            "Selected probes are descriptive only: no population accuracy, reliability, CER/WER, timing or release qualification."],
        "per_model_known_bit_readout": per_model, "paired_known_bit_readout": paired, "records": rows}


def write_exclusive(path, result):
    """Publish fully serialized bytes with an exclusive hard link, never replace a file."""
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(data) <= MAX_OUTPUT_BYTES, "readout byte limit exceeded")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".asr-evidence-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # EEXIST for files, directories, symlinks and hard-link aliases.
    finally:
        if temporary is not None:
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "rules", "characters", "phrases", "model-a", "model-b", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True, help="Previously declared manifest byte SHA256")
    parser.add_argument("--rules-sha256", required=True, help="Previously declared rules byte SHA256")
    args = parser.parse_args(argv)
    try:
        paths = {key: getattr(args, key) for key in FILE_LIMITS}
        output = args.output.resolve()
        require(not args.output.exists() and not args.output.is_symlink(), "output already exists")
        require(output not in {p.resolve() for p in paths.values()}, "output must be separate from inputs")
        loaded, hashes, inodes = {}, {}, set()
        for key in ("manifest", "rules"):
            value, digest, inode = load_input(paths[key], key, getattr(args, key + "_sha256"))
            loaded[key], hashes[key] = value, digest
            require(inode not in inodes, "input files must be distinct")
            inodes.add(inode)
        validate_manifest(loaded["manifest"])
        config = loaded["rules"]
        # Validate the nested SHA declarations before trusting them for external files.
        require(type(config) is dict and type(config.get("lexicons")) is dict, "missing lexicon declarations")
        exact_keys(config["lexicons"], ("characters", "phrases"), "lexicons")
        for key in ("characters", "phrases", "model_a", "model_b"):
            expected = None
            if key in ("characters", "phrases"):
                spec = config["lexicons"][key]
                exact_keys(spec, ("sha256", "entries"), "lexicon declaration")
                expected = spec["sha256"]
            value, digest, inode = load_input(paths[key], key, expected)
            loaded[key], hashes[key] = value, digest
            require(inode not in inodes, "input files must be distinct")
            inodes.add(inode)
        rules = Rules(config, hashes["rules"], loaded["characters"], loaded["phrases"],
                      loaded["manifest"]["target_order"])
        result = join(loaded["manifest"], hashes["manifest"], rules, loaded["model_a"], loaded["model_b"])
        result["input_sha256"] = hashes
        output_sha = write_exclusive(args.output, result)
    except (ContractError, OSError, ValueError, RecursionError) as exc:
        # No input text, private paths, or partial result is written on validation rejection.
        message = str(exc) if isinstance(exc, ContractError) else type(exc).__name__
        print(f"REJECTED: {message}", file=sys.stderr)
        return 2
    print(json.dumps({"schema_version": SCHEMAS["receipt"], "readout_sha256": output_sha,
        "manifest_sha256": hashes["manifest"], "rules_sha256": hashes["rules"],
        "rule_version": RULE_VERSION, "execution_kind": result["execution_kind"],
        "predeclared_models": result["predeclared_models"], "input_sha256": hashes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
