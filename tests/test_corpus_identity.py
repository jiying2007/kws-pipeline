from __future__ import annotations

import copy
import itertools
import json
import pathlib
import struct
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "training"), str(ROOT / "eval")]
from corpus_identity import (  # noqa: E402
    AUDIO_PATH_FIELDS, audio_identity, canonical_audio_path, corpus_digest,
    evaluation_corpus_identity, inspect_pcm16_wav, require_audited_corpora,
    require_audited_audio, sha256_file, training_corpus_identity,
    training_manifest_audio_bindings, validate_audio_identity, rebind_audio_path,
)
from audit_dataset import audit_splits, parse_jsonl  # noqa: E402
from run_corpus import audio_identity as execution_audio_identity, load_references  # noqa: E402
from diagnose_decoder_policy_replay import boundary_reference_contract  # noqa: E402
from render_domains import domain_reference_row  # noqa: E402
from long_far_stream import materialize_captures  # noqa: E402
from diagnose_frozen_far_source_replay import find_domain_row  # noqa: E402


def write_wav(path: pathlib.Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


def write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CorpusIdentityTests(unittest.TestCase):
    """Invented stdlib WAV/metadata fixtures, never a model or qualification run."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        write_wav(self.root / "train.wav", [0, 100, -100, 7])
        write_wav(self.root / "eval.wav", [0, 200, -200, 7])
        self.train = self.root / "train.jsonl"
        self.refs = self.root / "eval.jsonl"
        write_rows(self.train, [{"path": "train.wav", "target_ids": [1], "speaker_id": "train"}])
        self.reference = {"recording": "eval", "duration_s": 4 / 16000, "expected": [], "speaker_id": "eval"}
        write_rows(self.refs, [{**self.reference, "path": "eval.wav"}])

    def selected(self) -> list[dict]:
        digest = sha256_file(self.train)
        return [{"name": self.train.name, "sha256": digest, "lineage_sha256": digest}]

    def audited(self, roots: dict | None = None) -> dict:
        report = audit_splits([("train", self.train), ("eval", self.refs)], roots=roots)
        self.assertIs(report["clean"], True)
        # The qualification producer retains these same measured per-split bindings.
        return {"manifest_bindings": [
            {"name": pathlib.Path(split["manifest"]).name, "sha256": split["manifest_sha256"],
             "lineage_sha256": split["manifest_sha256"], "audio_identity": split["audio_identity"]}
            for split in report["splits"].values()]}

    def validate_selected(self, audit: dict, root: pathlib.Path | None = None) -> None:
        require_audited_corpora(audit, training_corpus_identity([self.train]), self.selected(),
                               evaluation_corpus_identity(self.refs, root or self.root)["recordings"],
                               sha256_file(self.refs))

    def test_pcm_and_file_identity(self) -> None:
        first = inspect_pcm16_wav(self.root / "train.wav")
        second = self.root / "copy.wav"
        second.write_bytes((self.root / "train.wav").read_bytes())
        self.assertEqual(first, inspect_pcm16_wav(second))
        self.assertEqual(len(corpus_digest([{"recording": "a", "path": "train.wav", **first}])), 64)
        write_wav(second, [0, 100, -101, 7])
        self.assertNotEqual(inspect_pcm16_wav(second)["pcm_sha256"], first["pcm_sha256"])

    def test_all_aliases_share_audit_execution_and_identity(self) -> None:
        for count in range(1, 4):
            for fields in itertools.combinations(AUDIO_PATH_FIELDS, count):
                with self.subTest(fields=fields):
                    write_rows(self.refs, [{**self.reference, **{field: "eval.wav" for field in fields}}])
                    audited = audit_splits([("eval", self.refs)])["splits"]["eval"]["audio_identity"]
                    evaluated = evaluation_corpus_identity(self.refs, self.root)["recordings"]
                    execution = load_references(self.refs)[0]
                    measured = execution_audio_identity(execution, self.root / execution["_execution_path"])
                    self.assertEqual(evaluated, [measured])
                    self.assertEqual(audited, audio_identity(evaluated))
                    trained = training_corpus_identity([self.refs])["recordings"]
                    self.assertEqual(audited, audio_identity(trained))
                    boundary_reference_contract(self.refs, positive=False)

    def test_every_conflicting_alias_pair_fails_every_reader(self) -> None:
        for left, right in itertools.combinations(AUDIO_PATH_FIELDS, 2):
            write_rows(self.refs, [{**self.reference, left: "eval.wav", right: "train.wav"}])
            for read in (lambda: parse_jsonl(self.refs), lambda: load_references(self.refs),
                         lambda: training_corpus_identity([self.refs]),
                         lambda: evaluation_corpus_identity(self.refs, self.root),
                         lambda: boundary_reference_contract(self.refs, positive=False)):
                with self.subTest(aliases=(left, right), reader=read):
                    with self.assertRaisesRegex(ValueError, "aliases disagree"):
                        read()

    def test_invalid_present_alias_is_not_ignored(self) -> None:
        for field in AUDIO_PATH_FIELDS:
            for bad in (None, "", "  ", 1, True, []):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValueError):
                        canonical_audio_path({"path": "eval.wav", field: bad})
        with self.assertRaises(ValueError):
            canonical_audio_path({})
        self.assertEqual(canonical_audio_path({"audio": " eval.wav ", "path": "eval.wav"}), "eval.wav")

    def test_selected_audio_matches_clean_audit(self) -> None:
        self.validate_selected(self.audited())

    def test_changed_wav_after_audit_fails(self) -> None:
        audit = self.audited()
        original_manifest = sha256_file(self.refs)
        write_wav(self.root / "eval.wav", [0, 300, -300, 7])
        self.assertEqual(sha256_file(self.refs), original_manifest)
        with self.assertRaisesRegex(ValueError, "selected WAV/PCM"):
            self.validate_selected(audit)

    def test_training_wav_after_audit_fails(self) -> None:
        audit = self.audited()
        write_wav(self.root / "train.wav", [0, 500, -500, 7])
        with self.assertRaisesRegex(ValueError, "selected WAV/PCM"):
            self.validate_selected(audit)

    def test_different_audio_root_binds_bytes(self) -> None:
        audit = self.audited()
        other = self.root / "other"
        other.mkdir()
        (other / "eval.wav").write_bytes((self.root / "eval.wav").read_bytes())
        self.validate_selected(audit, other)
        # Reuse training PCM under the same manifest path in a different root.
        (other / "eval.wav").write_bytes((self.root / "train.wav").read_bytes())
        with self.assertRaisesRegex(ValueError, "selected WAV/PCM"):
            self.validate_selected(audit, other)

    def test_container_change_with_identical_pcm_fails(self) -> None:
        audit = self.audited()
        wav = self.root / "eval.wav"
        pcm = inspect_pcm16_wav(wav)["pcm_sha256"]
        data = bytearray(wav.read_bytes())
        chunk = b"JUNK" + struct.pack("<I", 4) + b"meta"
        struct.pack_into("<I", data, 4, struct.unpack_from("<I", data, 4)[0] + len(chunk))
        wav.write_bytes(data[:12] + chunk + data[12:])
        self.assertEqual(inspect_pcm16_wav(wav)["pcm_sha256"], pcm)
        with self.assertRaisesRegex(ValueError, "selected WAV/PCM"):
            self.validate_selected(audit)

    def test_missing_or_tampered_audio_binding_fails(self) -> None:
        audit = self.audited()
        for mutation in (lambda row: row.pop("audio_identity"),
                         lambda row: row["audio_identity"]["recordings"][0].update(frames=12)):
            bad = copy.deepcopy(audit)
            mutation(bad["manifest_bindings"][0])
            with self.assertRaises(ValueError):
                self.validate_selected(bad)
        identity = audit["manifest_bindings"][0]["audio_identity"]
        self.assertEqual(identity, validate_audio_identity(identity))
        for frames in (0, True, 1.0):
            bad = copy.deepcopy(identity["recordings"])
            bad[0]["frames"] = frames
            with self.assertRaises(ValueError):
                audio_identity(bad)

    def test_clean_claim_cannot_hide_cross_split_pcm_overlap(self) -> None:
        audit = self.audited()
        train = audit["manifest_bindings"][0]["audio_identity"]
        evaluation = copy.deepcopy(train["recordings"])
        evaluation[0]["path"] = "eval.wav"
        audit["manifest_bindings"][1]["audio_identity"] = audio_identity(evaluation)
        with self.assertRaisesRegex(ValueError, "cross-split PCM overlap"):
            self.validate_selected(audit)

    def test_training_bindings_preserve_manifest_index_and_reuse(self) -> None:
        corpus = training_corpus_identity([self.train, self.train])
        bindings = training_manifest_audio_bindings(self.selected() * 2, corpus)
        self.assertEqual(bindings[0], bindings[1])
        require_audited_audio(self.audited(), bindings * 2)
        corpus["recordings"][1]["recording"] = "manifest-0:1"
        corpus["corpus_sha256"] = corpus_digest(corpus["recordings"])
        with self.assertRaisesRegex(ValueError, "index is inconsistent"):
            training_manifest_audio_bindings(self.selected() * 2, corpus)

    def test_rebinding_replaces_all_aliases_without_mutating_source(self) -> None:
        for count in range(4):
            for fields in itertools.combinations(AUDIO_PATH_FIELDS, count):
                source = {**{field: "source.wav" for field in fields},
                          "source_path": "original.wav", "speaker_id": "person",
                          "_execution_path": "obsolete.wav"}
                original = copy.deepcopy(source)
                for output_alias in AUDIO_PATH_FIELDS:
                    with self.subTest(fields=fields, output_alias=output_alias):
                        rebound = rebind_audio_path(source, " derived.wav ", field=output_alias)
                        self.assertEqual(canonical_audio_path(rebound), "derived.wav")
                        self.assertEqual(set(rebound) & set(AUDIO_PATH_FIELDS), {output_alias})
                        self.assertNotIn("_execution_path", rebound)
                        self.assertEqual(rebound["source_path"], "original.wav")
                        self.assertEqual(source, original)

    def test_rebinding_never_erases_conflicting_input(self) -> None:
        for left, right in itertools.combinations(AUDIO_PATH_FIELDS, 2):
            with self.assertRaisesRegex(ValueError, "aliases disagree"):
                rebind_audio_path({left: "one.wav", right: "two.wav"}, "derived.wav")
        for field in AUDIO_PATH_FIELDS:
            for value in (None, "", 4):
                with self.assertRaises(ValueError):
                    rebind_audio_path({field: value}, "derived.wav")
        with self.assertRaises(ValueError):
            rebind_audio_path({}, "derived.wav", field="input_path")
        with self.assertRaises(ValueError):
            rebind_audio_path({}, "")

    def test_domain_reference_metadata_uses_one_actual_audio_path(self) -> None:
        # Exercise the production metadata writer without domain rendering,
        # prototype generation, model execution or a training loop.
        for split in ("calibration", "test", "qualification"):
            for kind in ("positive", "background"):
                with self.subTest(split=split, kind=kind):
                    source = {**{field: "train.wav" for field in AUDIO_PATH_FIELDS},
                              "kind": kind, "keyword_id": 1,
                              "event_start_frame": 0, "event_end_frame": 1,
                              "scene": {"direct_delay_samples": 1, "afe_latency_samples": 1},
                              "domain_id": "fixture", "speaker_id": "person"}
                    derived = rebind_audio_path(source, str(self.root / "eval.wav"))
                    reference = domain_reference_row(derived, recording=f"domain-{split}-000000", frames=4)
                    self.assertEqual(reference["path"], str(self.root / "eval.wav"))
                    self.assertEqual(set(reference) & set(AUDIO_PATH_FIELDS), {"path"})
                    self.assertEqual(reference["duration_s"], 4 / 16000)
                    expected = ([{"keyword_id": 1, "start_s": 2 / 16000, "end_s": 3 / 16000}]
                                if kind == "positive" else [])
                    self.assertEqual(reference["expected"], expected)
                    write_rows(self.refs, [reference])
                    audited = audit_splits([(split, self.refs)])["splits"][split]["audio_identity"]
                    selected = evaluation_corpus_identity(self.refs, self.root)["recordings"]
                    self.assertEqual(audited, audio_identity(selected))
                    self.assertEqual(load_references(self.refs)[0]["_execution_path"], reference["path"])
        bad = {**derived, "audio_path": "different.wav"}
        with self.assertRaisesRegex(ValueError, "aliases disagree"):
            domain_reference_row(bad, recording="bad", frames=4)

    def test_capture_metadata_rebinds_optional_source_aliases(self) -> None:
        spool = self.root / "capture.pcm"
        spool.write_bytes(b"\x01\x00" * 4)
        for aliases in ({}, {field: "source.wav" for field in AUDIO_PATH_FIELDS}):
            detection = {**aliases, "recording": "stream", "keyword_id": 1,
                         "time_s": 2 / 16000, "confidence": 0.5, "_execution_path": "old.wav"}
            captures = materialize_captures(detections=[detection], spool_path=spool,
                output=self.root, context_samples=1, total_samples=4)
            capture = captures[0]
            self.assertEqual(set(capture) & set(AUDIO_PATH_FIELDS), {"path"})
            self.assertNotIn("_execution_path", capture)
            self.assertEqual(pathlib.Path(canonical_audio_path(capture)).read_bytes()[44:], b"\x01\x00" * 2)

    def test_replay_lookup_rebinds_without_stale_aliases(self) -> None:
        path = str(self.root / "eval.wav")
        digest = sha256_file(pathlib.Path(path))
        row = {**{field: path for field in AUDIO_PATH_FIELDS}, "tokens": ["wake"],
               "wav_sha256": digest, "_execution_path": "stale.wav"}
        found = find_domain_row([row], {"case_id": "fixture", "rendered_wav_sha256": digest,
                                        "expected_tokens": ["wake"]})
        self.assertEqual(set(found) & set(AUDIO_PATH_FIELDS), {"path"})
        self.assertNotIn("_execution_path", found)
        self.assertEqual(canonical_audio_path(found), path)

    def test_empty_and_truncated_audio_rejected(self) -> None:
        wav = self.root / "empty.wav"
        write_wav(wav, [])
        with self.assertRaisesRegex(ValueError, "empty PCM"):
            inspect_pcm16_wav(wav)
        wav.write_bytes((self.root / "eval.wav").read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "truncated PCM"):
            inspect_pcm16_wav(wav)


if __name__ == "__main__":
    unittest.main()
