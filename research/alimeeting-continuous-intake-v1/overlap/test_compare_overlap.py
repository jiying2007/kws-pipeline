#!/usr/bin/env python3
"""No audio, repository imports, external services, or third-party dependencies."""
import copy
import tempfile
from pathlib import Path
import unittest

from compare_overlap import compare, dataset, entry, INDEX_SCHEMA, write_new


def fixture():
    row = {"recording": "alimeeting-eval-R8001_M8001", "session_id": "R8001_M8001",
           "member": "Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8001_MS801.wav",
           "sample_rate_hz": 16000, "channel_index": 0,
           "source_wav_sha256": "a" * 64, "wav_sha256": "b" * 64, "pcm_sha256": "c" * 64}
    audit = {"schema": "alimeeting-eval-audit-v1", "rows": [row], "unique_session_count": 1}
    historical = {"path": "old/train/d00000.wav", "file_sha256": "d" * 64, "pcm_sha256": "e" * 64}
    index = {"schema": INDEX_SCHEMA, "datasets": [dataset("old", [entry(historical, "old:1")], [], "fixture")]}
    return index, audit


class ComparisonTests(unittest.TestCase):
    def test_no_match_never_means_independence(self):
        result = compare(*fixture())
        self.assertEqual(result["exact_hash_match_count"], 0)
        self.assertFalse(result["independence_verified"])
        self.assertFalse(result["sealed_qualification_allowed"])
        self.assertEqual(result["accepted_negative_duration_s_added"], 0)

    def test_derived_pcm_match(self):
        index, audit = fixture()
        index["datasets"][0]["records"][0]["hashes"]["pcm_sha256"] = "c" * 64
        result = compare(index, audit)
        self.assertEqual(result["exact_hash_match_count"], 1)
        self.assertEqual(result["comparison_status"], "exact_overlap_found")

    def test_original_source_wav_matches_file_alias(self):
        index, audit = fixture()
        index["datasets"][0]["records"][0]["hashes"]["file_sha256"] = "a" * 64
        self.assertEqual(compare(index, audit)["rows"][0]["exact_hash_matches"][0]["candidate_hash_field"], "source_wav_sha256")

    def test_ancestor_pcm_alias(self):
        index, audit = fixture()
        index["datasets"][0]["records"][0]["hashes"]["source_pcm_sha256"] = "c" * 64
        self.assertEqual(compare(index, audit)["exact_hash_match_count"], 1)

    def test_file_and_pcm_namespaces_do_not_cross(self):
        index, audit = fixture()
        index["datasets"][0]["records"][0]["hashes"]["file_sha256"] = "c" * 64
        self.assertEqual(compare(index, audit)["exact_hash_match_count"], 0)

    def test_session_name_in_source_filename(self):
        index, audit = fixture()
        index["datasets"][0]["records"][0]["path"] = "renamed/R8001_M8001_MS999.wav"
        self.assertGreater(compare(index, audit)["source_session_name_match_count"], 0)

    def test_name_substring_not_session_match(self):
        index, audit = fixture()
        index["datasets"][0]["records"][0]["path"] = "renamed/R8001_M80010_MS999.wav"
        self.assertEqual(compare(index, audit)["source_session_name_match_count"], 0)

    def test_missing_hash_is_rejected(self):
        index, audit = fixture()
        del audit["rows"][0]["source_wav_sha256"]
        with self.assertRaises(ValueError):
            compare(index, audit)

    def test_invalid_hash_is_rejected(self):
        index, audit = fixture()
        audit["rows"][0]["pcm_sha256"] = "A" * 64
        with self.assertRaises(ValueError):
            compare(index, audit)

    def test_unexpected_pcm_semantics_rejected(self):
        for field, value in (("sample_rate_hz", 8000), ("channel_index", 1)):
            index, audit = fixture()
            audit["rows"][0][field] = value
            with self.assertRaises(ValueError):
                compare(index, audit)

    def test_duplicate_sessions_rejected(self):
        index, audit = fixture()
        audit["rows"].append(copy.deepcopy(audit["rows"][0]))
        audit["rows"][-1]["recording"] += "-duplicate"
        audit["unique_session_count"] = 2
        with self.assertRaises(ValueError):
            compare(index, audit)

    def test_unknown_schemas_rejected(self):
        for target in (0, 1):
            values = list(fixture())
            values[target]["schema"] = "unreviewed"
            with self.assertRaises(ValueError):
                compare(*values)

    def test_historical_count_mismatch_rejected(self):
        index, audit = fixture()
        index["datasets"][0]["record_count"] = 9
        with self.assertRaises(ValueError):
            compare(index, audit)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_new(path, {"first": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {"first": False})


if __name__ == "__main__":
    unittest.main()
