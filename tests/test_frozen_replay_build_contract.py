#!/usr/bin/env python3
"""Fail-closed regressions for the historical replay CMake guard."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_frozen_replay_build_contract import INSERTIONS, verify_contract


class BuildContractTest(unittest.TestCase):
    def setUp(self):
        # Derive the historical fixture by undoing the exact reviewed additions.
        self.current = (ROOT / "CMakeLists.txt").read_bytes()
        self.historical = self.current
        for anchor, addition in reversed(INSERTIONS):
            self.assertEqual(self.historical.count(anchor + addition), 1)
            self.historical = self.historical.replace(anchor + addition, anchor, 1)

    def test_unchanged(self):
        self.assertEqual(verify_contract(self.historical, self.historical), "unchanged")
        self.assertEqual(verify_contract(self.current, self.current), "unchanged")

    def test_exact_additions(self):
        self.assertEqual(verify_contract(self.historical, self.current),
                         "exact-trace-slice-target-addition")

    def test_reject_semantic_and_unreviewed_changes(self):
        changes = [
            (b"PRIVATE KwsPipeline::core", b"PRIVATE another_runtime"),
            (b"PRIVATE kws_trace_io)", b"PRIVATE kws_trace_io another_library)"),
            (b"${KWS_STRICT_OPTIONS}", b"-ffast-math"),
            (b"if(KWS_BUILD_TESTS)", b"if(TRUE)"),
            (b"tools/kws_trace_slice.c", b"tools/other_source.c"),
            (b"COMMAND kws_trace_slice --self-test", b"COMMAND kws_trace_slice --rewrite"),
        ]
        for old, new in changes:
            with self.subTest(change=new):
                self.assertIn(old, self.current)
                with self.assertRaises(ValueError):
                    verify_contract(self.historical, self.current.replace(old, new, 1))
        for suffix in (b"\nadd_compile_options(-ffast-math)\n",
                       b"\nadd_dependencies(kws_pipeline kws_trace_slice)\n",
                       b"\n# unreviewed even if cosmetic\n"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                verify_contract(self.historical, self.current + suffix)

    def test_reject_partial_duplicate_reordered_or_removed_addition(self):
        anchor, addition = INSERTIONS[0]
        variants = [self.current.replace(addition, b"", 1),
                    self.current.replace(addition, addition + addition, 1),
                    addition + self.current.replace(addition, b"", 1),
                    self.historical.replace(anchor, b"", 1)]
        for value in variants:
            with self.subTest(value=value[-100:]), self.assertRaises(ValueError):
                verify_contract(self.historical, value)

    def test_historical_with_new_target_must_be_exact(self):
        with self.assertRaises(ValueError):
            verify_contract(self.current, self.current + b"\n")

    def test_anchor_missing_or_ambiguous(self):
        anchor = INSERTIONS[0][0]
        for historical in (self.historical.replace(anchor, b"", 1),
                           self.historical + anchor):
            with self.assertRaises(ValueError):
                verify_contract(historical, self.current)


if __name__ == "__main__":
    unittest.main()
