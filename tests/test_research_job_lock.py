#!/usr/bin/env python3
"""Real Linux descriptor inheritance, with tiny stdlib child processes only."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from research_job_lock import FD_ENV, PATH_ENV, SUPERVISED_ENV, heavy_model_slot


class HeavySlotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "heavy-model.lock"
        self.environment = {k: v for k, v in os.environ.items() if k not in (FD_ENV, PATH_ENV, SUPERVISED_ENV)}

    def test_supervised_flag_requires_and_preserves_a_verified_inherited_slot(self):
        with self.assertRaisesRegex(ValueError, "SUPERVISED_JOB_REQUIRES_INHERITED_SLOT"):
            with heavy_model_slot(self.path, environment={SUPERVISED_ENV: "1"}):
                pass
        with heavy_model_slot(self.path, environment=self.environment) as parent:
            environment = parent.child_environment(self.environment)
            environment[SUPERVISED_ENV] = "1"
            with heavy_model_slot(self.path, environment=environment) as child:
                self.assertTrue(child.supervised)
                self.assertTrue(child.inherited)
                self.assertEqual(child.child_environment(environment)[SUPERVISED_ENV], "1")

    def test_same_inode_other_open_is_busy_and_then_released(self):
        with heavy_model_slot(self.path, environment=self.environment):
            with self.assertRaisesRegex(BlockingIOError, "HEAVY_MODEL_SLOT_BUSY"):
                with heavy_model_slot(self.path, environment=self.environment):
                    self.fail("second owner acquired the same slot")
        with heavy_model_slot(self.path, environment=self.environment) as lease:
            self.assertFalse(lease.inherited)

    def test_child_reuses_descriptor_without_unlocking_parent(self):
        code = "\n".join((
            "import sys;sys.path.insert(0,sys.argv[1])",
            "from research_job_lock import heavy_model_slot",
            "with heavy_model_slot(sys.argv[2]) as lease:",
            " print('inherited='+str(lease.inherited))",
        ))
        with heavy_model_slot(self.path, environment=self.environment) as lease:
            child = subprocess.run([sys.executable, "-S", "-B", "-c", code, str(ROOT / "tools"), str(self.path)],
                                   env=lease.child_environment(self.environment), pass_fds=lease.pass_fds,
                                   capture_output=True, text=True, timeout=5, check=True)
            self.assertEqual(child.stdout.strip(), "inherited=True")
            with self.assertRaises(BlockingIOError):
                with heavy_model_slot(self.path, environment=self.environment):
                    pass

    def test_incomplete_closed_and_wrong_file_inheritance_fail(self):
        with self.assertRaisesRegex(ValueError, "INHERITANCE_INCOMPLETE"):
            with heavy_model_slot(self.path, environment={FD_ENV: "123"}):
                pass
        self.path.touch()
        with self.assertRaises(OSError):
            with heavy_model_slot(self.path, environment={FD_ENV: "999999", PATH_ENV: str(self.path)}):
                pass
        other = self.root / "another.lock"
        with heavy_model_slot(other, environment=self.environment) as lease:
            altered = {FD_ENV: str(lease.fd), PATH_ENV: str(self.path)}
            with self.assertRaisesRegex(ValueError, "DESCRIPTOR_IDENTITY"):
                with heavy_model_slot(self.path, environment=altered):
                    pass

    def test_alias_path_is_rejected(self):
        self.path.touch()
        link = self.root / "alias.lock"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "HEAVY_LOCK_ALIAS"):
            with heavy_model_slot(link, environment=self.environment):
                pass


if __name__ == "__main__":
    unittest.main()
