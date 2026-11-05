#!/usr/bin/env python3
"""No-download contracts for the separate pinned TTS setup."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import research_runtime
import setup_tts_runtime as setup


class TTSRuntimeContract(unittest.TestCase):
    def test_role_keeps_its_exact_packages_and_native_wheel(self):
        manifest = research_runtime.load_manifest()
        packages = setup.locked_packages(manifest)
        self.assertEqual(len(packages), 92)
        for name, version in manifest['roles']['tts']['packages'].items():
            self.assertEqual(packages[name]['version'], version)
        self.assertEqual(packages['torch']['version'], '2.11.0+cpu')
        self.assertTrue(packages['soundfile']['filename'].endswith('manylinux_2_28_x86_64.whl'))
        self.assertNotIn('qwen-asr', packages)
        self.assertEqual([row['name'] for row in packages.values() if not row['filename'].endswith('.whl')], ['sox'])

    def test_base_model_assets_pin_public_revision_and_full_roster(self):
        lock = json.loads(setup.ASSET_LOCK.read_text())
        self.assertEqual(lock['model_id'], 'Qwen/Qwen3-TTS-12Hz-0.6B-Base')
        self.assertEqual(lock['revision'], '5d83992436eae1d760afd27aff78a71d676296fc')
        self.assertTrue(lock['public'])
        self.assertFalse(lock['gated'])
        self.assertEqual(len(lock['files']), 13)
        self.assertEqual(sum(row['bytes'] for row in lock['files']), lock['total_bytes'])
        self.assertLess(lock['total_bytes'], 3 * 1024**3)
        for row in lock['files']:
            setup.safe_relative(row['filename'])
            self.assertIn('/resolve/' + lock['revision'] + '/', row['source_url'])
            self.assertRegex(row['sha256'], '^[0-9a-f]{64}$')
        weights = {row['filename']: row['sha256'] for row in lock['files']}
        self.assertEqual(weights['model.safetensors'], '180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6')


if __name__ == '__main__':
    unittest.main()
