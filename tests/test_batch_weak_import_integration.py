"""Public wiring regressions; artificial contracts only, no native runtime."""
from pathlib import Path
import hashlib
import importlib.util
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]
BUSINESS=('docs/batch-weak-import.md','tests/test_batch_weak_import.py',
    'tools/batch-source-contract.json','tools/batch_weak_import.py','tools/language_conditioned_vote.py')
TESTS=('test_batch_weak_import.py','test_batch_weak_import_integration.py')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class BatchIntegration(unittest.TestCase):
    def test_all_batch_paths_select_research_dependencies(self):
        changes=module('batch_integration_changes',ROOT/'tools/research_ci_changes.py')
        for name in (*BUSINESS,'tests/test_batch_weak_import_integration.py'):
            with self.subTest(path=name):self.assertTrue(changes.relevant([name]))
        self.assertFalse(changes.relevant(['unrelated-fixture.txt']))

    def test_existing_ci_block_has_three_ungated_modes_and_guard(self):
        inventory=module('batch_integration_inventory',ROOT/'tools/test_inventory.py')
        source=(ROOT/'.github/workflows/ci.yml').read_text()
        wanted={'python3 -S -B '+mode+'tests/'+test for test in TESTS for mode in ('','-O ','-OO ')}
        matching=[]
        for job,block,gated in inventory.run_blocks(source):
            found=wanted & {line.strip() for line in block.splitlines()}
            if found:matching.append((job,found,gated,block))
        self.assertEqual(len(matching),1)
        self.assertEqual(matching[0][1],wanted);self.assertFalse(matching[0][2])
        self.assertIn('tests/test_speech_data_queue.py',matching[0][3])
        self.assertIn('tests/test_language_conditioned_queue.py',matching[0][3])
        self.assertEqual(source.count('- name: Maintained speech data queue research-only contracts'),1)
        self.assertIn('Verify tracked worktree after Maintained speech data queue contracts',source)

    def test_inventory_declares_both_tests_without_claiming_execution(self):
        inventory=module('batch_integration_inventory_report',ROOT/'tools/test_inventory.py')
        report=inventory.inventory(ROOT/'tests',[ROOT/'.github/workflows/ci.yml'])
        self.assertFalse(report['execution_coverage_proven'])
        for name in TESTS:
            routes=report['files'][name]
            self.assertEqual(len(routes),3)
            self.assertEqual({r['scope'] for r in routes},{'mainline-required'})
            self.assertEqual({r['kind'] for r in routes},{'script'})

    def test_navigation_exposes_bounded_batch_entry_and_scope(self):
        tools=(ROOT/'tools/README.md').read_text();docs=(ROOT/'docs/README.md').read_text()
        self.assertIn('(batch_weak_import.py)',tools)
        self.assertIn('(batch-source-contract.json)',tools)
        self.assertIn('(../docs/batch-weak-import.md)',tools)
        self.assertIn('(batch-weak-import.md)',docs)
        guide=(ROOT/'docs/batch-weak-import.md').read_text()
        self.assertIn('Dylan/train and Serena/development_a',guide)
        self.assertIn('current_experiment_admission=false',guide)

    def test_public_retention_receipt_preserves_history_and_source_identities(self):
        retention=json.loads((ROOT/'research/consolidation/source-retention-2026-10-07.json').read_bytes())
        matches=[r for r in retention['active_workflow_maintenance']
            if r.get('id')=='bounded-batch-weak-import-integration-20261018']
        self.assertEqual(len(matches),1);record=matches[0]
        self.assertEqual(record['base_commit'],'fe8e78c2cff07a7b1c632deab15cf565e7b4d654')
        rows={r['path']:r for r in record['files']}
        self.assertTrue(set(BUSINESS)|{'tests/test_batch_weak_import_integration.py','.github/workflows/ci.yml',
            'tools/research_ci_changes.py','tools/README.md','docs/README.md'}<=set(rows))
        # A historical record is not a permanent runtime byte pin. Validate its
        # structure; the integration preparation separately binds current bytes.
        for row in rows.values():
            self.assertEqual(row['after']['path'],row['path'])
            self.assertEqual(len(row['after']['sha256']),64)
            self.assertEqual(len(row['after']['git_blob_sha1']),40)
        older=[r.get('id') for r in retention['active_workflow_maintenance'][:retention['active_workflow_maintenance'].index(record)]]
        self.assertIn('language-conditioned-weak-import-20261017',older)
        self.assertIn('maintained-speech-data-queue-20261017',older)


if __name__=='__main__':unittest.main()
