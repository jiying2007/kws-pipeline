"""B27 bridge tests use invented metadata and mocked execution boundaries only."""
import ast
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location('q1_delivery_contract', HERE / 'once_b27.py')
m = importlib.util.module_from_spec(definition)
definition.loader.exec_module(m)
o, d = m.original, m.d
definition = importlib.util.spec_from_file_location('existing_delivery_fixtures', HERE / 'test_delivery.py')
fixtures = importlib.util.module_from_spec(definition)
definition.loader.exec_module(fixtures)
definition = importlib.util.spec_from_file_location('b27_delivery_binder_contract', HERE / 'bind_b27.py')
binder = importlib.util.module_from_spec(definition)
definition.loader.exec_module(binder)


class B27DeliveryTests(unittest.TestCase):
    def complete_draft(self, root, stack):
        _, _, manifest = fixtures.DeliveryTests().fixture(stack)
        manifest['candidate_id'] = 'B27'
        manifest['candidate_pack'] = 'models/b27-challenger.pack'
        manifest['files'] = {name.replace('models/invented.pack', 'models/b27-challenger.pack')
            .replace('models/invented.', 'models/mero43-b27-challenger192.'): pin for name, pin in manifest['files'].items()}
        manifest['files'][binder.d.B0_PATH] = dict(bytes=binder.d.PACK_BYTES, sha256=binder.d.B0_SHA, mode=0o644)
        for lock in manifest['quality_locks'].values():
            lock.update(schema='english-v6-b27-original-quality-lock-v1', external_proof_binding_sha256='3' * 64)
        _, report, verifier = self.report_fixture(root)
        for key in ('candidate_id', 'candidate_pack', 'pipeline_commit', 'files', 'engineering', 'quality_locks', 'scope'):
            report[key] = manifest[key]
        source = root / 'complete-report.json'; source.write_bytes(d.encoded(report))
        script = root / 'complete-verifier.py'; script.write_bytes(verifier)
        return source, script

    def test_original_implementation_and_serialization_loader_callables_are_unchanged(self):
        self.assertEqual(hashlib.sha256((HERE / 'once.py').read_bytes()).hexdigest(), m.ORIGINAL_SHA)
        for name in ('baseline_container', 'npy_payload', 'load_checks', 'fetch_originals',
                     'pipeline_contract', 'guard', 'one_execution', 'api'):
            function = getattr(o, name)
            self.assertEqual(Path(function.__code__.co_filename), HERE / 'once.py')
        self.assertEqual(o.B0_NPZ, '02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696')
        self.assertEqual(d.B0_SHA, 'ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b')
        self.assertEqual(d.PIPELINE_COMMIT, '88a043c6c00bb3a1a62f45098a023c7601042ded')

    def test_null_selection_stops_before_paths_originals_network_or_installation(self):
        with patch.object(d, 'selection', return_value=dict(selected=None, manifest=None)), \
             patch.object(m, 'reviewed_readback', side_effect=AssertionError('no verifier path')), \
             patch.object(o, 'api', side_effect=AssertionError('no network')), \
             patch.object(o, 'baseline_container', side_effect=AssertionError('no B0 serialization')), \
             patch.object(o, 'load_checks', side_effect=AssertionError('no compiler')), \
             patch.object(d, 'package', side_effect=AssertionError('no package')):
            with self.assertRaisesRegex(ValueError, 'INERT'):
                m.execute('/unused', '0' * 40, '/unused/output')

    def test_inert_entry_has_exact_q1_schema_and_no_selection_authority(self):
        value = dict(schema=m.ENTRY_SCHEMA, enabled=False, pipeline_repository=o.PIPELINE_REPO,
                     pipeline_commit=None, manifest_sha256=None, pipeline_files=None)
        self.assertEqual(o.entry_contract(value), value)
        for key in ('pipeline_commit', 'manifest_sha256', 'pipeline_files'):
            changed = dict(value); changed[key] = 'invented'
            with self.assertRaises(ValueError):
                o.entry_contract(changed)
        changed = dict(value); changed['schema'] = 'english-v6-b24r1-software-delivery-entry-v1'
        with self.assertRaises(ValueError):
            o.entry_contract(changed)

    def test_pipeline_capture_includes_both_original_and_q1_entry_and_sha_named_readback(self):
        value = dict(readback=dict(report=dict(sha256='1' * 64), verifier=dict(sha256='2' * 64)))
        raw = b'invented manifest bytes'
        names = o.expected_pipeline_files(value, raw)
        self.assertEqual(len(names), 10)
        self.assertTrue({'research/english_v6_delivery/once.py',
                         'research/english_v6_delivery/once_b27.py',
                         'research/english_v6_delivery/delivery.py', o.PACKER} <= names)
        self.assertIn('research/english_v6_delivery/readbacks/' + '1' * 64 + '/completion.json', names)
        self.assertIn('research/english_v6_delivery/readbacks/' + '2' * 64 + '.py', names)

    def test_b27_external_proof_is_explicit_and_every_core_role_keeps_128mb(self):
        with ExitStack() as stack:
            _, _, manifest = fixtures.DeliveryTests().fixture(stack)
            manifest['candidate_id'] = 'B27'
            for lock in manifest['quality_locks'].values():
                lock.update(schema='english-v6-b27-original-quality-lock-v1', external_proof_binding_sha256='e' * 64)
            for panel in ('primary', 'replication'):
                for role in ('result', 'producer', 'consumed'):
                    manifest['quality_locks'][panel][role]['bytes'] = 128_000_000
            fixtures.d.manifest_contract(manifest)
            for panel in ('primary', 'replication'):
                for role in ('result', 'producer', 'consumed'):
                    bad = copy.deepcopy(manifest); bad['quality_locks'][panel][role]['bytes'] += 1
                    with self.subTest(panel=panel, role=role), self.assertRaises(ValueError):
                        fixtures.d.manifest_contract(bad)
                bad = copy.deepcopy(manifest); del bad['quality_locks'][panel]['external_proof_binding_sha256']
                with self.assertRaises(ValueError): fixtures.d.manifest_contract(bad)
                bad = copy.deepcopy(manifest); bad['quality_locks'][panel]['external_proof_binding_sha256'] = None
                with self.assertRaises(ValueError): fixtures.d.manifest_contract(bad)
            bad = copy.deepcopy(manifest); bad['candidate_id'] = 'B24R1'
            with self.assertRaisesRegex(ValueError, 'only to the B27 candidate'):
                fixtures.d.manifest_contract(bad)

    def test_legacy_and_lookalike_schemas_keep_exact_original_role_caps(self):
        with ExitStack() as stack:
            _, _, manifest = fixtures.DeliveryTests().fixture(stack)
            for schema in ('english-v6-b24r1-original-quality-lock-v1',
                           'english-v6-b999-original-quality-lock-v1',
                           'english-v6-b270-original-quality-lock-v1'):
                for lock in manifest['quality_locks'].values():
                    lock['schema'] = schema
                manifest['quality_locks']['replication']['producer']['bytes'] = 512_000_000
                fixtures.d.manifest_contract(manifest)
                manifest['quality_locks']['replication']['producer']['bytes'] += 1
                with self.subTest(schema=schema), self.assertRaises(ValueError):
                    fixtures.d.manifest_contract(manifest)

    def report_fixture(self, root):
        verifier = b'# invented exact verifier bytes; never executed\n'
        value = dict(candidate_id='B27', candidate_pack='models/b27-challenger.pack',
            pipeline_commit=d.PIPELINE_COMMIT, files={'invented': {}}, engineering={'invented': True},
            quality_locks={panel: dict(panel=panel, schema='english-v6-b27-original-quality-lock-v1', external_proof_binding_sha256='3' * 64)
                           for panel in ('primary', 'replication')},
            scope=copy.deepcopy(d.SCOPE), readback=dict(verifier=o.pin(verifier)))
        report = dict(schema=m.REPORT_SCHEMA,
            status='VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE',
            baseline_pack_requires_separate_byte_authentication=True, readback_verifier=o.pin(verifier),
            **{key: value[key] for key in ('candidate_id', 'candidate_pack', 'pipeline_commit',
                                         'files', 'engineering', 'quality_locks', 'scope')})
        report.update(checks=dict(complete_original_TRAIN_verified=True, complete_original_Q1_failure_verified=True,
            external_originals_retained=True), original_training_report_sha256='4' * 64,
            original_B24Q1_failure_report_sha256='9cd5df45c5b9ca22b0eb56f3fdda4788b2420e906a35bb754af9c23eb38be62f')
        report['quality'] = {panel: dict(schema='english-v6-b27-verified-original-quality-v1', panel=panel, panel_gate='PASS',
            original_training_report_sha256=report['original_training_report_sha256'],
            original_B24Q1_failure_report_sha256=report['original_B24Q1_failure_report_sha256'],
            external_proof_binding_sha256='3' * 64, all_original_training_proof_verified=True,
            complete_original_Q1_failure_verified=True) for panel in ('primary', 'replication')}
        self.save_report(root, value, report, verifier)
        return value, report, verifier

    def save_report(self, root, value, report, verifier):
        raw = d.encoded(report); value['readback']['report'] = o.pin(raw)
        directory = root / 'readbacks' / d.sha(raw); directory.mkdir(parents=True, exist_ok=True)
        (directory / 'completion.json').write_bytes(raw)
        (root / 'readbacks' / (d.sha(verifier) + '.py')).write_bytes(verifier)

    def test_exact_q1_report_and_verifier_are_reopened_but_never_executed(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(o, 'HERE', Path(temp)):
            value, _, verifier = self.report_fixture(Path(temp))
            with patch.object(o.subprocess, 'run', side_effect=AssertionError('no saved verifier execution')), \
                 patch.object(o.importlib.util, 'spec_from_file_location', side_effect=AssertionError('no saved verifier import')):
                self.assertEqual(m.reviewed_readback(value)['readback.py'], verifier)
            o.readback_paths(value)['readback.py'].write_bytes(b'changed verifier')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                m.reviewed_readback(value)

    def test_old_summary_candidate_schema_and_mismatched_reviewed_fields_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(o, 'HERE', Path(temp)):
            root = Path(temp)
            value, report, verifier = self.report_fixture(root)
            for key in ('candidate_pack', 'pipeline_commit', 'files', 'engineering', 'quality_locks', 'scope'):
                bad = copy.deepcopy(value); bad[key] = 'unrelated'
                with self.subTest(key=key), self.assertRaises(ValueError):
                    m.reviewed_readback(bad)
            for key, replacement in (('schema', 'english-v6-b24r1-verified-original-engineering-v1'),
                                     ('candidate_id', 'B24R1'), ('status', 'PASS'),
                                     ('baseline_pack_requires_separate_byte_authentication', False)):
                bad_report = copy.deepcopy(report); bad_report[key] = replacement
                bad_value = copy.deepcopy(value)
                self.save_report(root, bad_value, bad_report, verifier)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    m.reviewed_readback(bad_value)

    def test_missing_failed_qualification_training_or_either_pass_stays_closed(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(o, 'HERE', Path(temp)):
            root = Path(temp); value, report, verifier = self.report_fixture(root)
            changes = (
                lambda x: x['checks'].update(complete_original_TRAIN_verified=False),
                lambda x: x['checks'].update(complete_original_Q1_failure_verified=False),
                lambda x: x.update(original_B24Q1_failure_report_sha256='5' * 64),
                lambda x: x['quality']['primary'].update(panel_gate='NO_PROMOTION'),
                lambda x: x['quality']['replication'].update(all_original_training_proof_verified=False),
                lambda x: x['quality']['replication'].update(external_proof_binding_sha256='6' * 64),
                lambda x: x['quality']['replication'].update(original_training_report_sha256='7' * 64),
            )
            for mutate in changes:
                broken, selected = copy.deepcopy(report), copy.deepcopy(value)
                mutate(broken); self.save_report(root, selected, broken, verifier)
                with self.assertRaises(ValueError): m.reviewed_readback(selected)

    def test_original_engineering_metadata_must_be_q1_first_success_and_exact_role(self):
        pin = dict(id=55, run_id=66, run_head='a' * 40, bytes=77, sha256='b' * 64)
        run = dict(id=66, head_sha='a' * 40, status='completed', conclusion='success', run_attempt=1,
            repository=dict(full_name=o.DATA_REPO, private=False), head_repository=dict(full_name=o.DATA_REPO, private=False),
            event='push', path=m.ENGINEERING_WORKFLOW, head_branch=m.ENGINEERING_BRANCH)
        artifact = dict(id=55, expired=False, size_in_bytes=77, digest='sha256:' + 'b' * 64,
                        workflow_run=dict(id=66, head_sha='a' * 40), name=m.artifact_name('result', 66))
        commit = dict(sha='a' * 40)
        m.provenance(pin, run, artifact, commit, 'result')
        for change in (lambda r, a: r.update(conclusion='failure'), lambda r, a: r.update(run_attempt=2),
                       lambda r, a: r.update(path='.github/workflows/english-v6-b24r1-engineering-once.yml'),
                       lambda r, a: r.update(head_branch='research/en-v6-b24r1-engineering-once-20261124'),
                       lambda r, a: a.update(name='english-v6-b24r1-software-sdk-evidence-66'),
                       lambda r, a: a.update(expired=True), lambda r, a: a.update(size_in_bytes=78)):
            r, a = copy.deepcopy(run), copy.deepcopy(artifact); change(r, a)
            with self.assertRaises(ValueError):
                m.provenance(pin, r, a, commit, 'result')
        with self.assertRaisesRegex(ValueError, 'unchanged complete original B0'):
            m.provenance(pin, run, artifact, commit, 'B0')

    def test_q1_first_push_guard_keeps_two_substitutions_and_workflow_only_child(self):
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=o.DATA_REPO, GITHUB_EVENT_NAME='push',
            GITHUB_REF=m.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_RUN_ID='100', GITHUB_SHA='a' * 40)
        event = dict(repository=dict(full_name=o.DATA_REPO, private=False), before='b' * 40,
                     after='a' * 40, ref=m.BRANCH)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template = m.TOKEN + b'\n' + m.TOKEN + b'\n'
            for path, raw in ((m.TEMPLATE, template), (m.WORKFLOW, template.replace(m.TOKEN, b'b' * 40))):
                target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
            outputs = [b'a' * 40 + b'\n', b'a' * 40 + b' ' + b'b' * 40 + b'\n',
                       m.WORKFLOW.encode() + b'\n', b'']
            with patch.object(o, 'git', side_effect=outputs):
                identity = o.guard('b' * 40, root, env, event)
            self.assertEqual(identity['run_branch'], m.BRANCH)
            wrong = dict(env); wrong['GITHUB_RUN_ATTEMPT'] = '2'
            with patch.object(o, 'git', side_effect=AssertionError('no Git for retry')), self.assertRaises(ValueError):
                o.guard('b' * 40, root, wrong, event)
            wrong = dict(env); wrong['GITHUB_REF'] = 'refs/heads/research/en-v6-b24r1-delivery-once-20261125'
            with patch.object(o, 'git', side_effect=AssertionError('no Git for old branch')), self.assertRaises(ValueError):
                o.guard('b' * 40, root, wrong, event)

    def test_orchestration_uses_original_transport_and_exact_q1_receipt_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); data = root / 'data'; data.mkdir()
            bound = data / m.BINDING; bound.parent.mkdir(parents=True); bound.write_bytes(b'invented binding')
            output = root / 'output'
            value = dict(candidate_pack='models/b27-challenger.pack', files={'models/b27-challenger.pack': dict(bytes=5, sha256='1' * 64)},
                         pipeline_commit=d.PIPELINE_COMMIT, engineering={'fixture_only': True}, readback={'fixture_only': True})
            binding = dict(pipeline_commit='a' * 40, pipeline_files={'fixture_only': True})
            stack.enter_context(patch.object(d, 'trusted_manifest', return_value=(value, b'invented manifest')))
            stack.enter_context(patch.object(m, 'reviewed_readback', return_value={'completion.json': b'invented report', 'readback.py': b'# not executed'}))
            stack.enter_context(patch.object(d, 'document', return_value=binding))
            stack.enter_context(patch.object(o, 'entry_contract', return_value=binding))
            stack.enter_context(patch.object(o, 'pipeline_contract'))
            stack.enter_context(patch.object(o, 'guard', return_value={'fixture_only': True}))
            stack.enter_context(patch.object(o, 'one_execution', return_value={'fixture_only': True}))
            fetched = stack.enter_context(patch.object(o, 'fetch_originals', return_value={'fixture_only': True}))
            serialized = stack.enter_context(patch.object(o, 'baseline_container', return_value=(b'invented unchanged B0 bytes', {'fixture_only': True})))
            packaged = stack.enter_context(patch.object(d, 'package', return_value={'fixture_only': True}))
            installed = stack.enter_context(patch.object(d, 'install', return_value={'fixture_only': True}))
            stack.enter_context(patch.object(d, 'verify_installed', return_value=(value, output / 'installed/version')))
            loaded = stack.enter_context(patch.object(o, 'load_checks', return_value=dict(model_load_calls=2, model_forwards=0)))
            stack.enter_context(patch.object(o.subprocess, 'run', side_effect=AssertionError('no real compiler or scientific process')))
            result = m.execute(data, 'b' * 40, output)
            for operation in (fetched, serialized, packaged, installed, loaded):
                self.assertEqual(operation.call_count, 1)
            self.assertEqual(packaged.call_args.args[2].name, 'heym43-B27-software-delivery.tar.gz')
            self.assertEqual(result['schema'], 'english-v6-b27-software-delivery-result-v1')
            for key in ('model_forwards', 'optimizer_updates', 'pcm_samples_consumed', 'stream_init_calls'):
                self.assertEqual(result[key], 0)
            self.assertFalse(result['sdk_rebuilt'])
            self.assertFalse(result['weights_modified'])
            self.assertEqual(json.loads((output / 'receipts/RESULT.json').read_bytes()), result)

    def test_new_adapter_does_not_introduce_a_scientific_runtime_or_repacker(self):
        tree = ast.parse((HERE / 'once_b27.py').read_bytes())
        imports = {a.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
        self.assertFalse(imports & {'torch', 'numpy', 'onnxruntime', 'piper', 'soundfile', 'transformers'})
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(set(calls) & {'forward', 'launch', 'execv', 'pack', 'npy_payload'})

    def test_binder_produces_only_review_metadata_and_does_not_select_or_execute_verifier(self):
        before = binder.d.SELECTION.read_bytes()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); report, verifier = self.complete_draft(root, stack)
            stack.enter_context(patch.object(binder.original.subprocess, 'run', side_effect=AssertionError('no execution')))
            result = binder.manifest(report, verifier, root / 'draft')
            self.assertEqual(result['status'], 'READY_FOR_REVIEW_ONLY')
            self.assertFalse(result['actual_selection_changed'])
            chosen = json.loads((root / 'draft/selection.json').read_bytes())
            self.assertEqual(chosen['selected'], 'B27')
            selected = json.loads((root / 'draft' / chosen['manifest']['path']).read_bytes())
            binder.d.manifest_contract(selected)
            self.assertEqual(selected['readback']['report'], o.pin(report.read_bytes()))
            broken = json.loads(report.read_bytes()); broken['quality']['replication']['panel_gate'] = 'NO_PROMOTION'
            report.write_bytes(d.encoded(broken))
            with self.assertRaises(ValueError): binder.manifest(report, verifier, root / 'rejected')
            self.assertFalse((root / 'rejected').exists())
        self.assertEqual(binder.d.SELECTION.read_bytes(), before)

    def test_entry_binder_requires_already_committed_selection_and_exact_ten_git_files(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); report, verifier = self.complete_draft(root, stack)
            with patch.object(binder.original, 'git', side_effect=AssertionError('no Git before selection')):
                with self.assertRaisesRegex(ValueError, 'INERT'): binder.entry('a' * 40, root / 'closed')
            binder.manifest(report, verifier, root / 'draft')
            repository = root / 'repository'; here = repository / 'research/english_v6_delivery'
            shutil.copytree(root / 'draft', here)
            source_root = binder.runner.ROOT
            for name in binder.runner.DELIVERY_FILES - {'research/english_v6_delivery/selection.json'}:
                target = repository / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source_root / name).read_bytes())
            for obj, name, value in ((binder.runner, 'ROOT', repository), (binder.original, 'ROOT', repository),
                                     (binder.original, 'HERE', here), (binder.d, 'HERE', here),
                                     (binder.d, 'SELECTION', here / 'selection.json')):
                stack.enter_context(patch.object(obj, name, value))
            value, raw = binder.d.trusted_manifest(); names = binder.original.expected_pipeline_files(value, raw)
            def git(unused_root, *args):
                if args == ('rev-parse', 'HEAD'): return b'a' * 40 + b'\n'
                if args == ('status', '--porcelain', '--untracked-files=no'): return b''
                self.assertEqual(args[:3], ('ls-tree', '-z', 'HEAD'))
                return b''.join(b'100644 blob ' + hashlib.sha1(b'blob ' + str((repository / name).stat().st_size).encode()
                    + b'\0' + (repository / name).read_bytes()).hexdigest().encode() + b'\t' + name.encode() + b'\0' for name in sorted(names))
            stack.enter_context(patch.object(binder.original, 'git', side_effect=git))
            result = binder.entry('a' * 40, root / 'entry')
            self.assertFalse(result['actual_data_entry_changed'])
            actual = json.loads((root / 'entry/engineering_delivery_inputs.json').read_bytes())
            self.assertEqual(len(actual['pipeline_files']), 10)
            self.assertEqual(set(actual['pipeline_files']), names)
            self.assertEqual(actual['pipeline_commit'], 'a' * 40)
            with self.assertRaises(ValueError): binder.entry('b' * 40, root / 'wrong-commit')


if __name__ == '__main__':
    unittest.main()
