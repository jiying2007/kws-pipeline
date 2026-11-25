"""Invented B28 metadata and mocked byte transport; no PCM, model or compiler."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = load('b28_software_fixture_runner', HERE / 'once_b28.py')
binder = load('b28_software_fixture_binder', HERE / 'bind_b28.py')
o, d = m.original, m.d


def artifact(number, run):
    return dict(id=number, run_id=run, run_head=f'{run:040x}', bytes=100, sha256=f'{number:064x}')


def fixtures():
    files = {}
    names = {'LICENSE', 'README.md', 'models/b28-challenger.pack',
        *('models/fixture-b28.' + suffix for suffix in ('pt', 'npz', 'onnx')),
        *('sdk-' + backend + '/' + name for backend in d.BACKENDS for name in d.REQUIRED_SDK)}
    for name in names:
        files[name] = dict(bytes=d.PACK_BYTES if name.endswith('.pack') else 101,
            sha256=d.sha(('invented metadata only/' + name).encode()), mode=d.prescribed_mode(name))
    files[d.B0_PATH] = dict(bytes=d.PACK_BYTES, sha256=d.B0_SHA, mode=0o644)
    locks = {}
    for offset, panel in enumerate(('primary', 'replication'), 1):
        locks[panel] = dict(schema='english-v6-b28-original-quality-lock-v1', panel=panel,
            **{role: artifact(offset * 10 + i, offset + 100) for i, role in enumerate(('result', 'producer', 'consumed'), 1)},
            **{key: d.sha((panel + key).encode()) for key in ('evaluator_sha256', 'inputs_sha256',
                'producer_manifest_sha256', 'original_result_sha256', 'external_proof_binding_sha256')})
    verifier = b'# invented fixture verifier bytes; never executed or imported\n'
    value = dict(schema='heym43-software-delivery-v1', candidate_id='B28', candidate_pack='models/b28-challenger.pack',
        pipeline_commit=d.PIPELINE_COMMIT, files=files, quality_locks=locks, scope=copy.deepcopy(d.SCOPE),
        engineering=dict(repository=o.DATA_REPO, artifacts={role: artifact(500 + i, 500)
            for i, role in enumerate(sorted(d.ORIGIN_ROLES))}, result_sha256='a' * 64, manifest_sha256='b' * 64),
        readback=dict(verifier=o.pin(verifier)))
    certificate = dict(schema='english-v6-b28-original-b27-failure-certification-binding-v1',
        original_reader_path='automation/english_v6_b27_quality_readback.py',
        original_reader_sha256='87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa',
        original_report_sha256='2' * 64,
        adapter_path='automation/english_v6_b27_primary_no_promotion_readback.py', adapter_sha256='3' * 64,
        adapter_report_schema='english-v6-b27-primary-rooted-no-promotion-readback-v1', adapter_report_sha256='4' * 64,
        original_standalone_continuation_succeeded=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0)
    report = dict(schema=m.REPORT_SCHEMA, status='VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE',
        **{key: copy.deepcopy(value[key]) for key in ('candidate_id', 'candidate_pack', 'pipeline_commit',
            'files', 'engineering', 'quality_locks', 'scope')},
        baseline_pack_requires_separate_byte_authentication=True, readback_verifier=o.pin(verifier),
        original_training_report_sha256='5' * 64, original_B27_failure_report_sha256='2' * 64,
        original_B27_failure_certification=certificate,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, compiler_calls=0, packs_created=0,
        candidate_selection_changed=False,
        checks=dict(original_engineering_artifacts=4, raw_arrays=140, event_csvs=84, original_conditions=28,
            complete_original_quality_passes=2, frozen_pipeline_files=13, original_producer_files=200,
            natural_objective_returns_verified=384, external_training_originals=71,
            replication_external_total_originals=89,
            **{key: True for key in ('complete_original_TRAIN_verified', 'complete_original_Q1_failure_verified',
                'complete_original_B27_failure_verified', 'external_originals_retained', 'original_pack_tensor_bytes_verified',
                'unchanged_original_exports', 'installed_sdk_bytes_verified', 'every_saved_array_and_event_recomputed',
                'original_source_admission_recomputed', 'original_api_and_git_provenance_verified',
                'natural_no_event_terms_recomputed')}))
    report['quality'] = {}
    models = dict(B0='6' * 64, B9='7' * 64, control='8' * 64,
        challenger=files['models/fixture-b28.pt']['sha256'])
    for panel, lock in locks.items():
        report['quality'][panel] = dict(schema='english-v6-b28-verified-original-quality-v1', panel=panel, panel_gate='PASS',
            model_pt_sha256=copy.deepcopy(models),
            original_training_report_sha256=report['original_training_report_sha256'],
            original_B27_failure_report_sha256=report['original_B27_failure_report_sha256'],
            original_B27_failure_certification=copy.deepcopy(certificate),
            external_proof_binding_sha256=lock['external_proof_binding_sha256'],
            original_result_sha256=lock['original_result_sha256'],
            original_lock_sha256=d.sha(json.dumps(lock, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()),
            natural_objective_returns_verified=384, external_training_originals=71,
            external_total_originals=71 if panel == 'primary' else 89,
            primary_independently_verified=panel == 'replication', replication_open_allowed=panel == 'primary',
            candidate_engineering_validation_allowed=panel == 'replication',
            new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0,
            promotion_allowed=False, product_released=False, automatic_next_assessment_allowed=False,
            **{key: True for key in ('all_original_arrays_verified', 'original_source_admission_recomputed',
                'original_producer_git_verified', 'all_original_training_proof_verified', 'complete_original_Q1_failure_verified',
                'complete_original_B27_failure_verified', 'natural_no_event_terms_recomputed',
                'captured_code_matches_pinned_local_reader_root', 'old_stops_and_reserves_preserved')})
    return rebind_fixture(value, report, verifier)


def rebind_fixture(value, report, verifier):
    value = copy.deepcopy(value)
    raw = d.encoded(report)
    value['readback'] = dict(report=o.pin(raw), verifier=o.pin(verifier))
    saved = {'completion.json': raw, 'readback.py': verifier}
    bound = dict(schema=m.contract.CONFIG_SCHEMA, enabled=True,
        engineering_report=o.pin(raw), engineering_verifier=o.pin(verifier))
    return value, report, saved, bound


def authority(stack, runner, root, value, saved, bound, selected=False):
    root.mkdir(parents=True, exist_ok=True)
    config = root / 'b28_binding.json'
    config.write_bytes(d.encoded(bound))
    raw = d.encoded(value)
    pin = dict(path='manifests/' + d.sha(raw) + '.json', **o.pin(raw))
    chosen = dict(schema='heym43-software-selection-v1', selected='B28' if selected else None,
        manifest=pin if selected else None,
        baseline=dict(path=d.B0_PATH, bytes=d.PACK_BYTES, sha256=d.B0_SHA), scope=copy.deepcopy(d.SCOPE))
    (root / 'selection_b28.json').write_bytes(d.encoded(chosen))
    manifest = root / pin['path']; manifest.parent.mkdir(exist_ok=True); manifest.write_bytes(raw)
    report = root / 'readbacks' / value['readback']['report']['sha256'] / 'completion.json'
    report.parent.mkdir(parents=True); report.write_bytes(saved['completion.json'])
    (root / 'readbacks' / (value['readback']['verifier']['sha256'] + '.py')).write_bytes(saved['readback.py'])
    for obj, name, replacement in ((runner.contract, 'CONFIG', config), (runner.d, 'SELECTION', root / 'selection_b28.json'),
        (runner.d, 'HERE', root), (runner.original, 'HERE', root)):
        stack.enter_context(mock.patch.object(obj, name, replacement))
    return chosen


def inert_authority(stack, runner, root):
    """A closed fixture remains closed after the real checkout is bound."""
    root.mkdir(parents=True, exist_ok=True)
    config = dict(schema=runner.contract.CONFIG_SCHEMA, enabled=False,
        engineering_report=None, engineering_verifier=None)
    chosen = dict(schema='heym43-software-selection-v1', selected=None, manifest=None,
        baseline=dict(path=d.B0_PATH, bytes=d.PACK_BYTES, sha256=d.B0_SHA), scope=copy.deepcopy(d.SCOPE))
    config_path, selection_path = root / 'b28_binding.json', root / 'selection_b28.json'
    config_path.write_bytes(d.encoded(config)); selection_path.write_bytes(d.encoded(chosen))
    stack.enter_context(mock.patch.object(runner.contract, 'CONFIG', config_path))
    stack.enter_context(mock.patch.object(runner.d, 'SELECTION', selection_path))
    return config, chosen


class B28SoftwareTests(unittest.TestCase):
    def test_exact_original_implementations_and_b0_preserved(self):
        self.assertEqual(d.sha((HERE / 'once.py').read_bytes()), m.ORIGINAL_SHA)
        self.assertEqual(d.sha((HERE / 'delivery.py').read_bytes()), m.contract.ORIGINAL_SHA)
        for name in ('baseline_container', 'npy_payload', 'load_checks', 'fetch_originals', 'guard',
            'pipeline_contract', 'one_execution', 'api'):
            self.assertEqual(Path(getattr(o, name).__code__.co_filename), HERE / 'once.py')
        for name in ('prepare_payload', 'package', 'install', 'verify_installed', 'check_load', 'original_archive'):
            function = getattr(d, name)
            self.assertEqual(Path(getattr(function, '__wrapped__', function).__code__.co_filename), HERE / 'delivery.py')
        self.assertEqual(d.B0_SHA, 'ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b')
        self.assertEqual(d.PACK_BYTES, 216960)
        self.assertEqual(d.PIPELINE_COMMIT, '88a043c6c00bb3a1a62f45098a023c7601042ded')
        self.assertEqual(d.sha((HERE / 'selection.json').read_bytes()), '71fa5ea325ffd06ef2308b368e7c01c50e99c3e5eddb4759134ee434d6dbbec5')

    def test_committed_b28_binding_and_own_selection_are_coherent(self):
        bound = m.contract.configuration()
        selected = d.selection()
        if bound['enabled'] is False:
            self.assertIsNone(bound['engineering_report']); self.assertIsNone(bound['engineering_verifier'])
            self.assertIsNone(selected['selected']); self.assertIsNone(selected['manifest'])
        elif selected['selected'] is not None:
            value, _ = d.trusted_manifest()
            self.assertEqual(value['candidate_id'], 'B28')
            m.reviewed_readback(value)
        self.assertEqual(d.SELECTION.name, 'selection_b28.json')
        self.assertEqual(m.contract.CONFIG.name, 'b28_binding.json')

    def test_disabled_binding_rejects_partial_pins_or_nonboolean_enable(self):
        changes = [dict(engineering_report=dict(bytes=1, sha256='a' * 64)), dict(engineering_verifier={}),
            dict(enabled=1), dict(enabled=True), dict(schema='english-v6-b27-software-delivery-binding-v1'), dict(extra=None)]
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            original, _ = inert_authority(stack, m, Path(temp))
            for changed in changes:
                m.contract.CONFIG.write_bytes(d.encoded(dict(original, **changed)))
                with self.subTest(change=changed), self.assertRaises(ValueError): m.contract.configuration()

    def test_inert_proof_stops_all_entrypoints_before_external_effects(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            for index, runner in enumerate((m, binder.runner)):
                inert_authority(stack, runner, Path(temp) / str(index))
                for obj, name in ((runner, 'reviewed_readback'), (runner.original, 'api'),
                    (runner.original, 'baseline_container'), (runner.original, 'load_checks'),
                    (runner.d, 'package'), (runner.d, 'install')):
                    stack.enter_context(mock.patch.object(obj, name, side_effect=AssertionError('no effect before actual binding')))
            stack.enter_context(mock.patch.object(socket, 'socket', side_effect=AssertionError('no network')))
            stack.enter_context(mock.patch.object(o.subprocess, 'run', side_effect=AssertionError('no process')))
            for call in (lambda: m.execute('/missing', 'a' * 40, '/missing/out'),
                lambda: binder.manifest('/missing/report', '/missing/verifier', '/missing/out'),
                lambda: binder.entry('a' * 40, '/missing/out'), lambda: d.trusted_manifest()):
                with self.assertRaisesRegex(ValueError, 'INERT'): call()

    def test_private_transport_is_closed_even_with_caller_supplied_manifest(self):
        value, _, _, _ = fixtures()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            inert_authority(stack, m, Path(temp))
            stack.enter_context(mock.patch.object(d, 'original_archive', side_effect=AssertionError('no archive')))
            for call in (lambda: d.trusted_manifest(d.encoded(value)),
                lambda: d.prepare_payload('/missing', '/missing'), lambda: d.install('/missing', '/missing')):
                with self.assertRaisesRegex(ValueError, 'INERT'): call()

    def test_unbound_engineering_pin_cannot_be_bypassed_by_a_selected_b28_name(self):
        value, _, _, _ = fixtures()
        raw = d.encoded(value)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            _, chosen = inert_authority(stack, m, Path(temp))
            chosen.update(selected='B28', manifest=dict(path='manifests/' + d.sha(raw) + '.json', **o.pin(raw)))
            d.SELECTION.write_bytes(d.encoded(chosen))
            with self.assertRaisesRegex(ValueError, 'INERT'): d.selection()
            with self.assertRaisesRegex(ValueError, 'INERT'): d.trusted_manifest(raw)

    def test_reviewed_binding_supports_unselected_and_selected_states(self):
        value, _, saved, bound = fixtures()
        for selected in (False, True):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                authority(stack, m, Path(temp), value, saved, bound, selected)
                stack.enter_context(mock.patch.object(o.subprocess, 'run', side_effect=AssertionError('no process')))
                stack.enter_context(mock.patch.object(socket, 'socket', side_effect=AssertionError('no network')))
                self.test_committed_b28_binding_and_own_selection_are_coherent()
                self.assertTrue(m.contract.configuration(True)['enabled'])
                self.assertEqual(d.selection()['selected'], 'B28' if selected else None)
                if selected:
                    self.assertEqual(d.trusted_manifest()[0], value)
                    self.assertEqual(m.reviewed_readback(value), saved)
                else:
                    with self.assertRaisesRegex(ValueError, 'INERT'): d.trusted_manifest()

    def test_negative_entrypoints_remain_isolated_under_a_bound_checkout(self):
        value, _, saved, bound = fixtures()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            for index, runner in enumerate((m, binder.runner)):
                authority(stack, runner, Path(temp) / str(index), value, saved, bound, True)
            for check in (self.test_disabled_binding_rejects_partial_pins_or_nonboolean_enable,
                self.test_inert_proof_stops_all_entrypoints_before_external_effects,
                self.test_private_transport_is_closed_even_with_caller_supplied_manifest,
                self.test_unbound_engineering_pin_cannot_be_bypassed_by_a_selected_b28_name):
                check()
            self.assertTrue(m.contract.configuration(True)['enabled'])
            self.assertEqual(d.trusted_manifest()[0], value)

    def test_only_exact_b28_external_locks_and_candidate_are_admitted(self):
        value, _, _, _ = fixtures()
        d.manifest_contract(value)
        changes = [lambda x: x.update(candidate_id='B27'), lambda x: x.update(candidate_pack=d.B0_PATH),
            lambda x: x['quality_locks']['primary'].update(schema='english-v6-b27-original-quality-lock-v1'),
            lambda x: x['quality_locks']['replication'].update(schema='english-v6-b280-original-quality-lock-v1'),
            lambda x: x['quality_locks']['primary'].pop('external_proof_binding_sha256'),
            lambda x: x['quality_locks']['replication'].update(external_proof_binding_sha256=None)]
        for mutate in changes:
            bad = copy.deepcopy(value); mutate(bad)
            with self.assertRaises(ValueError): d.manifest_contract(bad)

    def test_every_b28_quality_artifact_role_retains_128mb_and_independent_runs(self):
        value, _, _, _ = fixtures()
        for panel in ('primary', 'replication'):
            for role in ('result', 'producer', 'consumed'):
                changed = copy.deepcopy(value); changed['quality_locks'][panel][role]['bytes'] = 128_000_000
                d.manifest_contract(changed)
                changed['quality_locks'][panel][role]['bytes'] += 1
                with self.subTest(panel=panel, role=role), self.assertRaises(ValueError): d.manifest_contract(changed)
        changed = copy.deepcopy(value)
        for role in ('result', 'producer', 'consumed'):
            changed['quality_locks']['replication'][role].update(run_id=101, run_head=f'{101:040x}')
        with self.assertRaisesRegex(ValueError, 'independent'): d.manifest_contract(changed)

    def test_actual_b0_sdk_modes_and_software_scope_cannot_change(self):
        value, _, _, _ = fixtures()
        changes = [lambda x: x['files'][d.B0_PATH].update(sha256='a' * 64),
            lambda x: x['files'][d.B0_PATH].update(bytes=1), lambda x: x['files'][d.B0_PATH].update(mode=0o755),
            lambda x: x['files'].pop('sdk-gcc/bin/heym43-live'),
            lambda x: x['files']['sdk-clang/bin/heym43-live'].update(mode=0o644),
            lambda x: x['scope'].update(physical_ssc305=True), lambda x: x['scope'].update(product_promotion_allowed=True)]
        for mutate in changes:
            bad = copy.deepcopy(value); mutate(bad)
            with self.assertRaises(ValueError): d.manifest_contract(bad)

    def test_only_configured_exact_report_and_verifier_bytes_are_reopened(self):
        value, _, saved, bound = fixtures()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); authority(stack, m, root, value, saved, bound, True)
            stack.enter_context(mock.patch.object(o.subprocess, 'run', side_effect=AssertionError('no saved verifier execution')))
            stack.enter_context(mock.patch.object(o.importlib.util, 'spec_from_file_location', side_effect=AssertionError('no saved verifier import')))
            self.assertEqual(m.reviewed_readback(value), saved)
            o.readback_paths(value)['readback.py'].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): m.reviewed_readback(value)

    def test_selected_manifest_cannot_replace_the_bound_engineering_report(self):
        value, _, saved, bound = fixtures()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); authority(stack, m, root, value, saved, bound, True)
            self.assertEqual(d.trusted_manifest()[0], value)
            bad = copy.deepcopy(value); bad['readback']['report']['sha256'] = 'f' * 64
            raw = d.encoded(bad)
            chosen = d.selection(); chosen['manifest'] = dict(path='manifests/' + d.sha(raw) + '.json', **o.pin(raw))
            d.SELECTION.write_bytes(d.encoded(chosen))
            with self.assertRaisesRegex(ValueError, 'exact reviewed engineering'): d.trusted_manifest(raw)

    def test_quality_lock_digest_uses_original_data_compact_canonical_bytes(self):
        value, report, saved, bound = fixtures()
        with mock.patch.object(m.contract, 'configuration', return_value=bound):
            self.assertEqual(m.validate_reviewed_readback(value, saved), saved)
        changes = []
        for panel in ('primary', 'replication'):
            pretty = d.sha(d.encoded(value['quality_locks'][panel]))
            self.assertNotEqual(pretty, report['quality'][panel]['original_lock_sha256'])
            changes.append((panel + ' pretty-JSON substitution', lambda x, panel=panel, pretty=pretty:
                x['quality'][panel].update(original_lock_sha256=pretty)))
        self.reject_reports(changes)

    def reject_reports(self, changes):
        value, report, saved, _ = fixtures()
        for label, mutate in changes:
            broken = copy.deepcopy(report); mutate(broken)
            selected, _, rebound, bound = rebind_fixture(value, broken, saved['readback.py'])
            with self.subTest(change=label), mock.patch.object(m.contract, 'configuration', return_value=bound), \
                 self.assertRaises(ValueError):
                m.validate_reviewed_readback(selected, rebound)

    def test_report_identity_and_payload_fields_reject_even_after_outer_repin(self):
        changes = [('old schema', lambda x: x.update(schema='english-v6-b27-verified-original-engineering-v1')),
            ('old candidate', lambda x: x.update(candidate_id='B27')), ('summary', lambda x: x.update(status='PASS')),
            ('baseline claim', lambda x: x.update(baseline_pack_requires_separate_byte_authentication=False))]
        for key in ('candidate_pack', 'pipeline_commit', 'files', 'engineering', 'quality_locks', 'scope'):
            changes.append((key, lambda x, key=key: x.update({key: 'unrelated'})))
        self.reject_reports(changes)

    def test_complete_engineering_counts_and_checks_are_required(self):
        _, report, _, _ = fixtures()
        changes = []
        for key, value in report['checks'].items():
            changed = False if type(value) is bool else (0 if key == 'original_producer_files' else value - 1)
            changes.append((key, lambda x, key=key, changed=changed: x['checks'].update({key: changed})))
        changes.append(('boolean count', lambda x: x['checks'].update(raw_arrays=True)))
        self.reject_reports(changes)

    def test_readback_has_zero_new_science_or_materialization(self):
        changes = [(key, lambda x, key=key: x.update({key: 1})) for key in
            ('new_model_calls', 'new_source_calls', 'new_observer_calls', 'compiler_calls', 'packs_created')]
        changes.extend([('selection changed', lambda x: x.update(candidate_selection_changed=True)),
            ('float zero', lambda x: x.update(new_model_calls=0.0))])
        self.reject_reports(changes)

    def test_same_b27_failure_certification_and_both_quality_passes_required(self):
        changes = [('missing panel', lambda x: x['quality'].pop('replication')),
            ('certificate mismatch', lambda x: x['original_B27_failure_certification'].update(original_report_sha256='f' * 64)),
            ('standalone relabel', lambda x: x['original_B27_failure_certification'].update(original_standalone_continuation_succeeded=True)),
            ('certificate new call', lambda x: x['original_B27_failure_certification'].update(new_model_calls=1))]
        for key, replacement in (('original_reader_sha256', 'f' * 64),
            ('adapter_path', 'automation/unrelated.py'), ('adapter_report_schema', 'unrelated-v1'),
            ('unregistered_permission', True)):
            def replace_all(report, key=key, replacement=replacement):
                for container in (report, *report['quality'].values()):
                    container['original_B27_failure_certification'][key] = replacement
            changes.append(('consistently changed certificate/' + key, replace_all))
        for panel in ('primary', 'replication'):
            fields = dict(panel_gate='NO_PROMOTION', original_training_report_sha256='e' * 64,
                original_B27_failure_report_sha256='f' * 64, external_proof_binding_sha256='a' * 64,
                original_result_sha256='b' * 64, original_lock_sha256='c' * 64,
                complete_original_B27_failure_verified=False, all_original_training_proof_verified=False,
                natural_no_event_terms_recomputed=False, natural_objective_returns_verified=383,
                external_training_originals=70, external_total_originals=88,
                captured_code_matches_pinned_local_reader_root=False, new_model_calls=1, product_released=True)
            for key, replacement in fields.items():
                changes.append((panel + '/' + key, lambda x, panel=panel, key=key, replacement=replacement:
                    x['quality'][panel].update({key: replacement})))
        self.reject_reports(changes)

    def test_both_quality_passes_belong_to_the_exact_exported_candidate(self):
        changes = []
        for panel in ('primary', 'replication'):
            for role in ('B0', 'B9', 'control', 'challenger'):
                changes.append((panel + '/' + role, lambda x, panel=panel, role=role:
                    x['quality'][panel]['model_pt_sha256'].update({role: 'f' * 64})))
        changes.append(('same wrong challenger in both panels', lambda x: [
            x['quality'][panel]['model_pt_sha256'].update(challenger='f' * 64)
            for panel in ('primary', 'replication')]))
        changes.append(('same duplicate model roles in both panels', lambda x: [
            x['quality'][panel]['model_pt_sha256'].update(control=x['quality'][panel]['model_pt_sha256']['B0'])
            for panel in ('primary', 'replication')]))
        self.reject_reports(changes)

    def test_exact_original_engineering_provenance_and_immutable_b0_role(self):
        pin = artifact(55, 66)
        run = dict(id=66, head_sha=pin['run_head'], status='completed', conclusion='success', run_attempt=1,
            repository=dict(full_name=o.DATA_REPO, private=False), head_repository=dict(full_name=o.DATA_REPO, private=False),
            event='push', path=m.ENGINEERING_WORKFLOW, head_branch=m.ENGINEERING_BRANCH)
        item = dict(id=pin['id'], expired=False, size_in_bytes=pin['bytes'], digest='sha256:' + pin['sha256'],
            workflow_run=dict(id=66, head_sha=pin['run_head']), name=m.artifact_name('result', 66))
        commit = dict(sha=pin['run_head'])
        m.provenance(pin, run, item, commit, 'result')
        for key, replacement in (('run_attempt', 2), ('conclusion', 'failure'),
            ('path', '.github/workflows/english-v6-b27-engineering-once.yml'),
            ('head_branch', 'research/en-v6-b27-engineering-once-20261125')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.provenance(pin, dict(run, **{key: replacement}), item, commit, 'result')
        with self.assertRaisesRegex(ValueError, 'unchanged complete original B0'):
            m.provenance(pin, run, item, commit, 'B0')

    def test_exact_twelve_file_capture_includes_separate_binding_and_selection(self):
        value, _, _, _ = fixtures()
        names = o.expected_pipeline_files(value, d.encoded(value))
        self.assertEqual(len(names), 12)
        self.assertTrue({'research/english_v6_delivery/delivery_b28.py', 'research/english_v6_delivery/once_b28.py',
            'research/english_v6_delivery/selection_b28.json', 'research/english_v6_delivery/b28_binding.json'} <= names)
        self.assertNotIn('research/english_v6_delivery/selection.json', names)
        self.assertFalse(any('SSC305_SDK_HANDOFF' in name or 'V6_QUALITY_EVIDENCE' in name for name in names))

    def test_disabled_data_entry_proposal_has_exact_unbound_shape(self):
        value = d.document(d.read(HERE / 'b28_entry.disabled.json'))
        self.assertEqual(o.entry_contract(value), value)
        self.assertFalse(value['enabled'])
        for key in ('pipeline_commit', 'manifest_sha256', 'pipeline_files'):
            self.assertIsNone(value[key])
            with self.assertRaises(ValueError): o.entry_contract(dict(value, **{key: 'invented'}))

    def test_review_draft_only_copies_metadata_and_keeps_actual_selection(self):
        value, _, saved, bound = fixtures()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); authority(stack, binder.runner, root / 'authority', value, saved, bound)
            before = binder.d.SELECTION.read_bytes()
            report = root / 'report.json'; report.write_bytes(saved['completion.json'])
            verifier = root / 'verifier.py'; verifier.write_bytes(saved['readback.py'])
            stack.enter_context(mock.patch.object(binder.original.subprocess, 'run', side_effect=AssertionError('no process')))
            result = binder.manifest(report, verifier, root / 'review')
            self.assertFalse(result['actual_selection_changed']); self.assertFalse(result['package_created'])
            self.assertFalse(result['saved_verifier_executed']); self.assertFalse(result['installed'])
            chosen = d.document((root / 'review/selection_b28.json').read_bytes())
            self.assertEqual(chosen['selected'], 'B28')
            self.assertFalse((root / 'review/selection.json').exists())
            self.assertEqual(binder.d.SELECTION.read_bytes(), before)

    def test_entry_draft_requires_exact_clean_committed_twelve_file_producer(self):
        value, _, saved, bound = fixtures()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); repository = root / 'repo'; here = repository / 'research/english_v6_delivery'
            source_root = binder.runner.ROOT
            for name in binder.runner.DELIVERY_FILES - {'research/english_v6_delivery/selection_b28.json',
                'research/english_v6_delivery/b28_binding.json'}:
                target = repository / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source_root / name).read_bytes())
            authority(stack, binder.runner, here, value, saved, bound, True)
            stack.enter_context(mock.patch.object(binder.runner, 'ROOT', repository))
            stack.enter_context(mock.patch.object(binder.original, 'ROOT', repository))
            names = binder.original.expected_pipeline_files(value, d.encoded(value))
            def git(unused_root, *args):
                if args == ('rev-parse', 'HEAD'): return b'a' * 40 + b'\n'
                if args == ('status', '--porcelain', '--untracked-files=no'): return b''
                self.assertEqual(args[:3], ('ls-tree', '-z', 'HEAD'))
                return b''.join(b'100644 blob ' + hashlib.sha1(b'blob ' + str((repository / name).stat().st_size).encode()
                    + b'\0' + (repository / name).read_bytes()).hexdigest().encode() + b'\t' + name.encode() + b'\0'
                    for name in sorted(names))
            stack.enter_context(mock.patch.object(binder.original, 'git', side_effect=git))
            result = binder.entry('a' * 40, root / 'entry')
            self.assertFalse(result['actual_data_entry_changed'])
            entry = d.document((root / 'entry/engineering_delivery_inputs.json').read_bytes())
            self.assertEqual(set(entry['pipeline_files']), names); self.assertEqual(len(names), 12)
            with self.assertRaises(ValueError): binder.entry('b' * 40, root / 'wrong')
            with mock.patch.object(binder.original, 'git', side_effect=[b'a' * 40 + b'\n', b' M tracked.py\n']), \
                 self.assertRaisesRegex(ValueError, 'clean'):
                binder.entry('a' * 40, root / 'dirty')

    def test_software_orchestration_uses_only_original_mocked_transport_and_loader(self):
        value, _, saved, bound = fixtures()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); data = root / 'data'; data.mkdir()
            entry = data / m.BINDING; entry.parent.mkdir(parents=True); entry.write_bytes(b'{}')
            output = root / 'output'
            binding = dict(pipeline_commit='a' * 40, pipeline_files={'invented': True})
            stack.enter_context(mock.patch.object(m.contract, 'configuration', return_value=bound))
            stack.enter_context(mock.patch.object(d, 'trusted_manifest', return_value=(value, d.encoded(value))))
            stack.enter_context(mock.patch.object(m, 'reviewed_readback', return_value=saved))
            stack.enter_context(mock.patch.object(o, 'entry_contract', return_value=binding))
            stack.enter_context(mock.patch.object(o, 'pipeline_contract'))
            stack.enter_context(mock.patch.object(o, 'guard', return_value={'invented': True}))
            stack.enter_context(mock.patch.object(o, 'one_execution', return_value={'invented': True}))
            fetched = stack.enter_context(mock.patch.object(o, 'fetch_originals', return_value={'invented': True}))
            serialized = stack.enter_context(mock.patch.object(o, 'baseline_container', return_value=(b'invented opaque bytes', {'invented': True})))
            packaged = stack.enter_context(mock.patch.object(d, 'package', return_value={'invented': True}))
            installed = stack.enter_context(mock.patch.object(d, 'install', return_value={'invented': True}))
            stack.enter_context(mock.patch.object(d, 'verify_installed', return_value=(value, output / 'version')))
            loaded = stack.enter_context(mock.patch.object(o, 'load_checks', return_value=dict(model_load_calls=2, model_forwards=0)))
            stack.enter_context(mock.patch.object(o.subprocess, 'run', side_effect=AssertionError('no real process')))
            result = m.execute(data, 'b' * 40, output)
            for operation in (fetched, serialized, packaged, installed, loaded): self.assertEqual(operation.call_count, 1)
            self.assertEqual(packaged.call_args.args[2].name, 'heym43-B28-software-delivery.tar.gz')
            self.assertEqual(result['schema'], 'english-v6-b28-software-delivery-result-v1')
            for key in ('model_forwards', 'pcm_samples_consumed', 'stream_init_calls', 'optimizer_updates'):
                self.assertEqual(result[key], 0)
            self.assertFalse(result['sdk_rebuilt']); self.assertFalse(result['weights_modified'])

    def test_b27_private_import_and_default_selection_stay_separate(self):
        old = load('unchanged_b27_software_comparison', HERE / 'once_b27.py')
        self.assertNotEqual(old.d, d)
        self.assertEqual(old.d.SELECTION, HERE / 'selection.json')
        self.assertEqual(d.SELECTION, HERE / 'selection_b28.json')
        value, _, _, _ = fixtures()
        with self.assertRaises(ValueError): old.d.manifest_contract(value)
        self.assertEqual(old.original.BRANCH, 'refs/heads/research/en-v6-b27-engineering-delivery-once-20261125')
        self.assertEqual(o.BRANCH, m.BRANCH)

    def test_new_adapters_have_no_scientific_imports_or_assert_authority(self):
        for name in ('delivery_b28.py', 'once_b28.py', 'bind_b28.py'):
            tree = ast.parse((HERE / name).read_bytes())
            imported = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
            self.assertFalse(imported & {'numpy', 'torch', 'onnxruntime', 'piper', 'soundfile', 'transformers'})
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
            calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            self.assertFalse(calls & {'forward', 'launch', 'execv', 'npy_payload', 'pack'})


if __name__ == '__main__':
    unittest.main()
