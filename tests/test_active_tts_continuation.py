#!/usr/bin/env python3
"""Real local queue/claim contracts with stdlib synthetic media and no models."""
from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tests')]
import active_tts_continuation as a
import test_active_tts_expand as fixture


def media(output, cell, ref, *, failed=False):
    value = fixture.raw_fixture(output, cell, ref, failed=failed)
    path = output / 'native' / cell['cell_id'] / 'call.json'
    call = a.read(path)
    call.update(source_family_id=cell['source_family_id'], reference_id=ref['reference_id'],
        intended_text=cell['prompt'], reference_mode='x-vector-only', reference_text_passed=False)
    path.write_bytes(a.q.canonical(call))
    value['files'][path.relative_to(output).as_posix()] = a.q.file_record(path)
    return value


def queue_probe():
    """Run the actual continuation entry under a real inherited queue lease.

    Only source/preparation inputs and the numerical backend are fixtures. The
    production run, network guard, queue verifier, durable claims, raw media
    verifier, summary/inventory writer and cache path all execute unchanged.
    """
    prepared = Path(sys.argv[1]); recipe = Path(sys.argv[2]); output = Path(sys.argv[3])
    d = a.read(prepared / 'declaration.json'); plan = a.read(prepared / 'fixture-plan.json')
    request = a.read(prepared / 'queue-request.json'); backend = SimpleNamespace(calls=0)

    def create(*args):
        guard = a.read(output / 'network-guard.json')
        a.require(guard['seccomp_installed'] is True and not any(m in sys.modules for m in a.ML_MODULES),
            'FIXTURE_BACKEND_PRE_GUARD')
        a.once(output / 'model-loaded.json', {'status': 'no-model-protocol-fixture'})
        return SimpleNamespace(prompt_creations=0), backend

    def callback(*args):
        def generate(cell, ref):
            backend.calls += 1
            return media(output, cell, ref, failed=cell['seed'] in (201, 241))
        return generate

    def handoff(*args):
        a.require((prepared / 'fixture-verified-asr-gate.json').is_file(), 'FIXTURE_GATE_REQUIRED')

    with patch.object(a, 'load_declaration', return_value=(d, {}, {}, prepared, plan, None)), \
            patch.object(a, 'queue_request', return_value=request), \
            patch.object(a, 'create_backend', side_effect=create), \
            patch.object(a, 'generate_callback', side_effect=callback), \
            patch.object(a, 'verify_handoff', side_effect=handoff):
        result = a.run(Namespace(recipe=recipe, output=output, heavy_lock=d['heavy_lock']))
    print(json.dumps({'fixture': True, 'new_generation_calls': result['new_generation_calls']}))


class ContinuationContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name); self.addCleanup(self.temporary.cleanup)
        self.plan = {'cells': [fixture.cell(i) for i in range(256)],
            'references': [dict(fixture.REF, reference_text_status='dataset-provided')], 'model_id': 'fixture'}

    def declaration_fixture(self):
        prepared = self.root / 'prepared'; prepared.mkdir()
        for name in ('protocol.json', 'old-stage.json', 'old-terminal.json'):
            a.once(prepared / name, {'fixture': name})
        d = {'prepared': str(prepared), 'queue_root': str(prepared / 'queue'),
            'continuation_id': 'fixture-continuation', 'chunks': a.chunks_for(self.plan),
            'runtime_python': sys.executable, 'heavy_lock': str(self.root / 'heavy.lock'),
            'ledger': str(self.root / 'ledger'), 'binding': fixture.BINDING,
            'old_stage': a.pin(prepared / 'old-stage.json'), 'old_terminal': a.pin(prepared / 'old-terminal.json'),
            'protocol_declaration': a.pin(prepared / 'protocol.json'),
            'consumed_unknown_ids': [fixture.cell(197)['cell_id']],
            'registration_path': str(self.root / 'ledger/continuations/original.json')}
        a.once(prepared / 'declaration.json', d)
        a.once(prepared / 'fixture-plan.json', self.plan)
        for i in range(5):
            job = a.expected_job(d, i); a.once(prepared / 'jobs' / (job['job_id'] + '.json'), job)
        a.once(prepared / 'queue-request.json', a.queue_request(d))
        return d, prepared

    def test_only58_unclaimed_ids_and_the_exact_pause_boundary(self):
        chunks = a.chunks_for(self.plan)
        self.assertEqual(list(map(len, chunks)), [12, 12, 10, 12, 12])
        self.assertEqual([n for chunk in chunks for n in chunk], [fixture.cell(i)['cell_id'] for i in range(198, 256)])
        self.assertNotIn(fixture.cell(197)['cell_id'], sum(chunks, []))
        self.assertEqual(sum(chunks[:3], []), [fixture.cell(i)['cell_id'] for i in range(198, 232)])
        bad = copy.deepcopy(self.plan); bad['cells'][-1] = bad['cells'][197]
        with self.assertRaisesRegex(ValueError, 'EXACT_256_PLAN'):
            a.chunks_for(bad)

    def test_actual_completion_count_unknown_and_both_overruns_are_honest(self):
        states = {'continuation-' + str(i).zfill(3): {'status': 'pending'} for i in range(1, 6)}
        first = a.budget(states, 5)
        self.assertEqual(first['completed_job_count'], 0)
        self.assertFalse(first['new_actual_total_complete'])
        states['continuation-001'] = {'status': 'completed', 'actual_elapsed_ns': 123}
        states['continuation-002'] = {'status': 'unknown', 'actual_elapsed_ns': None}
        current = a.budget(states, 5)
        self.assertEqual(current['completed_job_count'], 1)
        self.assertEqual(current['combined_actual_known_ns'], a.OLD_NS + 123)
        self.assertFalse(current['new_actual_total_complete'])
        states['continuation-001']['actual_elapsed_ns'] = a.NEW_NS + 7
        current = a.budget(states, 5)
        self.assertEqual(current['new_queue_overrun_ns'], 7)
        self.assertEqual(current['observed_combined_overrun_ns'], 0)
        self.assertFalse(current['within_new_queue_allocation'])
        self.assertTrue(current['within_combined_allocation'])
        states['continuation-001']['actual_elapsed_ns'] = a.COMBINED_NS
        self.assertEqual(a.budget(states, 5)['observed_combined_overrun_ns'], a.OLD_NS)

    def test_protocol_cannot_increase_budget_or_reclassify_unknown(self):
        path = self.root / 'pinned.json'; a.once(path, {'fixture': True})
        protocol = {'schema': 'kws-d1-post-timeout-continuation-predeclaration-v1',
            'new_generation_maximum_calls': 58, 'pause_after_jobs': 3, 'known_returned_cells': 197,
            'old_stage_returned_cells': 21, 'old_stage_consumed_calls': 22,
            'first_pause_requires_ASR_accounted232_before_last_two_jobs': True,
            'unknown_receives_asr_or_native_call': False, 'unknown_becomes_failed_native': False,
            'unknown_has_observed_text_or_pcm': False, 'unknown_is_negative': False,
            'budget': {'prior_actual_elapsed_ns': a.OLD_NS, 'new_queue_global_minutes': 29.9,
                'new_queue_global_ns': a.NEW_NS, 'combined_allocation_ns': a.COMBINED_NS,
                'per_job_maximum_ns': a.JOB_NS, 'combined_declared_actual_ceiling_ns': a.OLD_NS + a.NEW_NS,
                'cleanup_reserve_ns': a.COMBINED_NS - a.OLD_NS - a.NEW_NS}}
        protocol.update({name: a.pin(path) for name in ('old_tts_stage', 'old_queue_terminal', 'partial_audit',
            'recovery_contract_review', 'original_training_protocol')})
        a.verify_protocol(protocol)
        for key in protocol['budget']:
            changed = copy.deepcopy(protocol); changed['budget'][key] += 1
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'FIXED_CONTINUATION_BUDGET'):
                a.verify_protocol(changed)
        for key in ('unknown_receives_asr_or_native_call', 'unknown_becomes_failed_native',
                    'unknown_has_observed_text_or_pcm', 'unknown_is_negative'):
            changed = copy.deepcopy(protocol); changed[key] = True
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'CONTINUATION_PROTOCOL_SCOPE'):
                a.verify_protocol(changed)

    def test_registration_local_first_then_one_global_publication(self):
        d, prepared = self.declaration_fixture(); original = a.once
        def interrupted(path, value):
            if str(path) == d['registration_path']:
                raise RuntimeError('crash immediately before global publication')
            original(path, value)
        with patch.object(a, 'once', side_effect=interrupted), self.assertRaises(RuntimeError):
            a.publish_registration(d)
        self.assertTrue((prepared / 'registration.json').is_file())
        self.assertFalse(Path(d['registration_path']).exists())
        self.assertEqual(a.read(prepared / 'registration.json'), a.registration(d))
        a.once(d['registration_path'], a.registration(d))
        a.verify_registration(d)
        before = Path(d['registration_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            a.publish_registration(d)
        self.assertEqual(before, Path(d['registration_path']).read_bytes())

    def test_registration_binds_the58_and_original_timeout(self):
        d, _ = self.declaration_fixture(); a.publish_registration(d)
        a.verify_registration(d)
        changed = copy.deepcopy(d); changed['chunks'][0][0] = fixture.cell(197)['cell_id']
        with self.assertRaisesRegex(ValueError, 'REGISTRATION_CHANGED'):
            a.verify_registration(changed)
        changed = copy.deepcopy(d); changed['old_terminal']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'REGISTRATION_CHANGED'):
            a.verify_registration(changed)

    def test_unknown_never_reaches_callback_or_becomes_failed(self):
        ledger = self.root / 'ledger'; cell = fixture.cell(197); ref = self.plan['references'][0]
        def interrupt(*_):
            raise RuntimeError('one call consumed, no return')
        with self.assertRaises(RuntimeError):
            a.e.consume_cell(ledger, cell, ref, fixture.BINDING, fixture.OWNER, interrupt)
        claim = ledger / 'claims' / (cell['cell_id'] + '.json'); raw = claim.read_bytes()
        with self.assertRaisesRegex(ValueError, 'CONSUMED_WITHOUT_TERMINAL'):
            a.e.consume_cell(ledger, cell, ref, fixture.BINDING, fixture.OWNER,
                lambda *_: self.fail('unknown was called again'))
        self.assertEqual(raw, claim.read_bytes())
        self.assertFalse((ledger / 'cells' / (cell['cell_id'] + '.json')).exists())

    def test_new_return_keeps_actual_owner_and_rejects_native_corruption(self):
        d, prepared = self.declaration_fixture(); job = a.expected_job(d, 0)
        cell = self.plan['cells'][198]; ref = self.plan['references'][0]
        output = Path(d['queue_root']) / 'jobs' / job['job_id'] / 'outputs'
        owner = {'stage_sha256': job['declaration_sha256'], 'job_id': job['job_id'],
            'job_recipe_sha256': a.c.sha_file(prepared / 'jobs' / (job['job_id'] + '.json'))}
        a.e.consume_cell(d['ledger'], cell, ref, d['binding'], owner,
            lambda row, reference: media(output, row, reference, failed=True))
        value = a.returned_row(d, job, cell, ref)
        self.assertEqual(value['origin'], 'continuation-completed')
        self.assertEqual(value['record']['status'], 'FAILED_NO_RETRY')
        self.assertTrue(value['eligibility']['eligible'])
        self.assertEqual(value['ledger_record_sha256'], a.q.sha(a.q.canonical(value['record'])))
        (output / value['record']['native_wav']).write_bytes(b'changed raw waveform')
        with self.assertRaises(ValueError):
            a.returned_row(d, job, cell, ref)

    def test_last_two_jobs_gate_before_model_slot_or_cell_claim(self):
        d, prepared = self.declaration_fixture(); job = a.expected_job(d, 3)
        recipe = prepared / 'jobs' / (job['job_id'] + '.json')
        state = {'status': 'claimed-without-terminal', 'claim': {
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'deadline_monotonic_ns': time.monotonic_ns() + 10**9}}
        args = Namespace(recipe=recipe, output=Path(d['queue_root']) / 'jobs' / job['job_id'] / 'outputs',
            heavy_lock=d['heavy_lock'])
        with patch.object(a, 'load_declaration', return_value=(d, {}, {}, prepared, self.plan, None)), \
                patch.object(a, 'queue_state', return_value={job['job_id']: state}), \
                patch.object(a, 'verify_handoff', side_effect=ValueError('ASR232 not sealed')), \
                patch.object(a, 'heavy_model_slot', side_effect=AssertionError('heavy slot reached')), \
                patch.dict(os.environ, {'KWS_RESEARCH_RECIPE_SHA256': a.c.sha_file(recipe)}), \
                self.assertRaisesRegex(ValueError, 'ASR232 not sealed'):
            a.run(args)
        self.assertFalse((Path(d['ledger']) / 'claims').exists())

    def test_partial_invocation_is_not_restartable(self):
        d, prepared = self.declaration_fixture(); job = a.expected_job(d, 0)
        recipe = prepared / 'jobs' / (job['job_id'] + '.json')
        output = Path(d['queue_root']) / 'jobs' / job['job_id'] / 'outputs'
        a.once(output / 'invocation.json', {'fixture': 'interrupted'})
        with patch.object(a, 'load_declaration', return_value=(d, {}, {}, prepared, self.plan, None)), \
                patch.object(a, 'queue_state', return_value={job['job_id']: {'status': 'claimed-without-terminal'}}), \
                patch.object(a, 'heavy_model_slot', side_effect=AssertionError('restart acquired model slot')), \
                patch.dict(os.environ, {'KWS_RESEARCH_RECIPE_SHA256': a.c.sha_file(recipe)}), \
                self.assertRaisesRegex(ValueError, 'CONSUMED_INVOCATION_NO_RESTART'):
            a.run(Namespace(recipe=recipe, output=output, heavy_lock=d['heavy_lock']))

    def data_verifier_fixture(self):
        source = self.root / 'data-source'; marker = self.root / 'verifier-called.json'
        for name in a.DATA_SOURCES:
            path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
        code = 'from pathlib import Path\nimport hashlib,json\n'
        code += 'def verify_accounted_seal(path, expected_sha256, *, expected_count):\n'
        code += '    raw=Path(path).read_bytes()\n'
        code += '    if hashlib.sha256(raw).hexdigest()!=expected_sha256 or expected_count!=232 or json.loads(raw)!={"fixture_sealed":True}:raise ValueError("ACTUAL_SEAL_MISMATCH")\n'
        code += '    Path(' + repr(str(marker)) + ').write_text("verified")\n'
        code += '    return {"schema":"d1-accounted-selective-seal-verification-v3","status":"verified","verified":True,"accounted_prefix_count":232,"new_model_calls":0,"ml_imports":0,"full_replay_completed":True}\n'
        (source / 'automation/active_asr_weak_view.py').write_text(code)
        return {'source_root': str(source), 'files': {name: a.q.file_record(source / name) for name in a.DATA_SOURCES}}, marker

    def test_fresh_pinned_data_verifier_executes_real_api_and_rejects_unrelated_seal(self):
        verifier, marker = self.data_verifier_fixture()
        preflight = a.data_helper(verifier)
        self.assertTrue(preflight['source_preflight_passed'])
        self.assertFalse(preflight['verification_performed'])
        self.assertFalse(marker.exists())
        sealed = self.root / 'sealed.json'; a.once(sealed, {'fixture_sealed': True})
        result = a.data_helper(verifier, a.pin(sealed))
        self.assertTrue(result['verified']); self.assertEqual(result['new_model_calls'], 0)
        self.assertEqual(marker.read_text(), 'verified')
        marker.unlink(); sealed.write_bytes(a.q.canonical({'unrelated_seal': True}))
        with self.assertRaisesRegex(ValueError, 'ACTUAL_SEAL_MISMATCH'):
            a.data_helper(verifier, a.pin(sealed))
        self.assertFalse(marker.exists())

    def test_changed_data_source_is_rejected_before_verifier_import(self):
        verifier, marker = self.data_verifier_fixture()
        path = Path(verifier['source_root']) / 'automation/active_asr_weak_view.py'
        path.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("bad-import")\n')
        with self.assertRaisesRegex(ValueError, 'HANDOFF_SOURCE_CHANGED'):
            a.data_helper(verifier)
        self.assertFalse(marker.exists())

    def test_data_verifier_failure_dict_is_not_success(self):
        verifier, _ = self.data_verifier_fixture()
        path = Path(verifier['source_root']) / 'automation/active_asr_weak_view.py'
        path.write_text('def verify_accounted_seal(*args, **kwargs):\n    return {"verified":False,"status":"failed"}\n')
        verifier['files']['automation/active_asr_weak_view.py'] = a.q.file_record(path)
        sealed = self.root / 'sealed.json'; a.once(sealed, {'fixture_sealed': True})
        with self.assertRaisesRegex(ValueError, 'HANDOFF_VERIFIER_EXPLICIT_SUCCESS_REQUIRED'):
            a.data_helper(verifier, a.pin(sealed))

    def test_late_source_must_equal_control_snapshot_and_actual_view_producer(self):
        original, _ = self.data_verifier_fixture()
        contract = a.verifier_contract(self.root / 'prepared', self.root / 'asr-control')
        for root in (contract['control_source_root'], contract['retained_source_root']):
            for name in a.DATA_SOURCES:
                a.e.atomic_bytes(Path(root) / name, (Path(original['source_root']) / name).read_bytes())
        verifier = {'source_root': contract['retained_source_root'], 'control_source_root': contract['control_source_root'],
            'files': original['files'], 'local_commit': 'a' * 40, 'public_commit': 'b' * 40, 'git_tree': 'c' * 40}
        package = self.root / 'weak-view'
        producer = {name: row['sha256'] for name, row in original['files'].items()}
        a.once(package / 'report.json', {'producer': {'source_files_sha256': producer}})
        a.once(package / 'MANIFEST.json', {'report.json': a.c.sha_file(package / 'report.json')})
        entry = {'path': str(package), 'manifest_sha256': a.c.sha_file(package / 'MANIFEST.json')}
        a.verify_late_source({'asr232_verifier_contract': contract}, verifier, [entry])
        producer['automation/active_asr_weak_view.py'] = '0' * 64
        (package / 'report.json').write_bytes(a.q.canonical({'producer': {'source_files_sha256': producer}}))
        (package / 'MANIFEST.json').write_bytes(a.q.canonical({'report.json': a.c.sha_file(package / 'report.json')}))
        entry['manifest_sha256'] = a.c.sha_file(package / 'MANIFEST.json')
        with self.assertRaisesRegex(ValueError, 'NOT_ACTUAL_VIEW_PRODUCER'):
            a.verify_late_source({'asr232_verifier_contract': contract}, verifier, [entry])
        (Path(contract['control_source_root']) / a.DATA_SOURCES[0]).write_text('changed actual source')
        with self.assertRaisesRegex(ValueError, 'FROZEN_SOURCE_BYTES_CHANGED'):
            a.verify_late_source({'asr232_verifier_contract': contract}, verifier, [entry])

    def test_source_cannot_be_frozen_before_actual_asr_queue_completion(self):
        d, prepared = self.declaration_fixture(); d['asr232_control'] = str(self.root / 'future-control')
        d['asr232_gate_path'] = str(prepared / 'asr232-gate.json')
        d['asr232_verifier_contract'] = a.verifier_contract(prepared, d['asr232_control'])
        cp = self.root / 'cp.json'; a.once(cp, {'fixture': True})
        args = Namespace(declaration=prepared / 'declaration.json', checkpoint=cp,
            checkpoint_sha256=a.c.sha_file(cp), data_source_revision='a' * 40,
            data_public_commit='b' * 40, data_source_tree='c' * 40)
        states = {'continuation-' + str(i).zfill(3): {'status': 'completed' if i < 4 else 'pending'}
            for i in range(1, 6)}
        with patch.object(a, 'load_declaration', return_value=(d, {}, {}, prepared, self.plan, None)), \
                patch.object(a, 'queue_state', return_value=states), \
                patch.object(a, 'external_asr_queue', side_effect=ValueError('ASR not completed')), \
                self.assertRaisesRegex(ValueError, 'ASR not completed'):
            a.seal_handoff(args)
        self.assertFalse(Path(d['asr232_verifier_contract']['retained_source_root']).exists())
        self.assertFalse(Path(d['asr232_gate_path']).exists())

    def test_export_mixed_original_and_new_owners_is28plus27_and_unknown_absent(self):
        selected = list(range(176, 197)) + list(range(198, 232)); rows = []
        for index in selected:
            output = self.root / ('old-timeout' if index < 197 else 'new-completed')
            row = fixture.cell(index); value = media(output, row, self.plan['references'][0], failed=index == 200)
            rows.append({'cell_id': row['cell_id'], 'record': value,
                'origin': 'original-timed-out' if index < 197 else 'continuation-completed'})
        cp = {'schema': a.CHECKPOINT_SCHEMA, 'previous_checkpoint': None, 'prefix_count': 232,
            'plan_canonical_sha256': 'a' * 64, 'returned_rows': rows,
            'eligible_cell_ids': [row['cell_id'] for row in rows], 'excluded_native_rows': [],
            'unobservable_consumed': [{'cell_id': fixture.cell(197)['cell_id']}]}
        path = self.root / 'checkpoint.json'; a.once(path, cp)
        with patch.object(a, 'verify_checkpoint', return_value=cp):
            result = a.export(Namespace(checkpoint=path, checkpoint_sha256=a.c.sha_file(path), output=self.root / 'export'))
        self.assertEqual([len(row['cell_ids']) for row in result['partitions']], [28, 27])
        names = []
        for partition in result['partitions']:
            with zipfile.ZipFile(partition['archive']['path']) as stream:
                names += stream.namelist()
                self.assertIsNone(stream.testzip())
        self.assertEqual(len(names), 55)
        self.assertFalse(any(fixture.cell(197)['cell_id'] in name for name in names))
        index = a.read(self.root / 'export/index.json')
        self.assertEqual(index['unobservable_consumed_ids'], [fixture.cell(197)['cell_id']])
        self.assertEqual(index['asr_calls_executed'], 0)

    def test_real_queue_three_then_two_actual_run_claims_and_zero_repeat(self):
        d, prepared = self.declaration_fixture()
        wrapper = self.root / 'probe.py'
        wrapper.write_text('import sys\nsys.path.insert(0,' + repr(str(ROOT / 'tests')) + ')\n'
            'from test_active_tts_continuation import queue_probe\nqueue_probe()\n')
        request = a.read(prepared / 'queue-request.json')
        for job in request['jobs']:
            job['argv'] = [sys.executable, '-I', '-S', '-B', str(wrapper), str(prepared), '{recipe}', '{output_dir}']
        (prepared / 'queue-request.json').write_bytes(a.q.canonical(request))
        a.publish_registration(d)
        unknown = fixture.cell(197); ref = self.plan['references'][0]
        claim = {'schema': 'active-tts-plan-call-claim-v1', 'cell_id': unknown['cell_id'],
            'identity_sha256': a.e.cell_identity(d['binding'], unknown, ref), 'attempt': 1,
            'owner': {'fixture': 'old-timeout'}, 'started_at': 'original'}
        a.once(Path(d['ledger']) / 'claims' / (unknown['cell_id'] + '.json'), claim)
        original_claim = (Path(d['ledger']) / 'claims' / (unknown['cell_id'] + '.json')).read_bytes()
        base = [sys.executable, '-I', '-S', '-B', str(ROOT / 'tools/research_queue.py')]

        def cli(*argv):
            result = subprocess.run(base + list(argv), cwd=self.root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return json.loads(result.stdout)

        cli('init', '--request', str(prepared / 'queue-request.json'), '--root', d['queue_root'],
            '--heavy-lock', d['heavy_lock'])
        first = cli('run', '--root', d['queue_root'], '--max-jobs', '3')
        self.assertEqual(first['stop_reason'], 'max_jobs_reached')
        self.assertEqual([r['status'] for r in first['jobs'].values()], ['completed'] * 3 + ['pending'] * 2)
        self.assertFalse((Path(d['queue_root']) / 'jobs/continuation-004/claim.json').exists())
        old = {str(p): p.read_bytes() for i in range(1, 4)
            for p in (Path(d['queue_root']) / 'jobs' / ('continuation-' + str(i).zfill(3))).rglob('*') if p.is_file()}
        a.once(prepared / 'fixture-verified-asr-gate.json', {'fixture': True})
        final = cli('run', '--root', d['queue_root'], '--max-jobs', '2')
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(old, {name: Path(name).read_bytes() for name in old})
        with patch.object(a, 'queue_request', return_value=request):
            states = a.queue_state(d, required=True)
            actual = 0
            for row in states.values():
                self.assertEqual(row['claim']['global_actual_before_nanoseconds'], actual)
                actual += row['actual_elapsed_ns']
            self.assertEqual(actual, final['actual_known_job_nanoseconds'])
            checkpoint_budget = a.budget(states, 3)
            self.assertEqual(checkpoint_budget['completed_job_count'], 3)
        self.assertEqual(len(list((Path(d['ledger']) / 'cells').glob('*.json'))), 58)
        self.assertEqual(len(list((Path(d['ledger']) / 'claims').glob('*.json'))), 59)
        self.assertEqual(original_claim, (Path(d['ledger']) / 'claims' / (unknown['cell_id'] + '.json')).read_bytes())
        before = a.inventory(Path(d['queue_root']))
        again = cli('run', '--root', d['queue_root'])
        self.assertEqual(again['status'], 'completed')
        self.assertEqual(before, a.inventory(Path(d['queue_root'])))
        job = a.expected_job(d, 0); recipe = Path(d['queue_root']) / 'recipes' / (job['job_id'] + '.json')
        output = Path(d['queue_root']) / 'jobs' / job['job_id'] / 'outputs'
        with patch.object(a, 'load_declaration', return_value=(d, {}, {}, prepared, self.plan, None)), \
                patch.object(a, 'queue_request', return_value=request), \
                patch.object(a, 'heavy_model_slot', side_effect=AssertionError('cache acquired model slot')), \
                patch.object(a, 'create_backend', side_effect=AssertionError('cache loaded backend')), \
                patch.dict(os.environ, {'KWS_RESEARCH_RECIPE_SHA256': a.c.sha_file(recipe)}):
            result = a.run(Namespace(recipe=recipe, output=output, heavy_lock=d['heavy_lock']))
            self.assertEqual(result['new_model_calls'], 0)
            self.assertEqual(before, a.inventory(Path(d['queue_root'])))
            native = output / 'native' / fixture.cell(198)['cell_id'] / 'returned-native.wav'
            native.write_bytes(b'corrupted cached raw')
            with self.assertRaises(ValueError):
                a.run(Namespace(recipe=recipe, output=output, heavy_lock=d['heavy_lock']))
        with a.heavy_model_slot(d['heavy_lock']):
            pass

    def test_isolated_cli_never_imports_machine_learning(self):
        code = "import runpy,sys;sys.argv=[sys.argv[1],'--help']\ntry:runpy.run_path(sys.argv[0],run_name='__main__')\nexcept SystemExit:pass\n"
        code += "if any(x in sys.modules for x in ('torch','numpy','onnxruntime','qwen_tts','scipy','soundfile')):raise RuntimeError('ML import')"
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code,
            str(ROOT / 'tools/active_tts_continuation.py')], cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
