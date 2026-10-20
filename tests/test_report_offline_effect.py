import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('offline', ROOT / 'tools/report_offline_effect.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
H = '1' * 64
CONTEXT = dict(sample_rate_hz=16000, feed_chunk_samples=160, appended_context_samples=0,
    appended_context_kind='none', reset_frontend='per-recording', reset_model='per-recording',
    reset_decoder='per-recording', reset_clocks='per-recording', eof_partial_chunk='drop', eof_flush=False, eof_padding='none')

def fixture():
    identity = {key: H for key in m.IDENTITY}
    identity['context_policy'] = CONTEXT.copy()
    plan = dict(schema='offline-effect-plan-v1', references_sha256=H, source_split_sha256=H,
        frozen_plan_receipt_sha256=H, test_used_for_selection=False, comparison='fixed_operating_point',
        keyword_ids=[1,2], split_role='development', context_policy=CONTEXT.copy(), event_matching=dict(pre_tolerance_ms=0, post_tolerance_ms=500),
        arms=dict(control=copy.deepcopy(identity), candidate=copy.deepcopy(identity)))
    rows = [dict(recording='positive', source_family='fixture', root_recording_id='positive-root', session_id='positive-session', duration_s=3, expected=[dict(keyword_id=1, start_s=1, end_s=2)],
        evidence_tier='synthetic', scenario='near', pcm_sha256='2'*64, continuous_negative=False),
        dict(recording='negative', source_family='fixture', root_recording_id='negative-root', session_id='negative-session', duration_s=3600, expected=[], evidence_tier='public', scenario='speech',
        pcm_sha256='3'*64, continuous_negative=True, continuity=dict(kind='original_uninterrupted_recording',
        reset_count=0, execution_receipt_sha256=H))]
    for row in rows:
        row.update(sample_rate_hz=16000, sample_count=round(row['duration_s'] * 16000))
        if row['continuous_negative']:
            row['continuity'].pop('execution_receipt_sha256')
            row['continuity']['coverage_contract_sha256'] = m.coverage_contract(plan, row)
    return plan, rows

def execution(plan, rows):
    records=[]
    for row in rows:
        records.append(dict(recording=row['recording'], status='completed', terminal=True, exit_code=0,
            pcm_sha256=row['pcm_sha256'], sample_rate_hz=row['sample_rate_hz'], input_samples=row['sample_count'],
            consumed_samples=row['sample_count'], start_sample=0, end_sample=row['sample_count'],
            dropped_samples=0, padded_samples=0, reset_count=0,
            coverage_contract_sha256=m.coverage_contract(plan,row)))
    return dict(schema='offline-effect-execution-v2',status='completed',terminal=True,exit_code=0,
        plan_sha256=H,references_sha256=plan['references_sha256'],
        arms={arm:dict(identity=copy.deepcopy(identity),detections_sha256=H,status='completed',terminal=True,
            exit_code=0,recordings=copy.deepcopy(records)) for arm,identity in plan['arms'].items()})


class Tests(unittest.TestCase):
    def test_tiers_no_pooling_and_confidence(self):
        plan, rows = fixture()
        det = [dict(recording='positive', time_s=2, keyword_id=1)]
        out = m.build_report(plan, rows, [], det, execution(plan,rows))
        self.assertFalse(out['shipping_approved'])
        self.assertFalse(out['production_qualified'])
        self.assertFalse(out['tiers']['real_device']['present'])
        self.assertEqual(out['tiers']['synthetic']['candidate_minus_control']['false_rejects'], -1)
        public = out['tiers']['public']['arms']['candidate']
        self.assertIsNone(public['frr'])
        self.assertEqual(public['per_keyword']['2']['continuous_negative_false_accepts'], 0)
        self.assertAlmostEqual(public['continuous_negative']['far_upper_95_per_hour'], 2.99573227355)
        self.assertIsNone(out['tiers']['synthetic']['arms']['control']['continuous_negative']['far_per_hour'])

    def test_pre_onset_not_credited(self):
        p,r=fixture()
        out=m.build_report(p,r,[],[dict(recording='positive', time_s=.99, keyword_id=1)], execution(p,r))
        x=out['tiers']['synthetic']['arms']['candidate']
        self.assertEqual((x['false_rejects'],x['false_accepts']),(1,1))

    def test_clip_hours_not_continuous(self):
        plan, rows = fixture(); rows[1]['continuous_negative'] = False
        x = m.build_report(plan, rows, [], [], execution(plan,rows))['tiers']['public']['arms']['control']
        self.assertEqual(x['all_negative_recording_hours'], 1)
        self.assertEqual(x['continuous_negative']['hours'], 0)
        self.assertIsNone(x['continuous_negative']['far_per_hour'])

    def test_false_activation_keyword_scenario(self):
        plan, rows = fixture()
        out = m.build_report(plan, rows, [], [dict(recording='negative', time_s=2, keyword_id=2)], execution(plan,rows))
        item = out['tiers']['public']['scenarios']['speech']['candidate']
        self.assertEqual(item['per_keyword']['2']['continuous_negative_false_accepts'], 1)
        self.assertEqual(item['continuous_negative']['far_per_hour'], 1)

    def test_invalid_plans_fail_closed(self):
        for edit in [lambda p:p.update(test_used_for_selection=True),
                     lambda p:p['arms']['candidate'].update(thresholds_sha256='4'*64),
                     lambda p:p['event_matching'].update(pre_tolerance_ms=150),
                     lambda p:p.update(source_split_sha256='unknown'),
                     lambda p:p['arms']['candidate']['context_policy'].update(eof_flush=True)]:
            p,r=fixture(); edit(p)
            with self.assertRaises(ValueError): m.build_report(p,r,[],[], execution(p,r))

    def test_invalid_continuity_and_duplicates(self):
        for edit in [lambda r:r[1]['continuity'].update(kind='concatenated_clips'),
                     lambda r:r[1]['continuity'].update(reset_count=1),
                     lambda r:r[1].update(pcm_sha256=r[0]['pcm_sha256'])]:
            p,r=fixture(); edit(r)
            with self.assertRaises(ValueError): m.build_report(p,r,[],[], execution(p,r))

    def test_matched_far_requires_calibration_not_test(self):
        p,r=fixture(); p['comparison']='calibration_matched_far'
        p['arms']['candidate']['thresholds_sha256']='4'*64
        p['calibration']=dict(references_sha256='5'*64, selection_receipt_sha256=H,
            selection_split='calibration', test_used=False, source_groups_disjoint=True, target_far_per_hour=.1)
        m.build_report(p,r,[],[], execution(p,r))
        p['calibration']['references_sha256']=p['references_sha256']
        with self.assertRaises(ValueError): m.build_report(p,r,[],[], execution(p,r))

    def test_duplicate_root_or_session_exposure_rejected(self):
        for field in ('root_recording_id', 'session_id'):
            p,r=fixture()
            clone=copy.deepcopy(r[1]); clone.update(recording='channel2',pcm_sha256='7'*64,
                root_recording_id='other-root',session_id='other-session')
            clone[field]=r[1][field]; clone['continuity']['coverage_contract_sha256']=m.coverage_contract(p,clone); r.append(clone)
            with self.assertRaisesRegex(ValueError, 'duplicate continuous source exposure'):
                m.build_report(p,r,[],[], execution(p,r))

    def test_missing_lineage_or_padded_continuous_rejected(self):
        for edit in [lambda p,r:r[1].pop('root_recording_id'),
                     lambda p,r:r[1].update(session_id='unknown'),
                     lambda p,r:p['context_policy'].update(eof_flush=True),
                     lambda p,r:p['context_policy'].update(eof_padding='zero'),
                     lambda p,r:p['context_policy'].update(appended_context_samples=160,appended_context_kind='digital-zero')]:
            p,r=fixture(); edit(p,r)
            for arm in p['arms'].values(): arm['context_policy']=copy.deepcopy(p['context_policy'])
            with self.assertRaises(ValueError): m.build_report(p,r,[],[], execution(p,r))

    def test_split_group_overlap_and_reference_binding(self):
        for field in ('root_recording_id','session_id','speaker_id','preset_identity','rir_room_id','rir_config_id'):
            p,r=fixture()
            for i,row in enumerate(r): row[field]=f'group-{i}'
            split={'records':[dict(pcm_sha256=row['pcm_sha256'],split_role='development',source_family=row['source_family'],
                root_recording_id=row['root_recording_id'],session_id=row['session_id'],
                **({field:row[field]} if field not in ('root_recording_id','session_id') else {})) for row in r]}
            m.validate_source_split(p,r,split)
            extra=copy.deepcopy(split['records'][1]); extra.update(pcm_sha256='8'*64,split_role='training',
                root_recording_id='new-root',session_id='new-session')
            extra[field]=split['records'][1][field]; split['records'].append(extra)
            with self.assertRaisesRegex(ValueError,'source lineage crosses splits'):
                m.validate_source_split(p,r,split)
        p,r=fixture(); split={'records':[dict(row,split_role='development') for row in r]}
        r[0]['root_recording_id']='substituted'
        with self.assertRaisesRegex(ValueError,'reference lineage differs'):
            m.validate_source_split(p,r,split)

    def test_native_corpus_splits_allowed_unless_family_held_out(self):
        p,r=fixture(); split={'records':[dict(row,split_role='development') for row in r]}
        split['records'].append(dict(source_family='fixture',root_recording_id='training-root',
            session_id='training-session',pcm_sha256='9'*64,split_role='training'))
        m.validate_source_split(p,r,split)
        p['family_held_out']=True
        with self.assertRaisesRegex(ValueError,'source family crosses'):
            m.validate_source_split(p,r,split)

    def test_room_aliases_cannot_bypass_split_groups(self):
        for field, alias in [('room_id','RIRroom_id'),('room_config_id','RIRconfig_id')]:
            p,r=fixture()
            for row in r: row[field]='room'
            split={'records':[dict(row,split_role='development') for row in r]}
            split['records'].append(dict(source_family='fixture',root_recording_id='training-root',
                session_id='training-session',pcm_sha256='9'*64,split_role='training',**{alias:'room'}))
            with self.assertRaisesRegex(ValueError,'source lineage crosses'):
                m.validate_source_split(p,r,split)
            split['records'].pop(); r[0][alias]='different'
            with self.assertRaisesRegex(ValueError,'conflicting lineage aliases'):
                m.validate_source_split(p,r,split)

    def test_unknown_session_allowed_only_for_clip_reporting(self):
        p,r=fixture()
        for row in r: row.update(session_id=None, speaker_id=None, continuous_negative=False)
        split={'records':[dict(row,split_role='development') for row in r]}
        m.validate_source_split(p,r,split)
        report=m.build_report(p,r,[],[], execution(p,r))
        self.assertIsNone(report['tiers']['public']['arms']['control']['continuous_negative']['far_per_hour'])
        r[1]['continuous_negative']=True
        with self.assertRaises(ValueError): m.build_report(p,r,[],[], execution(p,r))

    def test_cli_binds_bytes_and_execution_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); p,r=fixture()
            for name, data in [('references',r),('control_detections',[]),('candidate_detections',[])]:
                (root/name).write_text(''.join(json.dumps(row)+'\n' for row in data))
            (root/'source_split').write_text(json.dumps({'records':[dict(pcm_sha256=row['pcm_sha256'],split_role='development',source_family=row['source_family'],root_recording_id=row['root_recording_id'],session_id=row['session_id']) for row in r]}))
            p.update(references_sha256=m.digest(root/'references'), source_split_sha256=m.digest(root/'source_split'))
            (root/'plan').write_text(json.dumps(p))
            receipt=execution(p,r)
            receipt['plan_sha256']=m.digest(root/'plan')
            for arm in p['arms']: receipt['arms'][arm]['detections_sha256']=m.digest(root/(arm+'_detections'))
            (root/'execution_receipt').write_text(json.dumps(receipt))
            cmd=[sys.executable,str(ROOT/'tools/report_offline_effect.py')]
            for name in ['plan','references','source_split','control_detections','candidate_detections','execution_receipt','output']:
                cmd += ['--'+name.replace('_','-'),str(root/name)]
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            receipt['arms']['candidate']['identity']['model_sha256']='6'*64
            (root/'execution_receipt').write_text(json.dumps(receipt))
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)
            (root/'references').write_text('tampered')
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)

class IntegrityTests(unittest.TestCase):
    def report(self,p,r,receipt):
        return m.build_report(p,r,[],[],receipt)

    def test_missing_legacy_failed_or_nonterminal_receipts_rejected(self):
        edits=[lambda x:x.update(schema='offline-effect-execution-v1'),
               lambda x:x.update(status='failed',exit_code=137),
               lambda x:x.update(terminal=False), lambda x:x.update(exit_code=False),
               lambda x:x.update(processed_recordings=[]),
               lambda x:x.pop('status'),lambda x:x['arms'].pop('candidate'),
               lambda x:x['arms']['control'].update(status='failed'),
               lambda x:x['arms']['candidate'].update(terminal=False),
               lambda x:x['arms']['candidate'].update(exit_code=1)]
        for edit in edits:
            p,r=fixture(); receipt=execution(p,r);edit(receipt)
            with self.subTest(edit=edit):
                with self.assertRaises(ValueError): self.report(p,r,receipt)
        p,r=fixture()
        with self.assertRaises(ValueError): self.report(p,r,None)

    def test_missing_duplicate_extra_and_partial_recordings_rejected(self):
        edits=[lambda a:a['recordings'].pop(), lambda a:a.update(recordings=[]),
            lambda a:a['recordings'].append(copy.deepcopy(a['recordings'][0])),
            lambda a:a['recordings'][0].update(recording='unknown'),
            lambda a:a['recordings'][0].update(status='failed',exit_code=137),
            lambda a:a['recordings'][0].update(terminal=False),
            lambda a:a['recordings'][0].update(pcm_sha256='9'*64),
            lambda a:a['recordings'][0].update(coverage_contract_sha256='9'*64)]
        for key in ('sample_rate_hz','input_samples','consumed_samples','start_sample',
                    'end_sample','dropped_samples','padded_samples','reset_count'):
            edits.extend([lambda a,key=key:a['recordings'][0].update({key:1}),
                          lambda a,key=key:a['recordings'][0].update({key:False})])
        for arm in ('control','candidate'):
            for edit in edits:
                p,r=fixture(); receipt=execution(p,r);edit(receipt['arms'][arm])
                with self.subTest(arm=arm,edit=edit):
                    with self.assertRaises(ValueError):self.report(p,r,receipt)

    def test_reference_samples_and_continuity_binding(self):
        for edit in [lambda r:r[1].update(duration_s=3599),lambda r:r[1].update(sample_count=0),
                     lambda r:r[1].update(sample_rate_hz=8000),
                     lambda r:r[1]['continuity'].update(coverage_contract_sha256='9'*64),
                     lambda r:r[1]['continuity'].update(execution_receipt_sha256=H)]:
            p,r=fixture(); receipt=execution(p,r);edit(r)
            with self.assertRaises(ValueError):self.report(p,r,receipt)

    def test_drop_tail_cannot_claim_complete_coverage(self):
        p,r=fixture()
        r[1]['sample_count']+=1;r[1]['duration_s']=r[1]['sample_count']/16000
        r[1]['continuity']['coverage_contract_sha256']=m.coverage_contract(p,r[1])
        with self.assertRaisesRegex(ValueError,'drop-tail policy'):
            self.report(p,r,execution(p,r))
        p['context_policy']['eof_partial_chunk']='process-retained'
        for arm in p['arms'].values():arm['context_policy']=copy.deepcopy(p['context_policy'])
        r[1]['continuity']['coverage_contract_sha256']=m.coverage_contract(p,r[1])
        out=self.report(p,r,execution(p,r))
        self.assertAlmostEqual(out['tiers']['public']['arms']['control']['continuous_negative']['hours'],
                               r[1]['sample_count']/16000/3600)

    def test_no_matched_latency_null_but_genuine_zero_preserved(self):
        p,r=fixture();out=self.report(p,r,execution(p,r))
        for tier in ('synthetic','public'):
            for item in list(out['tiers'][tier]['arms'].values())+[
                    a for scenario in out['tiers'][tier]['scenarios'].values() for a in scenario.values()]:
                self.assertEqual(item['matched_latency_samples'],0)
                for key,value in item.items():
                    if key.startswith(('p50_','p95_')) and ('latency' in key or 'offset' in key):
                        self.assertIsNone(value,key)
        det=[dict(recording='positive',time_s=2,keyword_id=1)]
        out=m.build_report(p,r,det,det,execution(p,r))
        item=out['tiers']['synthetic']['arms']['control']
        self.assertEqual(item['matched_latency_samples'],1)
        self.assertEqual(item['p95_post_end_latency_ms'],0)

    def test_output_aliases_rejected_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'plan';source.write_text('frozen bytes')
            symlink=root/'symlink';symlink.symlink_to(source)
            hardlink=root/'hardlink';os.link(source,hardlink)
            for output in [source,root/'sub/../plan',symlink,hardlink]:
                with self.assertRaisesRegex(ValueError,'aliases'):
                    m.validate_output_path(output,[source])
                self.assertEqual(source.read_text(),'frozen bytes')
            output=root/'new';m.validate_output_path(output,[source]);m.atomic_write(output,'report')
            self.assertEqual(output.read_text(),'report')
            self.assertEqual(source.read_text(),'frozen bytes')
            self.assertEqual(sorted(x.name for x in root.iterdir()),['hardlink','new','plan','symlink'])

    def test_cli_rejects_failed_execution_and_all_input_collisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p,r=fixture()
            for name,data in [('references',r),('control_detections',[]),('candidate_detections',[])]:
                (root/name).write_text(''.join(json.dumps(row)+'\n' for row in data))
            (root/'source_split').write_text(json.dumps({'records':[dict(row,split_role='development') for row in r]}))
            p.update(references_sha256=m.digest(root/'references'),source_split_sha256=m.digest(root/'source_split'))
            (root/'plan').write_text(json.dumps(p))
            receipt=execution(p,r);receipt['plan_sha256']=m.digest(root/'plan')
            for arm in p['arms']:receipt['arms'][arm]['detections_sha256']=m.digest(root/(arm+'_detections'))
            (root/'execution_receipt').write_text(json.dumps(receipt))
            names=['plan','references','source_split','control_detections','candidate_detections','execution_receipt']
            cmd=[sys.executable,str(ROOT/'tools/report_offline_effect.py')]
            for name in names:cmd+=['--'+name.replace('_','-'),str(root/name)]
            before={name:(root/name).read_bytes() for name in names}
            for name in names:
                result=subprocess.run(cmd+['--output',str(root/name)],capture_output=True)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn(b'aliases',result.stderr)
                self.assertEqual(before,{n:(root/n).read_bytes() for n in names})
            receipt.update(status='failed',exit_code=137)
            (root/'execution_receipt').write_text(json.dumps(receipt))
            result=subprocess.run(cmd+['--output',str(root/'out')],capture_output=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertFalse((root/'out').exists())

if __name__ == '__main__': unittest.main()
