"""Validate retained aggregate evidence without corpus, transcript, or inference."""
from pathlib import Path
import hashlib
import json
import unittest
import subprocess,sys,tempfile,shutil
HERE=Path(__file__).resolve().parent

def load(name):return json.loads((HERE/name).read_bytes())
def digest(name):return hashlib.sha256((HERE/name).read_bytes()).hexdigest()

class ReceiptTests(unittest.TestCase):
    def test_public_package_contains_only_reviewed_small_sources_and_receipts(self):
        inventory=load('publication-manifest.json')['files']
        expected=set(inventory)|{'publication-manifest.json'}
        actual={p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file()}
        self.assertEqual(actual,expected)
        for name,row in inventory.items():
            body=(HERE/name).read_bytes()
            self.assertLessEqual(len(body),300000,name)
            self.assertEqual(len(body),row['bytes'],name)
            self.assertEqual(hashlib.sha256(body).hexdigest(),row['sha256'],name)
            self.assertTrue(Path(name).suffix in ('.py','.json','.jsonl','.md','.sh','.log','.txt','.h','.exitcode') or Path(name).name=='SHA256SUMS' or name=='float-c-parity-observation-v1/requirements.lock',name)
            self.assertFalse(body.startswith(b'\x7fELF'),name)
    def test_actual_asr_archive_metadata_and_machine_label_boundary(self):
        evidence=HERE.parent/'tts-pilot-screening-v1/evidence/asr-run-38573117922'
        allow=json.loads((evidence/'archive-allowlist.json').read_bytes())
        self.assertEqual(len(allow['json_files']),13)
        self.assertEqual({p.name for p in evidence.iterdir()},{r['filename'] for r in allow['json_files']}|{'archive-allowlist.json'})
        for row in allow['json_files']:
            raw=(evidence/row['filename']).read_bytes()
            self.assertEqual(len(raw),row['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
        audit=json.loads((evidence/'audit-summary.json').read_bytes())
        self.assertEqual(audit['status'],'pass');self.assertEqual(audit['results'],94)
        self.assertEqual(audit['unique_sample_family'],94);self.assertEqual(audit['attempts_per_result'],1)
        self.assertEqual(audit['three_asr_candidates'],13)
        self.assertFalse(audit['human_gold']);self.assertFalse(audit['training_admitted'])
        recomputed=json.loads((evidence/'recomputed-report.json').read_bytes())
        canonical=json.dumps(recomputed,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.assertEqual((evidence/'report.json').read_bytes(),canonical)
    def test_machine_positive_control_is_failed_development_response_not_human_frr(self):
        control=load('machine-positive-control-v1/run-p4-11b93abe/summary.json')
        self.assertTrue(control['all_clips_complete']);self.assertEqual(control['clip_count'],13)
        self.assertEqual(control['input_samples'],444160);self.assertAlmostEqual(control['input_seconds'],27.76)
        self.assertEqual(control['trigger_event_count'],0)
        self.assertEqual(control['per_target']['你好小窝']['input_clips'],2)
        self.assertEqual(control['per_target']['小窝小窝']['input_clips'],11)
        for row in control['per_target'].values():self.assertEqual(row['clips_with_matching_trigger'],0)
        self.assertEqual(control['original_alimeeting_summary_sha256_unchanged'],digest('diagnostic-run-p4-11b93abe/summary.json'))
        self.assertEqual(control['build_receipt_sha256'],digest('diagnostic-build/build-receipt.json'))
        for key in ('far_available','human_frr_available','human_gold','shipping_approved','training_admitted','generalization_claim_available'):self.assertFalse(control[key])
    def test_trace_small_package_verifier_and_tamper_rejection(self):
        original=HERE/'positive-trace-observation-v1'
        flags=['-I','-S','-B']+(['-'+'O'*sys.flags.optimize] if sys.flags.optimize else [])
        command=[sys.executable,*flags,str(original/'verify_package.py')]
        result=subprocess.run(command,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        with tempfile.TemporaryDirectory() as temp:
            changed=Path(temp)/'package';shutil.copytree(original,changed)
            (changed/'diagnosis-summary.json').write_text('{}')
            result=subprocess.run([*command,str(changed)],capture_output=True,text=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
    def test_recovered_float_archive_verifier_and_tamper_rejection(self):
        original=HERE/'float-c-parity-observation-v1'
        flags=['-I','-S','-B']+(['-'+'O'*sys.flags.optimize] if sys.flags.optimize else [])
        command=[sys.executable,*flags,str(original/'verify_package.py')]
        result=subprocess.run(command,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        with tempfile.TemporaryDirectory() as temp:
            changed=Path(temp)/'package';shutil.copytree(original,changed)
            (changed/'run/receipt.json').write_text('{}')
            result=subprocess.run([*command,str(changed)],capture_output=True,text=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
    def test_real_archive_identity_is_observed_not_publisher_claim(self):
        a=load('raw/acquisition.json');b=load('audit/audit.json')
        self.assertEqual(a['archive_sha256'],b['archive_sha256'])
        self.assertEqual(a['archive_sha256'],'dc47343b2474b5ebcf458927e878155f6ddeb59c85e685b3645c32a1f9578d92')
        self.assertEqual(a['archive_bytes'],3673718355)
        self.assertIsNone(a['publisher_sha256']);self.assertIsNone(b['publisher_sha256'])
        self.assertEqual(b['license'],'CC-BY-SA-4.0')
    def test_fixed_eight_complete_meetings_not_negative_admitted(self):
        a=load('audit/audit.json');rows=a['rows']
        self.assertEqual(len(rows),8);self.assertEqual(len({r['session_id'] for r in rows}),8)
        self.assertEqual(sum(r['sample_count'] for r in rows),242211440)
        self.assertTrue(all(r['sample_rate_hz']==16000 and r['channel_index']==0 for r in rows))
        self.assertAlmostEqual(sum(r['duration_s'] for r in rows),15138.215)
        self.assertEqual(a['accepted_negative_duration_s'],0)
        self.assertFalse(a['independence_verified']);self.assertFalse(a['sealed_qualification'])
    def test_expected_labels_remain_null(self):
        rows=[json.loads(line) for line in (HERE/'audit/reference-candidates.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),8);self.assertTrue(all(r['expected'] is None for r in rows))
    def test_frozen_build_and_diagnostic_hash_chain(self):
        build=load('diagnostic-build/build-receipt.json')
        summary=load('diagnostic-run-p4-11b93abe/summary.json')
        self.assertEqual(summary['build_receipt_sha256'],digest('diagnostic-build/build-receipt.json'))
        self.assertEqual(summary['audit_sha256'],digest('audit/audit.json'))
        self.assertEqual(summary['source_revision'],build['source_revision'])
        self.assertEqual(summary['source_tree'],build['source_tree'])
        self.assertEqual(build['source_revision'],'11b93abe92427c3ece2a01f9a497cdb74bab0062')
        self.assertTrue(build['source_unchanged'])
    def test_zero_event_count_is_only_an_unadmitted_diagnostic(self):
        s=load('diagnostic-run-p4-11b93abe/summary.json')
        self.assertEqual(s['session_count'],8);self.assertTrue(s['all_sessions_complete'])
        self.assertEqual(s['input_samples'],242211440);self.assertEqual(s['trigger_event_count'],0)
        self.assertAlmostEqual(s['input_hours'],242211440/16000/3600)
        self.assertEqual(s['trigger_events_per_input_hour'],0)
        self.assertEqual(s['accepted_negative_duration_s'],0)
        for key in ('confirmed_far_available','frr_available','latency_available','shipping_approved'):self.assertFalse(s[key])
    def test_no_hash_overlap_does_not_establish_independence(self):
        s=load('overlap/alimeeting-overlap-result.json')
        self.assertEqual(s['exact_hash_match_count'],0);self.assertEqual(s['source_session_name_match_count'],0)
        self.assertFalse(s['independence_verified']);self.assertFalse(s['sealed_qualification_allowed'])
        self.assertEqual(s['accepted_negative_duration_s_added'],0)
if __name__=='__main__':unittest.main()
