import copy,hashlib,json,pathlib,struct,tempfile,unittest,shutil,zipfile,stat,os,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import remaining as r
BASE=pathlib.Path(os.environ.get('TQP_TEST_SOURCE_ROOT',str(pathlib.Path(__file__).resolve().parents[2])))
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
        self.incoming=self.root/'incoming';self.incoming.mkdir()
        self.source=self.root/'source';self.source.mkdir()
        for name in r.required_sources(BASE,'qwen06'):
            target=self.source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target)
        p=self.source/'research/tts-quality-pilot-v1'
        import importlib.util
        spec=importlib.util.spec_from_file_location('fixture_pilot',p/'pilot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.cell=m.plan('qwen06')[0];cell=self.cell['cell_id']
        pcm=struct.pack('<f',.1)*24
        fmt=struct.pack('<HHIIHH',3,1,24000,96000,4,32)
        wav=b'WAVE'+b'fmt '+struct.pack('<I',16)+fmt+b'data'+struct.pack('<I',len(pcm))+pcm
        wav=b'RIFF'+struct.pack('<I',len(wav))+wav
        row=dict(self.cell,status='generated',attempts=1,metrics={'technically_valid':True},raw_wav_sha256=r.digest(wav),native=dict(subtype='FLOAT',sample_rate_hz=24000,channels=1,frames=24,source_values_sha256=r.digest(pcm)))
        self.receipt=dict(status='probe_passed',engine='qwen06',model_id=m.MODELS['qwen06'][0],revision=m.MODELS['qwen06'][1],outcomes=[row])
        self.put('qwen06-receipt.json',self.receipt)
        self.put('host-receipt.json',dict(campaign='tts-quality-probe-20261014-v1',engine='qwen06',sha='a'*40,status='probe_completed_pending_human_review'))
        self.put('worker-status.json',dict(status='probe_completed'))
        self.put('container-tts.json',dict(status='complete',container_state=dict(ExitCode=0,OOMKilled=False)))
        (self.incoming/(cell+'.claim')).write_bytes(cell.encode())
        (self.incoming/(cell+'.raw-float.wav')).write_bytes(wav)
        self.archive=self.root/'archive.zip';self.rezip()
        self.manifest=self.root/'source.json';self.manifest.write_text(json.dumps(dict(source_sha='a'*40,files={str(f.relative_to(self.source)):r.digest(f.read_bytes()) for f in self.source.rglob('*') if f.is_file() and '__pycache__' not in str(f)})))
        self.binding=dict(schema='tts-remaining-binding-v1',state='reviewed_real_probe',engine='qwen06',repository='jiying2007/kws-pipeline',probe_run_id=123,artifact_id=456,probe_run_attempt=1,probe_run_status='completed',probe_run_conclusion='success',probe_job_id=789,probe_job_name='probe (qwen06)',probe_job_status='completed',probe_job_conclusion='success',probe_source_sha='a'*40,workflow_path='.github/workflows/tts-quality-probe-20261014-v1.yml',artifact_name='tts-quality-probe-qwen06',artifact_archive_sha256=r.digest(self.archive.read_bytes()),source_manifest_sha256=r.digest(self.manifest.read_bytes()),files={n:r.digest((self.incoming/n).read_bytes()) for n in r.filenames('qwen06',cell)})
    def tearDown(self):self.tmp.cleanup()
    def put(self,name,value):(self.incoming/name).write_text(json.dumps(value))
    def verify(self):return r.verify(self.binding,'qwen06',self.incoming,self.source,self.archive,self.manifest)
    def rezip(self):
        with zipfile.ZipFile(self.archive,'w') as z:
            for p in self.incoming.iterdir():z.write(p,p.name)
    def refresh(self,name):
        self.binding['files'][name]=r.digest((self.incoming/name).read_bytes());self.rezip()
        self.binding['artifact_archive_sha256']=r.digest(self.archive.read_bytes())
    def test_valid_restores_only_consumed_probe(self):
        out=self.root/'out'
        result=r.prepare(self.binding,'qwen06',self.incoming,self.source,self.archive,self.manifest,out)
        self.assertEqual(result['next_cell_index'],1);self.assertEqual(result['new_calls_max'],15)
        self.assertEqual(json.loads((out/'qwen06-receipt.json').read_text())['outcomes'],self.receipt['outcomes'])
        self.assertTrue((out/(self.cell['cell_id']+'.claim')).exists())
        with self.assertRaises(ValueError):r.prepare(self.binding,'qwen06',self.incoming,self.source,self.archive,self.manifest,out)
    def test_unfilled_template_rejected(self):
        self.binding['state']='unfilled_template_do_not_activate'
        with self.assertRaises(ValueError):self.verify()
    def test_native_tamper(self):
        p=self.incoming/(self.cell['cell_id']+'.raw-float.wav');p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaises(ValueError):self.verify()
    def test_failed_lane_even_with_rebound_hash(self):
        self.receipt['status']='stopped_no_retry';self.put('qwen06-receipt.json',self.receipt);self.refresh('qwen06-receipt.json')
        with self.assertRaises(ValueError):self.verify()
    def test_wrong_cell_even_with_rebound_hash(self):
        self.receipt['outcomes'][0]['text']='wrong';self.put('qwen06-receipt.json',self.receipt);self.refresh('qwen06-receipt.json')
        with self.assertRaises(ValueError):self.verify()
    def test_failed_container_even_with_rebound_hash(self):
        self.put('container-tts.json',dict(status='complete',container_state=dict(ExitCode=0,OOMKilled=True)));self.refresh('container-tts.json')
        with self.assertRaises(ValueError):self.verify()
    def test_source_drift(self):
        (self.source/'research/tts-quality-pilot-v1/pilot.py').write_text('# changed')
        with self.assertRaises(ValueError):self.verify()
    def test_wrong_archive(self):
        self.archive.write_bytes(b'other')
        with self.assertRaises(ValueError):self.verify()
    def test_probe_receipt_wrong_native_hash(self):
        self.receipt['outcomes'][0]['native']['source_values_sha256']='0'*64
        self.put('qwen06-receipt.json',self.receipt);self.refresh('qwen06-receipt.json')
        with self.assertRaises(ValueError):self.verify()
    def test_worker_delegates_only_remaining_and_blocks_reentry(self):
        from unittest.mock import patch,Mock
        out=self.root/'out'
        r.prepare(self.binding,'qwen06',self.incoming,self.source,self.archive,self.manifest,out)
        pilot,cell,bodies=self.verify()
        pilot.run_loaded=Mock(return_value={'status':'generated_16_pending_review'})
        args=(None,None,None,None,self.binding,'qwen06',self.incoming,self.source,self.archive,self.manifest,out)
        with patch.object(r,'verify',return_value=(pilot,cell,bodies)):
            r.run_verified_loaded(*args)
            self.assertEqual(pilot.run_loaded.call_args.args[-1],'remaining')
            with self.assertRaises(ValueError):r.run_verified_loaded(*args)
        self.assertEqual(pilot.run_loaded.call_count,1)
    def test_failure_before_retained_calls_cannot_retry(self):
        from unittest.mock import patch,Mock
        out=self.root/'out'
        r.prepare(self.binding,'qwen06',self.incoming,self.source,self.archive,self.manifest,out)
        pilot,cell,bodies=self.verify();pilot.run_loaded=Mock(side_effect=RuntimeError('fixture failure'))
        args=(None,None,None,None,self.binding,'qwen06',self.incoming,self.source,self.archive,self.manifest,out)
        with patch.object(r,'verify',return_value=(pilot,cell,bodies)):
            with self.assertRaises(RuntimeError):r.run_verified_loaded(*args)
            with self.assertRaises(ValueError):r.run_verified_loaded(*args)
        self.assertEqual(pilot.run_loaded.call_count,1)
    def test_nonzip_with_rebound_hash_rejected(self):
        self.archive.write_bytes(b'NOT_A_ZIP');self.binding['artifact_archive_sha256']=r.digest(self.archive.read_bytes())
        with self.assertRaises(zipfile.BadZipFile):self.verify()
    def test_archive_bytes_differ_from_incoming(self):
        with zipfile.ZipFile(self.archive,'w') as z:
            for p in self.incoming.iterdir():z.writestr(p.name,b'other' if p.name=='host-receipt.json' else p.read_bytes())
        self.binding['artifact_archive_sha256']=r.digest(self.archive.read_bytes())
        with self.assertRaises(ValueError):self.verify()
    def test_unsafe_duplicate_or_link_archive_rejected(self):
        import warnings
        for name,link in [('../escape',False),('host-receipt.json',False),('extra-link',True)]:
            self.rezip()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(self.archive,'a') as z:
                    info=zipfile.ZipInfo(name)
                    if link:info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
                    z.writestr(info,b'bad')
            self.binding['artifact_archive_sha256']=r.digest(self.archive.read_bytes())
            with self.assertRaises(ValueError):self.verify()
    def test_missing_runtime_scope_manifest_rejected(self):
        m=json.loads(self.manifest.read_text());del m['files']['research/source-screen32-v1/runtime_scope.py']
        self.manifest.write_text(json.dumps(m));self.binding['source_manifest_sha256']=r.digest(self.manifest.read_bytes())
        with self.assertRaises(ValueError):self.verify()
    def test_other_matrix_lane_failure_does_not_block_successful_target(self):
        self.binding['probe_run_conclusion']='failure'
        self.verify()
    def test_target_job_failure_or_missing_rejected(self):
        self.binding['probe_run_conclusion']='failure'
        for outcome in ('failure','cancelled','skipped',None):
            self.binding['probe_job_conclusion']=outcome
            with self.assertRaises(ValueError):self.verify()
    def test_other_lane_success_cannot_authorize_target(self):
        self.binding['probe_job_name']='probe (qwen17)'
        with self.assertRaises(ValueError):self.verify()
if __name__=='__main__':unittest.main()
