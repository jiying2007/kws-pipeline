"""Invented fixtures only. No model, dependency install, Docker or network."""
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace
import copy,hashlib,json,sys,tempfile,unittest,wave
sys.path.insert(0,str(Path(__file__).resolve().parent))
import host
class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def event(self):
        e=dict(GITHUB_REPOSITORY='jiying2007/kws-pipeline',GITHUB_REF='refs/heads/research/'+host.CAMPAIGN,GITHUB_EVENT_NAME='push',GITHUB_RUN_NUMBER='1',GITHUB_RUN_ATTEMPT='1',GITHUB_SHA='a'*40,GITHUB_RUN_ID='123')
        return e,dict(created=True,deleted=False,after='a'*40,repository=dict(full_name=e['GITHUB_REPOSITORY'],private=False))
    def test_first_event(self): host.admission(*self.event())
    def test_no_retry_event(self):
        for key,value in [('GITHUB_RUN_NUMBER','2'),('GITHUB_RUN_ATTEMPT','2'),('GITHUB_EVENT_NAME','pull_request'),('GITHUB_REF','refs/heads/main'),('GITHUB_REPOSITORY','other/repo')]:
            e,p=self.event();e[key]=value
            with self.assertRaises(ValueError):host.admission(e,p)
    def test_no_recreated_or_private_event(self):
        for key,value in [('created',False),('deleted',True),('after','b'*40)]:
            e,p=self.event();p[key]=value
            with self.assertRaises(ValueError):host.admission(e,p)
        e,p=self.event();p['repository']['private']=True
        with self.assertRaises(ValueError):host.admission(e,p)
    def blind(self):
        p=self.root/'blind';(p/'audio').mkdir(parents=True);clips=[]
        for i in range(1,5):
            q=p/'audio'/('tmp%d.wav'%i)
            with wave.open(str(q),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(bytes([i,0])*160)
            h=host.sha(q);q.rename(p/'audio'/(h+'.wav'));clips.append(dict(audio_id='clip-%06d'%i,audio_path='audio/'+h+'.wav',wav_sha256=h))
        (p/'job.json').write_text(json.dumps(dict(schema='blind-source-screen32-job-v1',clips=clips)));return p
    def test_blind_input(self):self.assertEqual(len(host.validate_input(self.blind())['clips']),4)
    def test_changed_audio(self):
        p=self.blind();next((p/'audio').glob('*')).write_bytes(b'bad')
        with self.assertRaises(ValueError):host.validate_input(p)
    def test_target_leak_rejected(self):
        p=self.blind();j=json.loads((p/'job.json').read_text());j['clips'][0]['text']='secret target';(p/'job.json').write_text(json.dumps(j))
        with self.assertRaises(ValueError):host.validate_input(p)
    def test_extra_file_rejected(self):
        p=self.blind();(p/'target.json').write_text('{}')
        with self.assertRaises(ValueError):host.validate_input(p)
    def test_fewer_than_four_rejected(self):
        p=self.blind();j=json.loads((p/'job.json').read_text());j['clips'].pop();(p/'job.json').write_text(json.dumps(j))
        with self.assertRaises(ValueError):host.validate_input(p)
    def test_path_escape_rejected(self):
        p=self.blind();j=json.loads((p/'job.json').read_text());j['clips'][0]['audio_path']='../x';(p/'job.json').write_text(json.dumps(j))
        with self.assertRaises(ValueError):host.validate_input(p)
    def test_source_projection(self):
        (self.root/'archive.py').write_text('public source');lock=dict(campaign=host.CAMPAIGN,files={'archive.py':host.sha(self.root/'archive.py')},stage={'research/a.py':'archive.py'})
        host.verify_sources(self.root,lock);host.stage(self.root,self.root/'stage',lock);self.assertEqual((self.root/'stage/research/a.py').read_text(),'public source')
    def test_source_drift(self):
        (self.root/'a').write_text('x')
        with self.assertRaises(ValueError):host.verify_sources(self.root,dict(campaign=host.CAMPAIGN,files={'a':'0'*64},stage={}))
    def results(self):
        m=dict(rows=[dict(opaque_id='clip-%06d'%i,derived_sha256=str(i)*64,sample_id='sample%d'%i,requested_text='你好小窝') for i in range(1,5)])
        r={f:[dict(opaque_id=x['opaque_id'],wav_sha256=x['derived_sha256'],status='success',completeness='complete',raw_text='你好，小窝。',quality_flags=[]) for x in m['rows']] for f in host.FAMILIES};return m,r
    def test_consensus_not_training(self):
        m,r=self.results();s=host.score(m,r);self.assertEqual(s['planned_asr_slots'],12);self.assertFalse(s['human_gold']);self.assertFalse(s['training_admitted']);self.assertTrue(all(x['requested_text_match'] for x in s['rows']))
    def test_disagreement_retained(self):
        m,r=self.results();r['whisper'][0]['raw_text']='你好小屋';s=host.score(m,r);self.assertFalse(s['rows'][0]['three_model_consensus']);self.assertEqual(s['rows'][0]['observations']['whisper']['raw_text'],'你好小屋')
    def test_failure_not_consensus(self):
        m,r=self.results();r['qwen06'][0]['status']='failed_no_retry';self.assertFalse(host.score(m,r)['rows'][0]['requested_text_match'])
    def test_wrong_audio_rejected(self):
        m,r=self.results();r['sensevoice'][0]['wav_sha256']='f'*64
        with self.assertRaises(ValueError):host.score(m,r)
    def test_missing_terminal_rejected(self):
        m,r=self.results();r['whisper'].pop()
        with self.assertRaises(ValueError):host.score(m,r)
    def test_export_no_runtime_or_logs(self):
        p=self.root/'work';(p/'runtime-primary').mkdir(parents=True);(p/'runtime-primary/secret.json').write_text('{}');(p/'primary/qwen06').mkdir(parents=True);(p/'primary/qwen06/terminal-outcomes.json').write_text('{}');(p/'primary/qwen06/raw.log').write_text('secret');(p/'host-receipt.json').write_text('{}');host.export_public(p);names={q.relative_to(p/'artifact').as_posix() for q in (p/'artifact').rglob('*') if q.is_file()};self.assertEqual(names,{'artifact-index.json','host-receipt.json','primary/qwen06/terminal-outcomes.json'})
    def test_deadline_prevents_docker(self):
        b=Mock()
        with self.assertRaises(ValueError):host.launch(b,self.root,self.root,self.root/'out','image','qwen06',0,self.root)
        b.host_capacity.assert_not_called();b.container_args.assert_not_called();b.supervise_container.assert_not_called()
    def test_unknown_public_receipt_excluded(self):
        self.assertFalse(host.public_member('qwen06/secret.json','qwen06'))
        self.assertFalse(host.public_member('clip-000001/unreviewed.json','whisper'))
        self.assertTrue(host.public_member('qwen06/clip-000001.decoder.json','qwen06'))
    def test_setup_failure_stops_all_later_stages(self):
        inputs=self.blind();here=self.root/'research/new';here.mkdir(parents=True);(here/'inputs').mkdir();inputs.rename(here/'inputs/blind')
        mapping,_=self.results();(here/'inputs/mapping.json').write_text(json.dumps(mapping));(here/'source-lock.json').write_text('{}')
        # Align mocked host mapping with real invented blind fixture digests.
        job=json.loads((here/'inputs/blind/job.json').read_text())
        for m,c in zip(mapping['rows'],job['clips']):m['derived_sha256']=c['wav_sha256']
        (here/'inputs/mapping.json').write_text(json.dumps(mapping))
        (self.root/'research/source-screen32-v1').mkdir();(self.root/'research/source-screen32-v1/container-lock.json').write_text('{"image":"fake"}')
        env,event=self.event();ep=self.root/'event.json';ep.write_text(json.dumps(event));env['GITHUB_EVENT_PATH']=str(ep)
        boundary=Mock();boundary.host_capacity.return_value={};terminal=Mock();loader=Mock();spec=SimpleNamespace(loader=loader)
        with patch.object(host,'ROOT',self.root),patch.object(host,'HERE',here),patch.object(host,'verify_sources'),patch.object(host,'stage'),patch.object(host.shutil,'disk_usage',return_value=SimpleNamespace(free=30*1024**3)),patch.object(host.importlib.util,'spec_from_file_location',return_value=spec),patch.object(host.importlib.util,'module_from_spec',return_value=terminal),patch.dict(sys.modules,{'boundary':boundary,'terminal':SimpleNamespace(whisper_rows=Mock())}),patch.object(host,'launch',side_effect=RuntimeError('invented uncertain cleanup')) as launch:
            with self.assertRaises(RuntimeError):host.run(env)
            self.assertEqual(launch.call_count,1)
        report=json.loads((self.root/('asr-'+host.CAMPAIGN)/'report.json').read_text())
        self.assertEqual(len(report['rows']),4);self.assertTrue(all(o['status']=='not_run' for r in report['rows'] for o in r['observations'].values()))
    def test_exact_source_lock(self):
        lockpath=host.HERE/'source-lock.json'
        if lockpath.exists():host.verify_sources(host.ROOT,json.loads(lockpath.read_text()))
        else:self.fail('source lock required')
if __name__=='__main__':unittest.main()
