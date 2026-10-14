import unittest,tempfile,json,hashlib,importlib.util
from pathlib import Path
from unittest.mock import patch
from screen import write
from terminal import whisper_rows
import host

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class TestTerminal(unittest.TestCase):
 def fixture(self,root):
  inp=root/'input';out=root/'output';inp.mkdir();out.mkdir()
  clips=[dict(audio_id=f'clip-{i:06d}',audio_path='audio/'+str(i)*64+'.wav',wav_sha256=str(i)*64) for i in range(1,4)]
  write(inp/'job.json',{'schema':'blind-source-screen32-job-v1','clips':clips});write(out/'native-config.json',{})
  for clip in clips[:2]:
   d=out/clip['audio_id'];d.mkdir();write(d/'started.json',{'attempt':1,'opaque_id':clip['audio_id'],'wav_sha256':clip['wav_sha256'],'job_sha256':sha(inp/'job.json'),'config_sha256':sha(out/'native-config.json')})
  d=out/clips[0]['audio_id'];write(d/'raw-output.json',{'raw_text':'你好小窝','decoder_completion_validated':True,'full_input_processed':True})
  write(d/'completed.json',dict(opaque_id=clips[0]['audio_id'],wav_sha256=clips[0]['wav_sha256'],status='success',raw_text='你好小窝',completeness='complete',quality_flags=[],started_sha256=sha(d/'started.json'),raw_sha256=sha(d/'raw-output.json')))
  return inp,out
 def test_whisper_hard_kill_preserves_completed(self):
  with tempfile.TemporaryDirectory() as t:
   i,o=self.fixture(Path(t));self.assertFalse((o/'outcomes.json').exists());rows=whisper_rows(i,o);self.assertEqual([r['status'] for r in rows],['success','failed_no_retry','not_run'])
 def test_whisper_tamper_fails(self):
  with tempfile.TemporaryDirectory() as t:
   i,o=self.fixture(Path(t));(o/'clip-000001/raw-output.json').write_text('{}');self.assertEqual(whisper_rows(i,o)[0]['status'],'failed_no_retry')
 def test_primary_uses_durable_validator_not_snapshot(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);i=root/'input';i.mkdir();write(i/'job.json',{'schema':'blind-source-screen32-job-v1','clips':[{'audio_id':'clip-000001','audio_path':'audio/'+'1'*64+'.wav','wav_sha256':'1'*64}]})
   out=root/'output';(out/'qwen06').mkdir(parents=True);write(out/'qwen06/outcomes.json',[{'opaque_id':'clip-000001','wav_sha256':'1'*64,'status':'success','raw_text':'fabricated'}])
   rows=host.primary_terminal(i,out,'qwen06');self.assertEqual(rows[0]['status'],'not_run');self.assertIsNone(rows[0]['raw_text'])
 def test_whisper_progress_shape(self):
  path=Path(host.HERE)/'whisper_worker.py'
  with patch.dict('sys.modules',{'runtime_scope':type('Scope',(),{'verify_runtime_scope':lambda:None})()}):
   spec=importlib.util.spec_from_file_location('test_whisper_progress',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
   with tempfile.TemporaryDirectory() as t:
    for stage,index in [('load',0),('cell',1),('cell',16)]:
     m.progress(Path(t),stage,index);r=json.loads((Path(t)/'progress.json').read_bytes());self.assertEqual(set(r),{'stage','index','started_monotonic'});self.assertEqual((r['stage'],r['index']),(stage,index))
if __name__=='__main__':unittest.main()
