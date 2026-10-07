"""Invented fixtures and source inspection only. Never import Torch or run a model."""
import ast,copy,gzip,hashlib,importlib.util,io,json,math,os,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import check_context,safe,read_json
from contracts import collapse,cohort_weights,cohort_loss,ctc_nll,D20,Q12,minimum_frames
from fetch_inputs import validate_request,receive,prepared_members,gunzip_exact
from artifacts import journal_counts,package,collect_phase_logs,publication_complete
from train_once import edit_distance,Journal
class PureTests(unittest.TestCase):
 def test_no_torch_import(self):self.assertNotIn('torch',sys.modules)
 def test_default_entrypoints_block(self):
  for name in ('train_once.py','launch_once.py','fetch_inputs.py','dependency_phase.py'):
   r=subprocess.run([sys.executable,'-B',str(ROOT/'src'/name)],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('PREPARATION_ONLY',r.stdout+r.stderr)
 def test_context_without_release_blocked(self):
  with self.assertRaises(ValueError):check_context(ROOT,{},'training')
 def test_target_repeat_frames(self):self.assertEqual(minimum_frames([1,1,2]),4)
 def test_collapse_blanks_preserve_repeats(self):self.assertEqual(collapse([1,1,0,1,2,2,0]),[1,1,2])
 def test_ctc_math(self):
  p=[[math.log(.5),math.log(.5)]+[-math.inf]*4]*2
  self.assertAlmostEqual(ctc_nll(p,[1]),-math.log(.75));self.assertTrue(math.isinf(ctc_nll(p,[1,1])))
 def test_balanced_cohorts(self):
  cs=[D20]*20+[Q12]*12;w=cohort_weights(cs);self.assertAlmostEqual(sum(w[:20]),.5);self.assertAlmostEqual(sum(w[20:]),.5)
  self.assertAlmostEqual(cohort_loss([4]*20+[8]*12,[[1,2,3,4]]*32,cs),1.5)
 def test_cohort_reweight_rejected(self):
  with self.assertRaises(ValueError):cohort_weights([D20]*19+[Q12]*13)
 def test_edit_diagnostics(self):self.assertEqual(edit_distance([1,2,3],[1,3]),1)
 def test_safe_paths(self):
  with tempfile.TemporaryDirectory()as d:
   for name in ('../a','/a','a/../b',''):
    with self.assertRaises(ValueError):safe(Path(d),name)
 def test_duplicate_json(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'a';p.write_text('{"x":1,"x":2}')
   with self.assertRaises(ValueError):read_json(p)
 def test_url_identity(self):
  e=dict(raw_url='https://raw.githubusercontent.com/r/p/'+'a'*40+'/f',repository='r/p',commit='a'*40,path='f',bytes=1)
  validate_request(e,{'r/p':{'a'*40}})
  for key,value in [('raw_url',e['raw_url']+'?x=1'),('bytes',1024**2+1),('commit','b'*40)]:
   bad={**e,key:value}
   with self.assertRaises(ValueError):validate_request(bad,{'r/p':{'a'*40}})
 def test_response_bounded(self):
  class R(io.BytesIO):
   status=200;headers={}
   def geturl(self):return 'u'
  raw=b'abc';e=dict(raw_url='u',bytes=3,sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob 3\0'+raw).hexdigest())
  self.assertEqual(receive(R(raw),e),raw)
  with self.assertRaises(ValueError):receive(R(raw+b'x'),e)
 def test_gzip_extra_stream_rejected(self):
  c=gzip.compress(b'a');e=dict(compressed_bytes=len(c),compressed_sha256=hashlib.sha256(c).hexdigest(),bytes=1,sha256=hashlib.sha256(b'a').hexdigest());self.assertEqual(gunzip_exact(c,e),b'a')
  c+=gzip.compress(b'b');e.update(compressed_bytes=len(c),compressed_sha256=hashlib.sha256(c).hexdigest())
  with self.assertRaises(ValueError):gunzip_exact(c,e)
 def fixture_tar(self,symlink=False):
  b=io.BytesIO()
  with tarfile.open(fileobj=b,mode='w')as t:
   m=tarfile.TarInfo('x');m.size=1
   if symlink:m.type=tarfile.SYMTYPE;m.linkname='../x';m.size=0
   t.addfile(m,None if symlink else io.BytesIO(b'a'))
  raw=gzip.compress(b.getvalue());manifest=dict(archive=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),members=[dict(path='x',bytes=1,sha256=hashlib.sha256(b'a').hexdigest())]));return raw,manifest
 def test_prepared_tar_exact(self):r,m=self.fixture_tar();self.assertEqual(prepared_members(r,m),{'x':b'a'})
 def test_prepared_tar_symlink_rejected(self):
  r,m=self.fixture_tar(True)
  with self.assertRaises(ValueError):prepared_members(r,m)
 def test_journal_exact_counts(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'journal';j=Journal(p)
   for i in range(1,301):
    for k in ('forward','backward','update'):j.emit(k,'started',i);j.emit(k,'completed',i)
   j.emit('forward','started',300);j.emit('forward','completed',300);j.close();c=journal_counts(p);self.assertEqual(c['completed'],dict(forward=301,backward=300,update=300));self.assertTrue(c['exact_completed_count'])
 def test_journal_interrupted_update(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'journal';j=Journal(p);j.emit('update','started',1);j.close();c=journal_counts(p);self.assertTrue(c['interrupted_call_may_have_executed']['update']);self.assertEqual(c['completed']['update'],0)
 def test_journal_partial_not_zero(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'journal';p.write_bytes(b'{"kind":');self.assertFalse(journal_counts(p)['exact_completed_count'])
 def test_unknown_artifact_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'work/artifact/private.wav').write_bytes(b'x')
   with self.assertRaises(ValueError):package(r,'FAILED_NO_RETRY')
 def test_unreceipted_partial_weights_hash_only(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'work/artifact/terminal-step300.pt').write_bytes(b'incomplete');x=package(r,'FAILED_NO_RETRY');self.assertEqual(x['manifest']['files'],[]);self.assertFalse(x['manifest']['checkpoint_receipt_valid']);self.assertEqual(x['manifest']['unpublished'][0]['file'],'terminal-step300.pt')
 def test_incomplete_json_preserves_other_evidence(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'work/artifact/initial-train-diagnostics.json').write_bytes(b'{"x":');(r/'work/artifact/call-journal.jsonl').write_bytes(b'')
   x=package(r,'FAILED_NO_RETRY');self.assertEqual(len(x['manifest']['files']),1);self.assertEqual(x['manifest']['unpublished'][0]['file'],'initial-train-diagnostics.json')
 def test_failed_complete_terminal_diagnostic_retained(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'work/artifact/terminal-train-logits.f32le').write_bytes(b'\0'*111720)
   x=package(r,'FAILED_NO_RETRY');self.assertEqual(x['manifest']['files'][0]['path'],'terminal-train-logits.f32le')
 def test_overlimit_logs_do_not_block_safe_receipt(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'work/logs').mkdir();(r/'work/logs/training.stderr').write_bytes(b'x'*(512*1024+1));collect_phase_logs(r);x=package(r,'FAILED_NO_RETRY');self.assertEqual(x['manifest']['files'][0]['path'],'private-log-identities.json')
 def test_interrupted_publication_quarantined_once(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'publication').mkdir();(r/'publication/partial').write_bytes(b'x');package(r,'FAILED_NO_RETRY');self.assertEqual((r/'work/publication-incomplete/partial').read_bytes(),b'x')
 def test_truncated_updates_keep_verified_prefix(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'work/artifact/updates.jsonl').write_bytes(b'{"step":1,"loss_before_update":1.0,"gradient_norm_before_clip":0.1}\n{"step":2,');x=package(r,'FAILED_NO_RETRY');v=read_json(r/'work/artifact/validated-jsonl-prefixes.json');self.assertEqual(v['derivations'][0]['known_complete_rows'],1);self.assertFalse(v['derivations'][0]['complete_source'])
 def test_ready_must_verify_not_merely_exist(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'work/artifact').mkdir(parents=True);(r/'work/PUBLICATION-READY.json').write_bytes(b'{"schema":');self.assertFalse(publication_complete(r));(r/'work/PUBLICATION-READY.json').unlink();package(r,'FAILED_NO_RETRY');self.assertTrue(publication_complete(r))
 def test_only_fixed_loop(self):
  s=(ROOT/'src/train_once.py').read_text();tree=ast.parse(s);ranges=[ast.unparse(n)for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='range'];self.assertIn('range(1, 301)',ranges)
  self.assertNotIn('initialization_control(',s);self.assertNotIn('load_state_dict(optimizer',s)
 def test_no_dev_transport(self):
  m=read_json(ROOT/'metadata/MODEL-INPUTS.json');self.assertEqual(m['audio_downloads'],0);self.assertEqual(len(m['objects']),3);self.assertTrue(all(not e['destination'].endswith('.wav')for e in m['objects']))
 def test_protocol_and_model_structure(self):
  p=read_json(ROOT/'PROTOCOL.json');self.assertEqual(p['calls'],dict(training_forwards=300,training_backwards=300,optimizer_updates=300,terminal_diagnostic_forwards=1,total_model_forwards=301,new_initial_control_forwards=0,feature_extractions=0,decoder_calls=0,development_calls=0))
 def test_all_existing_requests_admitted(self):
  m=read_json(ROOT/'metadata/MODEL-INPUTS.json')
  commits={'jiying2007/kws-data':{'7af8f8597b0b7fbe8761c9a2400f2e4028395551'},'jiying2007/kws-pipeline':{'804553286fd9caa294769cc4d44cc0c43dc66e88'}}
  for e in m['requests']:validate_request(e,commits)
 def test_dependency_lock_unchanged(self):
  self.assertEqual(hashlib.sha256((ROOT/'src/deps/exact-wheels.lock.json').read_bytes()).hexdigest(),'90a28180a51fae445941bd3c0c9fefb6fa569e2c622255d32c17d1021d6939fc')
if __name__=='__main__':unittest.main()
