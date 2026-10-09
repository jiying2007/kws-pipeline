"""End-to-end compiler routing and foreign-ELF blocking; no cross toolchain."""
import json, os, pathlib, subprocess, sys, tempfile, unittest, shutil, hashlib
from compiler_flags import CFLAGS, STRICT_FP_FLAGS
ROOT=pathlib.Path(__file__).resolve().parent
class DriverGuards(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.out=pathlib.Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def wrapper(self,foreign):
  path=self.out/('foreign-cc' if foreign else 'selected-cc');log=self.out/'compiler-calls.jsonl'
  code='#!'+sys.executable+'\nimport sys,pathlib,struct,subprocess,json\n'
  code+='with open('+repr(str(log))+',"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
  if foreign:
   code+='if "-o" not in sys.argv:print("mock foreign compiler");sys.exit(0)\n'
   code+='raw=bytearray(pathlib.Path("/proc/self/exe").read_bytes());endian="<" if raw[5]==1 else ">";machine=struct.unpack_from(endian+"H",raw,18)[0];struct.pack_into(endian+"H",raw,18,40 if machine!=40 else 62);p=pathlib.Path(sys.argv[sys.argv.index("-o")+1]);p.write_bytes(raw);p.chmod(0o755)\n'
  else:code+='sys.exit(subprocess.run(['+repr(os.environ.get("CC","cc"))+',*sys.argv[1:]],timeout=120).returncode)\n'
  path.write_text(code);path.chmod(0o755);return path,log
 def invoke(self,args):return subprocess.run([sys.executable,'-B',*map(str,args)],capture_output=True,text=True,timeout=120)
 def test_foreign_runtime_rejected_before_first_execution(self):
  cc,log=self.wrapper(True);target=self.out/'build';p=self.invoke([ROOT/'build.py','--cc',cc,'--output',target]);self.assertNotEqual(p.returncode,0);self.assertIn('ELF machine=',p.stderr);self.assertFalse((target/'RESULTS.json').exists());self.assertFalse((target/'sizes-original').exists());self.assertEqual(len(log.read_text().splitlines()),1)
 def test_foreign_differential_library_rejected_before_load(self):
  cc,log=self.wrapper(True);source=ROOT.parent/'native_a20/baseline/decoder';target=self.out/'diff';p=self.invoke([ROOT/'test_decoder.py','--cc',cc,'--original',source,'--optimized',source,'--output',target]);self.assertNotEqual(p.returncode,0);self.assertIn('ELF machine=',p.stderr);self.assertFalse((target/'decoder-test-results.json').exists());self.assertEqual(len(log.read_text().splitlines()),1)
 def source_copy(self):
  source=self.out/'source';shutil.copytree(ROOT.parent/'native_a20',source,ignore=shutil.ignore_patterns('__pycache__'));return source
 def test_self_consistent_changed_manifest_rejected(self):
  source=self.source_copy();p=source/'README.md';p.write_text(p.read_text()+'\nchanged\n');mp=source/'SOURCE_MANIFEST.json';manifest=json.loads(mp.read_text());manifest['files']['README.md']={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};mp.write_text(json.dumps(manifest));r=self.invoke([ROOT/'build.py','--source',source,'--output',self.out/'build']);self.assertNotEqual(r.returncode,0);self.assertIn('pinned source manifest mismatch',r.stderr);self.assertNotIn('PASS exact',r.stdout);self.assertFalse((self.out/'build').exists())
 def test_changed_checker_is_never_executed(self):
  source=self.source_copy();marker=self.out/'checker-was-executed';p=source/'tests/check_manifest.py';p.write_text('import pathlib\npathlib.Path('+repr(str(marker))+').write_text("executed")\n'+p.read_text());r=self.invoke([ROOT/'build.py','--source',source,'--output',self.out/'build']);self.assertNotEqual(r.returncode,0);self.assertIn('source manifest mismatch: tests/check_manifest.py',r.stderr);self.assertFalse(marker.exists());self.assertFalse((self.out/'build').exists())
 def test_unlisted_source_file_rejected(self):
  source=self.source_copy();(source/'extra.c').write_text('/* unlisted */');r=self.invoke([ROOT/'build.py','--source',source,'--output',self.out/'build']);self.assertNotEqual(r.returncode,0);self.assertIn('closed source file inventory mismatch',r.stderr);self.assertFalse((self.out/'build').exists())
 def test_selected_compiler_reaches_differential(self):
  cc,log=self.wrapper(False);target=self.out/'build';p=self.invoke([ROOT/'build.py','--cc',cc,'--output',target]);self.assertEqual(p.returncode,0,p.stdout+p.stderr);receipt=json.loads((target/'RESULTS.json').read_text());diff=json.loads((target/'differential/decoder-test-results.json').read_text());self.assertEqual(receipt['compiler_command'],str(cc));self.assertEqual(diff['compiler_command'],str(cc));calls=[json.loads(x) for x in log.read_text().splitlines()];outputs=[x[x.index('-o')+1] for x in calls if '-o' in x];self.assertTrue(any(x.endswith('original-decoder.so') for x in outputs));self.assertTrue(any(x.endswith('optimized-decoder.so') for x in outputs));self.assertEqual(len(outputs),8)
  for call in calls:
   if '-o' not in call:continue
   self.assertEqual(call[:len(CFLAGS)],list(CFLAGS))
   for flag in STRICT_FP_FLAGS:self.assertEqual(call.count(flag),1)
  self.assertEqual(receipt['compile_flags'],list(CFLAGS));self.assertEqual(diff['compile_flags'],list(CFLAGS));self.assertEqual(receipt['differential'],diff)
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  self.assertEqual(receipt['differential_receipt_sha256'],sha(target/'differential/decoder-test-results.json'))
  self.assertEqual(diff['compiler_flags_sha256'],sha(ROOT/'compiler_flags.py'))
  for name in ('original','optimized'):
   self.assertEqual(diff['binaries'][name]['library_sha256'],sha(target/'differential'/(name+'-decoder.so')))
  self.assertEqual(len(diff['commands']),2)
  for command in diff['commands']:
   self.assertEqual(command['command'][0],str(cc));self.assertEqual(command['command'][1:1+len(CFLAGS)],list(CFLAGS));self.assertEqual(command['exit'],0)
 def test_required_floating_point_policy(self):
  self.assertEqual(STRICT_FP_FLAGS,('-fno-fast-math','-ffp-contract=off','-frounding-math','-fexcess-precision=standard'))
  self.assertNotIn('-ffast-math',CFLAGS);self.assertIn('-Werror',CFLAGS)
 def test_unsupported_strict_flag_is_fatal(self):
  cc=self.out/'unsupported-cc';log=self.out/'attempted'
  cc.write_text('#!'+sys.executable+'\nimport pathlib,sys\npathlib.Path('+repr(str(log))+').write_text(" ".join(sys.argv[1:]))\nprint("strict flag unsupported",file=sys.stderr)\nsys.exit(1)\n');cc.chmod(0o755)
  source=ROOT.parent/'native_a20/baseline/decoder';target=self.out/'diff'
  p=self.invoke([ROOT/'test_decoder.py','--cc',cc,'--original',source,'--optimized',source,'--output',target])
  self.assertNotEqual(p.returncode,0);self.assertFalse((target/'decoder-test-results.json').exists())
  for flag in STRICT_FP_FLAGS:self.assertIn(flag,log.read_text())
  target=self.out/'build'
  p=self.invoke([ROOT/'build.py','--cc',cc,'--output',target])
  self.assertNotEqual(p.returncode,0);self.assertFalse((target/'RESULTS.json').exists())
  for flag in STRICT_FP_FLAGS:self.assertIn(flag,log.read_text())
if __name__=='__main__':unittest.main()
