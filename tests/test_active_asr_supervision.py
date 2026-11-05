#!/usr/bin/env python3
"""Real queue/batch/worker processes with fictional decoders; no ML imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_queue as queue
from research_job_lock import heavy_model_slot


CHILD = r'''
import hashlib,importlib.abc,json,os,struct,subprocess,sys,time,types
from pathlib import Path
tools_path=sys.argv[1]
sys.path.insert(0,tools_path)
import active_asr_batch as batch
from research_job_lock import heavy_model_slot,PATH_ENV
mode,runtime= sys.argv[2],Path(sys.argv[3])
output=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])
if mode=='batch':
    with (output/'decoder.log').open('w') as log:
        code=batch.run_worker([sys.executable,'-I','-S','-B',__file__,sys.argv[1],sys.argv[4],str(runtime)],
                              dict(os.environ),log,runtime,30)
    if code:raise SystemExit(code)
    original={str(p):[p.read_bytes().hex(),p.stat().st_ino,p.stat().st_mtime_ns]
              for folder in ('claims','native','results') for p in (output/folder).rglob('*.json')}
    with (output/'resume.log').open('w') as log:
        code=batch.run_worker([sys.executable,'-I','-S','-B',__file__,sys.argv[1],'resume',str(runtime)],
                              dict(os.environ),log,runtime,30)
    if code:raise SystemExit(code)
    after={str(p):[p.read_bytes().hex(),p.stat().st_ino,p.stat().st_mtime_ns]
           for folder in ('claims','native','results') for p in (output/folder).rglob('*.json')}
    if original!=after:raise RuntimeError('completed recognition was modified')
    (output/'answer.json').write_text(json.dumps({'reused_unchanged':True,'model_imports':0}))
    raise SystemExit(0)
if mode=='grandchild':
    with heavy_model_slot(os.environ[PATH_ENV]) as lease:
        info=os.fstat(lease.fd)
        (output/'grandchild.json').write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),
            'inherited':lease.inherited,'supervised':lease.supervised,'device':info.st_dev,'inode':info.st_ino}))
        time.sleep(30)
    raise SystemExit(0)
import active_asr_worker as worker
worker.research_runtime.check_packages=lambda *a:None
worker.verify_model_assets=lambda *a:{'fixture':'no model assets or model execution'}
if mode!='resume':
    # These tiny protocol doubles never import NumPy, Torch or a model package.
    class Finite:
        def all(self):return True
    np=types.ModuleType('numpy')
    np.fromfile=lambda path,dtype:[x[0] for x in struct.iter_unpack('<f',Path(path).read_bytes())]
    np.isfinite=lambda values:Finite()
    np.random=types.SimpleNamespace(seed=lambda value:None)
    torch=types.ModuleType('torch')
    torch.set_num_threads=lambda value:None
    torch.set_num_interop_threads=lambda value:None
    torch.manual_seed=lambda value:None
    torch.get_num_threads=lambda:1
    torch.get_num_interop_threads=lambda:1
    sys.modules.update(numpy=np,torch=torch)
    def load(runtime,threads):
        if mode=='timeout':
            with heavy_model_slot(os.environ[PATH_ENV]) as lease:
                subprocess.Popen([sys.executable,'-I','-S','-B',__file__,tools_path,'grandchild',str(runtime)],
                                 env=lease.child_environment(),pass_fds=lease.pass_fds)
                time.sleep(30)
        return object(),{'fixture':'fictional decoder; zero models'}
    worker.LOADERS['whisper']=load
    worker.DECODERS['whisper']=lambda *a:{'native':{'text':'fictional'},'text':'fictional',
        'quality_flags':[],'transcript_extraction':'fixture','decoder_terminated':True}
else:
    class NoModelImports(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in {'numpy','torch','qwen_asr','funasr','whisper'}:
                raise RuntimeError('completed cache tried to import a model library')
    sys.meta_path.insert(0,NoModelImports())
    worker.verify_model_assets=lambda *a:(_ for _ in ()).throw(RuntimeError('cache tried to verify/load model assets'))
attempt='resume' if mode=='resume' else 'first'
(output/'attempts'/attempt).mkdir(parents=True)
sys.argv=['active_asr_worker.py','--model','whisper','--runtime',str(runtime),'--inputs',str(runtime/'inputs.json'),
          '--output',str(output),'--attempt',attempt,'--threads','1','--ids','clip-000']
raise SystemExit(worker.main())
'''


class SupervisedAsrTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.lock = self.runtime / "heavy-model.lock"
        self.child = self.root / "fictional_asr.py"
        self.child.write_text(CHILD)
        (self.runtime / "setup.json").write_text(json.dumps({"status": "complete", "installed_packages": {},
            "lock_sha256": "f" * 64, "active_python": "fixture-only"}))
        pcm = self.runtime / "clip.f32"
        pcm.write_bytes(b"\0" * 16)
        self.clip = {"id": "clip-000", "original_member": "fictional.wav", "original_wav": {}, "original_pcm": {},
                     "derived_pcm": {"path": pcm.name, "frames": 4, "sample_rate": 16000, "channels": 1,
                                     "dtype": "<f4", "sha256": hashlib.sha256(pcm.read_bytes()).hexdigest()}}
        (self.runtime / "inputs.json").write_text(json.dumps({"schema": "active-asr-inputs-v1", "clips": [self.clip]}))
        self.recipe = self.root / "recipe.json"
        self.recipe.write_text('{"fixture":"fictional decoders only","real_model_calls":0}')

    def create(self, mode, budget):
        # The outer job writes the bounded ASR ledger before using the actual
        # batch launcher and actual worker entry point in fresh child processes.
        outer = "\n".join((
            "import json,os,runpy,sys,time;from pathlib import Path",
            "sys.argv=sys.argv[1:]",
            "out=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])",
            "(out/'job.json').write_text(json.dumps({'deadline_unix':time.time()+30,'identity':{'max_calls':3}}))",
            "runpy.run_path(sys.argv[0],run_name='__main__')",
        ))
        argv = [sys.executable, "-I", "-S", "-B", "-c", outer,
                str(self.child), str(ROOT / "tools"), "batch", str(self.runtime), mode]
        request = {"schema": queue.SCHEMA, "queue_id": "asr-fixture", "global_budget_minutes": budget,
            "jobs": [{"job_id": "recognize", "recipe": str(self.recipe),
                      "recipe_sha256": queue.sha(self.recipe.read_bytes()), "argv": argv,
                      "cwd": str(self.root), "depends_on": [], "timeout_minutes": 1,
                      "outputs": ["answer.json"]}]}
        path = self.root / "request.json"
        path.write_bytes(queue.canonical(request))
        output = self.root / "queue"
        queue.create(output, path, ROOT / "configs/research/active-plan-20261105.json", self.lock)
        return output

    def test_queue_batch_worker_share_lease_and_completed_resume_skips_models(self):
        root = self.create("complete", 1)
        status, code = queue.run(root)
        self.assertEqual((status["status"], code), ("completed", 0))
        folder = root / "jobs/recognize"
        output = folder / "outputs"
        command = json.loads((folder / "command-started.json").read_bytes())
        first = json.loads((output / "attempts/first/whisper.json").read_bytes())
        self.assertEqual(first["heavy_lock"]["process_group"], command["process_group"])
        self.assertTrue(first["heavy_lock"]["inherited"] and first["heavy_lock"]["supervised"])
        result = json.loads((output / "results/whisper/clip-000.json").read_bytes())
        self.assertEqual(result["decoder_source_sha256"], queue.sha((ROOT / "tools/active_asr_worker.py").read_bytes()))
        resumed = json.loads((output / "attempts/resume/whisper.json").read_bytes())
        self.assertEqual((resumed["new_calls"], resumed["reused"]), (0, 1))
        self.assertEqual(len(list((output / "claims").rglob("*.json"))), 1)
        before = (folder / "result.json").read_bytes()
        self.assertEqual(queue.run(root)[1], 0)
        self.assertEqual((folder / "result.json").read_bytes(), before)

    def test_queue_deadline_removes_worker_and_grandchild_without_releasing_early(self):
        root = self.create("timeout", .025)
        status, code = queue.run(root)
        self.assertEqual((status["status"], code), ("failed_or_blocked", 1))
        folder = root / "jobs/recognize"
        self.assertEqual(status["jobs"]["recognize"]["status"], "timed_out")
        command = json.loads((folder / "command-started.json").read_bytes())
        child = json.loads((folder / "outputs/grandchild.json").read_bytes())
        worker = json.loads((folder / "outputs/attempts/first/whisper.json").read_bytes())
        self.assertEqual(child["pgid"], command["process_group"])
        self.assertEqual(worker["heavy_lock"]["process_group"], command["process_group"])
        self.assertTrue(child["inherited"] and child["supervised"])
        self.assertEqual((child["device"], child["inode"]), (self.lock.stat().st_dev, self.lock.stat().st_ino))
        for pid in (child["pid"], worker["heavy_lock"]["process_id"]):
            proc = Path(f"/proc/{pid}/stat")
            deadline = time.monotonic() + 2
            while proc.exists() and proc.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                if time.monotonic() >= deadline:
                    self.fail("decoder or grandchild remained alive after queue timeout")
                time.sleep(.01)
        with heavy_model_slot(self.lock):
            pass


if __name__ == "__main__":
    unittest.main()
