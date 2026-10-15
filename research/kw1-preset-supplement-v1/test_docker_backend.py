"""Supervisor contracts tested without invoking Docker or models."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import docker_backend as b
from test_blind_asr import fixture


def fake_backend(root):
    obj=b.DockerBackend.__new__(b.DockerBackend)
    obj.work=Path(root);obj._failed=False;obj._next=0;obj.deadline=time.monotonic()+18000
    obj._summaries=[];obj._runtime_cleanups=[]
    st=obj.work.stat();obj._work_identity=(st.st_dev,st.st_ino)
    obj._save_summary=lambda:None
    obj._host_capacity=lambda:dict(available_memory_bytes=14*b.GIB,available_disk_bytes=25*b.GIB)
    return obj

class BackendTests(unittest.TestCase):
    def test_real_pinned_constructor_and_minimal_mounts_without_docker(self):
        import hashlib
        import worker
        import blind_asr_worker
        repo=Path(b.__file__).resolve().parents[2]
        files={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in b.dependency_paths()}
        with tempfile.TemporaryDirectory() as tmp:
            obj=b.DockerBackend(repo,Path(tmp)/'backend',time.monotonic()+18000,source_lock={'files':files})
            for engine in ('qwen06','qwen17'):
                selected=b.PILOT+'/'+engine+'-model-lock.json'
                obj._stage_code(engine,b.TTS_FILES+(selected,),aliases={b.PILOT+'/qwen-model-lock.json':selected},extra={b.NEW+'/plan.json':(repo/b.NEW/'plan.json').read_bytes()})
                worker.verify_sources(engine,obj.work/'code'/engine)
            obj._stage_code('asr',b.ASR_FILES+b.ASR_NEW)
            blind_asr_worker.verify_sources(obj.work/'code'/'asr')
            self.assertEqual(obj._next,0)
            self.assertEqual(obj._summaries,[])

    def test_fixed_resource_and_stage_limits(self):
        self.assertEqual(b.MEMORY,12*1024**3);self.assertEqual(b.GLOBAL_SECONDS,18000)
        self.assertEqual(b.STAGES,('image','setup_qwen06','tts_qwen06','setup_qwen17','tts_qwen17','setup_primary','qwen_asr','sensevoice','setup_whisper','whisper_small'))
    def test_fixed_blind_source_inventory(self):
        self.assertNotIn(b.NEW+'/plan.json',b.ASR_NEW)
        self.assertNotIn(b.NEW+'/worker.py',b.ASR_NEW)
        from blind_pcm import RETAINED_PINS
        self.assertEqual(set(b.ASR_FILES),set(RETAINED_PINS))
    def test_serial_once_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=fake_backend(tmp);calls=[]
            for stage in b.STAGES:obj._stage(stage,lambda _:calls.append(stage))
            self.assertEqual(calls,list(b.STAGES))
            with self.assertRaises(ValueError):obj._stage('image',lambda _:calls.append('retry'))
            self.assertEqual(calls,list(b.STAGES))
    def test_failed_capacity_consumes_terminal_stage(self):
        for memory,disk in [(14*b.GIB-1,25*b.GIB),(14*b.GIB,25*b.GIB-1),(True,25*b.GIB),(float(14*b.GIB),25*b.GIB)]:
            with tempfile.TemporaryDirectory() as tmp:
                obj=fake_backend(tmp);obj._host_capacity=lambda:dict(available_memory_bytes=memory,available_disk_bytes=disk);calls=[]
                with self.assertRaises(ValueError):obj._stage('image',lambda _:calls.append(1))
                self.assertEqual(calls,[]);self.assertTrue(obj._failed)
                self.assertTrue((Path(tmp)/'stages/image/stage.claim').is_file())
                with self.assertRaises(ValueError):obj._stage('image',lambda _:None)
    def test_expired_deadline_no_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=fake_backend(tmp);obj.deadline=time.monotonic()-1;calls=[]
            with self.assertRaises(ValueError):obj._stage('image',lambda _:calls.append(1))
            self.assertEqual(calls,[])
    def test_blind_snapshot_rejects_producer_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obj=fake_backend(root);inp=root/'input';fixture(inp)
            obj._blind_input(inp,root/'snapshot')
            (inp/'plan.json').write_text('{"target_text":"secret"}')
            with self.assertRaises(ValueError):obj._blind_input(inp,root/'bad')
    def test_cleanup_requires_exact_owned_terminal_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obj=fake_backend(root)
            for name in ('../','claims','runtime-qwen06'):
                with self.assertRaises(ValueError):obj._cleanup_runtime(name)
            obj._summaries=[dict(stage='tts_qwen06',status='complete')]
            runtime=root/'runtime-qwen06';runtime.mkdir();(runtime/'bytes').write_bytes(b'abc')
            obj._read_mountinfo=lambda:'1 0 0:1 / / rw - tmpfs tmpfs rw\n'
            obj._cleanup_runtime('runtime-qwen06')
            self.assertFalse(runtime.exists());self.assertEqual(obj._runtime_cleanups[0]['logical_regular_bytes'],3)
    def test_nested_mount_cleanup_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obj=fake_backend(root);obj._summaries=[dict(stage='tts_qwen06',status='complete')]
            runtime=root/'runtime-qwen06';runtime.mkdir()
            obj._read_mountinfo=lambda:f'1 0 0:1 / {runtime}/nested rw - tmpfs tmpfs rw\n'
            with self.assertRaises(ValueError):obj._cleanup_runtime('runtime-qwen06')
            self.assertTrue(runtime.exists())
if __name__=='__main__':unittest.main()
