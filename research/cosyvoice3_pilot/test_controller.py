import importlib.util,json,os,pathlib,signal,subprocess,sys,tempfile,time,unittest
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import controller as c
from unittest import mock
import shutil
class Cleanup(unittest.TestCase):
 def test_detached_grandchild_is_terminated(self):
  with tempfile.TemporaryDirectory() as d:
   target=pathlib.Path(d)/'pids.json'
   child="import os,time; open("+repr(str(target))+",'w').write(str(os.getpid()));time.sleep(60)"
   leader="import subprocess,sys,time;subprocess.Popen([sys.executable,'-I','-S','-c',"+repr(child)+"],start_new_session=True);time.sleep(60)"
   p=subprocess.Popen([sys.executable,'-I','-S','-c',leader],start_new_session=True)
   observed={}
   try:
    end=time.monotonic()+5
    while (not target.exists() or target.stat().st_size==0) and time.monotonic()<end:observed.update(c.descendants(p.pid));time.sleep(.02)
    self.assertTrue(target.exists());pid=int(target.read_text());observed.update(c.descendants(p.pid));self.assertIn(pid,observed)
    c.terminate(p,observed)
    end=time.monotonic()+2
    while time.monotonic()<end:
     try:state=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0]
     except FileNotFoundError:break
     if state=='Z':break
     time.sleep(.02)
    else:self.fail('detached descendant still active')
   finally:
    if p.poll() is None:c.terminate(p,observed)
 def test_reparented_child_still_counts_toward_rss(self):
  with tempfile.TemporaryDirectory() as d:
   target=pathlib.Path(d)/'pid';release=pathlib.Path(d)/'release'
   child="import os,time; data=bytearray(16*1024**2); open("+repr(str(target))+",'w').write(str(os.getpid()));time.sleep(60)"
   leader="import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-I','-S','-c',"+repr(child)+"],start_new_session=True);path=pathlib.Path("+repr(str(release))+");exec('while not path.exists(): time.sleep(.01)')"
   p=subprocess.Popen([sys.executable,'-I','-S','-c',leader],start_new_session=True)
   observed={}
   try:
    end=time.monotonic()+5
    while (not target.exists() or target.stat().st_size==0) and time.monotonic()<end:observed.update(c.descendants(p.pid));time.sleep(.02)
    self.assertTrue(target.exists());pid=int(target.read_text());observed.update(c.descendants(p.pid));self.assertIn(pid,observed)
    release.touch();p.wait(timeout=5)
    self.assertNotEqual(c.process_snapshot()[pid]['parent'],p.pid)
    self.assertGreaterEqual(c.tree_rss(p.pid,observed),16*1024**2)
    self.assertNotIn(pid,c.owned_processes(p.pid,{pid:'invalid-start-time'},{pid:{'parent':-1,'start':observed[pid],'state':'S','rss':1}}))
   finally:c.terminate(p,observed)
 def test_reparented_child_new_descendants_are_owned(self):
  with tempfile.TemporaryDirectory() as d:
   base=pathlib.Path(d);target=base/'pid';release=base/'release';spawn=base/'spawn';grandpid=base/'grandpid'
   grand="import os,time; data=bytearray(16*1024**2);open("+repr(str(grandpid))+",'w').write(str(os.getpid()));time.sleep(60)"
   child="import os,time,pathlib,subprocess,sys;data=bytearray(16*1024**2);open("+repr(str(target))+",'w').write(str(os.getpid()));path=pathlib.Path("+repr(str(spawn))+");exec('while not path.exists(): time.sleep(.01)');subprocess.Popen([sys.executable,'-I','-S','-c',"+repr(grand)+"],start_new_session=True);time.sleep(60)"
   leader="import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-I','-S','-c',"+repr(child)+"],start_new_session=True);path=pathlib.Path("+repr(str(release))+");exec('while not path.exists(): time.sleep(.01)')"
   p=subprocess.Popen([sys.executable,'-I','-S','-c',leader],start_new_session=True);observed={}
   try:
    end=time.monotonic()+5
    while (not target.exists() or target.stat().st_size==0) and time.monotonic()<end:observed.update(c.owned_processes(p.pid,observed));time.sleep(.02)
    self.assertTrue(target.exists());pid=int(target.read_text());observed.update(c.owned_processes(p.pid,observed));self.assertIn(pid,observed)
    release.touch();p.wait(timeout=5);spawn.touch();end=time.monotonic()+5
    while (not grandpid.exists() or grandpid.stat().st_size==0) and time.monotonic()<end:time.sleep(.02)
    self.assertTrue(grandpid.exists());grandchild=int(grandpid.read_text())
    self.assertNotIn(grandchild,observed);self.assertIn(grandchild,c.owned_processes(p.pid,observed))
    self.assertGreaterEqual(c.tree_rss(p.pid,observed),32*1024**2)
    self.assertTrue(c.terminate(p,observed));self.assertEqual(c.live_owned(p.pid,observed),{})
   finally:c.terminate(p,observed)
 def test_fast_detached_child_is_adopted(self):
  c.enable_subreaper()
  with tempfile.TemporaryDirectory() as d:
   target=pathlib.Path(d)/'pid'
   child="import os,time;data=bytearray(16*1024**2);open("+repr(str(target))+",'w').write(str(os.getpid()));time.sleep(60)"
   leader="import subprocess,sys,time;time.sleep(.04);subprocess.Popen([sys.executable,'-I','-S','-c',"+repr(child)+"],start_new_session=True)"
   p=subprocess.Popen([sys.executable,'-I','-S','-c',leader],start_new_session=True);observed=c.descendants(p.pid)
   try:
    p.wait(timeout=5);end=time.monotonic()+5
    while (not target.exists() or target.stat().st_size==0) and time.monotonic()<end:time.sleep(.02)
    self.assertTrue(target.exists());pid=int(target.read_text());self.assertNotIn(pid,observed)
    self.assertIn(pid,c.owned_processes(p.pid,observed));self.assertGreaterEqual(c.tree_rss(p.pid,observed),16*1024**2)
    self.assertTrue(c.terminate(p,observed));self.assertEqual(c.live_owned(p.pid,observed),{})
   finally:c.terminate(p,observed)
 def test_phase_log_is_hard_bounded_and_failed(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);(root/'logs').mkdir();o=c.Controller.__new__(c.Controller);o.root=root;o.free=shutil.disk_usage(root).free;o.started=time.monotonic();o.steps=[];o.state={'status':'running','steps':o.steps}
   with mock.patch.object(c,'LOG_MAX',1024):
    with self.assertRaises(RuntimeError):o.phase('burst',[sys.executable,'-I','-S','-c','import os;os.write(1,b"x"*(2*1024*1024))'],5)
   self.assertEqual((root/'logs/burst.log').stat().st_size,1024)
   self.assertEqual(o.steps[-1]['status'],'failed');self.assertTrue(o.steps[-1]['log_truncated']);self.assertTrue(o.steps[-1]['cleanup_verified'])
 def test_phase_normal_log_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);(root/'logs').mkdir();o=c.Controller.__new__(c.Controller);o.root=root;o.free=shutil.disk_usage(root).free;o.started=time.monotonic();o.steps=[];o.state={'status':'running','steps':o.steps}
   with mock.patch.object(c,'LOG_MAX',1024):o.phase('normal',[sys.executable,'-I','-S','-c','import os;os.write(1,b"hello")'],5)
   self.assertEqual((root/'logs/normal.log').read_bytes(),b'hello')
   self.assertEqual(o.steps[-1]['status'],'passed');self.assertFalse(o.steps[-1]['log_truncated']);self.assertTrue(o.steps[-1]['log_eof']);self.assertTrue(o.steps[-1]['cleanup_verified'])
 def test_owned_process_rss(self):
  self.assertGreater(c.tree_rss(os.getpid()),0)
if __name__=='__main__':unittest.main(verbosity=2)
