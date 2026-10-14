from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json,tempfile,unittest
from unittest.mock import patch
import resource_receipt as r
class Tests(unittest.TestCase):
    def test_available_read_only_cgroup(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'memory.peak').write_text('12345\n');(p/'cpu.stat').write_text('usage_usec 123\nnr_throttled 2\n')
            result=r.sample(cgroup_root=p)
            self.assertEqual(result['cgroup_memory_peak']['value'],12345)
            self.assertEqual(result['cgroup_cpu_stat']['usage_usec']['value'],123)
            self.assertEqual(result['cgroup_cpu_stat']['user_usec']['status'],'unknown')
            self.assertEqual((p/'memory.peak').read_text(),'12345\n')
    def test_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            result=r.sample(cgroup_root=Path(d))
            self.assertEqual(result['cgroup_memory_peak']['status'],'unknown')
            self.assertNotIn('value',result['cgroup_memory_peak'])
    def test_ru_maxrss_units_explicit(self):
        from types import SimpleNamespace
        with patch.object(r.resource,'getrusage',return_value=SimpleNamespace(ru_maxrss=123)),patch.object(r.platform,'system',return_value='Linux'):
            self.assertEqual(r.sample()['process_peak_rss']['value'],123*1024)
    def test_failure_finally_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'original'):
                try:raise RuntimeError('original')
                finally:r.worker_final(d,stage='generation',completed=False)
            v=json.loads((Path(d)/'remaining-worker-resources.json').read_text());self.assertFalse(v['completed'])
    def test_hardkill_missing_is_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            r.host_summary(d,container_state={'ExitCode':137,'OOMKilled':True})
            v=json.loads((Path(d)/'remaining-host-resource-status.json').read_text())
            self.assertEqual(v['worker_measurements'],'unknown');self.assertTrue(v['container_state']['OOMKilled'])
    def test_write_failure_best_effort(self):
        with patch.object(r,'atomic',side_effect=OSError('fixture')):
            self.assertFalse(r.worker_final('/unused'))
            self.assertFalse(r.host_summary('/unused'))
if __name__=='__main__':unittest.main()
