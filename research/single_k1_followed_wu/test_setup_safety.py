"""Standard-library subprocess regression; no installer or model execution."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from setup_safety import run_bounded,parse_smoke_stdout

class SetupSafetyTests(unittest.TestCase):
    def test_large_installed_file_and_separate_native_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            code='import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(bytes(8*1024**2));print("{\\\"session_count\\\":0}");print("native warning",file=sys.stderr)'
            result=run_bounded([sys.executable,'-c',code,str(root/'mock-installed')],root,'smoke',time.monotonic()+5)
            self.assertEqual((root/'mock-installed').stat().st_size,8*1024**2)
            self.assertEqual(parse_smoke_stdout(root/result['stdout'],{'session_count':0}),{'session_count':0})
            self.assertEqual((root/result['stderr']).read_text(),'native warning\n')
    def test_existing_logs_reduce_shared_cap_and_failure_consumes_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'previous.log').write_bytes(b'x'*100)
            with self.assertRaisesRegex(ValueError,'COMBINED_LOG_CAP'):
                run_bounded([sys.executable,'-c','print("a"*200)'],root,'overflow',time.monotonic()+5,combined_log_cap=256)
            self.assertLessEqual(sum(p.stat().st_size for p in root.glob('*.log')),256)
            with self.assertRaisesRegex(ValueError,'NO_LOG_RESUME'):
                run_bounded([sys.executable,'-c','pass'],root,'overflow',time.monotonic()+5)
    def test_deadline_stops_sleeping_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            start=time.monotonic()
            with self.assertRaisesRegex(ValueError,'WALL_DEADLINE'):
                run_bounded([sys.executable,'-c','import time;time.sleep(30)'],tmp,'slow',start+.1)
            self.assertLess(time.monotonic()-start,3)
    def test_missing_duplicate_extra_wrong_type_records_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'stdout.log'
            for raw in ['', '{}', '{"session_count":0}\n{"session_count":0}',
                        '{"session_count":false}', '{"session_count":0,"extra":1}',
                        '{"session_count":0,"session_count":0}', 'warning\n{"session_count":0}']:
                p.write_text(raw)
                with self.assertRaises(ValueError):parse_smoke_stdout(p,{'session_count':0})
            p.write_text(json.dumps({'session_count':0}))
            self.assertEqual(parse_smoke_stdout(p,{'session_count':0}),{'session_count':0})

if __name__=='__main__':unittest.main()
