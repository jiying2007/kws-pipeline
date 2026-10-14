import importlib.util
from pathlib import Path
import unittest

s=importlib.util.spec_from_file_location('diagnostic_runner',Path(__file__).with_name('diagnostic_runner.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class DiagnosticContractTests(unittest.TestCase):
    def stats(self,n=1000,count=0):
        return dict(processed_samples=n,processed_frames=m.frame_count(n),detections=count,discontinuities=0,lost_samples=0,external_vad_frames=0)
    def event(self,**change):
        return dict(dict(recording='session',keyword_id=1,time_s=.02,confidence=.9),**change)
    def test_frame_geometry(self):
        self.assertEqual([m.frame_count(n) for n in (0,399,400,719,720,1000)],[0,0,1,1,2,2])
    def test_zero_events_require_complete_stats(self):
        m.validate_outputs('session',1000,[],self.stats())
        bad=self.stats();bad['processed_samples']=999
        with self.assertRaises(ValueError):m.validate_outputs('session',1000,[],bad)
    def test_nonzero_events_and_stats(self):
        m.validate_outputs('session',1000,[self.event()],self.stats(count=1))
        with self.assertRaises(ValueError):m.validate_outputs('session',1000,[self.event()],self.stats())
    def test_bad_event_fields(self):
        for fields in ({'keyword_id':0},{'keyword_id':True},{'time_s':float('nan')},{'time_s':.1},{'confidence':1.01},{'confidence':True},{'recording':'other'},{'timing_contract':'unexpected'}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):m.validate_outputs('session',1000,[self.event(**fields)],self.stats(count=1))
    def test_out_of_order_events(self):
        with self.assertRaises(ValueError):m.validate_outputs('session',1000,[self.event(time_s=.03),self.event(time_s=.02)],self.stats(count=2))
    def test_discontinuity_and_external_vad_rejected(self):
        for key in ('discontinuities','lost_samples','external_vad_frames'):
            bad=self.stats();bad[key]=1
            with self.assertRaises(ValueError):m.validate_outputs('session',1000,[],bad)
    def test_boolean_stats_rejected(self):
        bad=self.stats();bad['detections']=False
        with self.assertRaises(ValueError):m.validate_outputs('session',1000,[],bad)
    def test_no_execution_without_flag(self):
        with self.assertRaisesRegex(ValueError,'requires --execute'):m.run('/absent','/absent','/absent',False)

if __name__=='__main__':unittest.main(verbosity=2)
