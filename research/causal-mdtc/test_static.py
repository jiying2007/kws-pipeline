import ast
import json
import math
from pathlib import Path
import unittest
from geometry_labels import geometry, frame_count, targets, pool_scalar, dependency_ranges

class StaticContracts(unittest.TestCase):
    def test_geometry(self):
        g=geometry()
        self.assertEqual(g['parameters'],42594)
        self.assertEqual(g['weight_macs_per_frame'],37952)
        self.assertEqual(g['cache_fp32_bytes'],48128)
        self.assertEqual(g['receptive_field_frames'],377)
        self.assertEqual(g['multiscale_rf_frames'],[129,253,377])
        self.assertEqual(g['pcm_span_samples'],60560)
    def test_schedule(self):
        self.assertEqual([frame_count(n) for n in [0,399,400,559,560,16000]], [0,0,1,1,2,98])
    def test_rf_causal(self):
        span=dependency_ranges(geometry()['cache_lengths'])
        for t in range(500): self.assertEqual(span(t),(t-376,t))
    def test_unknown_not_negative(self):
        self.assertEqual(targets(['human_positive','unknown']),((1,1.0),(0,0.0)))
        self.assertEqual(targets(['weak_negative','weak_positive']),((0,0.25),(1,0.25)))
        with self.assertRaises(ValueError): pool_scalar([[0,0]],1,['unknown','unknown'])
    def test_padding_and_late_positive(self):
        a=pool_scalar([[-9,-9],[9,-9],[999,999]],2,['human_positive','human_negative'])
        self.assertAlmostEqual(a,math.log1p(math.exp(-9)))
        self.assertEqual(a,pool_scalar([[-9,-9],[9,-9]],2,['human_positive','human_negative']))
    def test_two_simultaneous_keywords(self):
        self.assertLess(pool_scalar([[9,-9],[-9,9]],2,['human_positive','human_positive']),0.001)
    def test_no_import_or_execution_needed(self):
        for p in Path(__file__).parent.rglob('*.py'): ast.parse(p.read_text())
    def test_reject_empty_and_nonfinite(self):
        with self.assertRaises(ValueError): pool_scalar([],0,['human_negative','human_negative'])
        with self.assertRaises(ValueError): pool_scalar([[float('inf'),0]],1,['human_positive','unknown'])

if __name__=='__main__': unittest.main()
