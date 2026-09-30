import importlib.util,pathlib,unittest,math
from fractions import Fraction
import numpy as np
p=pathlib.Path(__file__).with_name('full_oracle.py');s=importlib.util.spec_from_file_location('oracle',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Bounds(unittest.TestCase):
 def test_gamma(self):
  for n in [2,14,27,129,257,501,801]:
   for bits in [24,53]:self.assertGreaterEqual(Fraction.from_float(m.gamma(n,bits)),Fraction(n,2**bits-n))
 def test_operations_upward(self):
  for a,b in [(0.,0.),(1.,2**-53),(2**-1022,2**-100),(1e-250,1e-250),(2.,3.)]:
   self.assertGreaterEqual(Fraction.from_float(float(m.add(a,b))),Fraction.from_float(a)+Fraction.from_float(b))
   self.assertGreaterEqual(Fraction.from_float(float(m.mul(a,b))),Fraction.from_float(a)*Fraction.from_float(b))
 def test_nonfinite(self):
  with self.assertRaises(ValueError):m.add(float('inf'),1)
  with self.assertRaises(ValueError):m.mul(1e300,1e300)
 def test_memory_indices(self):
  p=np.arange(15*128,dtype=float).reshape(15,128);l=np.ones((128,10));r=np.ones((128,2));o=m.memory_oracle(p,l,r)
  for t in range(15):
   expected=(p[t-2] if t>=2 else 0)+sum((p[j] for j in range(max(0,t-11),t-1)),start=np.zeros(128))+(p[t-1] if t>=1 else 0)+p[t]
   self.assertTrue(np.array_equal(o[t],expected))
if __name__=='__main__':unittest.main()
