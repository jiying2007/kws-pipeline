import ctypes as C,math,pathlib,struct,sys,unittest
lib=C.CDLL(str(pathlib.Path(sys.argv.pop(1)).resolve()));F=C.c_float;lib.donor_cmvn400.argtypes=[C.POINTER(F)]*4;lib.donor_cmvn400.restype=C.c_int
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
class CMVN(unittest.TestCase):
 def setUp(self):self.x=(F*400)(*[i/17 for i in range(400)]);self.mean=(F*400)(*[i/29 for i in range(400)]);self.std=(F*400)(*[.1+i/1000 for i in range(400)]);self.out=(F*400)(*([123]*400))
 def test_exact_scalar_order(self):
  self.assertEqual(lib.donor_cmvn400(self.x,self.mean,self.std,self.out),0)
  self.assertEqual(list(self.out),[f32(f32(self.x[i]-self.mean[i])*self.std[i]) for i in range(400)])
 def test_alias(self):
  self.assertEqual(lib.donor_cmvn400(self.x,self.mean,self.std,self.out),0);self.assertEqual(lib.donor_cmvn400(self.x,self.mean,self.std,self.x),0);self.assertEqual(list(self.x),list(self.out))
 def test_invalid_is_atomic(self):
  for field,value in [('x',math.nan),('mean',math.inf),('std',0),('std',-1)]:
   self.setUp();getattr(self,field)[399]=value;before=bytes(self.out);self.assertEqual(lib.donor_cmvn400(self.x,self.mean,self.std,self.out),1);self.assertEqual(bytes(self.out),before)
 def test_computed_overflow_atomic(self):
  self.x[399]=float.fromhex('0x1.fffffep127');self.mean[399]=-self.x[399];self.std[399]=1
  for output in [self.out,self.x]:
   before=bytes(output);self.assertEqual(lib.donor_cmvn400(self.x,self.mean,self.std,output),1);self.assertEqual(bytes(output),before)
 def test_partial_overlap(self):
  buf=(F*401)();out=C.cast(C.byref(buf,4),C.POINTER(F));before=bytes(buf);self.assertEqual(lib.donor_cmvn400(buf,self.mean,self.std,out),1);self.assertEqual(bytes(buf),before)
 def test_statistic_overlap(self):self.assertEqual(lib.donor_cmvn400(self.x,self.mean,self.std,self.mean),1)
 def test_null(self):self.assertEqual(lib.donor_cmvn400(None,self.mean,self.std,self.out),1)
if __name__=='__main__':unittest.main()
