"""Tests frozen before frontend implementation; reference is the stated frame grid."""
import ctypes as C,pathlib,sys,unittest
LIB=pathlib.Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 else None
F=C.c_float
CALLBACK=C.CFUNCTYPE(None,C.c_void_p,C.POINTER(F),C.c_uint64,C.c_uint64)
class FrameContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.lib=C.CDLL(str(LIB));cls.lib.donor_fbank_state_bytes.restype=C.c_size_t
  cls.lib.donor_fbank_init.argtypes=[C.c_void_p];cls.lib.donor_fbank_reset.argtypes=[C.c_void_p]
  cls.lib.donor_fbank_feed.argtypes=[C.c_void_p,C.POINTER(C.c_int16),C.c_size_t,CALLBACK,C.c_void_p];cls.lib.donor_fbank_feed.restype=C.c_int
  cls.lib.donor_fbank_finish.argtypes=[C.c_void_p];cls.lib.donor_fbank_finish.restype=C.c_int
 def stream(self,n,parts):
  # uint64 array provides sufficiently aligned caller-owned storage.
  state=(C.c_uint64*((self.lib.donor_fbank_state_bytes()+7)//8))();self.lib.donor_fbank_init(state);pcm=(C.c_int16*n)(*[(i*977)%65536-32768 for i in range(n)]);events=[]
  @CALLBACK
  def got(_,row,index,end):events.append((index,end,tuple(row[i] for i in range(80))))
  pos=0;k=0
  while pos<n:
   size=min(parts[k%len(parts)],n-pos);p=C.cast(C.byref(pcm,pos*2),C.POINTER(C.c_int16));self.assertEqual(self.lib.donor_fbank_feed(state,p,size,got,None),0);pos+=size;k+=1
  self.assertEqual(self.lib.donor_fbank_finish(state),0)
  self.assertEqual(self.lib.donor_fbank_finish(state),2)
  self.assertEqual(self.lib.donor_fbank_feed(state,pcm,0,got,None),2)
  expected=0 if n<400 else 1+(n-400)//160
  self.assertEqual([(i,e) for i,e,_ in events],[(j,400+160*j) for j in range(expected)])
  self.lib.donor_fbank_reset(state);again=[]
  @CALLBACK
  def got2(_,row,index,end):again.append((index,end,tuple(row[i] for i in range(80))))
  self.assertEqual(self.lib.donor_fbank_feed(state,pcm,n,got2,None),0);self.assertEqual(again,events)
  return events
 def test_boundaries_and_no_eof_padding(self):
  for n in [0,1,159,160,239,240,399,400,401,559,560,561,719,720,799,800,1599,1600]:
   with self.subTest(n=n):self.stream(n,[1,17,320,159,641])
 def test_partition_bit_parity(self):
  reference=self.stream(4800,[4800])
  for parts in [[1],[159,1],[320],[400],[641,17,3],[160]]:self.assertEqual(self.stream(4800,parts),reference)
 def test_partial_reset_discards(self):
  state=(C.c_uint64*((self.lib.donor_fbank_state_bytes()+7)//8))();self.lib.donor_fbank_init(state);pcm=(C.c_int16*400)(*range(400));seen=[]
  @CALLBACK
  def got(_,r,i,e):seen.append((i,e))
  self.assertEqual(self.lib.donor_fbank_feed(state,pcm,399,got,None),0);self.lib.donor_fbank_reset(state);self.assertEqual(self.lib.donor_fbank_feed(state,pcm,400,got,None),0);self.assertEqual(seen,[(0,400)])
 def test_invalid_input_atomic(self):
  state=(C.c_uint64*((self.lib.donor_fbank_state_bytes()+7)//8))();self.lib.donor_fbank_init(state)
  @CALLBACK
  def got(*args):pass
  before=bytes(state);dummy=(C.c_int16*1)();self.assertEqual(self.lib.donor_fbank_feed(state,dummy,16001,got,None),1);self.assertEqual(bytes(state),before);self.assertEqual(self.lib.donor_fbank_feed(state,None,1,got,None),1);self.assertEqual(bytes(state),before)
if __name__=='__main__':unittest.main()
