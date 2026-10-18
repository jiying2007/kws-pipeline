import unittest, math, sys
from event_head import ReferenceHead, Spec, timing

class SyntheticContractTests(unittest.TestCase):
    def test_budget(self):
        b=Spec().budget(); self.assertEqual(b['parameters'],10498);self.assertEqual(b['mac_per_frame'],10144)
        self.assertEqual(b['receptive_field_frames'],63);self.assertEqual(b['fp32_state_bytes'],7936)
    def test_chunks(self):
        s=Spec(input_dim=4,width=3,dilations=(1,2));x=[[math.sin(i+j) for j in range(4)] for i in range(19)]
        a=ReferenceHead(s).chunk(x);r=ReferenceHead(s);b=r.chunk(x[:1])+r.chunk(x[1:7])+r.chunk(x[7:])
        self.assertEqual(a,b);r.reset();self.assertEqual(a,r.chunk(x))
    def test_causal(self):
        s=Spec(input_dim=4,width=3,dilations=(1,2));x=[[i/20]*4 for i in range(20)]
        a=ReferenceHead(s).chunk(x);x[12:]=[[9.]*4]*8
        self.assertEqual(a[:12],ReferenceHead(s).chunk(x)[:12])
    def test_order_sensitive(self):
        s=Spec(input_dim=2,width=2,dilations=(1,2));r=ReferenceHead(s)
        # Controlled lag/current detector shows the head family can distinguish AB vs BA.
        r.proj=[[1.,0.],[0.,1.]];r.pb=[0.,0.]
        r.dw=[[[1.,0.,0.],[0.,0.,1.]],[[0.,0.,1.],[0.,0.,1.]]]
        r.pw=[[[1.,1.],[0.,0.]],[[1.,0.],[0.,0.]]];r.db=[[0.,0.],[0.,0.]];r.wb=[[0.,0.],[0.,0.]];r.out=[[1.,0.],[0.,1.]]
        a=r.chunk([[1.,0.],[0.,0.],[0.,1.]])[-1];r.reset()
        b=r.chunk([[0.,1.],[0.,0.],[1.,0.]])[-1];self.assertNotEqual(a,b)
    def test_timing_no_backdate(self):
        self.assertEqual(timing(8,16000)['available_ms'],285)
        self.assertEqual(timing(8,16000)['nominal_center_ms'],0)
        self.assertFalse(timing(40,16000)['pre_eof'])
    def test_no_tensor_runtime(self):self.assertNotIn('torch',sys.modules)

if __name__ == '__main__': unittest.main()
