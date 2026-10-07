"""Pure regression for the pinned graph's two permitted dimension refinements."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import run_single_melo as runner

META = json.loads((Path(__file__).parent/'onnx-static-metadata.json').read_bytes())

def session(shape=None):
    def nodes(rows):
        return [SimpleNamespace(name=r['name'], type='tensor(int64)' if r['element_type']=='INT64' else 'tensor(float)',
                  shape=[d['symbol'] if 'symbol' in d else d['value'] for d in r['shape']]) for r in rows]
    inputs, outputs = nodes(META['graph']['inputs']), nodes(META['graph']['outputs'])
    if shape is not None: outputs[0].shape=shape
    return SimpleNamespace(get_providers=lambda:['CPUExecutionProvider'],get_inputs=lambda:inputs,
                           get_outputs=lambda:outputs,get_modelmeta=lambda:SimpleNamespace(custom_metadata_map=META['metadata']))

class OutputSignatureTests(unittest.TestCase):
    def test_only_unit_batch_and_channel_refinements_pass(self):
        for batch in ['N',1]:
            for channels in ['S',1]:
                runner.validate_loaded_session(session([batch,channels,'T']),META)

    def test_wrong_fixed_dimensions_and_unknowns_fail(self):
        for shape in [[2,1,'T'],[1,2,'T'],[1,None,'T'],[None,1,'T'],['other',1,'T'],
                      [1,'other','T'],[1,1,'other'],[1,1,100],[True,1,'T'],[1,True,'T'],[1,1]]:
            with self.subTest(shape=shape),self.assertRaises(ValueError):
                runner.validate_loaded_session(session(shape),META)

    def test_output_name_dtype_and_count_remain_strict(self):
        for field,value in [('name','other'),('type','tensor(double)')]:
            s=session([1,1,'T']);setattr(s.get_outputs()[0],field,value)
            with self.assertRaises(ValueError):runner.validate_loaded_session(s,META)
        s=session();s.get_outputs().append(s.get_outputs()[0])
        with self.assertRaises(ValueError):runner.validate_loaded_session(s,META)

    def test_input_signature_remains_exact(self):
        s=session([1,1,'T']);s.get_inputs()[3].shape=[2]
        with self.assertRaisesRegex(ValueError,'LOADED_INPUT_SIGNATURE'):
            runner.validate_loaded_session(s,META)

    def test_capture_preserves_rejected_values_without_inference(self):
        s=session([1,2,'T']);snapshot=runner.loaded_session_signature(s)
        self.assertEqual(snapshot['outputs'][0]['shape'],[1,2,'T'])
        with self.assertRaises(ValueError):runner.validate_loaded_session(s,META,snapshot)
        self.assertEqual(snapshot['outputs'][0]['shape'],[1,2,'T'])

if __name__=='__main__':unittest.main()
