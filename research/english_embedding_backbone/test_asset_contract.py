"""Source-only frozen asset contract tests; no downloads or inference."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_openwakeword_assets as a
class AssetContractTests(unittest.TestCase):
 def test_trigger_and_exact_asset_roster(self):
  a.verify_trigger()
  self.assertEqual(len(a.EXPECTED),4)
  self.assertEqual(a.EXPECTED["melspectrogram.onnx"][0],129656233)
  self.assertEqual(a.EXPECTED["embedding_model.onnx"][0],129656245)
 def test_shared_feature_model_license_separated_from_classifiers(self):
  x=a.verify_trigger()
  self.assertEqual(x["feature_model_license"],"Apache-2.0")
  self.assertEqual(x["excluded_classifier_license"],"CC-BY-NC-SA-4.0")
  self.assertIs(x["training"],False)
  self.assertIs(x["model_promotion"],False)
 def test_size_budget_is_explicit(self):
  t=a.verify_trigger()
  sizes={r["name"]:r["bytes"] for r in t["assets"]}
  self.assertEqual(sizes["melspectrogram.tflite"]+sizes["embedding_model.tflite"],2422828)
  self.assertEqual(sizes["melspectrogram.onnx"]+sizes["embedding_model.onnx"],2414536)
 def test_synthetic_geometry_uses_upstream_audio_and_embedding_windows(self):
  for declared in ([None,"samples"],[1,"samples"],[1,16000]):
   self.assertEqual(a.synthetic_input_shape("melspectrogram.onnx","input",declared,
                                           "tensor(float)"),[1,16000])
  self.assertEqual(a.synthetic_input_shape("embedding_model.onnx","input_1",
                 ["batch",76,32,1],"tensor(float)"),[1,76,32,1])
  self.assertEqual(a.synthetic_input_shape("embedding_model.onnx","input_1",
                 [1,76,32,1],"tensor(float)"),[1,76,32,1])
 def test_synthetic_geometry_rejects_unknown_or_incompatible_graph(self):
  for asset,name,shape,dtype in (
      ("unknown.onnx","input",[None,None],"tensor(float)"),
      ("melspectrogram.onnx","unknown",[None,None],"tensor(float)"),
      ("melspectrogram.onnx","input",[None,None],"tensor(int16)"),
      ("melspectrogram.onnx","input",[1,1],"tensor(float)"),
      ("melspectrogram.onnx","input",[2,16000],"tensor(float)"),
      ("melspectrogram.onnx","input",[1,-1],"tensor(float)"),
      ("melspectrogram.onnx","input",[True,16000],"tensor(float)"),
      ("melspectrogram.onnx","input",[1,0],"tensor(float)"),
      ("embedding_model.onnx","input_1",[1,76,32],"tensor(float)"),
      ("embedding_model.onnx","input_1",[1,75,32,1],"tensor(float)"),
      ("embedding_model.onnx","input_1",[1,76,64,1],"tensor(float)")):
   with self.subTest(asset=asset,name=name,shape=shape,dtype=dtype):
    with self.assertRaises(ValueError):a.synthetic_input_shape(asset,name,shape,dtype)
if __name__=="__main__":unittest.main()
