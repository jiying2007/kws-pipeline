"""Artificial fixtures only: 3+3 updates plus one 1+1 capped pair, eight total."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.model_comparison.runner import runtime
RT=runtime(ROOT)
torch=RT[0]
from research.direct_kws.model import DirectKWSSpec
from research.direct_fit.data import REUSE_PURPOSE, project_actual_label, whole_source_features, validate_window
from research.direct_fit.runner import SCHEMA, fit_pair as raw_fit_pair, validate_config, paired_models, load_model, digest, trace_model
from research.direct_fit.compare import compare_events
from training_state import load_checkpoint_snapshot, atomic_save_checkpoint, state_identity


FIXTURE_UPDATES = {"ctc": 0, "direct": 0}

def fit_pair(*args, **kwargs):
    try:
        return raw_fit_pair(*args, **kwargs)
    finally:
        run_path = args[3] / "run.json"
        if run_path.exists():
            actual = json.loads(run_path.read_text())["optimizer_steps"]
            for arm in FIXTURE_UPDATES:
                FIXTURE_UPDATES[arm] += actual[arm]


def config():
    return dict(input_mode="reviewed-manifests",registry_inputs=None,schema=SCHEMA,development_only=True,manifests=["unused.jsonl"],tokens="unused.tokens",
        evidence_dir=None,label_reuse_purpose=REUSE_PURPOSE,coverage_mode="all-three-classes",
        direct_spec=DirectKWSSpec(2,(3,3),(1,2)).to_dict(),window_frames=7,seed=17,epochs=1,
        max_paired_steps=3,max_seconds=20,learning_rate=.001,chunk_frames=2,
        direct_event=dict(thresholds=[.5,.5],min_consecutive_frames=1,cooldown_frames=2),
        ctc_event=dict(thresholds=[.5,.5],max_frames=7,cooldown_frames=2))


def items_and_receipt():
    generator=torch.Generator().manual_seed(23)
    items=[(torch.randn(7,32,generator=generator),torch.tensor(tokens,dtype=torch.long),label)
           for tokens,label in (([],0),([1,2,3,4],1),([3,4,3,4],2))]
    receipt={"schema":"artificial-tensor-software-fixture-v1","recordings":[
        {"recording":"fixture-"+str(i)} for i in range(3)]}
    receipt["data_sha256"]=digest(receipt)
    return items,receipt


class DirectFitNumericTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory();cls.folder=Path(cls.temporary.name)
        cls.items,cls.receipt=items_and_receipt()
        cls.fit_result=fit_pair(cls.items,cls.receipt,config(),cls.folder/"completed",ROOT,RT)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_actual_labels_not_intended_text(self):
        row=dict(kind="confusable",keyword_id=None,actual_text="你好",tokens=["ni3","hao3"],
                 target_ids=[1,2],intended_text="你好小窝")
        self.assertEqual(project_actual_label(row),0)
        self.assertEqual(project_actual_label(dict(kind="positive",keyword_id=2,actual_text="小窝小窝",
            tokens=["xiao3","wo1","xiao3","wo1"],target_ids=[3,4,3,4])),2)
        for change in ({"kind":"unknown"},{"tokens":None},{"actual_text":None},
                       {"kind":"positive","keyword_id":1},{"target_ids":[1]}):
            with self.subTest(change=change),self.assertRaises(ValueError):project_actual_label(dict(row,**change))
        with self.assertRaises(ValueError):
            project_actual_label(dict(kind="negative",keyword_id=None,actual_text="你好小窝",
                tokens=["ni3","hao3","xiao3","wo1"],target_ids=[1,2,3,4]))

    def test_no_speech_background_is_explicit(self):
        self.assertEqual(project_actual_label(dict(kind="background",keyword_id=None,
            actual_text="",tokens=[],target_ids=[])),0)
        with self.assertRaises(ValueError):
            project_actual_label(dict(kind="negative",keyword_id=None,actual_text="",tokens=[],target_ids=[]))

    def test_window_full_pcm_tail_and_no_crop(self):
        raw=bytes(801*2)  # Artificial zero PCM geometry fixture, no speech.
        x,metadata=whole_source_features(raw,3,torch=torch,features=RT[3],frontend=RT[4])
        self.assertEqual(tuple(x.shape),(3,32));self.assertEqual(metadata["right_zero_samples"],239)
        self.assertEqual(metadata["window_pcm_sha256"],hashlib.sha256(raw+bytes(239*2)).hexdigest())
        with self.assertRaises(ValueError):whole_source_features(raw,2,torch=torch,features=RT[3],frontend=RT[4])

    def test_receptive_span_with_holes_rejected(self):
        self.assertEqual(validate_window(DirectKWSSpec(2,(3,3),(1,2)),7),2320)
        for spec,frames in ((DirectKWSSpec(2,(2,),(4,)),5),(DirectKWSSpec(2,(3,),(1,)),4)):
            with self.assertRaises(ValueError):validate_window(spec,frames)

    def test_config_fail_closed(self):
        for change in ({"development_only":False},{"label_reuse_purpose":"ctc-training-only"},
                       {"epochs":True},{"max_seconds":float("nan")},{"coverage_mode":"ignore"}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config(dict(config(),**change))
        bad=config();bad["ctc_event"]["cooldown_frames"]=3
        with self.assertRaises(ValueError):validate_config(bad)

    def test_backbone_initialization_is_exact(self):
        models=paired_models(validate_config(config()),17,torch)
        for key,value in models["ctc"].state_dict().items():
            if key.startswith(("in_proj.","blocks.")):
                self.assertTrue(torch.equal(value,models["direct"].state_dict()[key]))
        self.assertEqual(models["ctc"].out_proj.out_features,5)
        self.assertEqual(models["direct"].classifier.out_features,3)

    def test_complete_checkpoint_and_counted_updates(self):
        self.assertEqual(self.fit_result["optimizer_steps"],{"ctc":3,"direct":3})
        self.assertEqual(self.fit_result["status"],"completed")
        folder=self.folder/"completed"
        for arm in ("ctc","direct"):
            initial,_=load_checkpoint_snapshot(folder/f"{arm}.initial.pt")
            final,_=load_checkpoint_snapshot(folder/f"{arm}.final.pt")
            self.assertNotEqual(initial["float_state_identity"],final["float_state_identity"])
            self.assertEqual(final["initial_float_state_identity"],initial["float_state_identity"])
            completion=final["completion_state"]
            self.assertEqual(completion["completed_epochs"],1);self.assertEqual(completion["epoch_offset"],0)
            self.assertEqual(completion["optimizer_steps"],3);self.assertFalse(completion["resume_supported"])
            self.assertTrue(completion["optimizer_state_dict"]["state"])
            self.assertFalse(final["product_export_compatible"])
        direct,_=load_checkpoint_snapshot(folder/"direct.final.pt")
        self.assertEqual(direct["model_family"],"direct-kws")
        self.assertEqual(direct["objective"],"terminal-window-cross-entropy")

    def test_reload_trace_and_common_scoring(self):
        folder=self.folder/"completed";report=json.loads((folder/"comparison.json").read_text())
        self.assertIsNone(report["winner"]);self.assertFalse(report["qualification_allowed"])
        for arm in ("ctc","direct"):
            model,_,_=load_model(folder/f"{arm}.final.pt",ROOT,torch)
            names=[r["recording"] for r in self.receipt["recordings"]]
            traces,events=trace_model(model,self.items,names,arm=arm,config=config(),torch=torch)
            stored=[json.loads(line) for line in (folder/f"{arm}.frame-trace.jsonl").read_text().splitlines()]
            self.assertEqual(stored,traces);self.assertEqual(len(stored),21)
            self.assertEqual(len(stored[0]["probabilities"]),3 if arm=="direct" else 5)
            for width in (1,7):
                alternate,hits=trace_model(model,self.items,names,arm=arm,
                    config=dict(config(),chunk_frames=width),torch=torch)
                for x,y in zip(traces,alternate):
                    self.assertTrue(torch.allclose(torch.tensor(x["logits"]),torch.tensor(y["logits"]),atol=1e-6,rtol=1e-6))
                self.assertEqual(events,hits)

    def test_step_cap_saves_truthful_partial_epoch(self):
        result=fit_pair(self.items,self.receipt,dict(config(),max_paired_steps=1),self.folder/"capped",ROOT,RT)
        self.assertEqual(result["optimizer_steps"],{"ctc":1,"direct":1})
        self.assertEqual(result["status"],"paired-step-cap")
        saved,_=load_checkpoint_snapshot(self.folder/"capped/direct.final.pt")
        self.assertEqual(saved["completion_state"]["completed_epochs"],0)
        self.assertEqual(saved["completion_state"]["epoch_offset"],1)

    def test_zero_step_wall_cap_still_saves_initial_and_final(self):
        values=iter([0.,99.,100.])
        result=fit_pair(self.items,self.receipt,config(),self.folder/"timeout",ROOT,RT,clock=lambda:next(values))
        self.assertEqual(result["optimizer_steps"],{"ctc":0,"direct":0})
        self.assertEqual(result["status"],"wall-time-cap");self.assertEqual(len(result["checkpoints"]),4)

    def test_failed_initial_save_retains_terminal_record(self):
        with mock.patch("training_state.atomic_save_checkpoint",side_effect=OSError("fixture failure")):
            with self.assertRaises(RuntimeError):fit_pair(self.items,self.receipt,config(),self.folder/"failed",ROOT,RT)
        report=json.loads((self.folder/"failed/run.json").read_text())
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["optimizer_steps"],{"ctc":0,"direct":0})
        self.assertEqual(set(report["checkpoint_save_errors"]),{"ctc","direct"})

    def test_keyboard_interrupt_during_final_save_retains_run(self):
        original=atomic_save_checkpoint
        def interrupted(value, path):
            if path.name.endswith(".final.pt"):
                raise KeyboardInterrupt("software fixture")
            return original(value,path)
        values=iter([0.,99.,100.]);destination=self.folder/"interrupted-save"
        with mock.patch("training_state.atomic_save_checkpoint",side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                fit_pair(self.items,self.receipt,config(),destination,ROOT,RT,clock=lambda:next(values))
        report=json.loads((destination/"run.json").read_text())
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["optimizer_steps"],{"ctc":0,"direct":0})
        self.assertTrue((destination/"direct.initial.pt").is_file())
        self.assertIn("KeyboardInterrupt",report["checkpoint_save_errors"]["direct"])

    def test_keyboard_interrupt_during_trace_retains_run(self):
        values=iter([0.,99.,100.]);destination=self.folder/"interrupted-trace"
        with mock.patch("research.direct_fit.runner.trace_model",side_effect=KeyboardInterrupt("software fixture")):
            with self.assertRaises(KeyboardInterrupt):
                fit_pair(self.items,self.receipt,config(),destination,ROOT,RT,clock=lambda:next(values))
        report=json.loads((destination/"run.json").read_text())
        self.assertEqual(report["trace_status"],"failed")
        self.assertIn("KeyboardInterrupt",report["trace_error"])

    def test_checkpoint_tamper_rejected(self):
        original,_=load_checkpoint_snapshot(self.folder/"completed/direct.final.pt")
        for field,value in (("model_family","ctc"),("objective","ctc"),("source_hashes",{})):
            saved=copy.deepcopy(original);saved[field]=value
            path=self.folder/(field+".pt");atomic_save_checkpoint(saved,path)
            with self.subTest(field=field),self.assertRaises(ValueError):load_model(path,ROOT,torch)
        saved=copy.deepcopy(original);saved["state_dict"]["classifier.bias"][0]+=1
        path=self.folder/"bad-tensor.pt";torch.save(saved,path)
        with self.assertRaises(ValueError):load_model(path,ROOT,torch)

    def test_product_exporter_rejects_direct_before_writing(self):
        target=self.folder/"must-not-exist.kwm"
        process=subprocess.run([sys.executable,str(ROOT/"training/export_model.py"),"--checkpoint",
            str(self.folder/"completed/direct.final.pt"),"--tokens","unused","--output",str(target)],
            capture_output=True,text=True)
        self.assertNotEqual(process.returncode,0)
        self.assertIn("research/direct-class checkpoints cannot",process.stderr);self.assertFalse(target.exists())

    def test_live_admission_is_mandatory_and_original_receipt_preserved(self):
        from research.direct_fit.data import load_development
        from corpus_identity import inspect_pcm16_wav
        folder=self.folder/"admission-fixture";folder.mkdir()
        wav=folder/"zero-geometry.wav"
        with wave.open(str(wav),"wb") as stream:
            stream.setparams((1,2,16000,0,"NONE","not compressed"));stream.writeframes(bytes(1001*2))
        measured=inspect_pcm16_wav(wav)
        row=dict(path=str(wav),source_path=str(wav),source_wav_sha256=measured["file_sha256"],
            source_pcm_sha256=measured["pcm_sha256"],split="train",kind="background",keyword_id=None,
            actual_text="",tokens=[],target_ids=[],admission={"allowed_purpose":"ctc-training-only"})
        manifest=folder/"fixture.jsonl";manifest.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
        tokens=folder/"tokens.txt";tokens.write_text("<blk> 0\nni3 1\nhao3 2\nxiao3 3\nwo1 4\n")
        cfg=dict(config(),manifests=[str(manifest)],tokens=str(tokens),coverage_mode="incomplete-coverage-fit-only")
        admitted={"purpose":"reviewed-ctc-training","promotion_allowed":True,
            "manifests":[{"sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()}]}
        with mock.patch("training_admission.verify_training_manifests",return_value=admitted) as verify, \
             mock.patch("audit_dataset.verify_manifest_lineage",return_value=[row,row]):
            items,receipt=load_development(cfg,RT)
            self.assertFalse(verify.call_args.kwargs["diagnostic"])
            self.assertEqual(len(items),1)  # Original PCM aliases are not separate observations.
            self.assertEqual(len(receipt["recordings"][0]["manifest_bindings"]),2)
            self.assertEqual(receipt["original_ctc_admission"],admitted)
            self.assertFalse(receipt["promotion_allowed"])
            self.assertEqual(receipt["coverage"],{"0":1,"1":0,"2":0})
            with self.assertRaises(ValueError):load_development(dict(cfg,coverage_mode="all-three-classes"),RT)
        with mock.patch("training_admission.verify_training_manifests",side_effect=ValueError("unadmitted")):
            with self.assertRaisesRegex(ValueError,"unadmitted"):load_development(cfg,RT)

    def test_replaced_checkpoint_before_reload_is_rejected(self):
        actual_loader=load_model
        def swapped(path, repo, torch_module):
            if path.name == "direct.final.pt":
                value,_=load_checkpoint_snapshot(path)
                value["state_dict"]["classifier.bias"][0] += 1
                value["float_state_identity"]=state_identity(value["state_dict"])
                atomic_save_checkpoint(value,path)
            return actual_loader(path,repo,torch_module)
        values=iter([0.,99.,100.])
        destination=self.folder/"replaced-before-trace"
        with mock.patch("research.direct_fit.runner.load_model",side_effect=swapped):
            with self.assertRaisesRegex(ValueError,"endpoint bytes"):
                fit_pair(self.items,self.receipt,config(),destination,ROOT,RT,clock=lambda:next(values))
        report=json.loads((destination/"run.json").read_text())
        self.assertEqual(report["trace_status"],"failed")
        self.assertEqual(report["optimizer_steps"],{"ctc":0,"direct":0})

    def test_replaced_checkpoint_during_trace_is_rejected(self):
        destination=self.folder/"replaced-during-trace"
        actual_trace=trace_model
        def replaced(*args, **kwargs):
            result=actual_trace(*args, **kwargs)
            if kwargs["arm"] == "direct":
                path=destination/"ctc.initial.pt"
                value,_=load_checkpoint_snapshot(path)
                value["state_dict"]["out_proj.bias"][0] += 1
                value["float_state_identity"]=state_identity(value["state_dict"])
                atomic_save_checkpoint(value,path)
            return result
        values=iter([0.,99.,100.])
        with mock.patch("research.direct_fit.runner.trace_model",side_effect=replaced):
            with self.assertRaisesRegex(ValueError,"changed during"):
                fit_pair(self.items,self.receipt,config(),destination,ROOT,RT,clock=lambda:next(values))
        report=json.loads((destination/"run.json").read_text())
        self.assertEqual(report["trace_status"],"failed")

    def test_source_snapshot_mismatch_is_rejected(self):
        from research.direct_fit.data import read_source
        path=self.folder/"changed.wav";path.write_bytes(b"not the admitted bytes")
        with self.assertRaisesRegex(ValueError,"changed"):
            read_source(path,{"file_sha256":"0"*64,"pcm_sha256":"0"*64,"frames":400})

    def test_same_reference_event_adapter_and_missing_timing(self):
        references=[dict(recording="fixture",duration_s=1.,annotation_status="complete",
                         expected=[dict(keyword_id=1,start_s=.1,end_s=.3)])]
        hit=dict(recording="fixture",keyword_id=1,time_s=.2,confidence=.9)
        result=compare_events(references,{"ctc":[hit],"direct":[hit]},mode="event-windows")
        self.assertEqual(result["arms"]["ctc"],result["arms"]["direct"])
        with self.assertRaises(ValueError):
            compare_events([dict(recording="fixture",duration_s=1.,expected_keywords=[1])],
                {"ctc":[],"direct":[]},mode="clip-presence")


class RegistryAdapterFixtureTest(unittest.TestCase):
    def registry(self):
        return json.loads((ROOT/"research/kws-candidate-registry-v1/candidates.json").read_text())

    def test_public_metadata_coverage_is_three_kw1_only(self):
        from research.direct_fit.registry_data import select_actual_rows
        selected,excluded=select_actual_rows(self.registry())
        self.assertEqual([row["slot_id"] for row,_ in selected],["kw1s-05","kw1s-06","kw1s-07"])
        self.assertEqual([target for _,target in selected],[1,1,1]);self.assertEqual(len(excluded),13)
        self.assertTrue(all(row["training_admitted"] is False for row,_ in selected))

    def test_registry_actual_text_not_generation_target(self):
        from research.direct_fit.registry_data import select_actual_rows
        registry=self.registry()
        for row in registry["records"]:row["target_text"]="小窝小窝";row["target_class"]="KW2"
        selected,_=select_actual_rows(registry)
        self.assertEqual([target for _,target in selected],[1,1,1])
        selected[0][0]["asr"][0]["normalized_text"]="小窝小窝"
        fewer,excluded=select_actual_rows(registry)
        self.assertEqual(len(fewer),2);self.assertEqual(len(excluded),14)

    def test_historical_bindings_are_not_upgraded(self):
        from research.direct_fit.registry_data import select_actual_rows
        registry=self.registry()
        registry["records"]=[r for r in registry["records"] if not r["same_pcm_asr_binding_verified"]]
        with self.assertRaisesRegex(ValueError,"historical"):select_actual_rows(registry)
        registry=self.registry();registry["records"][0]["training_admitted"]=True
        with self.assertRaisesRegex(ValueError,"upgraded"):select_actual_rows(registry)

    def test_missing_paths_fail_before_archive_verifier(self):
        from research.direct_fit.registry_data import load_registry_development, PURPOSE, ARCHIVES
        selected=[r for r in self.registry()["records"] if r["same_pcm_asr_binding_verified"]]
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            cfg=dict(config(),input_mode="registry-exposed-fit",label_reuse_purpose=PURPOSE,
                manifests=[],coverage_mode="incomplete-coverage-fit-only",
                registry_inputs={"artifacts":str(folder),"rights_bindings":[
                    dict(record_id=r["record_id"],path=str(folder/(r["slot_id"]+".rights.json")),sha256="0"*64)
                    for r in selected]})
            validate_config(cfg)
            with self.assertRaises(ValueError) as error:load_registry_development(cfg,RT,ROOT)
            self.assertIn("missing exposed-fit",str(error.exception))
            for name in ARCHIVES:self.assertIn(name,str(error.exception))
            self.assertIn("kw1s-05.rights.json",str(error.exception))

    def test_artificial_registry_consumer_keeps_labels_and_evidence_scope(self):
        from research.direct_fit import registry_data as module
        from research.direct_fit.registry_data import load_registry_development, PURPOSE
        registry=self.registry()
        registry["records"]=[r for r in registry["records"] if r["same_pcm_asr_binding_verified"]]
        raw_pcm=bytes(1001*2);pcm_hash=hashlib.sha256(raw_pcm).hexdigest()
        for row in registry["records"]:
            row["derived_pcm"].update(bytes=len(raw_pcm),frames=1001,sha256=pcm_hash)
            row["target_text"]="小窝小窝";row["target_class"]="KW2"
            for observed in row["asr"]:observed["bound_pcm_sha256"]=pcm_hash
        # A deliberately artificial verifier seam, never the live ZIP verifier.
        verifier_source=("def verify_inputs(registry, sources):\n"
            "    if set(sources) != {'repo','artifacts'}: raise ValueError('bad inputs')\n"
            "    return [{'record_id':r['record_id'], 'pcm_bytes':bytes(r['derived_pcm']['bytes'])} for r in registry['records']]\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/"research/kws-candidate-registry-v1";package.mkdir(parents=True)
            registry_raw=json.dumps(registry).encode();(package/"candidates.json").write_bytes(registry_raw)
            (package/"build_registry.py").write_bytes(verifier_source)
            artifacts=root/"artifacts";artifacts.mkdir()
            for name in module.ARCHIVES:(artifacts/name).write_bytes(b"artificial-not-a-real-ZIP")
            rights=root/"rights.json";rights_raw=json.dumps(dict(schema="retained-asr-source-rights-v1",
                source_family_id="kw1-preset-supplement-20261014-v1",training_use_allowed=True,
                source_ids=[r["slot_id"] for r in registry["records"]],license_id="software-fixture-NOT-real-rights")).encode()
            rights.write_bytes(rights_raw)
            cfg=dict(config(),input_mode="registry-exposed-fit",label_reuse_purpose=PURPOSE,manifests=[],
                tokens=str(ROOT/"keywords/tokens.example.txt"),coverage_mode="incomplete-coverage-fit-only",
                registry_inputs=dict(artifacts=str(artifacts),rights_bindings=[dict(record_id=r["record_id"],
                    path=str(rights),sha256=hashlib.sha256(rights_raw).hexdigest()) for r in registry["records"]]))
            with mock.patch.object(module,"REGISTRY_SHA256",hashlib.sha256(registry_raw).hexdigest()), \
                 mock.patch.object(module,"VERIFIER_SHA256",hashlib.sha256(verifier_source).hexdigest()):
                items,receipt=load_registry_development(cfg,RT,root)
            self.assertEqual(len(items),3);self.assertTrue(all(item[2]==1 for item in items))
            self.assertEqual(receipt["coverage"],{"0":0,"1":3,"2":0})
            self.assertFalse(receipt["training_admitted"]);self.assertFalse(receipt["promotion_allowed"])
            self.assertTrue(all(tuple(item[0].shape)==(7,32) for item in items))

    def test_one_shared_rights_document_can_cover_three_sources(self):
        from research.direct_fit.registry_data import read_rights, select_actual_rows
        selected,_=select_actual_rows(self.registry())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"shared-rights.json"
            value=dict(schema="retained-asr-source-rights-v1",training_use_allowed=True,
                source_ids=[row["slot_id"] for row,_ in selected],
                source_family_id="kw1-preset-supplement-20261014-v1",
                license_id="software-fixture-NOT-real-source-rights")
            raw=json.dumps(value).encode();path.write_bytes(raw)
            bindings=[dict(record_id=row["record_id"],path=str(path),sha256=hashlib.sha256(raw).hexdigest())
                      for row,_ in selected]
            self.assertEqual(len(read_rights(bindings,selected)),3)

    def test_rights_bytes_scope_and_missing_permission(self):
        from research.direct_fit.registry_data import read_rights, select_actual_rows
        selected,_=select_actual_rows(self.registry())
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);bindings=[]
            for row,_ in selected:
                value=dict(schema="retained-asr-source-rights-v1",training_use_allowed=True,
                    source_id=row["slot_id"],source_family_id=row["source"]["campaign_id"],
                    license_id="software-fixture-NOT-real-source-rights")
                raw=json.dumps(value).encode();path=folder/(row["slot_id"]+".json");path.write_bytes(raw)
                bindings.append(dict(record_id=row["record_id"],path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            self.assertEqual(len(read_rights(bindings,selected)),3)
            with self.assertRaises(ValueError):read_rights(bindings[:-1],selected)
            first=bindings[0];path=Path(first["path"]);value=json.loads(path.read_bytes())
            for unknown in (1,"true",None):
                value["training_use_prohibited"]=unknown
                raw=json.dumps(value).encode();path.write_bytes(raw);first["sha256"]=hashlib.sha256(raw).hexdigest()
                with self.subTest(prohibition=unknown),self.assertRaisesRegex(ValueError,"permission"):
                    read_rights(bindings,selected)
            value.pop("training_use_prohibited")
            value["training_use_allowed"]=False
            raw=json.dumps(value).encode();path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,"bytes differ"):read_rights(bindings,selected)
            first["sha256"]=hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(ValueError,"permission"):read_rights(bindings,selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    successful = result.wasSuccessful() and not result.skipped and FIXTURE_UPDATES == {"ctc": 4, "direct": 4}
    if args.report:
        from research.direct_fit.runner import source_hashes
        hashes = source_hashes(ROOT)
        for path in ("tests/test_direct_fit_numeric.py", "training/export_model.py"):
            hashes[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"schema": "direct-fit-software-fixture-v1",
            "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "successful": successful, "skipped": len(result.skipped), "optimizer_steps": FIXTURE_UPDATES,
            "source_sha256": hashes, "python_version": sys.version.split()[0],
            "fixture": {"seed": 17, "feature_seed": 23, "rows": 3, "frames": 7,
                        "direct_spec": config()["direct_spec"], "expected_total_optimizer_updates": 8},
            "real_audio_training_calls": 0, "tts_calls": 0, "asr_calls": 0,
            "scope": "artificial tensor/zero-PCM software fixtures and public metadata only",
            "torch_version": str(torch.__version__)}, indent=2) + "\n")
    raise SystemExit(0 if successful else 1)

