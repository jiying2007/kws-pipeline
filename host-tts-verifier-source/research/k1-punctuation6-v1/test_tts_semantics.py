"""Stdlib deterministic FLOAT/PCM fixtures and strict archive mutation tests.

These prove byte/schema consistency only, never real model inference, native
SoundFile/SciPy behavior, speech quality, ASR truth, or training admission.
"""
from copy import deepcopy
import hashlib
import json
import math
import struct
import unittest
from unittest import mock

import contract
import tts_semantics as semantics
import tts_worker as worker
from test_tts_worker import eos_record, float_wav, pcm_wav, pcm_wav_samples, riff, TempCase, NumericArray, NumericOps, SoundFileFake
from test_tts_public_schema import source_fixture, resource_fixture, stage_fixture

IDENTITY = {"repository":"jiying2007/kws-data","executed_head_sha":"a"*40,
            "run_id":701,"run_attempt":1,"job_key":"k1-tts"}


def trusted_authorities():
    _,freeze,provenance=source_fixture()
    return {"source_freeze":freeze,"model_provenance":provenance,
            "plan_sha256":freeze["files"]["research/k1-punctuation6-v1/plan.proposed.json"]["sha256"]}


def set_json(members, name, value):
    members[name] = contract.canonical(value)


def update_json(members, name, fn):
    value = json.loads(members[name])
    fn(value)
    set_json(members, name, value)


def archive(completed=6, failed=False, partial_audio=False):
    """All-six denominator with complete prefix, optional failure, and not-run tail."""
    outcomes = worker.new_outcomes()
    outcomes["status"] = "completed_six_once" if completed == 6 else ("stopped_no_retry" if failed else "not_run")
    outcomes["consumed_attempts"] = completed + int(failed)
    members = {}
    set_json(members,"claim.json",{"schema":"k1-punctuation6-host-claim-v1","phase":"tts","identity":IDENTITY,
             "preflight_verified":True,"source_freeze_sha256":"b"*64,"source_tree":"c"*40,
             "source_branch":"research/k1-punctuation6-tts-v1","result_branch":"data/k1-punctuation6-tts-v1",
             "maximum_tts_attempts":6,"preflight":{"preflight_verified":True,"numeric_job_id":702,
             "source_branch_workflow_runs":1,"repository_public":True,"result_ref_absent":True,"read_only_requests":5}})
    source,_,_=source_fixture();source["identity"]=deepcopy(IDENTITY)
    resources=resource_fixture();resources["identity"]=deepcopy(IDENTITY)
    resources["status"]="started" if outcomes["status"]=="not_run" else outcomes["status"]
    if completed==6:resources["stages"]={name:stage_fixture(name) for name in ("setup","generation")}
    set_json(members,"source.json",source)
    set_json(members,"resources.json",resources)
    eos_rows = []
    for index, cell in enumerate(contract.cells()):
        row, ledger = outcomes["cells"][index], outcomes["ledger"][index]
        if index >= completed + int(failed): continue
        started = {"schema":"k1-punctuation6-tts-start-v1","cell_id":cell["cell_id"],"seed":cell["seed"],
                   "voice":cell["voice"],"text":cell["text"],"attempt":1,
                   "request_sha256":hashlib.sha256(contract.canonical(cell)).hexdigest()}
        claim_path = "claims/"+cell["cell_id"]+".started.json"
        set_json(members,claim_path,started)
        status = "complete" if index < completed else "failed_no_retry"
        row.update(status=status,attempts=1,claim=semantics.measured(members[claim_path]))
        ledger.update(status=status,attempts=1)
        if index == completed:
            row["error_class"] = "ValueError"
            eos_rows.append({"cell_id":cell["cell_id"],"failed_capture":{"capture_calls":1,"iterations":120}})
            if partial_audio: members["raw/"+cell["cell_id"]+".wav"] = float_wav([0.125]*3840)
            continue
        raw_path, derived_path = "raw/"+cell["cell_id"]+".wav", "derived/"+cell["cell_id"]+".wav"
        raw, pcm = float_wav([0.125 + index/64]*3840), pcm_wav(2560, 7+index)
        members[raw_path], members[derived_path] = raw, pcm
        observation = semantics.raw_wave(raw) | semantics.pcm_wave(pcm) | {
            "raw_path":raw_path,"derived_path":derived_path,"source_rate_hz":24000,"source_channels":1,
            "source_dtype":"little_endian_float32","sample_rate_hz":16000,"channels":1,"sample_width_bytes":2,
            "derivation":"scipy-resample-poly-2-to-3-no-gain-no-trim","quality_flags":[],
            "derivative_peak":0.125+index/64,"derivative_float_sha256":hashlib.sha256(struct.pack("<d",0.125+index/64)*2560).hexdigest()}
        row.update(observation=observation,eos=eos_record(),generation_seconds=0.25)
        eos_rows.append({"cell_id":cell["cell_id"],**row["eos"]})
    set_json(members,"outcomes.json",outcomes)
    set_json(members,"eos.json",{"schema":"k1-punctuation6-eos-v1","rows":eos_rows})
    return members


class WavTests(unittest.TestCase):
    def test_raw_finite_float24k_with_fact(self):
        raw = float_wav([0.0,0.25,-0.25])
        result = semantics.raw_wave(raw)
        self.assertEqual(result["source_frames"],3)
        self.assertEqual(result["raw_float_sha256"],hashlib.sha256(struct.pack("<3f",0.0,0.25,-0.25)).hexdigest())
        self.assertEqual(result["raw_wav_sha256"],hashlib.sha256(raw).hexdigest())

    def test_raw_fmt18_and_peak_chunk_supported(self):
        raw = riff([(b"fmt ",struct.pack("<HHIIHHH",3,1,24000,96000,4,32,0)),
                    (b"fact",struct.pack("<I",2)),(b"PEAK",struct.pack("<II f I",1,0,0.5,0)),
                    (b"data",struct.pack("<2f",0.5,0.0))])
        self.assertEqual(semantics.raw_wave(raw)["source_frames"],2)

    def test_raw_rejects_nonfinite_empty_cap_and_wrong_geometry(self):
        for values in ([math.nan],[math.inf],[-math.inf],[],[0.0]*(12*24000)):
            with self.subTest(length=len(values),first=values[:1]):
                with self.assertRaises(ValueError): semantics.raw_wave(float_wav(values))
        for fmt in ((1,1,24000,48000,2,16),(3,2,24000,192000,8,32),(3,1,16000,64000,4,32)):
            with self.subTest(fmt=fmt):
                with self.assertRaises(ValueError):
                    semantics.raw_wave(riff([(b"fmt ",struct.pack("<HHIIHH",*fmt)),(b"data",struct.pack("<2f",0.1,0.2))]))

    def test_raw_rejects_truncation_extra_bytes_and_false_riff_length(self):
        raw = float_wav([0.125]*10)
        for bad in (raw[:-1],raw+b"x",raw[:4]+struct.pack("<I",1)+raw[8:]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError): semantics.raw_wave(bad)

    def test_raw_rejects_unknown_duplicate_and_wrong_fact_chunks(self):
        fmt = (b"fmt ",struct.pack("<HHIIHH",3,1,24000,96000,4,32))
        data = (b"data",struct.pack("<2f",0.125,0.25))
        for chunks in ([fmt,data,(b"JUNK",b"x")],[fmt,data,data],[fmt,data,(b"fact",struct.pack("<I",3))],
                       [fmt,(b"data",b"\0"*5)]):
            with self.subTest(chunks=[x[0] for x in chunks]), self.assertRaises(ValueError): semantics.raw_wave(riff(chunks))

    def test_raw_peak_must_be_valid_mono_peak_metadata(self):
        fmt=(b"fmt ",struct.pack("<HHIIHH",3,1,24000,96000,4,32))
        data=(b"data",struct.pack("<2f",0.25,0.5))
        for peak in (b"arbitrary private text",b"",struct.pack("<II f I",1,0,math.nan,0),
                     struct.pack("<II f I",1,0,0.5,999)):
            with self.subTest(length=len(peak)),self.assertRaises(ValueError):
                semantics.raw_wave(riff([fmt,(b"PEAK",peak),data]))

    def test_pcm16k_exact_hashes(self):
        raw = pcm_wav(30,9)
        result = semantics.pcm_wave(raw)
        self.assertEqual(result["frames"],30)
        self.assertEqual(result["pcm_sha256"],hashlib.sha256(struct.pack("<h",9)*30).hexdigest())

    def test_pcm_rejects_truncated_or_unknown_trailing_chunk(self):
        raw = pcm_wav(10)
        bad = raw+b"JUNK"+struct.pack("<I",500)+b"x"
        bad = bad[:4]+struct.pack("<I",len(bad)-8)+bad[8:]
        with self.assertRaises(ValueError): semantics.pcm_wave(bad)

    def test_pcm_rejects_odd_data_byte_instead_of_ignoring_it(self):
        raw = riff([(b"fmt ",struct.pack("<HHIIHH",1,1,16000,32000,2,16)),(b"data",b"\0"*5)])
        with self.assertRaises(ValueError): semantics.pcm_wave(raw)

    def test_pcm_rejects_empty_cap_wrong_rate_and_duplicate_data(self):
        for raw in (pcm_wav(0),pcm_wav(12*16000),
                    riff([(b"fmt ",struct.pack("<HHIIHH",1,1,24000,48000,2,16)),(b"data",b"\0"*10)]),
                    riff([(b"fmt ",struct.pack("<HHIIHH",1,1,16000,32000,2,16)),(b"data",b"\0"*10),(b"data",b"\0"*10)])):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError): semantics.pcm_wave(raw)

    def test_json_requires_canonical_utf8_duplicate_free_finite_object(self):
        self.assertEqual(semantics.read_json(contract.canonical({"text":"你好，小窝"})),{"text":"你好，小窝"})
        for raw in (b'{ "a":1}',b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}',b'[]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): semantics.read_json(raw)


class EosTests(unittest.TestCase):
    def test_valid_eos(self):
        semantics.validate_eos(eos_record())

    def test_invalid_eos_values_rejected(self):
        for key,value in (("capture_calls",2),("capture_calls",True),("iterations",0),("iterations",120),
                          ("sequence_length",4),("last_token_ids",[999]),("last_token_ids",[2150,2150]),
                          ("eos",999),("ended_with_eos",False),("ended_with_eos",1),
                          ("hit_token_cap_without_eos",True),("earlier_eos_seen",True)):
            with self.subTest(key=key,value=value):
                eos=eos_record();eos[key]=value
                with self.assertRaises(ValueError): semantics.validate_eos(eos)

    def test_extra_or_missing_eos_field_rejected(self):
        for kind in ("extra","missing"):
            eos=eos_record()
            if kind=="extra":eos["new"]=1
            else:del eos["sequence_length"]
            with self.subTest(kind=kind),self.assertRaises(ValueError):semantics.validate_eos(eos)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        patcher=mock.patch.object(semantics,"local_authorities",side_effect=trusted_authorities)
        patcher.start();self.addCleanup(patcher.stop)

    def check(self,members):
        return semantics.validate_tts_members(members,IDENTITY)

    def test_complete_six_fixture_binds_every_receipt(self):
        members=archive()
        self.assertEqual(len(members),23)
        self.assertTrue(self.check(members))

    def test_not_run_and_each_partial_failure_preserve_six_rows(self):
        self.assertTrue(self.check(archive(0)))
        for completed in range(6):
            with self.subTest(completed=completed):
                members=archive(completed,True)
                self.assertTrue(self.check(members))
                out=json.loads(members["outcomes.json"])
                self.assertEqual(len(out["cells"]),6)
                self.assertEqual(out["consumed_attempts"],completed+1)
                self.assertEqual([r["status"] for r in out["cells"]], ["complete"]*completed+["failed_no_retry"]+["not_run"]*(5-completed))

    def test_failed_started_raw_only_is_truthful_partial_evidence(self):
        self.assertTrue(self.check(archive(2,True,True)))

    def test_fixed_cell_field_or_order_mutation_rejected(self):
        for field,value in (("voice","Vivian"),("text","你好沃"),("seed",1012099),("human_gold",True),
                            ("training_admitted",True),("attempts_max",2)):
            with self.subTest(field=field):
                members=archive()
                update_json(members,"outcomes.json",lambda out:out["cells"][0].update({field:value}))
                with self.assertRaises(ValueError):self.check(members)
        members=archive()
        update_json(members,"outcomes.json",lambda out:out["cells"].reverse())
        with self.assertRaises(ValueError):self.check(members)

    def test_denominator_and_consumed_accounting_mutations_rejected(self):
        for mutate in (lambda out:out["cells"].pop(),lambda out:out.update(consumed_attempts=5),
                       lambda out:out.update(consumed_attempts=True),lambda out:out["ledger"][0].update(attempts=2)):
            members=archive();update_json(members,"outcomes.json",mutate)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.check(members)

    def test_original_started_bytes_required_and_hash_bound(self):
        for kind in ("absent","seed","request","whitespace","receipt"):
            with self.subTest(kind=kind):
                members=archive();name="claims/k1p6-001.started.json"
                if kind=="absent":del members[name]
                if kind=="seed":update_json(members,name,lambda row:row.update(seed=9))
                if kind=="request":update_json(members,name,lambda row:row.update(request_sha256="0"*64))
                if kind=="whitespace":members[name]+=b"\n"
                if kind=="receipt":update_json(members,"outcomes.json",lambda out:out["cells"][0]["claim"].update(sha256="0"*64))
                with self.assertRaises(ValueError):self.check(members)

    def test_each_hash_and_frame_binding_rejects_receipt_mutation(self):
        for field,value in (("raw_wav_sha256","0"*64),("raw_float_sha256","0"*64),("wav_sha256","0"*64),
                            ("pcm_sha256","0"*64),("source_frames",3839),("frames",2559),
                            ("raw_path","raw/k1p6-002.wav"),("derived_path","derived/k1p6-002.wav"),
                            ("derivation","gain-normalized")):
            with self.subTest(field=field):
                members=archive()
                update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update({field:value}))
                with self.assertRaises(ValueError):self.check(members)

    def test_wav_mutation_and_truncation_rejected_without_receipt_update(self):
        for path in ("raw/k1p6-001.wav","derived/k1p6-001.wav"):
            for kind in ("mutate","truncate"):
                with self.subTest(path=path,kind=kind):
                    members=archive();raw=members[path]
                    members[path]=raw[:-1] if kind=="truncate" else raw[:-1]+bytes([raw[-1]^1])
                    with self.assertRaises(ValueError):self.check(members)

    def test_wrong_resample_frame_ratio_rejected_even_with_new_hashes(self):
        members=archive();path="derived/k1p6-001.wav"
        members[path]=pcm_wav(2561)
        update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update(semantics.pcm_wave(members[path])))
        with self.assertRaises(ValueError):self.check(members)

    def test_codec_frame_binding_rejected_even_with_self_consistent_wavs(self):
        members=archive()
        members["raw/k1p6-001.wav"]=float_wav([0.25]*1920)
        members["derived/k1p6-001.wav"]=pcm_wav(1280)
        observed=semantics.raw_wave(members["raw/k1p6-001.wav"])|semantics.pcm_wave(members["derived/k1p6-001.wav"])
        update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update(observed))
        with self.assertRaises(ValueError):self.check(members)

    def test_complete_cell_missing_raw_or_pcm_rejected(self):
        for path in ("raw/k1p6-001.wav","derived/k1p6-001.wav"):
            with self.subTest(path=path):
                members=archive();del members[path]
                with self.assertRaises(ValueError):self.check(members)

    def test_partial_derivative_without_raw_rejected(self):
        members=archive(0,True)
        members["derived/k1p6-001.wav"]=pcm_wav(2560)
        with self.assertRaises(ValueError):self.check(members)

    def test_unstarted_cell_cannot_acquire_claim_audio_or_eos(self):
        complete=archive()
        for name in ("claims/k1p6-006.started.json","raw/k1p6-006.wav","derived/k1p6-006.wav","eos.json"):
            with self.subTest(name=name):
                members=archive(0,True)
                if name=="eos.json":
                    update_json(members,name,lambda e:e["rows"].append({"cell_id":"k1p6-006",**eos_record()}))
                else:members[name]=complete[name]
                with self.assertRaises(ValueError):self.check(members)

    def test_eos_row_missing_duplicate_or_drift_rejected(self):
        for kind in ("missing","duplicate","drift"):
            members=archive()
            def mutate(e):
                if kind=="missing":e["rows"].pop(0)
                if kind=="duplicate":e["rows"][1]=deepcopy(e["rows"][0])
                if kind=="drift":e["rows"][0]["iterations"]=4
            update_json(members,"eos.json",mutate)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check(members)

    def test_source_resource_claim_run_identity_mismatch_rejected(self):
        for name in ("claim.json","source.json","resources.json"):
            with self.subTest(name=name):
                members=archive()
                update_json(members,name,lambda value:value["identity"].update(run_id=702))
                with self.assertRaises(ValueError):self.check(members)

    def test_source_freeze_and_tree_must_match_host_claim(self):
        for key,value in (("source_freeze_sha256","d"*64),("source_tree","d"*40)):
            with self.subTest(key=key):
                members=archive()
                update_json(members,"source.json",lambda src:src.update({key:value}))
                with self.assertRaises(ValueError):self.check(members)

    def test_source_freeze_and_tree_cannot_both_be_missing(self):
        for key in ("source_freeze_sha256","source_tree"):
            with self.subTest(key=key):
                members=archive()
                for name in ("claim.json","source.json"):
                    update_json(members,name,lambda value:value.pop(key))
                with self.assertRaises(ValueError):self.check(members)

    def test_source_identity_fields_cannot_be_matching_invalid_values(self):
        for key,value in (("source_freeze_sha256",None),("source_freeze_sha256","bad"),("source_tree",None),("source_tree","bad")):
            with self.subTest(key=key,value=value):
                members=archive()
                for name in ("claim.json","source.json"):
                    update_json(members,name,lambda record:record.update({key:value}))
                with self.assertRaises(ValueError):self.check(members)

    def test_complete_resources_status_must_match(self):
        members=archive()
        update_json(members,"resources.json",lambda value:value.update(status="stopped_no_retry"))
        with self.assertRaises(ValueError):self.check(members)

    def test_receipt_format_metadata_must_match_measured_wavs(self):
        for key,value in (("source_rate_hz",8000),("source_channels",2),("source_dtype","float64"),
                          ("sample_rate_hz",8000),("channels",2),("sample_width_bytes",1)):
            with self.subTest(key=key):
                members=archive()
                update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update({key:value}))
                with self.assertRaises(ValueError):self.check(members)

    def test_measured_frame_counts_require_integer_type(self):
        for key,value in (("source_frames",3840.0),("frames",2560.0)):
            with self.subTest(key=key):
                members=archive()
                update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update({key:value}))
                with self.assertRaises(ValueError):self.check(members)

    def test_quality_flags_cannot_be_cleared_added_or_reordered(self):
        for kind in ("silent_clear","source_clear","derivative_clear","invent"):
            with self.subTest(kind=kind):
                members=archive()
                if kind in ("silent_clear","source_clear"):
                    raw=float_wav([0.0 if kind=="silent_clear" else 1.25]*3840)
                    members["raw/k1p6-001.wav"]=raw
                    update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update(semantics.raw_wave(raw)))
                if kind=="derivative_clear":
                    update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update(derivative_peak=1.25))
                if kind=="invent":
                    update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update(quality_flags=["silent"]))
                with self.assertRaises(ValueError):self.check(members)

    def first_cell_pcm_fixture(self,raw_value,pcm_values,flags):
        """Deterministic bytes and receipts only; no numeric-library conversion."""
        members=archive()
        raw=float_wav([raw_value]*3840);pcm=pcm_wav_samples(pcm_values)
        members["raw/k1p6-001.wav"],members["derived/k1p6-001.wav"]=raw,pcm
        observed=semantics.raw_wave(raw)|semantics.pcm_wave(pcm)|{
            "quality_flags":list(flags),"derivative_peak":abs(raw_value),
            "derivative_float_sha256":hashlib.sha256(struct.pack("<d",raw_value)*len(pcm_values)).hexdigest()}
        update_json(members,"outcomes.json",lambda out:out["cells"][0]["observation"].update(observed))
        return members

    def test_nonzero_tiny_float_source_with_zero_pcm_requires_derived_silent_flag(self):
        tiny=struct.unpack("<f",struct.pack("<f",1e-10))[0]
        members=self.first_cell_pcm_fixture(tiny,[0]*2560,["derived_pcm_silent"])
        self.assertTrue(self.check(members))
        out=json.loads(members["outcomes.json"])
        self.assertEqual(out["cells"][0]["status"],"complete")
        self.assertGreater(out["cells"][0]["observation"]["source_peak"],0)
        self.assertGreater(out["cells"][0]["observation"]["derivative_peak"],0)

    def test_zero_raw_and_zero_pcm_dual_flags_are_valid_complete_cell(self):
        members=self.first_cell_pcm_fixture(0.0,[0]*2560,["silent","derived_pcm_silent"])
        self.assertTrue(self.check(members))
        self.assertEqual(json.loads(members["outcomes.json"])["cells"][0]["status"],"complete")

    def test_any_single_positive_or_negative_pcm_lsb_is_not_silent(self):
        tiny=struct.unpack("<f",struct.pack("<f",1e-10))[0]
        for sample in (1,-1):
            for position in (0,1,1279,2559):
                with self.subTest(sample=sample,position=position):
                    values=[0]*2560;values[position]=sample
                    self.assertTrue(self.check(self.first_cell_pcm_fixture(tiny,values,[])))

    def test_derived_pcm_silent_cannot_be_cleared_forged_duplicated_or_reordered(self):
        tiny=struct.unpack("<f",struct.pack("<f",1e-10))[0]
        cases=[(tiny,[0]*2560,[]),(0.0,[0]*2560,["silent"]),
               (tiny,[1]+[0]*2559,["derived_pcm_silent"]),
               (tiny,[0]*2559+[-1],["derived_pcm_silent"]),
               (tiny,[0]*2560,["derived_pcm_silent","derived_pcm_silent"]),
               (0.0,[0]*2560,["derived_pcm_silent","silent"])]
        for index,(raw,pcm,flags) in enumerate(cases):
            with self.subTest(index=index),self.assertRaises(ValueError):
                self.check(self.first_cell_pcm_fixture(raw,pcm,flags))

    def test_exact_metadata_fields_reject_extra_private_strings(self):
        for name,target in (("claim.json","top"),("outcomes.json","top"),("outcomes.json","cell"),
                            ("outcomes.json","observation"),("eos.json","top"),("eos.json","row")):
            with self.subTest(name=name,target=target):
                members=archive()
                def mutate(value):
                    destination=value
                    if target=="cell":destination=value["cells"][0]
                    if target=="observation":destination=value["cells"][0]["observation"]
                    if target=="row":destination=value["rows"][0]
                    destination["unexpected_private_note"]="sentinel"
                update_json(members,name,mutate)
                with self.assertRaises(ValueError):self.check(members)

    def test_public_errors_and_generation_seconds_bounded(self):
        for key,value in (("error_class","CustomSecretError"),("generation_seconds",-1),("generation_seconds",301),
                          ("generation_seconds",True)):
            with self.subTest(key=key,value=value):
                members=archive()
                update_json(members,"outcomes.json",lambda out:out["cells"][0].update({key:value}))
                with self.assertRaises(ValueError):self.check(members)

    def test_unrecognized_member_or_absent_required_metadata_rejected(self):
        members=archive();members["private.txt"]=b"not allowed"
        with self.assertRaises(ValueError):self.check(members)
        for name in ("claim.json","source.json","outcomes.json","eos.json","resources.json"):
            with self.subTest(name=name):
                members=archive();del members[name]
                with self.assertRaises(ValueError):self.check(members)

    def test_not_run_cannot_describe_consumed_cells(self):
        members=archive()
        update_json(members,"outcomes.json",lambda value:value.update(status="not_run"))
        with self.assertRaises(ValueError):self.check(members)

    def test_failure_cannot_continue_to_later_cell(self):
        members=archive()
        def mutate(out):
            out["cells"][0]["status"]="failed_no_retry"
            out["ledger"][0]["status"]="failed_no_retry"
            out["status"]="stopped_no_retry"
        update_json(members,"outcomes.json",mutate)
        with self.assertRaises(ValueError):self.check(members)


class EndToEndOfflineTests(TempCase):
    def setUp(self):
        super().setUp()
        patcher=mock.patch.object(semantics,"local_authorities",side_effect=trusted_authorities)
        patcher.start();self.addCleanup(patcher.stop)

    def complete_metadata(self):
        members=archive(0)
        update_json(members,"resources.json",lambda value:value.update(status="completed_six_once",stages={name:stage_fixture(name) for name in ("setup","generation")}))
        for path in self.root.rglob("*"):
            if path.is_file():members[path.relative_to(self.root).as_posix()]=path.read_bytes()
        return members

    def materialize_finalization(self,members):
        output=self.root/"publication-members";output.mkdir()
        for name,raw in members.items():
            path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        return output

    def finalization_members(self,output):
        return {path.relative_to(output).as_posix():path.read_bytes() for path in output.rglob("*") if path.is_file()}

    def test_finalization_reconciles_exclusive_claim_before_ledger_save_without_calls(self):
        import tts_finalize
        members=archive(0,True)
        outcomes=worker.new_outcomes();outcomes["status"]="started"
        set_json(members,"outcomes.json",outcomes)
        set_json(members,"eos.json",{"schema":"k1-punctuation6-eos-v1","rows":[]})
        members["outcomes.json.tmp"]=contract.canonical({"untrusted_incomplete_save":"never promote"})
        output=self.materialize_finalization(members)
        original=members["outcomes.json"]
        original_claim=members["claims/k1p6-001.started.json"]
        with mock.patch.object(worker,"execute_cells",side_effect=AssertionError("finalization must not resume")) as execute:
            result=tts_finalize.finalize_after_exit(output,self.root,worker_exited=True)
        execute.assert_not_called()
        self.assertEqual(result["status"],"stopped_no_retry")
        self.assertEqual(result["consumed_attempts"],1)
        self.assertEqual([row["status"] for row in result["cells"]],["failed_no_retry"]+["not_run"]*5)
        self.assertEqual((output/"claims/k1p6-001.started.json").read_bytes(),original_claim)
        receipt=result["finalization"]
        self.assertEqual(receipt["reconciled_cells"],[{"cell_id":"k1p6-001","previous_status":"not_run"}])
        self.assertEqual(receipt["original_outcomes"],semantics.measured(original))
        self.assertEqual(receipt["preserved_temporaries"],[{"path":"outcomes.json.tmp",**semantics.measured(members["outcomes.json.tmp"])}])
        self.assertFalse(receipt["calls_resumed"])
        self.assertFalse(receipt["completion_inferred"])
        self.assertFalse((output/"outcomes.json.tmp").exists())
        self.assertTrue(semantics.validate_tts_members(self.finalization_members(output),IDENTITY))
        self.assertEqual(tts_finalize.finalize_after_exit(output,self.root,worker_exited=True),result)

    def test_finalization_preserves_complete_prefix_and_quarantines_only_invalid_failed_audio(self):
        import tts_finalize
        members=archive(1,True,True)
        def interrupted(out):
            out["status"]="started"
            out["cells"][1]["status"]="started"
            out["cells"][1].pop("error_class")
            out["ledger"][1]["status"]="started"
        update_json(members,"outcomes.json",interrupted)
        path="derived/k1p6-002.wav";broken=b"RIFF truncated synthetic bytes"
        members[path]=broken
        output=self.materialize_finalization(members)
        first_before=json.loads(members["outcomes.json"])["cells"][0]
        result=tts_finalize.finalize_after_exit(output,self.root,worker_exited=True)
        self.assertEqual(result["cells"][0],first_before)
        self.assertEqual(result["consumed_attempts"],2)
        self.assertEqual([row["status"] for row in result["cells"]],["complete","failed_no_retry"]+["not_run"]*4)
        self.assertTrue((output/"raw/k1p6-002.wav").is_file())
        self.assertFalse((output/path).exists())
        self.assertEqual((self.root/"finalization-private"/path).read_bytes(),broken)
        self.assertEqual(result["finalization"]["quarantined_audio"],[{"path":path,**semantics.measured(broken),"reason":"invalid_waveform"}])
        self.assertTrue(semantics.validate_tts_members(self.finalization_members(output),IDENTITY))
        republished=self.finalization_members(output);republished[path]=pcm_wav(2560)
        with self.assertRaises(ValueError):semantics.validate_tts_members(republished,IDENTITY)

    def test_execute_write_pair_semantics_complete_six_with_stdlib_fakes(self):
        soundfile=SoundFileFake()
        def resample(array,up,down):return NumericArray([0.125]*((len(array)*2+2)//3),"float64")
        def generate(row):return NumericArray([0.25]*3840),eos_record(),0.1
        def write(array,cell):return worker.write_pair(array,cell,self.root,NumericOps,soundfile,resample)
        result=worker.execute_cells(self.root,generate,write,alarm=lambda _:None)
        self.assertEqual(result["consumed_attempts"],6)
        self.assertTrue(semantics.validate_tts_members(self.complete_metadata(),IDENTITY))

    def test_derived_pcm_silent_keeps_all_six_completed_once_without_regeneration(self):
        tiny=struct.unpack("<f",struct.pack("<f",1e-10))[0]
        soundfile=SoundFileFake([0]*2560)
        calls=[]
        def generate(row):
            calls.append(row["cell_id"])
            value=tiny if row["condition"]=="unpunctuated" else 0.0
            return NumericArray([value]*3840),eos_record(),0.1
        def resample(array,*_):return NumericArray([array.values[0]]*2560,"float64")
        def write(array,cell):return worker.write_pair(array,cell,self.root,NumericOps,soundfile,resample)
        result=worker.execute_cells(self.root,generate,write,alarm=lambda _:None)
        self.assertEqual(calls,[row["cell_id"] for row in contract.cells()])
        self.assertEqual(result["status"],"completed_six_once")
        self.assertEqual(result["consumed_attempts"],6)
        self.assertEqual([row["status"] for row in result["cells"]],["complete"]*6)
        for row in result["cells"]:
            expected=["derived_pcm_silent"] if row["condition"]=="unpunctuated" else ["silent","derived_pcm_silent"]
            self.assertEqual(row["observation"]["quality_flags"],expected)
        self.assertTrue(semantics.validate_tts_members(self.complete_metadata(),IDENTITY))

    def test_execute_partial_failure_retains_real_claim_and_failed_capture(self):
        state={}
        def generate(row):
            state.clear();state.update(capture_calls=1,iterations=120,ended_with_eos=False)
            raise ValueError("synthetic capture failure")
        with self.assertRaises(ValueError):
            worker.execute_cells(self.root,generate,lambda *_:None,alarm=lambda _:None,capture_state=state)
        members=self.complete_metadata()
        update_json(members,"resources.json",lambda value:value.update(status="stopped_no_retry"))
        self.assertTrue(semantics.validate_tts_members(members,IDENTITY))


if __name__ == "__main__":
    unittest.main(verbosity=2)
