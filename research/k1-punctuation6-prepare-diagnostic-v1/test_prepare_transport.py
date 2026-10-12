"""Strict local fixtures and fake Git APIs only; no model or network work."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock
import base64
import zipfile
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_publisher as p
import diagnostic_transport as t
import prepare_contract as pc

def tree_response(rows):
    digest, dirs = t.tree_identity(rows)
    return {"sha": digest, "truncated": False, "tree": [*copy.deepcopy(rows), *[{"path": name, "mode": "040000", "type": "tree", "sha": value} for name, value in sorted(dirs.items())]]}

class MockAPI:
    def __init__(self, env, event, release, source):
        self.identity = p.context(env, event, release); self.release = release
        self.paths = p.endpoints(self.identity, release); self.policy = t.policy(release["purpose"])
        self.calls = []; self.blobs = {}; self.source = source; self.mutations = {}; self.ref = None
        self.repo = {"full_name": t.REPOSITORY, "private": False}
        self.run = {"id": 123, "run_attempt": 1, "head_sha": release["source_sha"], "head_branch": self.policy["source_branch"],
            "event": "push", "path": self.policy["workflow"], "status": "in_progress", "repository": self.repo}
        self.history = {"total_count": 1, "workflow_runs": [self.run]}
        self.job = {"id": 456, "run_id": 123, "head_sha": release["source_sha"], "head_branch": self.policy["source_branch"],
            "name": self.policy["job_key"], "run_url": "https://api.github.com" + self.paths["run"], "status": "in_progress", "conclusion": None}
        self.jobs = {"total_count": 1, "jobs": [self.job]}

    def bind_result(self, commit, tree):
        self.expected_commit = commit; self.expected_tree = tree

    def __call__(self, method, route, body=None, expected=200):
        self.calls.append((method, route, copy.deepcopy(body), expected))
        mutation = self.mutations.get((method, route, expected))
        if isinstance(mutation, BaseException): raise mutation
        if mutation is not None: return mutation(method, route, body, expected)
        if method == "GET":
            if route == self.paths["repo"]: return self.repo
            if route == self.paths["run"]: return self.run
            if route == self.paths["history"]: return self.history
            if route == self.paths["jobs"]: return self.jobs
            if route == self.paths["source"]: return {"sha": self.release["source_sha"], "tree": {"sha": self.release["source_tree"]}}
            if route == self.paths["source_tree"]: return tree_response(self.source)
            if route == self.paths["ref"]:
                if expected == 404:
                    if self.ref: raise p.PublicationError(200)
                    return None
                return self.ref
            if route == p.BASE + "/git/commits/" + self.expected_commit: return self.commit
            if route == p.BASE + "/git/trees/" + self.expected_tree + "?recursive=1": return tree_response(self.result_rows)
        elif route == p.BASE + "/git/blobs":
            raw = base64.b64decode(body["content"], validate=True); digest = t.git_sha(raw)
            self.blobs[digest] = raw
            return {"sha": digest}
        elif route == p.BASE + "/git/trees":
            self.result_rows = self.source + body["tree"]
            return tree_response(self.result_rows)
        elif route == p.BASE + "/git/commits":
            expected_body, digest = p.commit_body(body["tree"], self.release)
            if body != expected_body: raise AssertionError("commit body differs")
            self.commit = {"sha": digest, "message": body["message"], "tree": {"sha": body["tree"]},
                "parents": [{"sha": v} for v in body["parents"]], "author": body["author"], "committer": body["committer"]}
            return self.commit
        elif route == p.BASE + "/git/refs":
            if self.ref: raise p.PublicationError(422)
            self.ref = {"ref": body["ref"], "object": {"type": "commit", "sha": body["sha"]}}
            return self.ref
        raise AssertionError("unexpected request")

    def raw_factory(self, commit, purpose):
        if commit != self.expected_commit: raise AssertionError("moving/mixed commit")
        self.raw_calls = []; prefix = t.policy(purpose)["directory"] + "/"
        files = {row["path"]: self.blobs[row["sha"]] for row in self.result_rows if row["path"].startswith(prefix)}
        def read(name):
            if name in self.raw_calls: raise AssertionError("duplicate read")
            self.raw_calls.append(name)
            return files[name]
        return read


def fixture():
    run = {"repository": t.REPOSITORY, "run_id": 123, "run_attempt": 1, "job_key": t.PURPOSE, "executed_head_sha": "a"*40}
    release = {**pc.FIXED, **pc.BINDING_PINS, "enabled": True, "independent_source_review": True,
        "source_closure_ready": True, "prior_prepare_history_sha256": "c"*64}
    code = {"research/"+t.EXPERIMENT+"/prepare-release.json": t.canonical(release), "README.md": b"public test fixture"}
    identities = {n: {"bytes": len(b), "sha256": t.sha(b)} for n,b in code.items()}
    freeze = {"schema": "k1-prepare-source-freeze-v1", "experiment": t.EXPERIMENT,
        **{k: identities for k in ("host_code", "setup_code", "inference_code", "worker_code")}}
    freeze_raw = t.canonical(freeze)
    code["research/"+t.EXPERIMENT+"/prepare-freeze.json"] = freeze_raw
    source = [{"path": n, "mode": "100644", "type": "blob", "sha": t.git_sha(raw)} for n,raw in sorted(code.items())]
    source_tree = t.tree_identity(source)[0]
    inp = (t.OLD/"input-freeze.json").read_bytes()
    screening = t.canonical(t.decode(inp)["actual_screening_receipt"])
    host = {"schema": "k1-prepare-diagnostic-host-receipt-v1", "identity": run, "source_freeze_sha256": t.sha(freeze_raw),
        "executed_tree_sha": source_tree, "source_claim_sha256": None, "screening_receipt_sha256": t.sha(screening),
        "input_verified": False, "status": "failed_no_retry", "stage": "input",
        "stages": {"setup": "not_run", **{f+"_prepare": "not_run" for f in t.FAMILIES}},
        "max_asr_calls": 0, "asr_forward_calls": None, "tts_calls": 0, "prepared_families": [],
        "primary_error": {"error_type": "ValueError", "error_text": "omitted_by_fixed_privacy_policy"}, "cleanup_error": None, "cleanup_verified": True}
    preflight = {"schema": "k1-prepare-run-preflight-v1", "identity": run, "numeric_job_id": 456,
        "workflow": t.POLICY["workflow"], "source_branch": t.POLICY["source_branch"], "evidence_branch": t.POLICY["result_branch"],
        "source_branch_workflow_runs": 1, "evidence_ref_absent": True, "prior_prepare_history_sha256": "c"*64,
        "public_standard_hosted_ubuntu24": True, "api_metadata_accounting": {"gets": 6, "response_body_bytes": 512,
            "response_body_cap_bytes": t.MIB, "scope": "public_api_metadata_separate_from_model_audio_bodies", "hard_wire_byte_cap": False}}
    files = {**{"source/"+n: raw for n,raw in code.items()}, t.FREEZE: freeze_raw,
        "input-freeze.json": inp, "public-screening-receipt.json": screening,
        "prepare-host-receipt.json": t.canonical(host), "prepare-run-preflight.json": t.canonical(preflight)}
    pub = {"schema": "k1-prepare-diagnostic-publisher-bound-release-v1", "purpose": t.PURPOSE,
        "source_sha": run["executed_head_sha"], "source_tree": source_tree, "source_closure_sha256": t.sha(freeze_raw),
        "prior_prepare_history_sha256": "c"*64, "commit_timestamp": "2026-10-12T00:00:00Z"}
    env = {"GITHUB_REPOSITORY": t.REPOSITORY, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123", "GITHUB_SHA": pub["source_sha"], "GITHUB_JOB": t.PURPOSE,
        "GITHUB_REF": "refs/heads/"+t.POLICY["source_branch"], "GITHUB_WORKFLOW_REF": t.REPOSITORY+"/"+t.POLICY["workflow"]+"@refs/heads/"+t.POLICY["source_branch"], "GITHUB_WORKFLOW_SHA": pub["source_sha"]}
    event = {"repository": {"full_name": t.REPOSITORY, "private": False}, "ref": env["GITHUB_REF"], "created": True,
        "deleted": False, "forced": False, "before": "0"*40, "after": pub["source_sha"]}
    return files, run, pub, env, event, source

class TransportTests(unittest.TestCase):
    def test_full_failed_prepare_package_roundtrip(self):
        files, run, release, env, event, source = fixture()
        t.validate_members(files, run)
        payload, _, _ = t._zip_files(files, run)
        manifest, measured = t.validate_archive(payload, run)
        parts = t.build_file_set(payload, run, release["source_closure_sha256"], 456)
        m = t.parse_manifest(parts[t.manifest_path()], run, release["source_closure_sha256"], 456)
        self.assertEqual(payload, t.reassemble(parts[t.manifest_path()], [(r["path"], parts[r["path"]]) for r in m["chunks"]], run, release["source_closure_sha256"], 456))
        t.validate_source_binding(payload, run, release["source_closure_sha256"], source, 456, release["source_tree"])
        self.assertEqual(manifest["schema"], "k1-prepare-diagnostic-member-manifest-v1")
    def test_fake_create_only_publication_roundtrip(self):
        files, run, release, env, event, source = fixture()
        payload = t._zip_files(files, run)[0]
        api = MockAPI(env,event,release,source)
        got = p.publish(env,event,release,payload,api,api.raw_factory)
        self.assertEqual(got["status"], "same_identity_readback_verified")
        self.assertFalse(got["recognition_result"])
        self.assertLessEqual(len(api.raw_calls),49)
        self.assertEqual([x[0] for x in api.calls if x[0] not in ("GET","POST")],[])
    def test_all_asr_or_unknown_paths_rejected(self):
        files, run, *_ = fixture()
        for name in ("terminal-outcomes.json", "host-receipt.json", "output/qwen06/clip-000001/started.json", "frozen/qwen06/native-config.json", "output/qwen06/unlisted.json", "notes.txt", "output/qwen06/prepared.json"):
            with self.subTest(name=name), self.assertRaises(ValueError): t.validate_members({**files,name:b'{}'},run)
    def test_source_freeze_cannot_rebind_or_omit(self):
        files, run, *_ = fixture()
        for operation in ("body", "missing", "extra", "old_schema", "worker_map"):
            bad = dict(files)
            if operation == "body": bad["source/README.md"] += b"!"
            if operation == "missing": bad.pop("source/README.md")
            if operation == "extra": bad["source/extra.py"] = b"x=1"
            if operation in ("old_schema", "worker_map"):
                f=t.decode(bad[t.FREEZE]); f["schema"]="k1-asr18-source-freeze-v1" if operation=="old_schema" else f["schema"]
                if operation=="worker_map":f["worker_code"]={"README.md":f["host_code"]["README.md"]}
                bad[t.FREEZE]=t.canonical(f);bad[t.SOURCE_FREEZE]=bad[t.FREEZE]
            with self.subTest(operation=operation), self.assertRaises(ValueError):t.validate_members(bad,run)
    def test_incomplete_failure_never_asserts_zero(self):
        files,run,*_=fixture();h=t.decode(files["prepare-host-receipt.json"]);h["asr_forward_calls"]=0;files["prepare-host-receipt.json"]=t.canonical(h)
        with self.assertRaises(ValueError):t.validate_members(files,run)
    def test_audio_secret_private_paths_and_types_rejected(self):
        files,run,*_=fixture()
        for value in ({"secret":"x"},{"x":"/workspace/private.wav"},{"x":"ghp_fake"}):
            with self.assertRaises(ValueError):t.validate_members({**files,"image-transfer.json":t.canonical(value)},run)
        with self.assertRaises(ValueError):t.validate_members({**files,"output/qwen06/model-load.json":b'RIFF0000WAVE00'},run)
    def test_fixed_archive_caps_and_unsafe_zip(self):
        files,run,*_=fixture();payload=t._zip_files(files,run)[0]
        for bad in (payload+b'junk',payload+b'\0',b'x'*(t.MAX_COMPRESSED+1)):
            with self.assertRaises(ValueError):t.validate_archive(bad,run)
        with self.assertRaises(ValueError):t._zip_files({"huge.txt":b'x'*(t.MAX_MEMBER+1)},run)
        self.assertEqual((t.MAX_COMPRESSED,t.MAX_UNCOMPRESSED,t.MAX_MEMBER,t.MAX_MEMBERS,t.MAX_BODY),(12*t.MIB,32*t.MIB,8*t.MIB,4096,16*t.MIB))
    def test_collector_fixed_private_dirs_only_and_failure_stdout(self):
        files,run,*_=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n,b in files.items():(root/n).parent.mkdir(parents=True,exist_ok=True);(root/n).write_bytes(b)
            (root/'input').mkdir();(root/'input'/'private.wav').write_bytes(b'RIFF0000WAVE')
            self.assertEqual(t._collect(root,run),files)
            (root/'unknown.txt').write_text('private canary')
            out=io.StringIO()
            with redirect_stdout(out),self.assertRaises(ValueError):t.build_package(root,run)
            record=json.loads(out.getvalue());self.assertEqual(record['status'],'failed_closed_no_pruning')
            self.assertIsNone(record['observed_asr_calls']);self.assertNotIn('canary',out.getvalue())
    def test_publisher_disabled_before_environment_or_token(self):
        class Poison(dict):
            def __getitem__(self,key):raise AssertionError('environment touched')
            def get(self,*args):raise AssertionError('environment touched')
            def __contains__(self,key):raise AssertionError('environment touched')
        with tempfile.TemporaryDirectory() as directory:
            local=Path(directory);(local/'publication-release.json').write_bytes(t.canonical(p.disabled_publication_release()))
            with mock.patch.object(p,'HERE',local),mock.patch.object(p.os,'environ',Poison()),mock.patch.object(p,'GitHubAPI',side_effect=AssertionError('network attempted')):
                with self.assertRaises(p.PublicationDisabled):p.main([])
    def test_active_gate_rejects_invalid_context_before_any_api(self):
        control=t.decode((HERE/'publication-release.json').read_bytes())
        self.assertTrue(all(control[k] is True for k in ('enabled','root_publication_release','independent_source_review')))
        p.publication_gate(control)
        files,run,release,env,event,source=fixture();payload=t._zip_files(files,run)[0]
        for key,value in (('GITHUB_REPOSITORY','other/repo'),('GITHUB_JOB','asr18'),('GITHUB_EVENT_NAME','workflow_dispatch'),('GITHUB_RUN_ATTEMPT','2')):
            bad=dict(env);bad[key]=value;api=mock.Mock(side_effect=AssertionError('API touched before context validation'))
            with self.subTest(key=key),self.assertRaises(ValueError):p.publish(bad,event,release,payload,api,mock.Mock())
            api.assert_not_called()
    def test_active_cli_rejects_extra_arguments_before_environment(self):
        p.publication_gate(t.decode((HERE/'publication-release.json').read_bytes()))
        class Poison(dict):
            def __getitem__(self,key):raise AssertionError('environment touched')
            def get(self,*args):raise AssertionError('token touched')
            def __contains__(self,key):raise AssertionError('environment touched')
        with mock.patch.object(p.os,'environ',Poison()),mock.patch.object(p,'GitHubAPI',side_effect=AssertionError('network attempted')):
            with self.assertRaises(ValueError):p.main(['--unapproved'])
    def test_old_identity_and_ref_cannot_enter(self):
        files,run,release,env,event,_=fixture()
        for key,value in (('job_key','asr18'),('run_attempt',2),('repository','other/repo')):
            bad=dict(run);bad[key]=value
            with self.assertRaises(ValueError):t.identity(bad)
        env['GITHUB_REF']='refs/heads/research/k1-punctuation6-asr18-r1'
        with self.assertRaises(ValueError):p.context(env,event,release)
    def test_publisher_exact_timestamp_equivalence(self):
        self.assertEqual(p._utc_timestamp('2026-10-12T08:00:00+08:00'),'2026-10-12T00:00:00Z')
        for bad in ('2026-10-12T00:00:00','2026-10-12T00:00:00.1Z','2026-10-12T00:00:00+24:00'):
            with self.assertRaises((ValueError,TypeError)):p._utc_timestamp(bad)



class CompletionRosterTests(unittest.TestCase):
    """Presence-only unit fixtures; full actual bodies are independently replayed.

    These placeholders never pass validate_members or claim native completion.
    Every present body's original validator remains required by that entrypoint.
    """
    def layout(self, *, final=True):
        host={"status":"all3_prepared_no_recognition" if final else "started", "identity":{"run_id":123},
            "stages":{"setup":"complete", **{f+"_prepare":"complete" for f in t.FAMILIES}}}
        names=set(t.SETUP_COMPLETE_MEMBERS)|set(t.PREPARED_HOST_MEMBERS)|set(t.legacy.official_whisper_source_inventory())
        for family in t.FAMILIES:
            names|={"output/"+family+"/"+leaf for leaf in t.PREPARED_COMMON|t.PREPARED_EXTRA[family]}
        if final:names.add("policy/capacity-proof.json")
        files={name:b"{}" for name in names}
        files["cpu-receipt.json"]=t.canonical({"all_final_usage_verified":True,"threshold_exceeded":False})
        files["cleanup-receipt.json"]=t.canonical({"all_removed":True,"all_cleanup_verified":True})
        ordered=[(t.c.EXPERIMENT+"-123-setup","setup",None)]+[(t.EXPERIMENT+"-123-"+f,f,"prepare") for f in t.FAMILIES]
        files["supervision-receipt.json"]=t.canonical({"stages":[{"name":name,"family":family,"mode":mode,
            "status":"complete","worker_exit_success":True,"primary_error":None,"cleanup_error":None} for name,family,mode in ordered]})
        return files,host

    def test_every_final_required_member_is_independently_required(self):
        files,host=self.layout();t._require_completed_layout(files,host,list(t.FAMILIES),"final")
        for name in files:
            bad=dict(files);bad.pop(name)
            with self.subTest(name=name),self.assertRaises(ValueError):
                t._require_completed_layout(bad,host,list(t.FAMILIES),"final")

    def test_prepared_phase_requires_real_roster_before_capacity_proof_exists(self):
        files,host=self.layout(final=False)
        self.assertNotIn("policy/capacity-proof.json",files)
        t._require_completed_layout(files,host,list(t.FAMILIES),"prepared")
        for name in files:
            bad=dict(files);bad.pop(name)
            with self.subTest(name=name),self.assertRaises(ValueError):
                t._require_completed_layout(bad,host,list(t.FAMILIES),"prepared")

    def test_failure_before_setup_requires_no_fabricated_future_receipts(self):
        host={"status":"failed_no_retry","stages":{"setup":"failed_no_retry", **{f+"_prepare":"not_run" for f in t.FAMILIES}}}
        t._require_completed_layout({},host,[],"final")
        files,run,*_=fixture()
        with mock.patch.object(t,"_require_completed_layout",wraps=t._require_completed_layout) as guard:
            t.validate_members(files,run)
        guard.assert_called_once()

    def test_failed_later_stage_still_requires_each_completed_family_originals(self):
        host={"status":"failed_no_retry","stages":{"setup":"complete", **{f+"_prepare":"not_run" for f in t.FAMILIES}}}
        for family in t.FAMILIES:
            host["stages"][family+"_prepare"]="complete"
            files={name:b"{}" for name in set(t.SETUP_COMPLETE_MEMBERS)|set(t.legacy.official_whisper_source_inventory())}
            files.update({"output/"+family+"/"+leaf:b"{}" for leaf in t.PREPARED_COMMON|t.PREPARED_EXTRA[family]})
            t._require_completed_layout(files,host,[family],"final")
            for downgrade in ("not_run","started","failed_no_retry"):
                bad_host=copy.deepcopy(host);bad_host["stages"]["setup"]=downgrade
                with self.subTest(family=family,setup=downgrade),self.assertRaises(ValueError):
                    t._require_completed_layout(files,bad_host,[family],"final")
            for name in files:
                bad=dict(files);bad.pop(name)
                with self.subTest(family=family,name=name),self.assertRaises(ValueError):
                    t._require_completed_layout(bad,host,[family],"final")

    def test_success_supervision_exact_order_identity_and_terminal_values(self):
        for phase in ("prepared","final"):
            files,host=self.layout(final=phase=="final")
            original=t.decode(files["supervision-receipt.json"])
            variants=[]
            for index in range(4):
                for key,value in (("status","failed"),("worker_exit_success",False),
                        ("primary_error",{"error_type":"ValueError"}),("cleanup_error",{"error_type":"ValueError"}),
                        ("name","other-run"),("mode","decode")):
                    bad=copy.deepcopy(original);bad["stages"][index][key]=value;variants.append(bad)
            for change in ("missing","duplicate","reordered"):
                bad=copy.deepcopy(original)
                if change=="missing":bad["stages"].pop()
                elif change=="duplicate":bad["stages"].append(bad["stages"][0])
                else:bad["stages"].reverse()
                variants.append(bad)
            for index,value in enumerate(variants):
                bad=dict(files);bad["supervision-receipt.json"]=t.canonical(value)
                with self.subTest(phase=phase,index=index),self.assertRaises(ValueError):
                    t._require_completed_layout(bad,host,list(t.FAMILIES),phase)

    def test_final_and_prepared_resource_success_flags_must_agree(self):
        for phase in ("prepared","final"):
            files,host=self.layout(final=phase=="final")
            for name,key,value in (("cpu-receipt.json","all_final_usage_verified",False),
                    ("cpu-receipt.json","threshold_exceeded",True),("cleanup-receipt.json","all_removed",False),
                    ("cleanup-receipt.json","all_cleanup_verified",False)):
                bad=dict(files);row=t.decode(bad[name]);row[key]=value;bad[name]=t.canonical(row)
                with self.subTest(phase=phase,key=key),self.assertRaises(ValueError):
                    t._require_completed_layout(bad,host,list(t.FAMILIES),phase)


class SizingUnitTests(unittest.TestCase):
    def synthetic_prepared(self):
        # Explicit synthetic unit fixture. Collection admission is replaced only
        # in the sizing test; these bytes are never an admissible real run packet.
        sys.path.insert(0,str(t.ROOT/'tests'))
        from asr_native_fixtures import native_fixture
        import worker_receipts as wr
        files,run,*_=fixture()
        frozen=t.decode(files['input-freeze.json'])
        job,plan,binding=(frozen[n] for n in ('actual_blind_job','actual_opaque_order','actual_worker_binding'))
        policy={'schema':'recognizer-native-policy-v1','recognizers':[]}
        # build_policy consumes the same exact native source/model/config receipts.
        with tempfile.TemporaryDirectory() as directory:
            prepared={}
            for family in t.FAMILIES:
                folder=Path(directory)/family;folder.mkdir();prepared[family]=folder
                config,_=native_fixture(wr.ADAPTERS[family])
                source={'schema':'retained-asr-source-lock-v1','executed_head_sha':run['executed_head_sha'],'workflow_run_id':str(run['run_id']),
                    'files':[{'path':'invented-'+family+'.py','bytes':1,'sha256':'1'*64}]}
                assets=[{'path':'invented-'+family+'-asset','bytes':1,'sha256':'2'*64}]
                model={'schema':'retained-asr-model-lock-v1','assets':assets,'model_sha256':t.sha(t.canonical(assets)),
                    'provider_id':'invented-provider','model_id':'invented-'+family,'model_revision':'invented-v1','model_family':config['model_family']}
                config['source_lock_sha256']=t.sha(t.canonical(source));config['asset_lock_sha256']=t.sha(t.canonical(model))
                for name,value in zip(wr.FILES,(source,model,config)):
                    raw=t.canonical(value);(folder/name).write_bytes(raw);files['output/'+family+'/'+name]=raw
                files['output/'+family+'/prepare-receipt.json']=t.canonical({'status':'prepared_configuration_only',
                    'audio_forward_calls':0,'native_decode_attempts':0,'recognition_result':False})
            policy=wr.build_policy(prepared)
        files['policy/policy.json']=t.canonical(policy)
        return files,run,job,plan,binding

    def test_full_private_pressure_fixture_and_actual_prepared_measurements(self):
        files,run,job,plan,binding=self.synthetic_prepared()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'policy').mkdir();(root/'policy/policy.json').write_bytes(files['policy/policy.json'])
            snapshots=[];original=t._zip_files
            def pack(values,identity):snapshots.append(dict(values));return original(values,identity)
            with mock.patch.object(t,'_collect',return_value=files),mock.patch.object(t,'_zip_files',side_effect=pack):
                proof=t.package_budget_preflight(root,run,job=job,execution_plan=plan,input_binding=binding,policy_path=root/'policy/policy.json')
            self.assertEqual(len(snapshots),2);self.assertEqual(snapshots[0],files)
            pressure=snapshots[1]
            self.assertEqual(proof['status'],'actual_prepared_and_private_pressure_fit')
            self.assertTrue(proof['synthetic_pressure_is_not_asr_evidence'])
            self.assertLessEqual(proof['compressed_bytes'],12*t.MIB);self.assertLessEqual(proof['expanded_member_bytes'],32*t.MIB)
            for family in t.FAMILIES:
                for name,raw in files.items():
                    if name.startswith('output/'+family+'/'):self.assertEqual(pressure['frozen/'+name[7:]],raw)
            for clip in job['clips']:
                key='output/qwen06/'+clip['audio_id']+'.native-generation-return.json'
                row=t.decode(pressure[key]);self.assertEqual(len(row['returned_sequence_ids']),65536);self.assertEqual(row['prompt_token_count'],65536-256)
                prefix='output/whisper/'+clip['audio_id']+'/'
                self.assertEqual(sum(n.startswith(prefix+'update-') for n in pressure),224)
                for family in t.FAMILIES:
                    for leaf in ('raw-native-output.json','raw-transcript.json'):
                        self.assertEqual(len(t.decode(pressure['output/'+family+'/'+clip['audio_id']+'/'+leaf])['raw_text']),32768)
            with self.assertRaises(ValueError):t.validate_members(pressure,run)
            self.assertFalse(any('clip-' in n for n in snapshots[0]))
    def test_oversized_pressure_reports_without_pruning(self):
        files,run,job,plan,binding=self.synthetic_prepared();out=io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'policy').mkdir();(root/'policy/policy.json').write_bytes(files['policy/policy.json'])
            with mock.patch.object(t,'_collect',return_value=files),mock.patch.object(t,'MAX_UNCOMPRESSED',128),redirect_stdout(out),self.assertRaises(ValueError):
                t.package_budget_preflight(root,run,job=job,execution_plan=plan,input_binding=binding,policy_path=root/'policy/policy.json')
        row=json.loads(out.getvalue());self.assertEqual(row['stage'],'capacity_proof');self.assertEqual(row['status'],'failed_closed_no_pruning')
    def test_independent_prepare_receipt_strict_zero_guard(self):
        _,run,*_=fixture();claim={'metadata_packet_sha256':'1'*64,'source_freeze_sha256':'2'*64}
        row={'schema':'k1-prepare-worker-result-v1','identity':run,'family':'qwen06','status':'prepared_configuration_only',
            'model_loads_started':1,'model_loads_completed':1,'maximum_new_asr_calls':0,'native_decode_attempts':0,
            'audio_forward_calls':0,'audio_mounted':False,'recognition_result':False,'legacy_execution_contract_grants_asr':False,
            **claim,'guard':{'schema':'k1-prepare-forward-guard-v1','forbidden_attempts':0,
                'python_entry_guard':'active_through_constructor_and_config','coverage':'reviewed_fixed_python_native_and_module_forward_entrypoints',
                'universal_native_extension_proof':False},'wall_seconds':1.0}
        t.validate_prepare_receipt(row,'qwen06',run,claim)
        for k,v in (('audio_forward_calls',1),('audio_mounted',True),('recognition_result',True),('model_loads_started',2),('legacy_execution_contract_grants_asr',True)):
            bad=copy.deepcopy(row);bad[k]=v
            with self.subTest(key=k),self.assertRaises(ValueError):t.validate_prepare_receipt(bad,'qwen06',run,claim)

if __name__=='__main__':unittest.main()
