"""Invented byte fixtures and fake APIs only; no live runtime/model/network calls."""
import base64
import ast
import copy
import io
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in (ROOT / "tests", ROOT / "tools", HERE): sys.path.insert(0, str(folder))
import campaign_contract as c
import evidence_transport as t
import evidence_publisher as p
import worker_receipts as wr
from asr_native_fixtures import native_fixture

def tree_response(rows):
    digest, directories = t.tree_identity(rows)
    return {"sha": digest, "truncated": False, "tree": [*copy.deepcopy(rows),
        *[{"path": name, "mode": "040000", "type": "tree", "sha": value} for name, value in sorted(directories.items())]]}


def setup(purpose="asr18"):
    source = [{"path": ".github/workflows/k1-punctuation6-asr18-v1.yml", "type": "blob", "mode": "100644", "sha": t.git_sha(b"disabled")},
              {"path": "README.md", "type": "blob", "mode": "100644", "sha": t.git_sha(b"public fixture repository") }]
    release = {"schema": "k1-asr18-publisher-bound-release-v1", "purpose": purpose,
        "source_sha": "a" * 40, "source_tree": tree_response(source)["sha"],
        "source_closure_sha256": "b" * 64, "prior_claim_history_sha256": "c" * 64,
        "commit_timestamp": "2026-10-12T00:00:00Z"}
    policy = t.policy(purpose)
    env = {"GITHUB_REPOSITORY": t.REPOSITORY, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123", "GITHUB_SHA": release["source_sha"], "GITHUB_JOB": policy["job_key"],
        "GITHUB_REF": "refs/heads/" + policy["source_branch"],
        "GITHUB_WORKFLOW_REF": t.REPOSITORY+"/"+policy["workflow"]+"@refs/heads/"+policy["source_branch"],
        "GITHUB_WORKFLOW_SHA": release["source_sha"]}
    event = {"repository": {"full_name": t.REPOSITORY, "private": False}, "ref": env["GITHUB_REF"],
             "created": True, "deleted": False, "forced": False, "before": "0"*40, "after": release["source_sha"]}
    return env, event, release, source


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



def invented_files(identity):
    source = b"# Invented public source for an offline transport fixture.\n"
    row = {"bytes": len(source), "sha256": t.sha(source)}
    freeze = {"schema": "k1-asr18-source-freeze-v1", **{key: {"research/invented-fixture.py": row} for key in
        ("inference_code", "setup_code", "host_code")}}
    raw = t.canonical(freeze)
    return {"source/research/invented-fixture.py": source,
        "source/research/k1-punctuation6-asr18-v1/execution-freeze.json": raw,
        "authority/execution-freeze.json": raw,
        "host-receipt.json": t.canonical({"schema": "invented-test-host-only", "identity": identity})}


def fake_host(files):
    def validate(actual, identity, *, phase):
        if actual.get("host-receipt.json") != files["host-receipt.json"]:
            raise ValueError("exact test-only host schema")
        if phase not in ("prepared", "final"): raise ValueError("test phase")
    return types.SimpleNamespace(collect_public=lambda root: sorted(files), public_setup_members=lambda: set(),
        validate_public_evidence=validate)


def blind(count=6):
    job = {"schema": "blind-source-screen32-job-v1", "clips": [{"audio_id": f"clip-{i:06d}",
        "audio_path": "audio/" + str(i) * 64 + ".wav", "wav_sha256": str(i) * 64} for i in range(1, count+1)]}
    order = list(reversed([row["audio_id"] for row in job["clips"]]))
    plan = {"schema": "k1-asr18-anonymous-order-metadata-proposal-v1", "execution_enabled": False,
        "job_sha256": t.sha(t.canonical(job)), "family_order": list(c.FAMILIES), "audio_order": order}
    binding = {"schema": "k1-asr18-worker-input-binding-v1", "job_sha256": plan["job_sha256"],
        "execution_plan_sha256": t.sha(t.canonical(plan)), "screening_receipt_sha256": "a"*64,
        "tts_authority_sha256": "b"*64, "audio": [{"audio_id": row["audio_id"], "wav_sha256": row["wav_sha256"],
            "pcm_sha256": "c"*64, "audio_frames": 1280} for row in reversed(job["clips"])]}
    return job, plan, binding


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="asr18-invented-transport-")
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.env, self.event, self.release, self.source = setup()
        self.identity = p.context(self.env, self.event, self.release)
        self.files = invented_files(self.identity)
        self.source = [{"path": name[len("source/"):], "type": "blob", "mode": "100644", "sha": t.git_sha(raw)}
            for name, raw in self.files.items() if name.startswith("source/")]
        self.release["source_tree"] = tree_response(self.source)["sha"]
        self.release["source_closure_sha256"] = t.sha(self.files["authority/execution-freeze.json"])
        self.files["host-receipt.json"] = t.canonical({"schema": "invented-test-host-only", "identity": self.identity,
            "executed_tree_sha": self.release["source_tree"]})
        self.files["run-preflight.json"] = t.canonical({"schema": "invented-test-preflight-only", "numeric_job_id": 456})
        self.host = fake_host(self.files)
        self.guard = mock.patch("http.client.HTTPSConnection", side_effect=AssertionError("no live network"))
        self.guard.start(); self.addCleanup(self.guard.stop)
        self.hpatch = mock.patch.object(t, "_host", return_value=self.host)
        self.hpatch.start(); self.addCleanup(self.hpatch.stop)
        self.materialize()

    def materialize(self):
        for name, raw in self.files.items():
            dest = self.root / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)

    def package(self):
        return t.build_package(self.root, self.identity)[0]

    def test_exact_complete_source_package_roundtrip(self):
        payload = self.package()
        manifest, header = t.validate_archive(payload, self.identity)
        self.assertEqual(set(manifest["members"]), set(self.files))
        self.assertEqual(header["member_count"], len(self.files))
        self.assertEqual(header["archive_bytes"], len(payload))
        files = t.build_file_set(payload, self.identity, self.release["source_closure_sha256"], 456)
        outer = files[t.manifest_path()]
        parsed = t.parse_manifest(outer, self.identity, self.release["source_closure_sha256"], 456)
        chunks = [(r["path"], files[r["path"]]) for r in parsed["chunks"]]
        self.assertEqual(t.reassemble(outer, chunks, self.identity, self.release["source_closure_sha256"], 456), payload)
        self.assertNotIn(b"KWS_NATIVE_EVIDENCE_V1", payload)

    def test_complete_source_missing_or_mutated_rejected(self):
        for mode in ("missing", "changed"):
            files = dict(self.files)
            if mode == "missing": del files["source/research/invented-fixture.py"]
            else: files["source/research/invented-fixture.py"] += b"extra"
            with self.subTest(mode=mode), self.assertRaises(ValueError): t.validate_members(files, self.identity)

    def test_private_audio_binaries_unknown_paths_never_admitted(self):
        values = [("private-history.json", b"{}"), ("output/qwen06/arbitrary.json", b"{}"),
            ("output/qwen06/clip-000007/started.json", b"{}"), ("output/whisper/clip-000001/update-225.json", b"{}"),
            ("output/qwen06/clip-000001/update-001.json", b"{}"), ("output/qwen06/clip-000001/raw-native-output.json", b"RIFFxxxxWAVE"),
            ("hidden.wav", b"not audio"), ("weights.safetensors", b"invented"), ("public.bin", b"invented"),
            ("retained-assets/new.npz", b"PK"), ("../outside.json", b"{}")]
        for name, raw in values:
            with self.subTest(name=name), self.assertRaises(ValueError): t.validate_members({**self.files, name: raw}, self.identity)

    def test_nested_private_keys_and_paths_rejected(self):
        for value in ({"credentials":"invented"}, {"nested":{"history_count":1}}, {"nested":{"inventory_sha256":"a"*64}},
            {"nested":{"env":{"X":"invented"}}}, {"path":"/workspace/private"}, {"value":"github_pat_invented"}):
            with self.subTest(value=value), self.assertRaises(ValueError): t._public_metadata(value)

    def test_source_manifest_cannot_smuggle_audio_or_binary(self):
        for name in ("research/fixture.wav", "research/fixture.safetensors", "research/fixture.zip", "research/fixture.bin"):
            raw = b"invented"
            freeze = {"schema":"k1-asr18-source-freeze-v1", **{k:{name:{"bytes":len(raw),"sha256":t.sha(raw)}} for k in
                ("inference_code","setup_code","host_code")}}
            fraw=t.canonical(freeze)
            files={"source/"+name:raw,"authority/execution-freeze.json":fraw,
                "source/research/k1-punctuation6-asr18-v1/execution-freeze.json":fraw}
            with self.subTest(name=name), self.assertRaises(ValueError):t.validate_members(files,self.identity)

    def test_only_three_exact_lock_bound_binary_assets(self):
        for name in t.BINARY_ASSETS:
            family, suffix = name.split("/", 1)
            member = "output/"+name; raw=b"invented-public-asset\0"
            lock={"schema":"retained-asr-model-lock-v1","assets":[{"path":member,"bytes":len(raw),"sha256":t.sha(raw),"body_retained":True}]}
            files={member:raw,"output/"+family+"/native-asset-lock.json":t.canonical(lock)}
            self.assertIn(member,t._native_body_members(files))
            files[member]+=b"changed"
            with self.assertRaises(ValueError):t._native_body_members(files)
        member="output/whisper/retained-assets/arbitrary.npz"
        files={member:b"invented", "output/whisper/native-asset-lock.json":t.canonical({"schema":"retained-asr-model-lock-v1",
            "assets":[{"path":member,"bytes":8,"sha256":t.sha(b"invented"),"body_retained":True}]})}
        with self.assertRaises(ValueError):t._native_body_members(files)

    def test_zip_trailer_comment_duplicate_and_foreign_identity_fail(self):
        payload=self.package()
        for raw in (payload+b"trailer",payload[:-1],payload[:-2]+b"\x01\0x"):
            with self.assertRaises((ValueError,zipfile.BadZipFile)):t.validate_archive(raw,self.identity)
        altered=dict(self.identity,run_id=124)
        with self.assertRaises(ValueError):t.validate_archive(payload,altered)
        buffer=io.BytesIO(payload)
        with zipfile.ZipFile(buffer,"a") as z:
            with self.assertWarns(UserWarning):z.writestr("host-receipt.json",self.files["host-receipt.json"])
        with self.assertRaises(ValueError):t.validate_archive(buffer.getvalue(),self.identity)

    def test_caps_not_raised_or_pruned(self):
        self.assertEqual((t.MAX_COMPRESSED,t.MAX_UNCOMPRESSED,t.MAX_MEMBER,t.MAX_MEMBERS),
            (12*1024**2,32*1024**2,8*1024**2,4096))
        for files in ({"large.json":b"x"*(t.MAX_MEMBER+1)}, {str(i)+".json":b"x" for i in range(4097)},
            {str(i)+".json":b"x"*(8*1024**2) for i in range(5)}):
            with self.assertRaises(ValueError):t._zip_files(files,self.identity)

    def test_chunk_identity_bounds_order_and_no_missing(self):
        files=t.build_file_set(self.package(),self.identity,self.release["source_closure_sha256"],456)
        raw=files[t.manifest_path()]
        for key,value in (("chunk_count",True),("numeric_job_id",True),("member_count",4097),("archive_bytes",t.MAX_ARCHIVE+1)):
            m=t.decode(raw);m[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):t.parse_manifest(t.canonical(m),self.identity,self.release["source_closure_sha256"],456)
        with self.assertRaises(ValueError):t.reassemble(raw,[],self.identity,self.release["source_closure_sha256"],456)
        m=t.decode(raw);chunks=[(m['chunks'][0]['path'],files[m['chunks'][0]['path']]+b"x")]
        with self.assertRaises(ValueError):t.reassemble(raw,chunks,self.identity,self.release["source_closure_sha256"],456)

    def test_package_stash_is_exclusive_and_no_symlinks(self):
        payload=self.package();dest=t.write_package(self.root,payload,self.identity)
        self.assertEqual(t.read_package(self.root,self.identity),payload)
        with self.assertRaises(FileExistsError):t.write_package(self.root,payload,self.identity)
        dest.unlink();dest.symlink_to(self.root/'host-receipt.json')
        with self.assertRaises((ValueError,OSError)):t.read_package(self.root,self.identity)

    def prepared(self, count=6):
        job,plan,binding=blind(count)
        policy={"schema_version":2,"evidence_class":wr.POLICY_CLASS_V2,"normalization":wr.NORMALIZATION,
            "frozen_before_recognition":True,"target_conditioning_allowed":False,"recognizers":[]}
        for family in c.FAMILIES:
            config,_=native_fixture(wr.ADAPTERS[family],audio_frames=1280)
            prefix="output/"+family+"/"
            self.files[prefix+"native-config.json"]=t.canonical(config)
            self.files[prefix+"prepared.json"]=t.canonical({"schema":"k1-asr18-worker-result-v1","family":family,"mode":"prepare",
                "status":"prepared_no_recognition","audio_forward_calls":0,"completed":0,"audio_ids":[],
                "job_sha256":plan["job_sha256"],"execution_plan_sha256":t.sha(t.canonical(plan))})
            policy["recognizers"].append({"recognizer_id":family,"provider_id":"invented","model_id":"invented-"+family,
                "model_revision":"invented-v1","model_family":config["model_family"],"model_sha256":t.sha(family.encode()),
                "native_decode_config":config,"native_decode_config_sha256":t.sha(t.canonical(config))})
        self.files['policy/policy.json']=t.canonical(policy)
        self.materialize();(self.root/'policy/policy.json').chmod(0o444)
        return job,plan,binding

    def test_actual_six_config_fixture_budget_bound_before_forward(self):
        job,plan,binding=self.prepared(6)
        with mock.patch.object(t,'_zip_files',wraps=t._zip_files) as pack:
            result=t.package_budget_preflight(self.root,self.identity,job=job,execution_plan=plan,input_binding=binding,
                policy_path=self.root/'policy/policy.json')
        pack.assert_called_once()
        trial=pack.call_args.args[0]
        self.assertEqual(result['eligible_clip_count'],6)
        self.assertEqual(result['actual_prepared_members'],len(self.files))
        self.assertEqual(result['worst_fixture_members'],len(self.files)+6+6*3*4+6+6*224+1)
        for clip in job['clips']:
            opaque=clip['audio_id']
            name='output/qwen06/'+opaque+'.native-generation-return.json'
            captured=t.decode(trial[name]);ids=captured['returned_sequence_ids']
            self.assertEqual(captured['schema'],'synthetic-size-only-not-ASR-evidence')
            self.assertEqual(captured['sequence_shape'],[1,65536])
            self.assertEqual(len(ids),65536)
            self.assertEqual(captured['prompt_token_count'],65536-256)
            self.assertFalse(captured['capture_truncated'])
            self.assertTrue(all(type(value) is int and 1<=value<151000 for value in ids))
            self.assertGreater(len(set(ids)),50000)
            with self.assertRaises(ValueError):t.validate_member_json(name,captured)
            qwen=t.decode(trial['output/qwen06/'+opaque+'/raw-native-output.json'])
            self.assertEqual(len(qwen['token_ids']),256)
            self.assertEqual(qwen['config']['native_request']['max_new_tokens'],256)
            prefix='output/whisper/'+opaque+'/update-'
            self.assertEqual(sum(name.startswith(prefix) for name in trial),224)
            for step in range(1,225):
                row=t.decode(trial[prefix+f'{step:03d}.json'])
                self.assertEqual(len(row['token_ids']),4+step)
                self.assertEqual(row['completed'],step==224)
        self.assertGreater(result['expanded_member_bytes'],7*1024**2)
        for family in c.FAMILIES:
            self.assertEqual(c.validate_decode_proof(result,job=job,execution_plan=plan,input_binding=binding,
                policy_path=self.root/'policy/policy.json',family=family),result)
        self.assertFalse(any(self.root.rglob('update-*.json')))
        self.assertFalse(any(self.root.rglob('*.native-generation-return.json')))
        with self.assertRaises(ValueError):t.validate_members({**self.files,'size-reserve.json':b'{}'},self.identity)

    def test_qwen_capture_stress_is_deterministic_and_eligible_only(self):
        job,plan,binding=self.prepared(1)
        with mock.patch.object(t,'_zip_files',wraps=t._zip_files) as pack:
            results=[t.package_budget_preflight(self.root,self.identity,job=job,execution_plan=plan,input_binding=binding,
                policy_path=self.root/'policy/policy.json') for _ in range(2)]
        self.assertEqual(results[0],results[1])
        first,second=[call.args[0] for call in pack.call_args_list]
        self.assertEqual(first,second)
        self.assertEqual({name for name in first if name.endswith('.native-generation-return.json')},
            {'output/qwen06/clip-000001.native-generation-return.json'})
        self.assertEqual(results[0]['worst_fixture_members'],len(self.files)+6+3*4+1+224+1)

    def test_budget_missing_actual_configuration_or_already_called_fails(self):
        job,plan,binding=self.prepared(1)
        del self.files['output/whisper/native-config.json']
        with self.assertRaises(ValueError):t.package_budget_preflight(self.root,self.identity,job=job,execution_plan=plan,input_binding=binding,
            policy_path=self.root/'policy/policy.json')
        for count in (0,7):
            job,plan,binding=blind(count)
            with self.assertRaises(ValueError):t.package_budget_preflight(self.root,self.identity,job=job,execution_plan=plan,input_binding=binding,
                policy_path=self.root/'policy/policy.json')

    def test_native_partial_failed_observation_kept_but_extra_nested_rejected(self):
        value={"schema":"source-screen-native-v2-observation-v1","opaque_id":"clip-000001","input_binding_sha256":"a"*64,
            "inference_returned":False,"acoustic_verbatim_completeness":"unknown","exception_type":"RuntimeError"}
        t.validate_member_json('output/qwen06/clip-000001.native-observation.json',value)
        value['extra']='private'
        with self.assertRaises(ValueError):t.validate_member_json('output/qwen06/clip-000001.native-observation.json',value)
        raw={"schema":"retained-asr-raw-transcript-v1","request_id":"invented","actual_text":"invented",
            "output_scope":"complete-decoder-transcript","truncated":False,"native_output":{"path":"raw-native-output.json","bytes":1,"sha256":"a"*64},
            "native_row":None,"acoustic_verbatim_completeness":"unknown"}
        t.validate_member_json('output/whisper/clip-000001/raw-transcript.json',raw)
        raw['native_output']['extra']='unexpected'
        with self.assertRaises(ValueError):t.validate_member_json('output/whisper/clip-000001/raw-transcript.json',raw)

    def test_full_whisper_prefix_shape_no_truncation(self):
        row={"step":224,"token_ids":[50258,50260,50359,50363]+[1]*224,"completed":True}
        t.validate_member_json('output/whisper/clip-000001/update-224.json',row)
        for mutation in ({**row,'token_ids':row['token_ids'][-1:]},{**row,'step':True},{**row,'extra':'x'}):
            with self.assertRaises(ValueError):t.validate_member_json('output/whisper/clip-000001/update-224.json',mutation)


class PublisherTests(TransportTests):
    def api(self):return MockAPI(self.env,self.event,self.release,self.source)
    def publish(self,api):return p.publish(self.env,self.event,self.release,self.package(),api,api.raw_factory)

    def test_fake_create_only_whole_immutable_readback(self):
        api=self.api();receipt=p.run_once(self.env,self.event,self.release,self.package(),api,api.raw_factory,self.root)
        self.assertEqual(receipt['status'],'same_identity_readback_verified')
        self.assertEqual(receipt['raw_gets'],2)
        self.assertFalse(receipt['retry_authorized'])
        self.assertFalse(any(call[0] in ('PUT','PATCH','DELETE') for call in api.calls))
        self.assertEqual([call[0] for call in api.calls[:7]],['GET']*7)
        with self.assertRaises(FileExistsError):p.run_once(self.env,self.event,self.release,self.package(),api,api.raw_factory,self.root)

    def test_only_exact_commit_message_or_one_removed_terminal_lf(self):
        for methods in ({'POST'},{'GET'},{'POST','GET'}):
            class LFAPI(MockAPI):
                def __call__(self,method,route,body=None,expected=200):
                    value=super().__call__(method,route,body,expected)
                    if method in methods and (route==p.BASE+'/git/commits' or route.startswith(p.BASE+'/git/commits/') and route!=self.paths['source']):
                        value=copy.deepcopy(value);value['message']=value['message'][:-1]
                    return value
            api=LFAPI(self.env,self.event,self.release,self.source)
            self.assertEqual(self.publish(api)['result_commit'],api.expected_commit)

    def test_commit_sha_parent_tree_author_timestamp_and_noncanonical_text_fail(self):
        body,digest=p.commit_body('d'*40,self.release)
        good={'sha':digest,'message':body['message'],'tree':{'sha':body['tree']},'parents':[{'sha':body['parents'][0]}],
            'author':body['author'],'committer':body['committer']}
        changes=[('sha','e'*40),('message',body['message'].rstrip()+' '),('message',body['message']+'\n'),
            ('message',body['message'][:-1]+'\r\n'),('tree',{'sha':'e'*40}),('parents',[{'sha':'e'*40}]),
            ('author',{**body['author'],'date':'2026-10-13T00:00:00Z'}),('committer',{**body['committer'],'name':'Other'})]
        for key,value in changes:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):p.verify_commit({**good,key:value},digest,body)

    def test_fresh_push_same_repo_attempt_one_only(self):
        for target,key,value in [('env','GITHUB_REPOSITORY','jiying2007/kws-data'),('env','GITHUB_RUN_ATTEMPT','2'),
            ('env','GITHUB_REF','refs/heads/research/qwen12-native-admission-remaining35-r1-v1'),('env','GITHUB_JOB','qwen12'),
            ('env','GITHUB_RUN_ID','00123'),('event','created',1),('event','created',False),('event','deleted',True)]:
            env,event=copy.deepcopy(self.env),copy.deepcopy(self.event);(env if target=='env' else event)[key]=value
            api=self.api()
            with self.assertRaises(ValueError):p.publish(env,event,self.release,self.package(),api,api.raw_factory)
            self.assertEqual(api.calls,[])

    def test_run_history_job_mutations_and_existing_ref_stop_before_posts(self):
        for target,key,value in [('history','total_count',2),('run','run_attempt',True),('run','status','completed'),
            ('run','head_branch','old'),('job','id',True),('job','name','old'),('repo','private',True)]:
            api=self.api();getattr(api,target)[key]=value
            with self.assertRaises(ValueError):self.publish(api)
            self.assertFalse(any(call[0]=='POST' for call in api.calls))
        api=self.api();api.ref={'ref':'already exists'}
        with self.assertRaises(p.PublicationError):self.publish(api)
        self.assertFalse(any(call[0]=='POST' for call in api.calls))

    def test_false_blob_tree_commit_refs_fail_no_retry(self):
        for noun in ('blobs','trees','commits','refs'):
            api=self.api();api.mutations[('POST',p.BASE+'/git/'+noun,201)]=lambda *_:{'sha':'e'*40}
            with self.assertRaises(ValueError):self.publish(api)
            self.assertEqual(sum(call[1]==p.BASE+'/git/'+noun for call in api.calls),1)

    def disabled_cli_source(self):
        # Exact current Python bytes, with explicit test-only disabled controls.
        # Never rewrite the production controls or replace a production gate.
        folder=self.root/'disabled-source-copy'/'research'/'k1-punctuation6-asr18-v1'
        folder.mkdir(parents=True)
        for name in ('evidence_publisher.py','evidence_transport.py','campaign_contract.py','asr18_metadata.py'):
            body=(HERE/name).read_bytes();(folder/name).write_bytes(body)
            self.assertEqual((folder/name).read_bytes(),body)
        (folder/'publication-release.json').write_bytes(t.canonical(p.disabled_publication_release()))
        (folder/'execution-release.json').write_bytes(t.canonical(c.disabled_release()))
        return folder

    def test_disabled_cli_gates_before_env_token_git_network(self):
        disabled=self.disabled_cli_source()
        class ForbiddenEnv(dict):
            def __iter__(self):raise AssertionError('disabled touched environment')
            def __getitem__(self,key):raise AssertionError('disabled touched environment')
            def __contains__(self,key):raise AssertionError('disabled touched environment')
            def keys(self):raise AssertionError('disabled touched environment')
            def get(self,*args):raise AssertionError('disabled touched environment')
        with mock.patch.object(p,'HERE',disabled),mock.patch.object(p.os,'environ',ForbiddenEnv()), \
             mock.patch.object(p.c,'read_json',wraps=p.c.read_json) as read, \
             mock.patch.object(p,'_source_git',side_effect=AssertionError('no Git')) as git, \
             mock.patch.object(p,'GitHubAPI',side_effect=AssertionError('no API/token')) as api, \
             mock.patch.object(p,'RawReader',side_effect=AssertionError('no network')) as raw, \
             mock.patch.object(p.t,'read_package',side_effect=AssertionError('no package read')) as package, \
             mock.patch.object(p.c,'durable_mkdir',side_effect=AssertionError('no claim directory')) as claim:
            with self.assertRaisesRegex(p.PublicationDisabled,'^ASR18 publication disabled or review absent$'):p.main([])
            read.assert_called_once_with(p.HERE/'publication-release.json',cap=65536)
            for untouched in (git,api,raw,package,claim):untouched.assert_not_called()
        control=p.disabled_publication_release()
        for key in ('enabled','root_publication_release','independent_source_review'):
            changed=dict(control);changed[key]=True
            with self.assertRaises(ValueError):p.publication_gate(changed)
        for key in ('enabled','root_publication_release','independent_source_review'):control[key]=True
        p.publication_gate(control)

    def test_actual_isolated_publisher_cli_reaches_disabled_gate_without_side_effects(self):
        disabled=self.disabled_cli_source()
        cwd=self.root/'isolated-cli';cwd.mkdir()
        for name in ('evidence_transport.py','campaign_contract.py','sitecustomize.py'):
            (cwd/name).write_text("raise RuntimeError('untrusted cwd or PYTHONPATH import')\n")
        sentinel='ghs_invented-isolated-cli-sentinel'
        env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/nonexistent','LC_ALL':'C',
            'PYTHONPATH':str(cwd),'GITHUB_EVENT_PATH':str(cwd/'must-not-read-event.json'),
            'RUNNER_TEMP':str(cwd),**{key:sentinel for key in ('GITHUB_TOKEN','GH_TOKEN','KWS_EVIDENCE_TOKEN',
                'ACTIONS_RUNTIME_TOKEN','ACTIONS_ID_TOKEN_REQUEST_TOKEN','ACTIONS_FUTURE_TEST_TOKEN')}}
        before={str(path.relative_to(self.root)):path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        result=subprocess.run(['python3','-I','-B',str(disabled/'evidence_publisher.py')],cwd=cwd,env=env,
            stdin=subprocess.DEVNULL,capture_output=True,timeout=15,check=False)
        self.assertEqual(result.returncode,1)
        self.assertEqual(result.stdout,b'')
        self.assertEqual(result.stderr,b'ASR18 publication disabled or review absent.\n')
        self.assertNotIn(sentinel.encode(),result.stdout+result.stderr)
        after={str(path.relative_to(self.root)):path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(after,before)
        self.assertEqual(sorted(path.name for path in cwd.iterdir()),
            ['campaign_contract.py','evidence_transport.py','sitecustomize.py'])

    def test_current_released_isolated_cli_rejects_missing_authority_before_side_effects(self):
        publication=c.read_json(HERE/'publication-release.json')
        execution=c.read_json(HERE/'execution-release.json')
        self.assertTrue(all(publication[key] is True for key in
            ('enabled','root_publication_release','independent_source_review')))
        self.assertTrue(all(execution[key] is True for key in c.RELEASE_APPROVALS))
        p.publication_gate(publication)
        cwd=self.root/'released-cli';cwd.mkdir()
        event=cwd/'event.json';event.write_bytes(b'{}')
        for name in ('evidence_transport.py','campaign_contract.py','sitecustomize.py'):
            (cwd/name).write_text("raise RuntimeError('untrusted cwd or PYTHONPATH import')\n")
        sentinel='ghs_invented-released-cli-sentinel'
        env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/nonexistent','LC_ALL':'C','PYTHONPATH':str(cwd),
            'GITHUB_EVENT_PATH':str(event),'RUNNER_TEMP':str(cwd),
            **{key:sentinel for key in ('GITHUB_TOKEN','GH_TOKEN','KWS_EVIDENCE_TOKEN','ACTIONS_RUNTIME_TOKEN',
                'ACTIONS_ID_TOKEN_REQUEST_TOKEN','ACTIONS_FUTURE_TEST_TOKEN')}}
        # This local digest is a negative-test input, never execution authority:
        # no trusted Actions host is supplied, and Git is prohibited below.
        freeze_sha=t.sha(c.read_bytes(HERE/'execution-freeze.json'))
        before={str(path.relative_to(self.root)):path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        probe=r'''
import os, sys
source, expected = sys.argv[1:]
violations = []
native_roots = {'torch','numpy','whisper','funasr','qwen_asr','native_worker','runtime_session','worker_receipts','prepare_freeze'}
def reject(label):
    violations.append(label)
    raise RuntimeError('prohibited test boundary')
def audit(event, args):
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.fork','os.exec','os.posix_spawn','ctypes.dlopen'):
        reject('process_network_native')
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.truncate'):
        reject('filesystem_write')
    if event == 'open':
        mode, flags = args[1:3]
        if (isinstance(mode,str) and any(char in mode for char in 'wax+')) or (
            isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
            reject('filesystem_write')
    if event == 'import' and args[0].split('.')[0] in native_roots:
        reject('native_import')
sys.addaudithook(audit)
class GuardedEnv(dict):
    def protect(self,key):
        if 'TOKEN' in key: reject('token_read')
    def __getitem__(self,key):
        self.protect(key); return super().__getitem__(key)
    def __contains__(self,key):
        self.protect(key); return super().__contains__(key)
    def get(self,key,*args):
        self.protect(key); return super().get(key,*args)
    def __iter__(self): reject('environment_inventory')
    def keys(self): reject('environment_inventory')
    def items(self): reject('environment_inventory')
    def values(self): reject('environment_inventory')
os.environ = GuardedEnv(os.environ)
sys.path.insert(0,source)
import evidence_publisher as publisher
try:
    publisher.main([])
except ValueError as error:
    if type(error) is not ValueError or str(error) != expected:
        raise RuntimeError('wrong production gate') from None
else:
    raise RuntimeError('production gate unexpectedly passed')
if violations or any(name.split('.')[0] in native_roots for name in sys.modules):
    raise RuntimeError('side effect observed')
print('exact production authority gate; zero side effects')
'''
        for extra,reason in (({},'lowercase SHA256 required'),
                ({'KWS_SOURCE_FREEZE_SHA256':freeze_sha},'trusted Actions poststep only')):
            with self.subTest(reason=reason):
                isolated_env={**env,**extra}
                direct=subprocess.run(['python3','-I','-B',str(HERE/'evidence_publisher.py')],cwd=cwd,
                    env=isolated_env,stdin=subprocess.DEVNULL,capture_output=True,timeout=15,check=False)
                self.assertEqual((direct.returncode,direct.stdout,direct.stderr),
                    (1,b'',b'ASR18 publication blocked or unconfirmed; no retry.\n'))
                guarded=subprocess.run(['python3','-I','-B','-c',probe,str(HERE),reason],cwd=cwd,
                    env=isolated_env,stdin=subprocess.DEVNULL,capture_output=True,timeout=15,check=False)
                self.assertEqual((guarded.returncode,guarded.stdout,guarded.stderr),
                    (0,b'exact production authority gate; zero side effects\n',b''))
                self.assertNotIn(sentinel.encode(),direct.stdout+direct.stderr+guarded.stdout+guarded.stderr)
        after={str(path.relative_to(self.root)):path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(after,before)

    def test_git_subprocess_has_exact_safe_environment_and_no_token_inheritance(self):
        sentinel = "invented-secret-sentinel-never-forward"
        names = ("GITHUB_TOKEN", "GH_TOKEN", "KWS_EVIDENCE_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_FUTURE_TEST_TOKEN")
        with mock.patch.dict(p.os.environ, {key:sentinel for key in names}), \
             mock.patch.object(p.subprocess,"check_output",return_value=b"a"*40+b"\n") as process:
            for args in (('status','--porcelain','--untracked-files=all'),('rev-parse','HEAD'),('rev-parse','HEAD^{tree}')):
                self.assertEqual(p._source_git(*args),"a"*40)
                sent=process.call_args.kwargs['env'];argv=process.call_args.args[0]
                self.assertEqual(sent,{'PATH':'/usr/bin:/bin','HOME':'/nonexistent','GIT_CONFIG_NOSYSTEM':'1',
                    'GIT_CONFIG_GLOBAL':'/dev/null','GIT_TERMINAL_PROMPT':'0','LC_ALL':'C'})
                self.assertFalse(set(sent)&set(names));self.assertNotIn(sentinel,str(sent)+str(argv))
                self.assertIn('core.fsmonitor=false',argv);self.assertIn('core.hooksPath=/dev/null',argv)
                self.assertEqual(process.call_args.kwargs['stderr'],p.subprocess.DEVNULL)
                self.assertEqual(process.call_args.kwargs['timeout'],15)
            self.assertFalse(set(p._CONTEXT_KEYS)&set(names))
        tree=ast.parse((HERE/'evidence_publisher.py').read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess']
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].func.attr,'check_output')
        self.assertEqual([(kw.arg,kw.value.id) for kw in calls[0].keywords if isinstance(kw.value,ast.Name)],
            [('env','safe_env')])
        with self.assertRaises(ValueError):p._source_git("fetch")

    def test_sentinel_tokens_never_enter_public_receipts_calls_or_logs(self):
        sentinel="invented-token-sentinel-for-output-negative-test"
        env={**self.env,**{key:sentinel for key in ('GITHUB_TOKEN','GH_TOKEN','KWS_EVIDENCE_TOKEN','ACTIONS_RUNTIME_TOKEN','ACTIONS_ID_TOKEN_REQUEST_TOKEN')}}
        api=self.api()
        with mock.patch('builtins.print') as printer:
            receipt=p.run_once(env,self.event,self.release,self.package(),api,api.raw_factory,self.root)
            printer.assert_not_called()
        retained=''.join((self.root/name).read_text() for name in ('publication.started.json','publication.receipt.json'))
        self.assertNotIn(sentinel,repr(receipt)+retained+repr(api.calls))

    def test_api_fixed_host_methods_routes_caps_and_ephemeral_token(self):
        for token in ('github_pat_invented','ghp_invented','',None):
            with self.assertRaises(ValueError):p.GitHubAPI(token,self.identity,self.release)
        api=p.GitHubAPI('ghs_invented',self.identity,self.release)
        for method,route in [('PATCH',p.BASE+'/git/refs'),('GET','/repos/other/repo'),('GET','https://evil.invalid')]:
            with self.assertRaises(ValueError):api(method,route)
        api.count=p.MAX_API_REQUESTS
        with self.assertRaises(ValueError):api('GET',p.BASE)
        api=p.GitHubAPI('ghs_invented',self.identity,self.release);api.deadline=0
        with self.assertRaises(ValueError):api('GET',p.BASE)


class FailureSchemaTests(unittest.TestCase):
    def setUp(self):
        import hosted_run
        self.guard=mock.patch("http.client.HTTPSConnection",side_effect=AssertionError("live network prohibited"))
        self.guard.start();self.addCleanup(self.guard.stop)
        self.hpatch=mock.patch.object(t,"_host",return_value=types.SimpleNamespace(_safe_failure=hosted_run._safe_failure))
        self.hpatch.start();self.addCleanup(self.hpatch.stop)

    def boundary(self):return {"source_role":"source_screen_only","training_admitted":False,"shipping_approved":False,
        "human_gold":False,"acoustic_verbatim_completeness":"unknown"}

    def test_full_phased_failure_retained_without_error_body(self):
        value={"schema":"whisper-screen-phased-failure-v1","phase":"model","status":"failed_no_retry",
            "model_calls":0,"retry_count":0,"reason_code":"http_error","http_status":503,**self.boundary()}
        t.validate_member_json("whisper-setup/phased-setup-failure.json",value)
        for extra in ({"message":"forbidden arbitrary body"},{"reason_code":"invented unknown"},{"model_calls":1}):
            with self.assertRaises(ValueError):t.validate_member_json("whisper-setup/phased-setup-failure.json",{**value,**extra})

    def test_partial_setup_failure_asset_roster_not_deleted(self):
        asset={"group":"model","index":1,"expected_sha256":"a"*64,"expected_bytes":123,"status":"started",
            "response_body_bytes_observed":2,"observed_sha256":None,"http_status":503}
        value={"schema":"whisper-screen-phased-progress-v1","status":"failed_no_retry","stage":"model_download",
            "completed_stages":["entry"],"lock_sha256":{},"asset":asset,"assets":[asset],"response_body_bytes_observed":2,
            "verified_assets":0,"model_calls":0,"retry_count":0,"network_transfer_total_known":False,
            "byte_scope":"source-projected observed bodies","phase":"model","failure":{"reason_code":"http_error","http_status":503},**self.boundary()}
        t.validate_member_json("whisper-setup/phase-model-progress.json",value)
        changed=copy.deepcopy(value);changed['asset']['request_headers']={'Authorization':'invented'}
        with self.assertRaises(ValueError):t.validate_member_json("whisper-setup/phase-model-progress.json",changed)

    def test_broker_failed_asset_conservative_charge_preserved(self):
        value={"schema":"bounded-host-asset-failure-v1","asset_id":"wheel-01","status":"failed_no_retry",
            "broker_failure_code":"byte_cap","broker_observed_ciphertext_bytes":None,"budget_charge_kind":"conservative_reserved_allowance"}
        t.validate_member_json("whisper-setup/bounded-acquisition/wheel-01-failure.json",value)
        for changes in ({'asset_id':'model'},{'broker_failure_code':'arbitrary text'},{'credentials':'invented'}):
            with self.assertRaises(ValueError):t.validate_member_json("whisper-setup/bounded-acquisition/wheel-01-failure.json",{**value,**changes})

    def test_actual_supervision_three_gate_variants_and_setup_none(self):
        for gate in ({'status':'valid','schema':'qwen12-gated-command-v1','exit_code':0},{'status':'missing'},
            {'status':'invalid','error_type':'ValueError'}):
            row={'name':'k1-asr18-invented-setup','family':'setup','mode':None,'status':'failed','worker_exit_success':False,
                'primary_error':{'error_type':'ValueError'},'cleanup_error':None,'wall_seconds':0.1,'gate_result':gate}
            value={'schema':'k1-asr18-supervision-v1','stages':[row]}
            t.validate_member_json('supervision-receipt.json',value)
            changed=copy.deepcopy(value);changed['stages'][0]['gate_result']['body']='no'
            with self.assertRaises(ValueError):t.validate_member_json('supervision-receipt.json',changed)

    def test_cleanup_failure_complete_record_cannot_silently_drop_error(self):
        row={'name':'k1-asr18-invented-qwen06','container_id':None,'status':'removal_unconfirmed','stage':'qwen06',
            'ownership':'unknown','operations':[],'cleanup_budget_seconds':15,'host_deadline_expired_at_start':False,
            'host_deadline_overrun':False,'pause_acknowledged':False,'errors':[{'operation':'ownership','error_type':'ValueError'}],
            'accounting_status':'final_usage_unverified','gate_result':{'status':'missing'},'cleanup_seconds':0.1,
            'cleanup_overrun_seconds':0,'cleanup_ok':False,'receipt_errors':[]}
        value={'containers':[row],'all_removed':False,'all_cleanup_verified':False}
        t.validate_member_json('cleanup-receipt.json',value)
        changed=copy.deepcopy(value);changed['containers'][0]['errors'][0]['message']='forbidden'
        with self.assertRaises(ValueError):t.validate_member_json('cleanup-receipt.json',changed)

    def test_new_setup_summary_no_command_or_environment(self):
        value={'schema':'k1-asr18-setup-summary-v1','status':'complete','model_calls':0,'actual_download_bytes':13,'setup_receipt_sha256':'a'*64}
        t.validate_member_json('asr-setup-public.json',value)
        for change in ({'commands':['invented']},{'environment':{}},{'model_calls':True}):
            with self.assertRaises(ValueError):t.validate_member_json('asr-setup-public.json',{**value,**change})


class FakeHTTPTests(unittest.TestCase):
    def setUp(self):
        self.env,self.event,self.release,_=setup();self.identity=p.context(self.env,self.event,self.release)

    def connection(self,code,raw):
        response=types.SimpleNamespace(status=code,read=lambda n:raw[:n])
        return types.SimpleNamespace(request=mock.Mock(),getresponse=lambda:response,close=mock.Mock())

    def test_redirect_error_body_charged_and_no_retry(self):
        api=p.GitHubAPI('ghs_invented',self.identity,self.release)
        conn=self.connection(302,b'invented error body')
        with mock.patch.object(p.http.client,'HTTPSConnection',return_value=conn) as connect:
            with self.assertRaises(p.PublicationError):api('GET',p.BASE)
            self.assertEqual(api.body_bytes,len(b'invented error body'))
            with self.assertRaises(ValueError):api('GET',p.BASE)
            self.assertEqual(connect.call_count,1)
            conn.close.assert_called_once()

    def test_api_response_cap_counts_overlimit_byte(self):
        api=p.GitHubAPI('ghs_invented',self.identity,self.release)
        conn=self.connection(200,b'x'*(p.MAX_RESPONSE+1))
        with mock.patch.object(p.http.client,'HTTPSConnection',return_value=conn):
            with self.assertRaises(p.PublicationError):api('GET',p.BASE)
        self.assertEqual(api.body_bytes,p.MAX_RESPONSE+1)
        self.assertEqual(api.count,1)

    def test_raw_reader_one_commit_fixed_paths_and_error_once(self):
        read=p.RawReader('a'*40,t.PURPOSE)
        with self.assertRaises(ValueError):read('../private')
        conn=self.connection(404,b'invented error')
        with mock.patch.object(p.http.client,'HTTPSConnection',return_value=conn) as connect:
            with self.assertRaises(p.PublicationError):read(t.manifest_path())
            with self.assertRaises(ValueError):read(t.manifest_path())
            self.assertEqual(connect.call_count,1)
        self.assertEqual(read.body_bytes,len(b'invented error'))
        self.assertEqual(conn.request.call_args.args[0],'GET')
        self.assertIn('/jiying2007/kws-pipeline/'+'a'*40+'/',conn.request.call_args.args[1])


class RealHostTransportIntegration(unittest.TestCase):
    """Invented authority files through the real collector and all validators."""
    def setUp(self):
        from test_asr18_host import public_fixture
        self.files,self.identity=public_fixture()
        self.guard=mock.patch('http.client.HTTPSConnection',side_effect=AssertionError('live network prohibited'))
        self.guard.start();self.addCleanup(self.guard.stop)
        self.temp=tempfile.TemporaryDirectory(prefix='asr18-real-validator-invented-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=[{'path':name[len('source/'):],'mode':'100644','type':'blob','sha':t.git_sha(raw)}
            for name,raw in self.files.items() if name.startswith('source/')]
        self.source_tree=tree_response(self.source)['sha']
        host=t.decode(self.files['host-receipt.json']);host['executed_tree_sha']=self.source_tree
        self.files['host-receipt.json']=t.canonical(host)
        self.materialize()

    def materialize(self):
        for name,raw in self.files.items():
            dest=self.root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)

    def test_real_preclaim_failure_full_source_18_rows_roundtrip_and_fake_publish(self):
        payload,manifest=t.build_package(self.root,self.identity)
        verified,_=t.validate_archive(payload,self.identity)
        self.assertEqual(manifest,verified)
        self.assertNotIn('authority/source-claim.json',manifest['members'])
        self.assertEqual(len(t.decode(self.files['terminal-outcomes.json'])['slots']),18)
        env,event,release,_=setup()
        release.update(source_sha=self.identity['executed_head_sha'],source_tree=self.source_tree,
            source_closure_sha256=t.sha(self.files['authority/execution-freeze.json']))
        env.update(GITHUB_SHA=release['source_sha'],GITHUB_WORKFLOW_SHA=release['source_sha'])
        event['after']=release['source_sha']
        api=MockAPI(env,event,release,self.source)
        result=p.publish(env,event,release,payload,api,api.raw_factory)
        self.assertEqual(result['status'],'same_identity_readback_verified')
        self.assertEqual(result['result_commit'],api.expected_commit)
        self.assertFalse(result['retry_authorized'])

    def test_real_host_missing_denominator_source_extra_private_never_packs(self):
        terminal=t.decode(self.files['terminal-outcomes.json']);terminal['slots'].pop()
        self.files['terminal-outcomes.json']=t.canonical(terminal);self.materialize()
        with self.assertRaises(ValueError):t.build_package(self.root,self.identity)

    def test_actual_immutable_source_authority_mismatch_prevents_all_posts(self):
        payload,_=t.build_package(self.root,self.identity)
        env,event,release,_=setup()
        release.update(source_sha=self.identity['executed_head_sha'],source_tree=self.source_tree,source_closure_sha256='f'*64)
        env.update(GITHUB_SHA=release['source_sha'],GITHUB_WORKFLOW_SHA=release['source_sha']);event['after']=release['source_sha']
        api=MockAPI(env,event,release,self.source)
        with self.assertRaises(ValueError):p.publish(env,event,release,payload,api,api.raw_factory)
        self.assertFalse(any(call[0]=='POST' for call in api.calls))


class CompleteSourceLicenseIntegration(unittest.TestCase):
    """Full real source roster; native results are explicitly synthetic only.

    All-notrun uses all 113 current source bodies and actual input metadata.
    Native semantic cases reuse the existing zero-model host fixture and replace
    exactly three metadata bodies, listed below, to bind its invented inputs.
    No actual ASR identity/result, model, download or remote claim is created.
    """
    LICENSE='source/research/qwen6_asr/FUNASR_MODEL_LICENSE'
    LICENSE_ID={'bytes':5306,'sha256':'7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8'}
    # Parent-rebuilt candidate freeze after the reviewed narrow license delta.
    FROZEN_SOURCE_SHA256='ca7f5298e222f57de373fa50ed1c647712430513f76c2738052493d7d2a6bb64'
    SYNTHETIC_CONTROLS=frozenset(('execution-release.json','input-freeze.json','tts-source-authority.json'))

    def setUp(self):
        import hosted_run
        import prepare_freeze
        import test_asr18_host
        self.host=hosted_run;self.freezer=prepare_freeze;self.host_tests=test_asr18_host
        self.guard=mock.patch('http.client.HTTPSConnection',side_effect=AssertionError('no live network'))
        self.guard.start();self.addCleanup(self.guard.stop)
        self.roster=self.freezer.declared_rosters(ROOT)['host_code']
        self.assertEqual(len(self.roster),113)
        self.assertEqual(t.SOURCE_TEXT_EXCEPTIONS,{self.LICENSE:self.LICENSE_ID})

    def full_fixture(self, *, synthetic_native=False):
        checked=c.read_bytes(HERE/'execution-freeze.json')
        self.assertEqual(t.sha(checked),self.FROZEN_SOURCE_SHA256)
        self.assertEqual(checked,t.canonical(self.freezer.build(ROOT)))
        freeze=t.decode(checked)
        source={'source/'+name:c.read_bytes(ROOT/name) for name in freeze['host_code']}
        if synthetic_native:
            files,run=self.host_tests.public_fixture()
            inp=t.decode(files['input-freeze.json'])
            substitutions={'execution-release.json':files['source/'+self.host.NEW+'execution-release.json'],
                'input-freeze.json':files['input-freeze.json'],
                'tts-source-authority.json':t.canonical(inp['actual_screening_receipt']['tts_authority'])}
            self.assertEqual(set(substitutions),self.SYNTHETIC_CONTROLS)
            for leaf,body in substitutions.items():
                name=self.host.NEW+leaf;source['source/'+name]=body
                for role in ('host_code','setup_code','inference_code'):
                    if name in freeze[role]:freeze[role][name]={'bytes':len(body),'sha256':t.sha(body)}
            files={name:raw for name,raw in files.items() if not name.startswith('source/')}
        else:
            _,_,run=self.host_tests.context()  # Invented run only; no source authority claim.
            inp=c.read_json(HERE/'input-freeze.json')
            release=c.read_json(HERE/'execution-release.json')
            projection=self.host.projection_for(inp['actual_new_cells'])
            receipt=inp['actual_screening_receipt']
            host=self.host.host_receipt(run,t.sha(t.canonical(freeze)),'d'*40,t.sha(t.canonical(receipt)))
            host.update(status='stopped_no_retry',stage='package')
            values={'host-receipt.json':host,'input-freeze.json':inp,'public-screening-receipt.json':receipt,
                'new-cell-projection.json':projection,'terminal-outcomes.json':{
                    'schema':'k1-asr18-terminal-outcomes-v1','identity':run,
                    'projection_sha256':t.sha(t.canonical(projection)),
                    'slots':c.initial_ledger(projection),'accounting_valid':True},
                'run-preflight.json':{'schema':'k1-asr18-run-preflight-v1','identity':run,'numeric_job_id':456,
                    'workflow':self.host.WORKFLOW,'source_branch':self.host.SOURCE_REF,
                    'evidence_branch':self.host.EVIDENCE_REF,'source_branch_workflow_runs':1,
                    'evidence_ref_absent':True,'prior_public_claim_history_sha256':release['prior_public_claim_history_sha256'],
                    'public_standard_hosted_ubuntu24':True,'api_metadata_accounting':{'gets':6,'response_body_bytes':1,
                        'response_body_cap_bytes':c.MIB,'scope':'public_api_metadata_separate_from_model_audio_bodies',
                        'hard_wire_byte_cap':False}}}
            files={name:t.canonical(value) for name,value in values.items()}
        raw=t.canonical(freeze)
        files.update(source);files['authority/execution-freeze.json']=raw
        files['source/'+self.host.NEW+'execution-freeze.json']=raw
        host=t.decode(files['host-receipt.json']);host['source_freeze_sha256']=t.sha(raw)
        files['host-receipt.json']=t.canonical(host)
        self.assert_complete_source(files,synthetic_native=synthetic_native)
        return files,run

    def assert_complete_source(self, files, *, synthetic_native):
        self.assertEqual({name[7:] for name in files if name.startswith('source/')},
            self.roster|{self.host.NEW+'execution-freeze.json'})
        self.assertEqual({'bytes':len(files[self.LICENSE]),'sha256':t.sha(files[self.LICENSE])},self.LICENSE_ID)
        files[self.LICENSE].decode('utf-8')
        differences={name for name in self.roster if files['source/'+name]!=c.read_bytes(ROOT/name)}
        self.assertEqual(differences,{self.host.NEW+leaf for leaf in self.SYNTHETIC_CONTROLS} if synthetic_native else set())

    def materialize(self, root, files):
        for name,body in files.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)

    def rebind_test_freeze(self, files, member):
        """Deliberately coherent adversarial fixture; never trusted authority."""
        freeze=t.decode(files['authority/execution-freeze.json']);name=member[7:]
        row={'bytes':len(files[member]),'sha256':t.sha(files[member])}
        freeze['host_code'][name]=row
        for role in ('setup_code','inference_code'):
            if name in freeze[role]:freeze[role][name]=row
        raw=t.canonical(freeze);files['authority/execution-freeze.json']=raw
        files['source/'+self.host.NEW+'execution-freeze.json']=raw
        host=t.decode(files['host-receipt.json']);host['source_freeze_sha256']=t.sha(raw)
        if 'authority/source-claim.json' in files:
            claim=t.decode(files['authority/source-claim.json']);claim['source_freeze_sha256']=t.sha(raw)
            files['authority/source-claim.json']=t.canonical(claim)
            host['source_claim_sha256']=t.sha(files['authority/source-claim.json'])
        files['host-receipt.json']=t.canonical(host)

    def reject_source_mutations(self, files, run, operation, *, phase):
        for mutation in ('single_byte','rebound_single_byte','same_body_suffixless','same_name_other_directory',
                'missing_body','rebound_non_utf8','extra_private_suffixless','relative_alias'):
            with self.subTest(phase=phase,mutation=mutation),tempfile.TemporaryDirectory(prefix='synthetic-license-negative-') as temp:
                changed=dict(files);member=self.LICENSE;reason=None
                if mutation in ('single_byte','rebound_single_byte'):
                    changed[member]=b'X'+changed[member][1:]
                    if mutation=='rebound_single_byte':
                        self.rebind_test_freeze(changed,member);reason='exact reviewed license'
                elif mutation=='missing_body':del changed[member]
                elif mutation=='rebound_non_utf8':
                    changed[member]=b'\xff'+changed[member][1:]
                    self.rebind_test_freeze(changed,member);reason='exact reviewed license'
                else:
                    member={'same_body_suffixless':self.LICENSE+'_COPY',
                        'same_name_other_directory':'source/research/another/FUNASR_MODEL_LICENSE',
                        'extra_private_suffixless':'source/research/qwen6_asr/PRIVATE_TEST_ONLY',
                        'relative_alias':'source/research/qwen6_asr/./FUNASR_MODEL_LICENSE'}[mutation]
                    changed[member]=b'invented private text must not publish' if mutation=='extra_private_suffixless' else files[self.LICENSE]
                    self.rebind_test_freeze(changed,member)
                    if mutation!='relative_alias':reason='exact reviewed license'
                root=Path(temp);self.materialize(root,changed)
                if reason:
                    with self.assertRaisesRegex(ValueError,reason):operation(root,run)
                else:
                    with self.assertRaises(ValueError):operation(root,run)

    def test_exact_113_real_source_bodies_all_notrun_package_and_license_negatives(self):
        files,run=self.full_fixture()
        with tempfile.TemporaryDirectory(prefix='synthetic-run-real-source-all-notrun-') as temp:
            root=Path(temp);self.materialize(root,files)
            payload,manifest=t.build_package(root,run)
            verified,_=t.validate_archive(payload,run);self.assertEqual(manifest,verified)
            self.assertEqual(set(self.host.collect_public(root)),set(files))
            self.assertEqual(t.decode(files['terminal-outcomes.json'])['slots'],
                c.initial_ledger(t.decode(files['new-cell-projection.json'])))
            self.assertNotIn('authority/source-claim.json',files)
            self.assertFalse(any(name.startswith(('output/','frozen/')) for name in files))
        self.reject_source_mutations(files,run,t.build_package,phase='all_notrun')

    def full_native_case(self, fail_native):
        files,run=self.full_fixture(synthetic_native=True)
        original_proof=t.package_budget_preflight;original_package=t.build_package
        observed=[]
        def proof(root,identity,**kwargs):
            result=original_proof(root,identity,**kwargs)
            actual={name:c.read_bytes(root/name) for name in self.host.collect_public(root)}
            self.assert_complete_source(actual,synthetic_native=True)
            self.assertEqual(result['new_native_calls'],0)
            self.assertEqual(result['actual_prepared_members'],len(actual))
            def operation(copy_root,copy_identity):
                return original_proof(copy_root,copy_identity,**{**kwargs,'policy_path':copy_root/'policy/policy.json'})
            self.reject_source_mutations(actual,identity,operation,phase='prepared')
            observed.append('prepared');return result
        def package(root,identity):
            result=original_package(root,identity)
            actual={name:c.read_bytes(root/name) for name in self.host.collect_public(root)}
            self.assert_complete_source(actual,synthetic_native=True)
            terminal=t.decode(actual['terminal-outcomes.json'])
            self.assertEqual([row['status'] for row in terminal['slots']],
                ['failed_no_retry']+['not_run']*17 if fail_native else ['complete']*18)
            self.reject_source_mutations(actual,identity,original_package,
                phase='first_failure_18_denominator' if fail_native else 'synthetic_18_success')
            observed.append('final');return result
        case=self.host_tests.CompleteSyntheticEndToEnd()
        with mock.patch.object(self.host_tests,'public_fixture',return_value=(files,run)), \
             mock.patch.object(t,'package_budget_preflight',side_effect=proof), \
             mock.patch.object(t,'build_package',side_effect=package):
            case.run_case(fail_native=fail_native)
        self.assertEqual(observed,['prepared','final'])

    def test_full_113_roster_prepared_and_synthetic_18_success_license_negatives(self):
        self.full_native_case(False)

    def test_full_113_roster_prepared_and_synthetic_first_failure_license_negatives(self):
        self.full_native_case(True)


if __name__=='__main__':unittest.main()
