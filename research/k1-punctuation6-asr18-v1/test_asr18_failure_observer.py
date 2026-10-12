"""Pure stdlib synthetic failure tests; no model, HTTP, Docker, or remote write."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import campaign_contract as c
import asr18_metadata as m
import failure_observer as f
import hosted_run as h

IDENTITY={'repository':'jiying2007/kws-pipeline','job_key':'asr18','run_id':123,'run_attempt':1,'executed_head_sha':'c'*40}
SECRET='TOKEN_SENTINEL_PRIVATE_PATH_OLD_HISTORY_AUDIO_TEXT'

def envelope(output):
    return next(json.loads(line) for line in output.getvalue().splitlines() if json.loads(line)['schema']=='k1-asr18-independent-process-diagnostic-envelope-v1')

def synthetic_start(family,audio_id):
    hashes={'file_sha256','pcm_sha256','job_sha256','execution_plan_sha256','worker_input_binding_sha256',
        'screening_receipt_sha256','tts_authority_sha256','recognizer_identity_sha256','recognizer_policy_sha256',
        'model_lock_sha256','source_lock_sha256','native_decode_config_sha256'}
    return {**{key:'a'*64 for key in hashes},'schema':'retained-asr-started-v1','source_id':'k1asr18-'+'a'*64,
        'audio_id':audio_id,'audio_frames':1280,'request_id':'k1asr18-123-'+family+'-'+audio_id,
        'recognizer_id':family,'run_id':'123','run_attempt':1,'executed_head_sha':'c'*40}


class ObserverTests(unittest.TestCase):
    def test_unconfigured_failure_unknown_not_zero(self):
        output=io.StringIO();o=f.Observation('host')
        with contextlib.redirect_stdout(output):
            o.stage('release');o.failure(ValueError(SECRET));self.assertTrue(o.finish())
        value=envelope(output)
        self.assertIsNone(value['record']['claims']['consumed_attempts'])
        self.assertIsNone(value['record']['claims']['reported_terminal_attempts'])
        self.assertNotIn(SECRET,output.getvalue())
        self.assertEqual(value['persistence']['status'],'unconfigured')

    def test_original_whitelisted_error_type_without_content(self):
        for error in (FileNotFoundError(2,SECRET),PermissionError(13,SECRET),ValueError(SECRET),KeyError(SECRET),
                subprocess.CalledProcessError(7,SECRET,output=SECRET,stderr=SECRET)):
            value=f.error_metadata(error)
            self.assertEqual(value['error_type'],type(error).__name__)
            self.assertNotIn(SECRET,json.dumps(value))
        self.assertEqual(f.error_metadata(FileNotFoundError(2,SECRET))['errno'],2)

    def test_hostile_exception_never_reads_dynamic_content(self):
        class Hostile(Exception):
            def __str__(self):raise AssertionError('no str')
            def __repr__(self):raise AssertionError('no repr')
            @property
            def __class__(self):raise AssertionError('no class property')
        error=Hostile(SECRET);error.__cause__=ValueError(SECRET)
        self.assertEqual(f.error_metadata(error),{'error_type':'UnlistedError','error_text':'omitted_by_fixed_privacy_policy'})

    def test_explicit_disabled_cli_keeps_safe_release_diagnostic_before_sideeffects(self):
        from test_asr18_entrypoints import disabled_source
        with disabled_source() as source:
            proc=subprocess.run([sys.executable,'-I','-B',str(source/'hosted_run.py')],
                env={'PATH':'/usr/bin:/bin','GITHUB_TOKEN':SECRET,'GH_TOKEN':SECRET,'RUNNER_TEMP':SECRET},
                capture_output=True,text=True,timeout=10)
        self.assertEqual(proc.returncode,2);self.assertNotIn(SECRET,proc.stdout+proc.stderr)
        rows=[json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(rows[0]['stage'],'release')
        self.assertEqual(rows[1]['record']['stage'],'release')
        self.assertIsNone(rows[1]['record']['identity'])
        self.assertEqual(rows[-1],{'status':'blocked_no_retry'})

    def test_actual_disabled_cli_rejects_before_event_or_sideeffects(self):
        proc=subprocess.run([sys.executable,'-I','-B',str(HERE/'hosted_run.py')],env={'PATH':'/usr/bin:/bin',
            'GITHUB_TOKEN':SECRET,'GH_TOKEN':SECRET,'RUNNER_TEMP':SECRET},capture_output=True,text=True,timeout=10)
        self.assertEqual(proc.returncode,2)
        self.assertNotIn(SECRET,proc.stdout+proc.stderr)
        rows=[json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(rows[0]['stage'],'release')
        self.assertEqual([row['stage'] for row in rows if row.get('schema')=='k1-asr18-safe-checkpoint-v1'],
                         ['release'])
        self.assertEqual(rows[-2]['record']['stage'],'release')
        self.assertIsNone(rows[-2]['record']['identity'])
        self.assertEqual(rows[-1],{'status':'blocked_no_retry'})

    def test_complete_claim_observations_do_not_assert_actual_native_success(self):
        with tempfile.TemporaryDirectory() as temp:
            o=f.Observation('host');o.attach(IDENTITY,temp);root=o.root
            (root/'authority').mkdir(parents=True);(root/'output').mkdir()
            c.write_once(root/'authority/source-claim.json',{'synthetic':True})
            for family in c.FAMILIES:
                for index in range(1,7):
                    path=root/'output'/family/f'clip-{index:06d}';path.mkdir(parents=True)
                    c.write_once(path/'started.json',synthetic_start(family,f'clip-{index:06d}'))
            output=io.StringIO()
            with contextlib.redirect_stdout(output):o.stage('package_build');o.failure(ValueError(SECRET));self.assertTrue(o.finish())
            value=envelope(output);claim=value['record']['claims']
            self.assertEqual(claim['observed_slot_directories'],18)
            self.assertEqual(claim['validated_start_files_observed'],18)
            self.assertEqual(len(claim['slots']),18)
            self.assertIsNone(claim['consumed_attempts']);self.assertIsNone(claim['reported_terminal_attempts'])
            self.assertFalse(value['record']['native_success_established'])
            raw=(o.receipt_dir/'process-diagnostic.json').read_bytes()
            self.assertEqual(c.digest(raw),value['record_sha256']);self.assertEqual(len(raw),value['record_bytes'])
            self.assertEqual(json.loads(raw),value['record'])

    def test_missing_started_file_retains_uncertain_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'output/qwen06/clip-000001').mkdir(parents=True)
            value=f.claim_snapshot(root,IDENTITY)
            self.assertEqual(value['observed_slot_directories'],1)
            self.assertEqual(value['validated_start_files_observed'],0)
            self.assertEqual(value['slots'][0]['started']['status'],'absent_at_observation')
            self.assertIsNone(value['consumed_attempts'])

    def test_missing_root_and_ledger_do_not_turn_into_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            for root in (None,Path(temp)/'absent',Path(temp)):
                value=f.claim_snapshot(root,IDENTITY)
                self.assertIsNone(value['consumed_attempts']);self.assertIsNone(value['reported_terminal_attempts'])
                self.assertIsNone(value['observed_slot_directories'])

    def test_synthetic_terminal_reported_count_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            ledger={'schema':'k1-asr18-terminal-outcomes-v1','identity':IDENTITY,'projection_sha256':'a'*64,
                'slots':[{'id':slot['slot_id'],'status':'not_run','attempts':0} for slot in m.slots()],'accounting_valid':True}
            c.write_once(root/'terminal-outcomes.json',ledger)
            value=f.claim_snapshot(root,IDENTITY)
            self.assertEqual(value['reported_terminal_attempts'],0)
            self.assertTrue(value['reported_terminal_accounting_valid'])
            self.assertIsNone(value['consumed_attempts'])

    def test_invalid_ledger_retains_hash_but_no_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c.write_once(root/'terminal-outcomes.json',{'private':SECRET})
            value=f.claim_snapshot(root,IDENTITY)
            self.assertEqual(value['terminal_ledger']['status'],'observed_invalid_ledger')
            self.assertIsNone(value['terminal_ledger']['sha256'])
            self.assertIsNone(value['reported_terminal_attempts']);self.assertNotIn(SECRET,json.dumps(value))

    def test_fixed_paths_reject_symlink_and_foreign_members_are_not_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'output/qwen06').mkdir(parents=True)
            (root/'secret').write_text(SECRET)
            (root/'output/qwen06/clip-000001').symlink_to(root,target_is_directory=True)
            (root/'output/qwen06/private.wav').write_text(SECRET)
            value=f.claim_snapshot(root,IDENTITY)
            self.assertEqual(value['slots'][0]['directory'],'unreadable_or_aliased')
            self.assertNotIn(SECRET,json.dumps(value));self.assertIsNone(value['consumed_attempts'])

    def test_persistence_failure_does_not_replace_primary_and_stdout_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            o=f.Observation('host');o.attach(IDENTITY,temp);output=io.StringIO()
            with contextlib.redirect_stdout(output),patch.object(c,'write_once',side_effect=PermissionError(13,SECRET)):
                o.stage('package_write');o.failure(ValueError(SECRET));self.assertFalse(o.finish())
            value=envelope(output)
            self.assertEqual(value['record']['errors'][0]['detail']['error_type'],'ValueError')
            self.assertEqual(value['persistence']['error']['error_type'],'PermissionError')
            self.assertNotIn(SECRET,output.getvalue())

    def test_checkpoint_stdout_failure_does_not_interrupt_cleanup(self):
        o=f.Observation('host')
        with patch('builtins.print',side_effect=BrokenPipeError(32,SECRET)):
            o.stage('cleanup');o.failure(ValueError(SECRET));self.assertFalse(o.finish())
        self.assertEqual(o.current_stage,'cleanup')
        self.assertNotIn(SECRET,json.dumps(o.checkpoint_output_error))

    def test_fifo_fixed_name_never_blocks_and_discloses_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'authority').mkdir();os.mkfifo(root/'authority/source-claim.json')
            value=f.claim_snapshot(root,IDENTITY)
            self.assertEqual(value['source_claim']['status'],'unreadable_or_aliased')
            self.assertIsNone(value['source_claim']['sha256']);self.assertIsNone(value['consumed_attempts'])

    def test_hardlinked_fixed_file_and_nonregular_device_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'authority').mkdir();(root/'private').write_text(SECRET)
            os.link(root/'private',root/'authority/source-claim.json')
            value=f.claim_snapshot(root,IDENTITY)
            self.assertEqual(value['source_claim']['status'],'unreadable_or_aliased')
            self.assertIsNone(value['source_claim']['sha256']);self.assertNotIn(SECRET,json.dumps(value))

    def test_final_diagnostic_memory_error_does_not_replace_primary(self):
        primary=ValueError(SECRET);canonical=c.canonical
        def broken(value):
            if type(value) is dict and value.get('schema')=='k1-asr18-independent-process-diagnostic-v1':
                raise MemoryError(SECRET)
            return canonical(value)
        with patch.object(h,'release_first',side_effect=primary),patch.object(c,'canonical',side_effect=broken), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError) as caught:h.main([])
        self.assertIs(caught.exception,primary)

    def test_fixed_file_read_error_closes_all_descriptors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);file=root/'test';file.write_bytes(b'{}')
            before=len(os.listdir('/proc/self/fd'))
            with patch.object(os,'read',side_effect=OSError(5,SECRET)):
                metadata,body=f._fixed_file(file,16384)
            self.assertEqual(len(os.listdir('/proc/self/fd')),before)
            self.assertEqual(metadata['status'],'unreadable_or_aliased');self.assertIsNone(body)

    def test_final_serializer_failure_emits_literal_unknown_fallback(self):
        output=io.StringIO();o=f.Observation('host');primary=ValueError(SECRET)
        with contextlib.redirect_stdout(output):
            o.stage('package_build');o.failure(primary)
            with patch.object(c,'canonical',side_effect=MemoryError(SECRET)):
                self.assertFalse(o.finish())
        last=json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(last['schema'],'k1-asr18-independent-diagnostic-fallback-v1')
        self.assertEqual(last['stage'],'package_build');self.assertIsNone(last['consumed_attempts'])
        self.assertNotIn(SECRET,output.getvalue())

    def test_envelope_serializer_failure_emits_literal_unknown_fallback(self):
        output=io.StringIO();o=f.Observation('host');canonical=c.canonical
        def broken(value):
            if type(value) is dict and value.get('schema')=='k1-asr18-independent-process-diagnostic-envelope-v1':
                raise MemoryError(SECRET)
            return canonical(value)
        with contextlib.redirect_stdout(output):
            o.stage('package_write');o.failure(ValueError(SECRET))
            with patch.object(c,'canonical',side_effect=broken):
                self.assertFalse(o.finish())
        last=json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(last['schema'],'k1-asr18-independent-diagnostic-fallback-v1')
        self.assertEqual(last['stage'],'package_write');self.assertIsNone(last['consumed_attempts'])
        self.assertNotIn(SECRET,output.getvalue())

    def test_checkpoint_and_role_bounds(self):
        for role in ('model',SECRET,None):
            with self.assertRaises(ValueError):f.Observation(role)
        o=f.Observation('host')
        with self.assertRaises(ValueError):o.stage(SECRET)
        with patch('builtins.print') as out:
            for _ in range(f.MAX_CHECKPOINTS):o.stage('prepare')
            o.stage('cleanup')
            self.assertTrue(o.checkpoint_limit_reached)
            self.assertEqual(o.current_stage,'cleanup')
        self.assertEqual(out.call_count,f.MAX_CHECKPOINTS)

    def test_identity_exact_and_attach_once(self):
        with tempfile.TemporaryDirectory() as temp:
            o=f.Observation('publisher')
            with self.assertRaises(ValueError):o.attach({**IDENTITY,'token':SECRET},temp)
            o.attach(IDENTITY,temp)
            with self.assertRaises(ValueError):o.attach(IDENTITY,temp)
            with contextlib.redirect_stdout(io.StringIO()):self.assertTrue(o.finish());self.assertFalse(o.finish())


class HostFaultObservationTests(unittest.TestCase):
    def test_after_zero_one_eighteen_synthetic_claims_package_failure_is_distinguishable(self):
        import test_asr18_host as old
        import native_worker as nw
        import evidence_transport as transport
        original=h.execute
        for count in (0,1,18):
            with self.subTest(synthetic_claims=count):
                observed=[]
                def execute(*args,**kwargs):
                    o=f.Observation('host');o.attach(args[1],args[0]['RUNNER_TEMP'])
                    try:return original(*args,**kwargs,observer=o)
                    except BaseException as error:
                        o.failure(error);o.finish(False)
                        observed.append(c.read_json(o.receipt_dir/'process-diagnostic.json'))
                        raise
                def fail_package(*args):raise PermissionError(13,SECRET)
                with patch.object(h,'execute',side_effect=execute),patch.object(transport,'build_package',side_effect=fail_package):
                    with self.assertRaises(PermissionError):
                        if count==0:
                            with patch.object(nw,'_execute_family',side_effect=ValueError(SECRET)):
                                old.CompleteSyntheticEndToEnd().run_case()
                        else:old.CompleteSyntheticEndToEnd().run_case(fail_native=count==1)
                record=observed[0]
                self.assertEqual(record['stage'],'package_build')
                self.assertEqual(record['claims']['reported_terminal_attempts'],count)
                self.assertEqual(record['claims']['observed_slot_directories'],count)
                self.assertIsNone(record['claims']['consumed_attempts'])
                self.assertEqual(record['errors'][-1]['detail']['error_type'],'PermissionError')
                if count!=18:self.assertGreaterEqual(len(record['errors']),2)
                self.assertNotIn(SECRET,json.dumps(record))

    def test_cleanup_checkpoint_failure_never_prevents_cleanup_or_replaces_primary(self):
        import test_asr18_host as old
        original=h.execute;canonical=c.canonical;records=[]
        def broken(value):
            if type(value) is dict and value.get('schema')=='k1-asr18-safe-checkpoint-v1' and value.get('stage')=='cleanup':
                raise MemoryError(SECRET)
            return canonical(value)
        def execute(*args,**kwargs):
            o=f.Observation('host');o.attach(args[1],args[0]['RUNNER_TEMP'])
            result=original(*args,**kwargs,observer=o);o.finish(result==0)
            records.append(c.read_json(o.receipt_dir/'process-diagnostic.json'))
            return result
        with patch.object(h,'execute',side_effect=execute),patch.object(c,'canonical',side_effect=broken):
            old.CompleteSyntheticEndToEnd().run_case(fail_native=True)
        record=records[0]
        self.assertEqual(record['checkpoint_output_error']['error_type'],'MemoryError')
        self.assertEqual(record['errors'][0]['detail']['error_type'],'OSError')
        self.assertEqual(record['claims']['reported_terminal_attempts'],1)

    def test_terminal_write_failure_keeps_one_claim_without_reporting_zero(self):
        import test_asr18_host as old
        original=h.execute;write=h.write_once;records=[]
        def failing_write(path,value):
            if Path(path).name=='terminal-outcomes.json':raise PermissionError(13,SECRET)
            return write(path,value)
        def execute(*args,**kwargs):
            o=f.Observation('host');o.attach(args[1],args[0]['RUNNER_TEMP'])
            try:return original(*args,**kwargs,observer=o)
            except BaseException as error:
                o.failure(error);o.finish(False)
                records.append(c.read_json(o.receipt_dir/'process-diagnostic.json'));raise
        with patch.object(h,'execute',side_effect=execute),patch.object(h,'write_once',side_effect=failing_write):
            with self.assertRaises(PermissionError):old.CompleteSyntheticEndToEnd().run_case(fail_native=True)
        record=records[0];self.assertEqual(record['stage'],'terminal_write')
        self.assertEqual(record['claims']['observed_slot_directories'],1)
        self.assertIsNone(record['claims']['reported_terminal_attempts'])
        self.assertIsNone(record['claims']['consumed_attempts'])
        self.assertEqual(record['errors'][0]['detail']['error_type'],'OSError')

    def test_terminal_accounting_failure_keeps_observable_claim_hashes(self):
        import test_asr18_host as old
        original=h.execute;records=[]
        def execute(*args,**kwargs):
            o=f.Observation('host');o.attach(args[1],args[0]['RUNNER_TEMP'])
            try:return original(*args,**kwargs,observer=o)
            except BaseException as error:
                o.failure(error);o.finish(False);records.append(c.read_json(o.receipt_dir/'process-diagnostic.json'));raise
        with patch.object(h,'execute',side_effect=execute),patch.object(h,'terminal_accounting',side_effect=KeyError(SECRET)):
            with self.assertRaises(KeyError):old.CompleteSyntheticEndToEnd().run_case(fail_native=True)
        record=records[0];self.assertEqual(record['stage'],'terminal_accounting')
        self.assertEqual(record['claims']['validated_start_files_observed'],1)
        self.assertIsNone(record['claims']['reported_terminal_attempts'])
        self.assertRegex(record['claims']['slots'][0]['started']['sha256'],'^[0-9a-f]{64}$')
        self.assertNotIn(SECRET,json.dumps(record))

    def test_package_write_failure_is_recorded_separately_after_all_eighteen(self):
        import test_asr18_host as old
        import evidence_transport as transport
        original=h.execute;records=[]
        def execute(*args,**kwargs):
            o=f.Observation('host');o.attach(args[1],args[0]['RUNNER_TEMP'])
            try:return original(*args,**kwargs,observer=o)
            except BaseException as error:
                o.failure(error);o.finish(False);records.append(c.read_json(o.receipt_dir/'process-diagnostic.json'));raise
        with patch.object(h,'execute',side_effect=execute),patch.object(transport,'write_package',side_effect=OSError(28,SECRET)):
            with self.assertRaises(OSError):old.CompleteSyntheticEndToEnd().run_case()
        record=records[0];self.assertEqual(record['stage'],'package_write')
        self.assertEqual(record['claims']['reported_terminal_attempts'],18)
        self.assertEqual(record['claims']['validated_start_files_observed'],18)
        self.assertIsNone(record['claims']['consumed_attempts'])



class StaticValidationCodeTests(unittest.TestCase):
    def test_codes_are_unique_static_gate_messages_in_reviewed_sources(self):
        import ast
        messages=set()
        for name in ('hosted_run.py','evidence_transport.py','evidence_publisher.py'):
            for node in ast.walk(ast.parse((HERE/name).read_bytes())):
                if isinstance(node,ast.Call) and len(node.args)>1 and isinstance(node.args[1],ast.Constant):
                    if isinstance(node.func,ast.Name) and node.func.id=='require' or isinstance(node.func,ast.Attribute) and node.func.attr=='require':
                        messages.add(node.args[1].value)
        self.assertEqual(len(f.VALIDATION_CODES),len(set(f.VALIDATION_CODES.values())))
        self.assertTrue(set(f.VALIDATION_CODES)<=messages)
        for message,code in f.VALIDATION_CODES.items():
            self.assertRegex(code,r'^[a-z][a-z0-9_]{1,63}$')
            result=f.error_metadata(ValueError(message))
            self.assertEqual(result['validation_code'],code)
            self.assertNotIn(message,json.dumps(result))

    def test_unknown_prefix_subclass_multiarg_and_hostile_content_stay_generic(self):
        known=next(iter(f.VALIDATION_CODES))
        class Derived(ValueError):
            @property
            def args(self):raise AssertionError('do not read subclass property')
            def __str__(self):raise AssertionError('do not render')
        for value in (known+' '+SECRET,SECRET,known+'\n',known[:5], 'x'*100000):
            self.assertNotIn('validation_code',f.error_metadata(ValueError(value)))
        for error in (Derived(known),ValueError(known,SECRET),ValueError(),TypeError(known)):
            self.assertNotIn('validation_code',f.error_metadata(error))

    def test_real_metadata_gates_emit_only_fixed_codes_in_complete_envelope(self):
        import evidence_transport as transport
        cases=({'path':'/home/'+SECRET},{'secret':SECRET},{'anything':'Bearer '+SECRET})
        for value,expected in zip(cases,('metadata_private_path','metadata_forbidden_key','metadata_credential')):
            output=io.StringIO();observer=f.Observation('host')
            with contextlib.redirect_stdout(output):
                observer.stage('capacity_proof')
                try:transport._public_metadata(value)
                except ValueError as error:observer.failure(error)
                self.assertTrue(observer.finish())
            result=envelope(output)
            self.assertEqual(result['record']['errors'][0]['detail']['validation_code'],expected)
            self.assertNotIn(SECRET,output.getvalue())
            self.assertIsNone(result['record']['claims']['consumed_attempts'])
            self.assertEqual(result['record_sha256'],c.digest(c.canonical(result['record'])))

if __name__=='__main__':unittest.main()

