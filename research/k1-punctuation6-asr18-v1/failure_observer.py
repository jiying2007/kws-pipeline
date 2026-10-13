"""Bounded public diagnostics independent of the complete ASR evidence package.

No model, network, token, transcript, audio, arbitrary error text, or environment
is read. These are local observations, never native success or retry authority.
A missing ledger is unknown, not zero. A SIGKILL/runner loss can still prevent a
final record; earlier bounded stdout checkpoints then remain only partial proof.
"""
import errno
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import campaign_contract as c

STAGES = frozenset(('entry','release','arguments','event_read','context','api_preflight','source_authority',
    'workspace','input_authorities','source_stage','input','source_claim','setup','prepare','policy_freeze',
    'capacity_proof','decode','cleanup','terminal_accounting','terminal_write','host_receipt_write',
    'package_build','package_write','complete','publication_gate','source_release','source_git_clean',
    'source_git_identity','source_freeze','source_timestamp','package_read','token_admission',
    'receipt_directory','publication_started','remote_publish','receipt_write','validate_local_package',
    'verify_actual_run_source','verify_destination_absent','create_blobs','create_tree','create_commit',
    'create_ref','immutable_postpublish_readback','readback_verified',
    'create_checkpoint_ref','immutable_checkpoint_readback','checkpoint_verified',
    'prepare_qwen06','prepare_sensevoice','prepare_whisper','decode_qwen06','decode_sensevoice','decode_whisper'))
MAX_RECORD = 16384
MAX_CHECKPOINTS = 96
IDENTITY_KEYS = frozenset(('repository','run_id','run_attempt','job_key','executed_head_sha'))
ERROR_TYPES = (AssertionError,FileNotFoundError,FileExistsError,NotADirectoryError,IsADirectoryError,
    PermissionError,TimeoutError,OSError,ValueError,TypeError,KeyError,RuntimeError,ImportError,
    ModuleNotFoundError,MemoryError,RecursionError,UnicodeError,UnicodeDecodeError,
    json.JSONDecodeError,subprocess.TimeoutExpired,subprocess.CalledProcessError,KeyboardInterrupt,SystemExit)
ERRNOS = frozenset(errno.errorcode)


# Only exact, source-fixed validation messages map to public enum codes.
# No exception message, prefix, path, variable, subclass property or traceback
# is copied. Unlisted conditions keep the original generic private-text policy.
VALIDATION_CODES = {
    # Fixed prepare-only launch predicates; never infer a code for an old run.
    'bounded_private_local_logs': 'prepare_launch_logs',
    'exact_mount_roster': 'prepare_launch_mount_roster',
    'exact_prepare_command_no_mode': 'prepare_launch_command',
    'exact_prepare_image': 'prepare_launch_image',
    'fixed_tmpfs_arguments': 'prepare_launch_tmpfs_arguments',
    'fixed_tmpfs_options': 'prepare_launch_tmpfs_options',
    'fixed_tmpfs_roster': 'prepare_launch_tmpfs_roster',
    'metadata_only_mount_contract': 'prepare_launch_metadata_mounts',
    'no_extra_host_resource': 'prepare_launch_host_resources',
    'no_extra_volume': 'prepare_launch_extra_volume',
    'no_secret_or_label_environment': 'prepare_launch_environment',
    'nonroot_effective_worker_user': 'prepare_launch_nonroot_user',
    'observed_mount_identity': 'prepare_launch_mount_identity',
    'prepare_privilege_boundary': 'prepare_launch_privileges',
    'prepare_resource_network_boundary': 'prepare_launch_resources_network',
    'unique_mount_destination': 'prepare_launch_mount_unique',
    'unique_tmpfs_destination': 'prepare_launch_tmpfs_unique',
    'unique_worker_environment': 'prepare_launch_environment_unique',
    'unknown worker member cannot be dropped or published': 'collector_worker_roster',
    'only fixed public authority/policy/order members': 'collector_fixed_roster',
    'whole actual public source closure required': 'collector_source_roster',
    'complete public source identity': 'collector_source_identity',
    'exact bounded Whisper setup metadata': 'collector_whisper_setup_roster',
    'complete public evidence bound; never prune to fit': 'collector_total_bytes',
    'complete actual retained source/assets': 'collector_retained_body_identity',
    'required public evidence missing': 'host_required_member',
    'whole actual prepared metadata and lock closure': 'host_prepared_roster',
    'prepared actual run/source closure': 'host_prepared_identity',
    'prepared measured config/model/source identity': 'host_prepared_measurement',
    'complete policy and actual prepared identity': 'host_prepared_policy',
    'durable complete prepare-only result': 'host_prepared_result',
    'prepared exact anonymous inputs': 'host_prepared_inputs',
    'no prepared forward traces': 'host_prepared_has_forward',
    'fixed evidence phase': 'host_evidence_phase',
    'complete source-bound evidence core': 'host_evidence_core',
    'fixed host identity/boundary/caps': 'host_identity_or_limits',
    'actual source freeze/head binding': 'host_source_binding',
    'actual metadata bodies separately counted within remaining6GiB reserve': 'host_input_body_reservation',
    'ASR actual body reservation': 'host_asr_body_reservation',
    'bounded complete evidence file map': 'package_file_map',
    'member byte cap': 'package_member_bytes',
    'member count cap': 'package_member_count',
    'member cap': 'package_member_size',
    'complete expanded evidence cap; do not omit records': 'package_expanded_bytes',
    'complete compressed evidence cap; do not omit records': 'package_compressed_bytes',
    'member outside exact public allowlist': 'package_member_allowlist',
    'public metadata object required': 'package_json_object',
    'nonpublic field prohibited': 'metadata_forbidden_key',
    'private host path prohibited': 'metadata_private_path',
    'credential payload prohibited': 'metadata_credential',
    'bounded JSON nesting': 'metadata_depth',
    'public JSON scalar': 'metadata_scalar_type',
    'exact retained public metadata schema required': 'metadata_retained_schema',
    'complete retained native nested shape': 'metadata_native_shape',
    'fixed bounded supervision': 'metadata_supervision_shape',
    'bounded observed supervision': 'metadata_supervision_value',
    'complete cleanup failure/success fields': 'metadata_cleanup_shape',
    'complete bounded image observation': 'metadata_image_shape',
    'exact bounded setup reason': 'metadata_setup_reason',
    'exact phased progress shape': 'metadata_setup_progress',
    'no setup model calls/retries or false all-wire claim': 'metadata_setup_claim',
    'full exact native source/dependency bytes required': 'native_retained_body_identity',
    'all three actual prepared configurations': 'capacity_prepared_roster',
    'actual prepared native policy/config binding': 'capacity_policy_identity',
    'complete prepare/frozen identity': 'capacity_prepare_identity',
    'zero forward before budget proof': 'capacity_forward_residue',
    'prepare-only zero native calls': 'capacity_prepare_result',
    'actual prepared policy bytes': 'capacity_policy_bytes',
    'fixed commit timestamp': 'publisher_timestamp_syntax',
    'actual push source timestamp': 'publisher_timestamp_missing',
    'whole-second source timestamp with explicit offset': 'publisher_timestamp_offset_syntax',
    'timestamp from exact checked-out source commit': 'publisher_timestamp_commit_identity',
    'push timestamp matches source committer instant': 'publisher_timestamp_instant_mismatch',
}


def validation_code(error):
    if type(error) is not ValueError:return None
    try:
        args=BaseException.args.__get__(error)
        if type(args) is tuple and len(args)==1 and type(args[0]) is str and len(args[0])<=160:
            return VALIDATION_CODES.get(args[0])
    except BaseException:pass
    return None


def error_metadata(error):
    """Never render error content; only exact builtin args may select a fixed enum."""
    kind = type(error)
    name = next((known.__name__ for known in ERROR_TYPES if kind is known), 'UnlistedError')
    result = {'error_type':name, 'error_text':'omitted_by_fixed_privacy_policy'}
    code=validation_code(error)
    if code is not None:result['validation_code']=code
    if issubclass(kind,OSError):
        try:
            number=OSError.errno.__get__(error)
            if type(number) is int and number in ERRNOS:result['errno']=number
        except BaseException:pass
    if kind is subprocess.CalledProcessError:
        try:
            fields=BaseException.__dict__['__dict__'].__get__(error)
            number=dict.get(fields,'returncode')
            if type(number) is int and -255<=number<=255:result['returncode']=number
        except BaseException:pass
    return result


def _identity(value):
    c.require(type(value) is dict and set(value)==IDENTITY_KEYS,'fixed diagnostic identity')
    c.require(value['repository']=='jiying2007/kws-pipeline' and value['job_key']=='asr18' and
        type(value['run_id']) is int and 0<value['run_id']<10**20 and type(value['run_attempt']) is int and
        value['run_attempt']==1 and type(value['executed_head_sha']) is str and
        re.fullmatch('[0-9a-f]{40}',value['executed_head_sha']) is not None,'fixed diagnostic source/run')
    c.require(value['run_id'] not in c.SEALED_ASR_RUNS and value['executed_head_sha'] not in c.SEALED_ASR_SOURCES,
        'sealed earlier ASR identity is not reusable')
    return {key:value[key] for key in sorted(IDENTITY_KEYS)}


def _directory(path):
    try:
        fd=c._open_directory(path)
        try: info=os.fstat(fd)
        finally:os.close(fd)
        return 'observed_directory' if stat.S_ISDIR(info.st_mode) else 'unreadable_or_aliased'
    except FileNotFoundError:return 'absent_at_observation'
    except BaseException:return 'unreadable_or_aliased'


def _regular_bytes(path,cap):
    # O_NONBLOCK is essential: a fixed-name FIFO must fail fstat, not hang open.
    path=c.safe_path(path);parent=c._open_directory(path.parent);fd=None
    try:
        fd=os.open(path.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)
        before=os.fstat(fd)
        c.require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and 0<=before.st_size<=cap,
            'fixed bounded unaliased diagnostic file')
        body=os.read(fd,cap+1)
        c.require(len(body)==before.st_size and c._stamp(os.fstat(fd))==c._stamp(before) and
            c._stamp(os.stat(path.name,dir_fd=parent,follow_symlinks=False))==c._stamp(before),
            'diagnostic bytes and identity stable')
        c.require(c._directory_identity(os.fstat(parent))==c._directory_identity(path.parent.stat(follow_symlinks=False)),
            'diagnostic parent stable')
        return body
    finally:
        try:
            if fd is not None:os.close(fd)
        finally:os.close(parent)


def _fixed_file(path, cap):
    try:
        body=_regular_bytes(path,cap)
        return {'status':'observed_file','bytes':len(body),'sha256':c.digest(body)},body
    except FileNotFoundError:return {'status':'absent_at_observation','bytes':None,'sha256':None},None
    except BaseException:return {'status':'unreadable_or_aliased','bytes':None,'sha256':None},None


def _sha(value):
    c.require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'fixed public digest')


def _safe_source_claim(raw,identity):
    value=c.decode_json(raw)
    keys={'schema','identity','source_freeze_sha256','release_sha256','source_branch','result_branch',
        'workflow','numeric_job_id','prior_public_claim_history_sha256','api_preflight_verified',
        'source_branch_workflow_runs','maximum_new_asr_calls','old_calls_allowed','new_audio_forwards_at_claim'}
    c.require(type(value) is dict and set(value)==keys and value['schema']=='k1-asr18-once-source-claim-v1' and
        value['identity']==identity,'fixed new source claim only')
    _identity(value['identity'])
    fixed={'source_branch':'research/k1-punctuation6-asr18-r2','result_branch':'evidence/k1-punctuation6-asr18-r2',
        'workflow':'.github/workflows/k1-punctuation6-asr18-v1.yml','api_preflight_verified':True,
        'source_branch_workflow_runs':1,'maximum_new_asr_calls':18,'old_calls_allowed':0,'new_audio_forwards_at_claim':0}
    c.require(all(type(value[k]) is type(v) and value[k]==v for k,v in fixed.items()),'fixed source claim fields')
    c.require(type(value['numeric_job_id']) is int and 0<value['numeric_job_id']<10**20,'numeric job only')
    for name in ('source_freeze_sha256','release_sha256','prior_public_claim_history_sha256'):_sha(value[name])


def _safe_start(raw,identity,family,audio_id):
    value=c.decode_json(raw)
    hashes={'file_sha256','pcm_sha256','job_sha256','execution_plan_sha256','worker_input_binding_sha256',
        'screening_receipt_sha256','tts_authority_sha256','recognizer_identity_sha256','recognizer_policy_sha256',
        'model_lock_sha256','source_lock_sha256','native_decode_config_sha256'}
    keys=hashes|{'schema','source_id','audio_id','audio_frames','request_id','recognizer_id','run_id','run_attempt','executed_head_sha'}
    c.require(type(value) is dict and set(value)==keys,'only fixed safe start fields')
    for name in hashes:_sha(value[name])
    expected={'schema':'retained-asr-started-v1','source_id':'k1asr18-'+value['file_sha256'],'audio_id':audio_id,
        'request_id':'k1asr18-'+str(identity['run_id'])+'-'+family+'-'+audio_id,'recognizer_id':family,
        'run_id':str(identity['run_id']),'run_attempt':1,'executed_head_sha':identity['executed_head_sha']}
    c.require(all(type(value[k]) is type(v) and value[k]==v for k,v in expected.items()) and
        type(value['audio_frames']) is int and 0<value['audio_frames']<192000,'fixed new start identity only')


def _invalid_file(status):
    return {'status':status,'bytes':None,'sha256':None}


def unknown_claims():
    return {'scope':'fixed_new_campaign_local_observations_only','snapshot_atomic':False,
        'consumed_attempts':None,'consumed_attempts_authority':'unknown',
        'source_claim':{'status':'unobserved','bytes':None,'sha256':None},
        'terminal_ledger':{'status':'unobserved','bytes':None,'sha256':None},
        'reported_terminal_attempts':None,'reported_terminal_accounting_valid':None,
        'observed_slot_directories':None,'validated_start_files_observed':None,'slots':[]}


def claim_snapshot(root, identity):
    """Read only the fixed new source claim/ledger and18 fixed anonymous starts.

    Counts are observations, not a replacement for native/source validation.
    Even an existing directory without started.json is conservatively visible.
    No absent ledger or failed read becomes a zero consumption assertion.
    """
    result=unknown_claims()
    if root is None or _directory(root)!='observed_directory':return result
    source,source_raw=_fixed_file(root/'authority/source-claim.json',16384)
    if source_raw is not None:
        try:_safe_source_claim(source_raw,identity)
        except BaseException:source=_invalid_file('observed_invalid_source_claim')
    terminal,raw=_fixed_file(root/'terminal-outcomes.json',16384)
    result.update(source_claim=source,terminal_ledger=terminal)
    if raw is not None:
        try:
            value=c.decode_json(raw,require_canonical=False)
            c.require(set(value)=={'schema','identity','projection_sha256','slots','accounting_valid'} and
                value['schema']=='k1-asr18-terminal-outcomes-v1' and value['identity']==identity and
                type(value['accounting_valid']) is bool and type(value['slots']) is list and len(value['slots'])==18,
                'exact diagnostic ledger')
            _sha(value['projection_sha256'])
            _identity(value['identity'])
            import asr18_metadata as m
            for row,slot in zip(value['slots'],m.slots()):
                c.require(type(row) is dict and set(row)=={'id','status','attempts'} and row['id']==slot['slot_id'] and
                    row['status'] in ('not_run','not_run_duplicate','not_run_quality','complete','failed_no_retry') and
                    type(row['attempts']) is int and row['attempts'] in (0,1),'fixed diagnostic ledger rows')
            result['reported_terminal_attempts']=sum(row['attempts'] for row in value['slots'])
            result['reported_terminal_accounting_valid']=value['accounting_valid']
        except BaseException:
            result['terminal_ledger']=_invalid_file('observed_invalid_ledger')
    output=root/'output'
    if _directory(output)!='observed_directory':return result
    rows=[];directories=starts=0;uncertain=False
    for family in c.FAMILIES:
        for index in range(1,7):
            audio_id=f'clip-{index:06d}';folder=output/family/audio_id
            state=_directory(folder)
            start={'status':'unobserved','bytes':None,'sha256':None}
            if state=='observed_directory':
                directories+=1;start,start_raw=_fixed_file(folder/'started.json',16384)
                if start_raw is not None:
                    try:_safe_start(start_raw,identity,family,audio_id)
                    except BaseException:start=_invalid_file('observed_invalid_start')
                if start['status']=='observed_file':starts+=1
            if state=='unreadable_or_aliased' or start['status']=='unreadable_or_aliased':uncertain=True
            rows.append({'family':family,'audio_id':audio_id,'directory':state,'started':start})
    result.update(slots=rows,observed_slot_directories=directories,validated_start_files_observed=starts)
    # These remain non-atomic lower-bound observations even when every read worked.
    result['consumed_attempts_authority']='unknown_partial_local_observation' if uncertain else 'unknown_local_observations_only'
    return result


class Observation:
    """One fixed process, bounded checkpoint lines and one separate final record."""
    def __init__(self,role):
        c.require(role in ('host','publisher'),'fixed diagnostic role')
        self.role=role;self.current_stage='entry';self.identity=None;self.root=None
        self.receipt_dir=None;self.errors=[];self._seen_errors=[];self.checkpoints=0;self.finished=False
        self.attachment_error=None;self.checkpoint_output_error=None;self.checkpoint_limit_reached=False;self.errors_truncated=False

    def stage(self,stage):
        c.require(type(stage) is str and stage in STAGES,'fixed diagnostic stage code')
        if self.finished:return
        self.current_stage=stage
        if self.checkpoints>=MAX_CHECKPOINTS:
            self.checkpoint_limit_reached=True
            return
        self.checkpoints+=1
        value={'schema':'k1-asr18-safe-checkpoint-v1','role':self.role,'stage':stage,
            'sequence':self.checkpoints,'identity':self.identity,'native_calls':'not_asserted'}
        try:
            body=c.canonical(value);c.require(len(body)<=1024,'checkpoint byte cap')
            print(body.decode('ascii'),flush=True)
        except BaseException as error:self.checkpoint_output_error=error_metadata(error)

    def attach(self,identity,runner_temp):
        c.require(self.identity is None,'diagnostic identity set once')
        self.identity=_identity(identity)
        parent=c.safe_path(runner_temp)
        fd=c._open_directory(parent);os.close(fd)
        self.root=parent/(c.EXPERIMENT+'-'+str(identity['run_id']))
        target=parent/('k1-asr18-diagnostic-'+str(identity['run_id'])+'-'+self.role)
        try:self.receipt_dir=c.durable_mkdir(target)
        except BaseException as error:
            self.attachment_error=error_metadata(error)
            raise

    def failure(self,error):
        if any(error is old for old in self._seen_errors):return
        if len(self.errors)<4:
            self._seen_errors.append(error)
            self.errors.append({'stage':self.current_stage,'detail':error_metadata(error),
                'explanation':'operation_failed_at_fixed_stage; exception_content_omitted'})
        else:self.errors_truncated=True

    def finish(self,success=False):
        if self.finished:return False
        self.finished=True
        try:return self._finish(success)
        except BaseException:
            # No serializer, file, exception text, or candidate value is needed
            # for this fixed final fallback. Role/stage were enum-validated.
            try:
                print('{"schema":"k1-asr18-independent-diagnostic-fallback-v1","role":"'+self.role+
                    '","stage":"'+self.current_stage+'","status":"diagnostic_failed",'+
                    '"consumed_attempts":null,"retry_authorized":false}',flush=True)
            except BaseException:pass
            return False

    def _finish(self,success):
        try: claims=claim_snapshot(self.root,self.identity)
        except BaseException:claims=unknown_claims()
        value={'schema':'k1-asr18-independent-process-diagnostic-v1','role':self.role,
            'identity':self.identity,'stage':self.current_stage,'status':'returned_success' if success else 'failed_or_unconfirmed',
            'errors':self.errors,'attachment_error':self.attachment_error,
            'checkpoint_output_error':self.checkpoint_output_error,
            'checkpoint_limit_reached':self.checkpoint_limit_reached,'errors_truncated':self.errors_truncated,'claims':claims,
            'checkpoint_count':self.checkpoints,'original_writer_receipt_included':False,
            'native_success_established':False,'retry_authorized':False,
            'limit':'ordinary_exception_observation; abrupt_process_or_runner_loss_may_prevent_final_record'}
        raw=c.canonical(value)
        if len(raw)>MAX_RECORD:
            # Never dump an oversized/unknown object. Emit only a fixed failure shell.
            value={'schema':'k1-asr18-independent-process-diagnostic-v1','role':self.role,
                'identity':self.identity,'stage':self.current_stage,'status':'diagnostic_size_rejected',
                'consumed_attempts':None,'retry_authorized':False}
            raw=c.canonical(value)
        persistence={'status':'unconfigured','error':None}
        if self.receipt_dir is not None:
            try:
                c.write_once(self.receipt_dir/'process-diagnostic.json',value)
                persistence['status']='durable_local_record_written'
            except BaseException as error:
                persistence={'status':'local_record_failed','error':error_metadata(error)}
        envelope={'schema':'k1-asr18-independent-process-diagnostic-envelope-v1','record':value,
            'record_bytes':len(raw),'record_sha256':c.digest(raw),'persistence':persistence}
        encoded=c.canonical(envelope);c.require(len(encoded)<=MAX_RECORD+1024,'diagnostic envelope cap')
        print(encoded.decode('ascii'),flush=True)
        return persistence['status']!='local_record_failed'
