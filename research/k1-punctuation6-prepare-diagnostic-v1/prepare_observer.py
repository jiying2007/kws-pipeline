"""Bounded diagnostics for the fixed load-only run; independent of its archive.

Only fixed stage/error codes and public run identities are emitted. No original
exception text, audio, tokens, environment, private paths or ASR ledger is read.
Partial observations never establish zero consumption for either sealed run.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ASR=HERE.parent/'k1-punctuation6-asr18-v1'
if str(ASR) not in sys.path:sys.path.insert(0,str(ASR))
import campaign_contract as c
from failure_observer import error_metadata
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import prepare_contract as pc

STAGES=frozenset(('entry','release','arguments','event_read','context','api_preflight','source_authority','workspace',
    'input_authorities','source_stage','input','metadata','source_claim','setup','prepare_qwen06','prepare_sensevoice',
    'prepare_whisper','policy_freeze','capacity_proof','cleanup','host_receipt_write','package_build','package_write','complete',
    'publication_gate','source_release','source_git_clean','source_git_identity','source_freeze','source_timestamp',
    'package_read','token_admission','receipt_directory','publication_started','remote_publish','receipt_write',
    'validate_local_package','verify_actual_run_source','verify_destination_absent','create_blobs','create_tree',
    'create_commit','create_ref','immutable_postpublish_readback','readback_verified'))
MAX_RECORD=8192


def identity(value):
    c.require(type(value) is dict and set(value)=={'repository','run_id','run_attempt','job_key','executed_head_sha'},'fixed prepare observation identity')
    c.require(value['repository']=='jiying2007/kws-pipeline' and value['job_key']=='prepare_diagnostic' and
        type(value['run_id']) is int and 0<value['run_id']<10**20 and type(value['run_attempt']) is int and
        value['run_attempt']==1 and type(value['executed_head_sha']) is str and c.HEAD.fullmatch(value['executed_head_sha']),
        'fixed prepare observation source')
    return pc.validate_identity(value)


class Observation:
    def __init__(self,role):
        c.require(role in ('host','publisher'),'fixed prepare observation role')
        self.role=role;self.identity=None;self.current_stage='entry';self.count=0;self.errors=[];self.seen=[]
        self.finished=False;self.receipt_dir=None;self.attachment_error=None;self.output_error=False;self.truncated=False
    def stage(self,stage):
        c.require(type(stage) is str and stage in STAGES,'fixed prepare observation stage')
        if self.finished:return
        self.current_stage=stage
        if self.count>=96:self.truncated=True;return
        self.count+=1
        try:
            value={'schema':'k1-prepare-safe-checkpoint-v1','role':self.role,'identity':self.identity,
                'stage':stage,'sequence':self.count,'native_calls':'not_asserted'}
            raw=c.canonical(value);c.require(len(raw)<=1024,'prepare checkpoint bound')
            print(raw.decode('ascii'),flush=True)
        except BaseException:self.output_error=True
    def attach(self,value,runner_temp):
        c.require(self.identity is None,'prepare identity once')
        self.identity=identity(value)
        try:self.receipt_dir=c.durable_mkdir(Path(runner_temp)/(pc.EXPERIMENT+'-observation-'+str(value['run_id'])+'-'+self.role))
        except BaseException as error:self.attachment_error=error_metadata(error);raise
    def failure(self,error):
        if any(error is old for old in self.seen):return
        if len(self.errors)<4:
            self.seen.append(error);self.errors.append({'stage':self.current_stage,'detail':error_metadata(error)})
        else:self.truncated=True
    def finish(self,success=False):
        if self.finished:return False
        self.finished=True
        try:return self._finish(success)
        except BaseException:
            try:print('{"schema":"k1-prepare-diagnostic-fallback-v1","role":"'+self.role+'","stage":"'+self.current_stage+'","consumed_asr_attempts":null,"retry_authorized":false}',flush=True)
            except BaseException:pass
            return False
    def _finish(self,success):
        record={'schema':'k1-prepare-independent-process-diagnostic-v1','role':self.role,'identity':self.identity,
            'stage':self.current_stage,'status':'returned_success' if success else 'failed_or_unconfirmed',
            'errors':self.errors,'attachment_error':self.attachment_error,'checkpoint_count':self.count,
            'checkpoint_output_error':self.output_error,'truncated':self.truncated,
            'maximum_authorized_asr_calls':0,'consumed_asr_attempts':None,'retry_authorized':False,
            'recognition_success_established':False,'original_writer_receipt_included':False,
            'limit':'partial_process_observation; abrupt_loss_may_prevent_final_record'}
        raw=c.canonical(record);c.require(len(raw)<=MAX_RECORD,'prepare diagnostic record bound')
        persistence={'status':'unconfigured','error':None}
        if self.receipt_dir is not None:
            try:c.write_once(self.receipt_dir/'process-diagnostic.json',record);persistence['status']='durable_local_record_written'
            except BaseException as error:persistence={'status':'local_record_failed','error':error_metadata(error)}
        envelope={'schema':'k1-prepare-independent-diagnostic-envelope-v1','record':record,
            'record_bytes':len(raw),'record_sha256':c.digest(raw),'persistence':persistence}
        encoded=c.canonical(envelope);c.require(len(encoded)<=MAX_RECORD+1024,'prepare diagnostic envelope bound')
        print(encoded.decode('ascii'),flush=True)
        return persistence['status']!='local_record_failed'
