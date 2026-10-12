"""Fixed independently gated prepare-only hosted diagnostic; no ASR dispatch exists."""
from pathlib import Path
import json
import os
import re
import shutil
import signal
import sys
import time
import urllib.parse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
ASR=HERE.parent/'k1-punctuation6-asr18-v1'
if str(ASR) not in sys.path:sys.path.insert(0,str(ASR))
import prepare_contract as pc
import prepare_host as ph
import prepare_observer as observation
import campaign_contract as c
import hosted_run as retained_host
from campaign_contract import require,canonical,digest,read_json,read_bytes,write_once
REPOSITORY='jiying2007/kws-pipeline'
NEW=pc.REL
WORKFLOW=pc.FIXED['workflow']
SOURCE_REF=pc.FIXED['source_ref']
EVIDENCE_REF=pc.FIXED['evidence_ref']
PublicAPI=retained_host.PublicAPI
_unique_page=retained_host._unique_page
git_read=retained_host.git_read


def observe(observer,stage):
    try:observer.stage(stage)
    except BaseException:pass


def failure(observer,error):
    try:observer.failure(error)
    except BaseException:pass


def prior_history_identity():
    return {'schema':'k1-prepare-prior-public-history-v1','repository':REPOSITORY,'source_branch':SOURCE_REF,
        'workflow':WORKFLOW,'prior_runs':[],'prior_runs_scope':'only_new_diagnostic_source_branch_and_workflow',
        'observation_status':'public_read_observed','observed_at':'2026-10-12T11:34:40.014Z',
        'source_ref_absent':True,'evidence_ref_absent':True,'all_branch_push_history_count':0,
        'sealed_prepare_runs':[{'run_id':38234429437,'source_branch':'research/k1-punctuation6-prepare-diagnostic-v1',
            'source_sha':pc.SEALED_PREPARE_SOURCE,'result_branch':'evidence/k1-punctuation6-prepare-diagnostic-v1',
            'result_sha':pc.SEALED_PREPARE_RESULT,'conclusion':'failure','consumed':True,'retry_authorized':False,
            'reusable_slots':0,'completed_constructor_receipts':0,'prepare_completion_established':False,
            'capacity_success_established':False,'asr_forward_calls':None}],
        'sealed_asr_runs':[{'run_id':38207026177,'actual_consumed_attempts':None,'reusable_slots':0},
            {'run_id':38219171568,'actual_consumed_attempts':None,'reusable_slots':0}]}

def event_identity(env, event):
    fixed = {'GITHUB_REPOSITORY': REPOSITORY, 'GITHUB_REF': 'refs/heads/' + SOURCE_REF,
        'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_JOB': 'prepare_diagnostic',
        'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64', 'ImageOS': 'ubuntu24'}
    require(all(env.get(k) == v for k, v in fixed.items()), 'fixed public standard ubuntu24.04 first-attempt job')
    head, run = env.get('GITHUB_SHA', ''), env.get('GITHUB_RUN_ID', '')
    require(c.HEAD.fullmatch(head) and re.fullmatch('[1-9][0-9]{0,19}', run), 'actual run/source identity')
    require(int(run) not in pc.SEALED_RUN_IDS, 'sealed earlier campaign run cannot become a new authority')
    require(env.get('GITHUB_WORKFLOW_REF') == REPOSITORY + '/' + WORKFLOW + '@refs/heads/' + SOURCE_REF and
            env.get('GITHUB_WORKFLOW_SHA') == head, 'fixed source workflow identity')
    require(type(event) is dict and event.get('created') is True and event.get('deleted') is False and
        event.get('forced') is False and event.get('before') == '0' * 40 and event.get('after') == head and
        event.get('ref') == 'refs/heads/' + SOURCE_REF and
        event.get('repository', {}).get('full_name') == REPOSITORY and
        event.get('repository', {}).get('private') is False, 'first public source creation event')
    return pc.validate_identity({'repository': REPOSITORY, 'run_id': int(run), 'run_attempt': 1,
            'job_key': 'prepare_diagnostic', 'executed_head_sha': head})


def api_preflight(api, run, release):
    pc.validate_identity(run)
    base = '/repos/' + REPOSITORY
    current = api.get(base + '/actions/runs/' + str(run['run_id']))
    require(current.get('id') == run['run_id'] and type(current.get('run_attempt')) is int and current.get('run_attempt') == 1 and
        current.get('status')=='in_progress' and current.get('conclusion') is None and
        current.get('event') == 'push' and current.get('head_branch') == SOURCE_REF and
        current.get('head_sha') == run['executed_head_sha'] and current.get('path') == WORKFLOW and
        current.get('repository', {}).get('full_name') == REPOSITORY and
        current.get('repository', {}).get('private') is False and
        current.get('head_repository', {}).get('full_name') == REPOSITORY and
        type(current.get('workflow_id')) is int and current['workflow_id'] > 0, 'actual current API run identity')
    workflow = api.get(base + '/actions/workflows/' + str(current['workflow_id']))
    require(workflow.get('id') == current['workflow_id'] and workflow.get('path') == WORKFLOW, 'actual workflow path')
    branch = urllib.parse.quote(SOURCE_REF, safe='')
    history = _unique_page(api, base + '/actions/workflows/' + str(current['workflow_id']) + '/runs?branch=' + branch, 'workflow_runs')
    require(len(history) == 1 and history[0].get('id') == run['run_id'] and
            history[0].get('head_sha') == run['executed_head_sha'] and
            type(history[0].get('run_attempt')) is int and history[0].get('run_attempt') == 1 and
            history[0].get('head_branch') == SOURCE_REF and history[0].get('path') == WORKFLOW and
            history[0].get('event') == 'push',
            'unique complete workflow/source-branch history required; deleted refs never reset authority')
    require(digest(canonical(prior_history_identity())) == release['prior_prepare_history_sha256'],
            'independently released prior public claim history')
    jobs = _unique_page(api, base + '/actions/runs/' + str(run['run_id']) + '/attempts/1/jobs', 'jobs')
    require(len(jobs) == 1, 'single fixed public hosted job')
    job = jobs[0]
    require(job.get('name') == 'prepare_diagnostic' and type(job.get('id')) is int and job['id'] > 0 and
        job.get('run_id') == run['run_id'] and job.get('status')=='in_progress' and job.get('conclusion') is None and job.get('head_sha') == run['executed_head_sha'] and
        job.get('labels') == ['ubuntu-24.04'] and type(job.get('runner_name')) is str and
        job['runner_name'].startswith('GitHub Actions ') and job.get('runner_group_name') == 'GitHub Actions',
        'actual standard hosted ubuntu24.04 runner')
    ref = api.get(base + '/git/ref/heads/' + SOURCE_REF)
    require(ref.get('ref') == 'refs/heads/' + SOURCE_REF and
            ref.get('object', {}).get('type') == 'commit' and ref['object'].get('sha') == run['executed_head_sha'], 'actual source ref/head')
    api.get(base + '/git/ref/heads/' + EVIDENCE_REF, absent=True)
    return {'schema': 'k1-prepare-run-preflight-v1', 'identity': run, 'numeric_job_id': job['id'],
            'workflow': WORKFLOW, 'source_branch': SOURCE_REF, 'evidence_branch': EVIDENCE_REF,
            'source_branch_workflow_runs': 1, 'evidence_ref_absent': True,
            'prior_prepare_history_sha256': release['prior_prepare_history_sha256'],
            'public_standard_hosted_ubuntu24': True, 'api_metadata_accounting':api.accounting()}

def source_authority(run,release):
    head=run['executed_head_sha']
    require(git_read('rev-parse','HEAD').decode().strip()==head and
        git_read('status','--porcelain=v1','--untracked-files=all')==b'','clean actual diagnostic source')
    tree=git_read('rev-parse',head+'^{tree}').decode().strip();require(c.HEAD.fullmatch(tree),'actual diagnostic tree')
    raw=git_read('cat-file','blob',head+':'+NEW+'prepare-freeze.json')
    freeze=c.decode_json(raw)
    require(read_bytes(HERE/'prepare-freeze.json')==raw,'exact committed diagnostic freeze')
    require(read_bytes(HERE/'prepare-release.json')==git_read('cat-file','blob',head+':'+NEW+'prepare-release.json'),
        'exact committed diagnostic release')
    pc.validate_release(release)
    from prepare_freeze_diagnostic import build
    require(c.exact_json(freeze,build(ROOT)),'complete diagnostic source roles and bytes')
    require(read_bytes(ROOT/WORKFLOW)==read_bytes(HERE/'workflow.yml')==git_read('cat-file','blob',head+':'+WORKFLOW),
        'exact diagnostic workflow')
    return freeze,digest(raw),tree


def input_authorities(release,freeze):
    inp=read_json(ASR/'input-freeze.json',128*1024);authority=read_json(ASR/'tts-source-authority.json',2*c.MIB)
    require(digest(canonical(inp))==release['input_freeze_sha256'],'exact six diagnostic input authority')
    from raw_data_reader import validate_authority
    validate_authority(authority,expected_authority_sha256=release['tts_authority_sha256'])
    receipt=inp['actual_screening_receipt']
    require(digest(canonical(receipt))==release['screening_receipt_sha256'] and inp['actual_new_cells']==receipt['cells'],
        'exact accepted diagnostic screening receipt')
    policy=retained_host.screening_policy_identity(freeze)
    require(receipt['screening_policy_sha256']==policy['sha256'],'exact diagnostic screening algorithm')
    c.validate_input_binding(inp['actual_worker_binding'],inp['actual_blind_job'],inp['actual_opaque_order'])
    require(digest(canonical(inp['actual_worker_binding']))==release['worker_input_binding_sha256'] and
        digest(canonical(inp['actual_opaque_order']))==release['execution_plan_sha256'],'exact anonymous diagnostic inputs')
    return inp,authority,policy['sha256']


def host_receipt(run,freeze_sha,tree,release):
    return {'schema':'k1-prepare-diagnostic-host-receipt-v1','identity':run,'source_freeze_sha256':freeze_sha,
        'executed_tree_sha':tree,'source_claim_sha256':None,'screening_receipt_sha256':release['screening_receipt_sha256'],
        'input_verified':False,'status':'started','stage':'admitted',
        'stages':{name:'not_run' for name in ('setup','qwen06_prepare','sensevoice_prepare','whisper_prepare')},
        'max_asr_calls':0,'asr_forward_calls':None,'tts_calls':0,'prepared_families':[],
        'primary_error':None,'cleanup_error':None,'cleanup_verified':False}


def execute(env,run,release,freeze,freeze_sha,tree,preflight,observer):
    import runtime_session as runtime
    from resource_host import MeteredDocker,atomic
    import input_binding
    import worker_receipts
    import diagnostic_transport as transport
    started=time.monotonic()
    def interrupt(_sig,_frame):raise TimeoutError('fixed diagnostic host deadline')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupt)
    signal.setitimer(signal.ITIMER_REAL,c.LIMITS['host_seconds'])
    observe(observer,'workspace')
    root=c.durable_mkdir(Path(env['RUNNER_TEMP'])/(pc.EXPERIMENT+'-'+str(run['run_id'])))
    receipt=host_receipt(run,freeze_sha,tree,release)
    docker=None;failed=False
    try:
        observe(observer,'input_authorities');inp,authority,policy_sha=input_authorities(release,freeze)
        write_once(root/'prepare-run-preflight.json',preflight)
        write_once(root/'input-freeze.json',inp)
        write_once(root/'public-screening-receipt.json',inp['actual_screening_receipt'])
        c.durable_mkdir(root/'authority');write_once(root/'authority/prepare-freeze.json',freeze)
        observe(observer,'source_stage');runtime.stage_source(freeze['host_code'],root/'source')
        write_once(root/'source'/NEW/'prepare-freeze.json',freeze)
        atomic(root/'prepare-host-receipt.json',receipt)
        observe(observer,'input');receipt['stage']='input';atomic(root/'prepare-host-receipt.json',receipt)
        require(shutil.disk_usage(root).free>=c.LIMITS['initial_free_bytes'],'fixed diagnostic initial disk reserve')
        # This mapping supplies only input hashes to the immutable raw reader;
        # it is never an ASR release or a worker permission object.
        read_pins={'actual_'+key:value for key,value in release.items() if key in
            ('tts_authority_sha256','screening_receipt_sha256')}
        bound,accounting=retained_host.verify_download(read_pins,inp,authority,expected_policy_sha256=policy_sha)
        input_binding.stage_worker_inputs(root/'input',bound['verification'],bound['eligible_audio'])
        receipt['input_verified']=True
        observe(observer,'metadata')
        packet=ph.build_metadata_packet(root/'input',release,inp['actual_worker_binding'],inp['actual_opaque_order'],run)
        c.durable_mkdir(root/'metadata')
        for name,raw in packet.items():
            destination=root/'metadata'/name
            with destination.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            destination.chmod(0o444)
        inventory={name:{'bytes':len(raw),'sha256':digest(raw)} for name,raw in sorted(packet.items())}
        observe(observer,'source_claim')
        claim={'schema':'k1-prepare-once-claim-v1','experiment':pc.EXPERIMENT,'identity':run,
            'source_freeze_sha256':freeze_sha,'release_sha256':digest(canonical(release)),
            'metadata_packet_sha256':digest(canonical(inventory)),'maximum_model_constructions':3,
            'maximum_new_asr_calls':0,'prior_prepare_history_sha256':release['prior_prepare_history_sha256']}
        claim_sha=write_once(root/'authority/prepare-claim.json',claim);receipt['source_claim_sha256']=claim_sha
        acquisition={'schema':'k1-asr18-reserved-acquisition-v1','data':accounting,'asr_response_body_bytes':None,
            'whisper':None,'reserved_total_bytes':c.LIMITS['acquisition_reserved_bytes'],'cap_bytes':6*c.GIB,'hard_wire_byte_cap':False}
        atomic(root/'acquisition-receipt.json',acquisition)
        docker=MeteredDocker(root,started+c.LIMITS['host_seconds'])
        observe(observer,'setup');receipt['stage']='setup';receipt['stages']['setup']='started'
        atomic(root/'prepare-host-receipt.json',receipt)
        # Even setup receives metadata, never the host-only input WAV directory.
        setup=runtime.setup_runtimes(root,freeze,docker,root/'metadata',run)
        receipt['stages']['setup']='complete'
        acquisition.update(asr_response_body_bytes=setup['asr_body_bytes'],whisper=setup['whisper'])
        atomic(root/'acquisition-receipt.json',acquisition)
        c.durable_mkdir(root/'output');c.durable_mkdir(root/'policy')
        receipt['stage']='prepare';atomic(root/'prepare-host-receipt.json',receipt)
        result=ph.run_prepares(root,setup,docker,run,freeze_sha,claim_sha,setup['seconds'],observer=observer)
        receipt['prepared_families']=result['prepared_families']
        for family in pc.FAMILIES:receipt['stages'][family+'_prepare']='complete'
        observe(observer,'policy_freeze');receipt['stage']='policy_freeze'
        policy=worker_receipts.build_policy({family:root/'output'/family for family in pc.FAMILIES})
        policy_sha=write_once(root/'policy/policy.json',policy)
        write_once(root/'policy/policy-freeze.json',{'schema':'k1-prepare-all3-policy-freeze-v1','identity':run,
            'recognizer_policy_sha256':policy_sha,'asr_forward_calls':0,
            'prepared_config_files':{family:c.identity(root/'output'/family/'native-config.json') for family in pc.FAMILIES}})
        receipt['asr_forward_calls']=0
        observe(observer,'capacity_proof');receipt['stage']='capacity_proof';atomic(root/'prepare-host-receipt.json',receipt)
        proof=transport.package_capacity_preflight(root,run)
        write_once(root/'policy/capacity-proof.json',proof)
        receipt['status']='all3_prepared_no_recognition'
    except BaseException as error:
        failure(observer,error);failed=True;receipt['status']='failed_no_retry';receipt['primary_error']=observation.error_metadata(error)
    finally:
        observe(observer,'cleanup');receipt['stage']='cleanup'
        if docker is not None:
            try:docker.finish();receipt['cleanup_verified']=True
            except BaseException as error:
                failure(observer,error);failed=True;receipt['status']='failed_no_retry';receipt['cleanup_error']=observation.error_metadata(error)
        else:receipt['cleanup_verified']=True
        # Successful original per-family receipts remain available after any
        # later failure; no missing or invalid receipt becomes observed zero.
        completed=[]
        for family in pc.FAMILIES:
            folder=root/'output'/family
            if not (folder/'prepare-receipt.json').exists():continue
            try:
                raw=read_bytes(folder/'prepare-receipt.json');require(raw==read_bytes(folder/'worker-complete.json'),'matching diagnostic prepare result')
                value=ph.validate_result(c.decode_json(raw),family,run,freeze_sha)
                require(value['metadata_packet_sha256']==claim['metadata_packet_sha256'],'same claimed diagnostic metadata')
                completed.append(family);receipt['stages'][family+'_prepare']='complete'
            except BaseException:failed=True;receipt['status']='failed_no_retry'
        receipt['prepared_families']=completed
        if completed!=list(pc.FAMILIES):receipt['asr_forward_calls']=None
        if docker is not None:
            for family in pc.FAMILIES:
                name=pc.EXPERIMENT+'-'+str(run['run_id'])+'-'+family
                if name in docker.names and family not in completed:
                    receipt['stages'][family+'_prepare']='failed_no_retry'
        for key,value in receipt['stages'].items():
            if value=='started':receipt['stages'][key]='failed_no_retry'
        observe(observer,'host_receipt_write');receipt['stage']='package';atomic(root/'prepare-host-receipt.json',receipt)
        observe(observer,'package_build');payload,_=transport.build_package(root,run)
        observe(observer,'package_write');transport.write_package(env['RUNNER_TEMP'],payload,run)
        print(json.dumps({'schema':'k1-prepare-local-evidence-ready-v1','archive_bytes':len(payload),
            'archive_sha256':digest(payload),'prepare_failed':failed,'recognition_result':False},sort_keys=True),flush=True)
        signal.setitimer(signal.ITIMER_REAL,0)
    return int(failed)


def main(argv=None):
    observer=observation.Observation('host')
    try:
        observe(observer,'release');release=pc.release_first()
        require(not (sys.argv[1:] if argv is None else argv),'fixed prepare host has no arguments')
        observe(observer,'event_read');event=read_json(Path(os.environ['GITHUB_EVENT_PATH']),2*c.MIB)
        observe(observer,'context');run=event_identity(dict(os.environ),event)
        observer.attach(run,os.environ['RUNNER_TEMP'])
        observe(observer,'source_authority');freeze,freeze_sha,tree=source_authority(run,release)
        observe(observer,'api_preflight');preflight=api_preflight(PublicAPI(),run,release)
        result=execute(dict(os.environ),run,release,freeze,freeze_sha,tree,preflight,observer)
        try:observer.finish(result==0)
        except BaseException:pass
        return result
    except BaseException as error:
        failure(observer,error)
        try:observer.finish(False)
        except BaseException:pass
        raise


if __name__=='__main__':
    try:status=main()
    except BaseException:
        print('{"status":"prepare_diagnostic_failed_or_disabled_no_retry"}',flush=True)
        status=2
    raise SystemExit(status)
