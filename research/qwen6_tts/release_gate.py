"""Shared, read-only gate for the two separately frozen source-screen jobs."""
import hashlib,json,re
from pathlib import Path

EXPERIMENT='qwen-stock-source-screen-v1'
RECOVERY={'prior_runs': [{'run_id': 37335852099, 'status': 'FAILED_NO_RETRY', 'cause': 'UNKNOWN_NOT_RETAINED', 'tts_calls': 0, 'asr_calls': 0}, {'run_id': 37339583157, 'status': 'FAILED_NO_RETRY', 'cause': 'SOUNDFILE_IMPORT_NATIVE_RESOLUTION_FAILED', 'tts_calls': 0, 'asr_calls': 0}], 'tts_setup_attempt': 3, 'asr_setup_attempt': 1, 'cumulative_max_tts_calls': 6, 'cumulative_max_asr_calls': 12, 'prior_controlled_download_bytes': 5867332928}
SHA=re.compile(r'[0-9a-f]{64}\Z')

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_release(root,component,release_path):
    root=Path(root).resolve();path=Path(release_path)
    if component not in ('tts','asr') or root.name!='qwen6_'+component or root.parent.name!='research':raise ValueError('Component deployment scope')
    if path.is_symlink() or path.resolve()!=(root.parent/'qwen6-source-screen-release.json').resolve():raise ValueError('Shared release location')
    release=json.loads(path.read_bytes())
    fields={'schema','approved','experiment_id','tts_candidate_sha256','asr_candidate_sha256','plan_sha256','recovery'}
    if set(release)!=fields or release['schema']!='qwen6-source-screen-native-recovery-release-v3' or release['approved'] is not True or release['experiment_id']!=EXPERIMENT:raise ValueError('Reviewed paired release required')
    if release['recovery']!=RECOVERY:raise ValueError('Exact one-recovery identity required')
    if any(type(release[k]) is not str or not SHA.fullmatch(release[k]) for k in ('tts_candidate_sha256','asr_candidate_sha256','plan_sha256')):raise ValueError('Paired release identities')
    freeze_path=root/'candidate-freeze.json'
    if file_sha(freeze_path)!=release[component+'_candidate_sha256']:raise ValueError('Reviewed component hash mismatch')
    freeze=json.loads(freeze_path.read_bytes())
    if freeze.get('schema')!='qwen6-component-code-freeze-v1' or freeze.get('component')!=component:raise ValueError('Wrong component freeze')
    scope=[p for p in root.rglob('*') if p.relative_to(root).parts[0]!='runtime']
    if any(p.is_symlink() or not(p.is_file() or p.is_dir()) for p in scope):raise ValueError('Source aliases/special files')
    actual={p.relative_to(root).as_posix() for p in scope if p.is_file() and p!=freeze_path}
    if actual!=set(freeze['files']):raise ValueError('Unexpected component source membership')
    for name,sha in freeze['files'].items():
        if file_sha(root/name)!=sha:raise ValueError('Component source drift')
    workflow=root.parent.parent/'.github/workflows/qwen6-source-screen.yml'
    if workflow.is_symlink() or workflow.read_bytes()!=(root/'workflow.yml').read_bytes():raise ValueError('Workflow/template drift')
    # This is a byte-identity check only. ASR never decodes or passes the plan to a model.
    if file_sha(root.parent/'qwen6_tts/plan.json')!=release['plan_sha256']:raise ValueError('Preregistered plan drift')
    return release
