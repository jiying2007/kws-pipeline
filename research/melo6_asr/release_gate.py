"""Read-only Melo paired-release gate. Prepared releases are unapproved."""
import hashlib
import json
import re
from pathlib import Path

EXPERIMENT = 'melo-six-phrase-source-screen-v1'
SHA = re.compile(r'[0-9a-f]{64}\Z')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_release(root, component, release_path):
    root = Path(root).resolve()
    path = Path(release_path)
    if component != 'asr' or root.name != 'melo6_asr' or root.parent.name != 'research':
        raise ValueError('Melo ASR deployment scope')
    if path.is_symlink() or path.resolve() != (root.parent / 'melo6-source-screen-release.json').resolve():
        raise ValueError('Melo release location')
    release = json.loads(path.read_bytes())
    fields = {'schema','approved','experiment_id','tts_candidate_sha256','asr_candidate_sha256',
              'plan_sha256','blind_archive_sha256','blind_freeze_sha256','max_tts_calls','max_asr_calls'}
    if set(release) != fields or release['schema'] != 'melo6-source-screen-release-v1' or release['approved'] is not True or release['experiment_id'] != EXPERIMENT:
        raise ValueError('Explicitly approved Melo release required')
    if release['max_tts_calls'] != 6 or release['max_asr_calls'] != 12:
        raise ValueError('Fixed call budget')
    for key in fields:
        if key.endswith('_sha256') and (type(release[key]) is not str or not SHA.fullmatch(release[key])):
            raise ValueError('Frozen source/input identity missing')
    freeze_path = root / 'candidate-freeze.json'
    if file_sha(freeze_path) != release['asr_candidate_sha256']:
        raise ValueError('Melo ASR source hash mismatch')
    freeze = json.loads(freeze_path.read_bytes())
    if freeze.get('schema') != 'melo6-component-code-freeze-v1' or freeze.get('component') != 'asr':
        raise ValueError('Wrong component freeze')
    scope = [p for p in root.rglob('*') if p.relative_to(root).parts[0] != 'runtime']
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in scope):
        raise ValueError('Source aliases/special files')
    actual = {p.relative_to(root).as_posix() for p in scope if p.is_file() and p != freeze_path}
    if actual != set(freeze['files']):
        raise ValueError('Unexpected ASR source membership')
    for name, digest in freeze['files'].items():
        if file_sha(root / name) != digest:
            raise ValueError('ASR source drift')
    workflow = root.parent.parent / '.github/workflows/melo6-source-screen.yml'
    if workflow.is_symlink() or workflow.read_bytes() != (root / 'workflow.yml').read_bytes():
        raise ValueError('Melo workflow drift')
    incoming = root / 'runtime/incoming'
    if file_sha(incoming / 'blind-inputs.zip') != release['blind_archive_sha256'] or file_sha(incoming / 'blind-input-freeze.json') != release['blind_freeze_sha256']:
        raise ValueError('Blind input differs from approved exact bytes')
    # The private plan is not read by the recognizer stage. Its published hash
    # binds later comparison to the same six-input generation plan.
    return release
