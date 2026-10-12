"""Real Docker create/inspect/remove contract. No container may ever start.

This is infrastructure CI, independent of every model execution/publication
release. Six fresh empty directories stand in for mounts; no repository tree,
audio, model, credentials or authority packet is mounted. The fixed official
Python image is pulled, but no command in it is executed.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare_host as host
c = host.c
IMAGE = 'docker.io/library/python@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a'
LABEL = 'org.kws.prepare-launch-ci-owner'
CID = re.compile(r'[0-9a-f]{64}\Z')
VERSION_FORMAT = '{{json .}}'
EMPTY_TIME = '0001-01-01T00:00:00Z'
HOST_FIELDS = ('Memory', 'MemorySwap', 'NanoCpus', 'PidsLimit', 'NetworkMode',
    'ReadonlyRootfs', 'Privileged', 'CgroupnsMode', 'PidMode', 'IpcMode', 'CapAdd',
    'CapDrop', 'SecurityOpt', 'LogConfig', 'Binds', 'VolumesFrom', 'Devices',
    'DeviceRequests', 'ExtraHosts', 'Links', 'GroupAdd', 'PortBindings',
    'PublishAllPorts', 'Tmpfs')


FAILURE_PREDICATES = frozenset((
    'bounded_private_local_logs',
    'ci_arguments_once',
    'ci_container_never_started',
    'ci_container_ownership',
    'ci_create_output_cid_match',
    'ci_created_cid_required',
    'ci_docker_command_denied',
    'ci_docker_output_bound',
    'ci_empty_metadata_only_fixture',
    'ci_exact_created_cid',
    'ci_exact_gate_owner',
    'ci_exact_single_create',
    'ci_fixed_official_image_lock',
    'ci_fresh_cidfile',
    'ci_immutable_created_cid',
    'ci_no_arguments',
    'ci_nonroot_host',
    'ci_one_inspect_record',
    'ci_owned_cid_absence_proof',
    'ci_owned_cid_removal_only',
    'ci_public_standard_hosted_only',
    'ci_removal_proof_only',
    'ci_removed_exact_cid',
    'docker_version_fields',
    'exact_mount_roster',
    'exact_prepare_command_no_mode',
    'exact_prepare_image',
    'fixed_tmpfs_arguments',
    'fixed_tmpfs_options',
    'fixed_tmpfs_roster',
    'metadata_only_mount_contract',
    'no_extra_host_resource',
    'no_extra_volume',
    'no_secret_or_label_environment',
    'nonroot_effective_worker_user',
    'observed_mount_identity',
    'prepare_privilege_boundary',
    'prepare_resource_network_boundary',
    'unique_mount_destination',
    'unique_tmpfs_destination',
    'unique_worker_environment',
))


def failure(error):
    # Exact builtin types and fixed source predicates only. Never access an
    # exception subclass's args/property, class name, str or repr.
    name = 'Exception'
    for kind, fixed_name in ((ValueError, 'ValueError'), (OSError, 'OSError'),
            (FileNotFoundError, 'FileNotFoundError'), (PermissionError, 'PermissionError'),
            (TimeoutError, 'TimeoutError'), (MemoryError, 'MemoryError'),
            (KeyboardInterrupt, 'KeyboardInterrupt'), (SystemExit, 'SystemExit'),
            (subprocess.CalledProcessError, 'CalledProcessError'),
            (subprocess.TimeoutExpired, 'TimeoutExpired')):
        if type(error) is kind:
            name = fixed_name
            break
    predicate = 'unclassified_error'
    if type(error) is ValueError:
        args = BaseException.args.__get__(error)
        if type(args) is tuple and args and type(args[0]) is str and args[0] in FAILURE_PREDICATES:
            predicate = args[0]
    return {'type': name, 'predicate': predicate}


def version_fields(raw):
    value = json.loads(raw)
    result = {}
    for side in ('Client', 'Server'):
        row = value[side]
        names = ('Version', 'ApiVersion', 'MinAPIVersion', 'Os', 'Arch')
        result[side] = {key: row[key] for key in names if key in row and type(row[key]) is str and
            re.fullmatch(r'[a-zA-Z0-9_.+() /:-]{1,128}', row[key])}
        c.require(all(key in result[side] for key in ('Version', 'ApiVersion', 'Os', 'Arch')),
            'docker_version_fields')
    return result


def inspect_projection(record, args, root):
    """Sanitized projection of actual API fields, never a passing fixture.

    Full argv, environment values, source paths and daemon error text are not
    published. Tmpfs preserves the exact raw safe option strings; mount source
    equality and six explicit destinations retain the boundary observation.
    """
    cfg, actual = record.get('Config', {}), record.get('HostConfig', {})
    image_index = args.index(IMAGE)
    supplied = {args[i + 1] for i, item in enumerate(args[:image_index]) if item == '--env'}
    environment = cfg.get('Env', [])
    default_keys = {'PATH', 'LANG', 'GPG_KEY', 'PYTHON_VERSION', 'PYTHON_SHA256'}
    expected_sources = {'/' + name: str(root / name) for name in
        ('code', 'runtime', 'output', 'metadata', 'authority', 'gate')}
    mounts = []
    for row in record.get('Mounts', []):
        destination = row.get('Destination')
        mounts.append({'Type': row.get('Type') if row.get('Type') in ('bind', 'tmpfs', 'volume') else '<unknown>',
            'Destination': destination if destination in expected_sources else '<unexpected-destination>',
            'RW': row.get('RW') if type(row.get('RW')) is bool else None,
            'Propagation': row.get('Propagation') if row.get('Propagation') in
                ('rprivate', 'private', 'rshared', 'shared', 'rslave', 'slave') else '<unexpected-propagation>',
            'Source_matches_expected': destination in expected_sources and
                row.get('Source') == expected_sources[destination]})
    projection = {}
    for key in ('Memory', 'MemorySwap', 'NanoCpus', 'PidsLimit'):
        projection[key] = actual.get(key) if type(actual.get(key)) is int else None
    for key in ('ReadonlyRootfs', 'Privileged', 'PublishAllPorts'):
        projection[key] = actual.get(key) if type(actual.get(key)) is bool else None
    for key in ('NetworkMode', 'CgroupnsMode', 'PidMode', 'IpcMode'):
        value = actual.get(key)
        projection[key] = value if value in ('', 'none', 'bridge', 'private', 'host', 'shareable') else '<unexpected>'
    for key in ('CapAdd', 'CapDrop', 'SecurityOpt'):
        value = actual.get(key)
        projection[key] = value if value is None or (type(value) is list and all(type(v) is str and
            (re.fullmatch(r'[A-Z_]{1,40}', v) or v in ('no-new-privileges', 'no-new-privileges:true')) for v in value)) else '<unexpected>'
    log = actual.get('LogConfig', {})
    projection['LogConfig'] = log if log == {'Type': 'local', 'Config': {
        'max-size': '1m', 'max-file': '1', 'compress': 'false'}} else '<unexpected-log-config>'
    projection['Unexpected_host_resources_present'] = [key for key in ('Binds', 'VolumesFrom', 'Devices',
        'DeviceRequests', 'ExtraHosts', 'Links', 'GroupAdd', 'PortBindings') if actual.get(key)]
    tmpfs = actual.get('Tmpfs')
    if type(tmpfs) is dict:
        safe = re.compile(r'(rw|ro|noexec|exec|nosuid|suid|nodev|dev|size=[0-9]+|uid=[0-9]+|gid=[0-9]+|mode=[0-7]+)\Z')
        projection['Tmpfs'] = {key: value if type(value) is str and len(value) <= 512 and
            all(safe.fullmatch(part) for part in value.split(',')) else '<unexpected-options>'
            for key, value in tmpfs.items() if key in ('/tmp', '/scratch')}
        projection['Unexpected_tmpfs_destination_count'] = len(set(tmpfs) - {'/tmp', '/scratch'})
    else:
        projection['Tmpfs'] = None
    return {'Id': record.get('Id') if type(record.get('Id')) is str and CID.fullmatch(record['Id']) else None,
        'State': {key: record.get('State', {}).get(key) for key in ('Status', 'Running', 'Pid', 'StartedAt', 'FinishedAt')},
        'Config': {'Image_matches_locked': cfg.get('Image') == IMAGE,
            'Entrypoint_empty': cfg.get('Entrypoint') in (None, []),
            'Cmd_matches_production_argv': cfg.get('Cmd') == list(args[image_index + 1:]),
            'User': cfg.get('User') if type(cfg.get('User')) is str and re.fullmatch(r'[0-9]+:[0-9]+', cfg['User']) else '<unexpected-user>',
            'Supplied_environment_matches': supplied <= set(environment),
            'Unexpected_environment_count': sum(entry not in supplied and entry.split('=', 1)[0] not in default_keys
                for entry in environment)},
        'HostConfig': projection, 'Mounts': mounts}


class NoStartDocker:
    """Closed Docker vocabulary; start/run/exec and aliases cannot be invoked."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.owner = uuid.uuid4().hex
        self.name = 'k1-prepare-launch-ci-' + self.owner
        self.cidfile = self.root / 'created.cid'
        self.cid = None
        self.owned = False
        self.create_attempted = False
        self.removed = False
        self.commands = []
        self.args = None
        for name in ('code', 'runtime', 'output', 'metadata', 'authority', 'gate', 'docker-home', 'docker-config'):
            (self.root / name).mkdir(mode=0o700)
        self.env = {'PATH': '/usr/bin:/bin', 'HOME': str(self.root / 'docker-home'),
            'DOCKER_CONFIG': str(self.root / 'docker-config'), 'LANG': 'C.UTF-8'}

    def gate(self, name):
        c.require(name == self.name, 'ci_exact_gate_owner')
        return self.root / 'gate'

    def arguments(self):
        c.require(self.args is None, 'ci_arguments_once')
        # No execution or publication release is consulted, altered or enabled.
        args = host._worker_arguments(name=self.name, image=IMAGE, code=self.root / 'code',
            runtime=self.root / 'runtime', output=self.root / 'output', metadata=self.root / 'metadata',
            authority=self.root / 'authority', family='qwen06',
            run_identity={'executed_head_sha': '0' * 40, 'run_id': '0'},
            freeze_sha='0' * 64, claim_sha='0' * 64)
        from resource_host import gated_arguments
        args = gated_arguments(args, self.name, self, '/code/' + c.OLD + 'cpu_gate.py')
        # Only ownership metadata is added to the actual production argv path.
        index = args.index(IMAGE)
        args[index:index] = ['--cidfile', str(self.cidfile), '--label', LABEL + '=' + self.owner]
        self.args = tuple(args)
        return list(self.args)

    def _allowed(self, args):
        c.require(type(args) is tuple and args, 'ci_docker_command_denied')
        fixed = {('version', '--format', VERSION_FORMAT), ('pull', '--platform', 'linux/amd64', IMAGE)}
        if args in fixed:
            return
        if args[0] == 'create':
            c.require(args == self.args and not self.create_attempted, 'ci_exact_single_create')
            for name in ('code', 'runtime', 'output', 'metadata', 'authority', 'gate'):
                folder = self.root / name
                c.require(not folder.is_symlink() and folder.is_dir() and not any(folder.iterdir()),
                    'ci_empty_metadata_only_fixture')
            c.require(not self.cidfile.exists(), 'ci_fresh_cidfile')
            self.create_attempted = True
            return
        c.require(self.cid is not None and CID.fullmatch(self.cid), 'ci_docker_command_denied')
        if args == ('container', 'inspect', self.cid):
            return
        if args == ('container', 'rm', self.cid):
            c.require(self.owned and not self.removed, 'ci_owned_cid_removal_only')
            return
        if args == ('container', 'ls', '--all', '--no-trunc', '--filter', 'id=' + self.cid, '--format', '{{.ID}}'):
            c.require(self.removed, 'ci_removal_proof_only')
            return
        c.require(False, 'ci_docker_command_denied')

    def __call__(self, *args):
        self._allowed(args)
        self.commands.append(list(args[:2]) if args[0] == 'container' else [args[0]])
        result = subprocess.run(['/usr/bin/docker', '--host', 'unix:///var/run/docker.sock', *args],
            env=self.env, stdin=subprocess.DEVNULL, capture_output=True, timeout=180 if args[0] == 'pull' else 30,
            check=True)
        c.require(len(result.stdout) <= c.MIB and len(result.stderr) <= c.MIB, 'ci_docker_output_bound')
        return result.stdout

    def capture_cid(self, output=None):
        c.require(self.create_attempted and self.cidfile.is_file() and not self.cidfile.is_symlink(),
            'ci_created_cid_required')
        raw = c.regular(self.cidfile, 65).decode('ascii').strip()
        c.require(CID.fullmatch(raw), 'ci_exact_created_cid')
        if output is not None:
            c.require(output.decode('ascii').strip() == raw, 'ci_create_output_cid_match')
        if self.cid is not None:
            c.require(self.cid == raw, 'ci_immutable_created_cid')
        self.cid = raw

    def inspect_owned(self):
        records = json.loads(self('container', 'inspect', self.cid))
        c.require(type(records) is list and len(records) == 1, 'ci_one_inspect_record')
        record = records[0]
        c.require(record.get('Id') == self.cid and record.get('Name') == '/' + self.name and
            record.get('Config', {}).get('Labels') == {LABEL: self.owner}, 'ci_container_ownership')
        state = record.get('State', {})
        c.require(state.get('Status') == 'created' and state.get('Running') is False and
            state.get('Paused') is False and state.get('Restarting') is False and state.get('Dead') is False and
            state.get('Pid') == 0 and state.get('StartedAt') == EMPTY_TIME and
            state.get('FinishedAt') == EMPTY_TIME, 'ci_container_never_started')
        self.owned = True
        return record

    def cleanup(self):
        if self.cid is None:
            if not self.create_attempted:
                return {'status': 'create_not_attempted', 'removed_by_name': False}
            if not self.cidfile.exists():
                return {'status': 'create_outcome_unresolved_no_owned_cid', 'removed_by_name': False}
            self.capture_cid()
        # Re-prove exact ID, random owner, name and never-started state at cleanup.
        # A colliding name or unowned ID is NEVER a deletion target.
        self.inspect_owned()
        output = self('container', 'rm', self.cid)
        c.require(output.decode('ascii').strip() == self.cid, 'ci_removed_exact_cid')
        self.removed = True
        absent = self('container', 'ls', '--all', '--no-trunc', '--filter', 'id=' + self.cid, '--format', '{{.ID}}')
        c.require(absent.strip() == b'', 'ci_owned_cid_absence_proof')
        return {'status': 'removed_and_absent', 'removed_cid': self.cid, 'removed_by_name': False}


def run(root):
    docker = NoStartDocker(root)
    receipt = {'schema': 'k1-prepare-launch-ci-v1', 'status': 'failed',
        'scope': 'create_inspect_remove_only_no_worker_execution', 'image': IMAGE,
        'model_loads': 0, 'container_start_calls': 0, 'asr_calls': 0, 'tts_calls': 0,
        'inference_evidence': False, 'real_docker_versions': None, 'inspect': None,
        'launch_validator_passed': False, 'primary_error': None, 'cleanup_error': None,
        'cleanup': None}
    try:
        lock = json.loads((c.ROOT / 'research/source-screen32-v1/container-lock.json').read_bytes())
        c.require(lock['image'] == IMAGE and lock['platform'] == 'linux/amd64', 'ci_fixed_official_image_lock')
        args = docker.arguments()
        receipt['real_docker_versions'] = version_fields(docker('version', '--format', VERSION_FORMAT))
        docker('pull', '--platform', 'linux/amd64', IMAGE)
        docker.capture_cid(docker(*args))
        # Capture the sanitized actual API evidence before the launch predicate.
        record = docker.inspect_owned()
        receipt['inspect'] = inspect_projection(record, args, docker.root)
        host.validate_launch(record, args)
        receipt['launch_validator_passed'] = True
    except BaseException as error:
        receipt['primary_error'] = failure(error)
    finally:
        try:
            receipt['cleanup'] = docker.cleanup()
        except BaseException as error:
            receipt['cleanup_error'] = failure(error)
        receipt['docker_operations'] = docker.commands
    if receipt['primary_error'] is None and receipt['cleanup_error'] is None and \
            receipt['launch_validator_passed'] and receipt['cleanup']['status'] == 'removed_and_absent':
        receipt['status'] = 'passed'
    return receipt


def main():
    c.require(not sys.argv[1:], 'ci_no_arguments')
    expected = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline',
        'GITHUB_JOB': 'python-contracts', 'RUNNER_ENVIRONMENT': 'github-hosted',
        'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64', 'ImageOS': 'ubuntu24'}
    c.require(all(os.environ.get(k) == v for k, v in expected.items()), 'ci_public_standard_hosted_only')
    c.require(os.getuid() != 0, 'ci_nonroot_host')
    # tempfile owns removal of only its new empty fixture tree, never a checkout.
    with tempfile.TemporaryDirectory(prefix='k1-prepare-launch-ci-', dir=os.environ['RUNNER_TEMP']) as temp:
        receipt = run(Path(temp))
        print(c.canonical(receipt).decode(), flush=True)
    return int(receipt['status'] != 'passed')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(c.canonical({'schema': 'k1-prepare-launch-ci-entry-failure-v1', 'error': failure(error)}).decode(), flush=True)
        raise SystemExit(1)
