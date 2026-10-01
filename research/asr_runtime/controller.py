#!/usr/bin/env python3
"""Admitted Docker lifecycle, frozen inputs and failure collection.

No CLI is currently exposed: source-only candidate. The caller must complete all
admission checks before calling qualify(). Host never imports installed packages.
"""
import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
import signal
import errno
import stat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify as q


class OwnershipChanged(q.GateError):
    pass


def command_output(result):
    q.require(result['returncode'] == 0 and not result['timeout'] and not result['truncated'], 'Docker client command failed: ' + result['output'][-8192:])
    return result['output']


def measure_tree(root, allow_links, sampled):
    """FD-relative scan, never follows links; sampled ENOENT/rename races skip."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    stack = [os.open(root, flags)]
    total, count = 0, 0
    transient = (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
    try:
        while stack:
            fd = stack.pop()
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:
                        count += 1
                        q.require(count <= 200000, 'scratch/output entry cap exceeded')
                        try:
                            info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                            if stat.S_ISDIR(info.st_mode):
                                stack.append(os.open(entry.name, flags, dir_fd=fd))
                            elif stat.S_ISLNK(info.st_mode):
                                q.require(allow_links, 'symlink output rejected')
                                total += info.st_size
                            else:
                                q.require(stat.S_ISREG(info.st_mode), 'special work/output file rejected')
                                q.require(allow_links or info.st_nlink == 1, 'hardlinked output rejected')
                                total += info.st_size
                        except OSError as exc:
                            if not sampled or exc.errno not in transient:
                                raise
            finally:
                os.close(fd)
    finally:
        for fd in stack:
            os.close(fd)
    return total


def folder_size(root, sampled=False):
    return measure_tree(root, allow_links=False, sampled=sampled)


def work_size(root):
    return measure_tree(root, allow_links=True, sampled=True)


def validate_stage_receipt(inner, which, output, before):
    q.require(inner.get('schema') == 'kws.asr-runtime-container-stage.v1' and inner.get('stage') == which,
              'wrong stage receipt schema/stage')
    q.require(type(inner.get('target_speech_model_weights_loaded')) is int and inner['target_speech_model_weights_loaded'] == 0 and
              type(inner.get('user_audio_loaded')) is int and inner['user_audio_loaded'] == 0, 'stage exceeded no-model scope')
    names = ('admission.json', 'inventory.json', 'build-plan.json', 'container_stage.py',
             'build-requirements.txt', 'runtime-requirements.txt', 'runtime-versions.json')
    contracts = {n: before[n]['sha256'] for n in names if n in before}
    q.require(inner.get('input_contracts_sha256') == contracts, 'stage contract binding mismatch')
    if which.startswith('build-'):
        q.require(inner.get('built_source_count') == 5, 'source build count mismatch')
    if which == 'runtime':
        q.require(inner.get('cpu_probe_completed') is True, 'runtime CPU probe incomplete')
        cpu_path = Path(output) / 'cpu-probe.json'
        q.require(cpu_path.is_file() and not cpu_path.is_symlink() and q.sha256_file(cpu_path) == inner.get('cpu_probe_sha256'), 'CPU probe receipt identity mismatch')
        cpu = q.load_json(cpu_path, q.MAX_RECEIPT)
        q.require(cpu.get('schema') == 'kws.asr-runtime-cpu-probe.v1' and cpu.get('device') == 'cpu' and
                  cpu.get('float32_matmul') is True and cpu.get('bf16_matmul_layer_norm_conv1d') is True and
                  cpu.get('target_speech_model_weights_loaded') == 0 and cpu.get('user_audio_loaded') == 0, 'CPU probe receipt fields incomplete')
        q.require(isinstance(cpu.get('installed_versions'), dict) and cpu['installed_versions'], 'actual package versions missing')
        expected_version_hash = before.get('runtime-versions.json', {}).get('sha256')
        q.require(hashlib.sha256(json.dumps(cpu['installed_versions'], sort_keys=True).encode()).hexdigest() == expected_version_hash, 'actual installed versions do not match locked runtime')
        inner['cpu_probe'] = cpu
    return inner


def stage(docker, image, image_id, name, inputs, output, limits, which, run=q.bounded_command, clock=time.monotonic, sleep=time.sleep):
    output = Path(output)
    output.mkdir(mode=0o777)
    output.chmod(0o777)
    work = output.parent / (which + '-work')
    work.mkdir(mode=0o777)
    work.chmod(0o777)
    before = q.regular_tree(inputs)
    args = q.create_command(docker, image, name, inputs, output, work, limits, which)
    prefix = [docker, '--host=unix:///var/run/docker.sock']
    cid, ownership_verified = None, False
    record = {'stage': which, 'status': 'failed', 'stop_reason': None,
              'input_manifest_sha256': hashlib.sha256(q.canonical(before)).hexdigest(), 'cleanup_errors': []}

    def inspect_owned():
        value = json.loads(command_output(run(prefix + ['inspect', cid])))[0]
        if value.get('Config', {}).get('Labels', {}).get('com.kws.asr-runtime.owner') != name:
            raise OwnershipChanged('container ownership changed')
        return value

    try:
        candidate_id = command_output(run(args)).strip()
        q.require(q.HEX.fullmatch(candidate_id), 'invalid container ID')
        cid = candidate_id
        inspected = json.loads(command_output(run(prefix + ['inspect', cid])))
        q.require(isinstance(inspected, list) and len(inspected) == 1, 'invalid ownership inspect')
        ownership_verified = inspected[0].get('Config', {}).get('Labels', {}).get('com.kws.asr-runtime.owner') == name
        q.require(ownership_verified, 'container ownership was not verified')
        q.verify_container_inspect(inspected, args, image_id, limits)
        command_output(run(prefix + ['start', cid]))
        start = clock()
        while True:
            observed = inspect_owned()
            if not observed['State']['Running']:
                break
            if clock() - start > limits['wall_seconds']:
                record['stop_reason'] = 'wall_timeout'
            elif work_size(work) > limits['work_disk_bytes']:
                record['stop_reason'] = 'sampled_work_disk_byte_cap'
            elif folder_size(output, sampled=True) > limits['output_bytes']:
                record['stop_reason'] = 'sampled_output_byte_cap'
            elif shutil.disk_usage(output).free < limits['minimum_free_disk_bytes']:
                record['stop_reason'] = 'sampled_free_disk_floor'
            if record['stop_reason']:
                break
            sleep(1)
    except BaseException as exc:
        record['controller_error'] = type(exc).__name__ + ': ' + str(exc)
        if cid is None:
            record.update(create_outcome_unknown=True, owned_container_name=name, retry_permitted=False)
    finally:
        if cid and ownership_verified:
            record['container_id'] = cid
            final = None
            # Diagnostics cannot suppress bounded cleanup of the immutable CID that
            # was already proven ours. No unverified response is used as a CID.
            for operation in ('inspect', 'stop', 'inspect', 'kill', 'inspect'):
                try:
                    if operation == 'inspect':
                        final = inspect_owned()
                        if not final['State']['Running']:
                            break
                    else:
                        command_output(run(prefix + ([operation, '--time=2', cid] if operation == 'stop' else [operation, cid]), timeout=10))
                except BaseException as exc:
                    record['cleanup_errors'].append(operation + ': ' + type(exc).__name__ + ': ' + str(exc))
                    if isinstance(exc, OwnershipChanged):
                        record['ownership_lost'] = True
                        break
            if final is not None:
                record.update(q.summarize_state(final, record['stop_reason']))
            else:
                record['terminal_state_unknown'] = True
            if record.get('controller_error') or record['cleanup_errors']:
                record['status'] = 'failed'
            try:
                log = run(prefix + ['logs', cid], maximum=q.MAX_LOG)
                record['log'] = {k: v for k, v in log.items() if k != 'output'}
                print(json.dumps({'stage': which, 'container_log': log['output']}), flush=True)
                q.require(log['returncode'] == 0 and not log['timeout'], 'container logs incomplete')
                record['input_snapshot_unchanged'] = q.regular_tree(inputs) == before
                q.require(record['input_snapshot_unchanged'], 'read-only inputs changed')
                record['work_bytes'], record['output_bytes'] = work_size(work), folder_size(output)
                q.require(record['work_bytes'] <= limits['work_disk_bytes'] and record['output_bytes'] <= limits['output_bytes'], 'final work/output byte cap exceeded')
                record['inner'] = validate_stage_receipt(q.load_json(output / 'stage-receipt.json', q.MAX_RECEIPT), which, output, before)
            except BaseException as exc:
                record['status'] = 'failed'
                record['diagnostic_error'] = type(exc).__name__ + ': ' + str(exc)
            try:
                record['early_receipt_sha256'] = q.write_receipt(output.parent / (which + '.receipt.json'), record)
            except BaseException as exc:
                record['status'] = 'failed'
                record['early_receipt_error'] = type(exc).__name__ + ': ' + str(exc)
            try:
                # Remove our exact already-verified CID even if diagnostics failed.
                # --force closes a running-container uncertainty; absent State is
                # retained as unknown, never fabricated as exit 0 or not-OOM.
                q.require(not record.get('ownership_lost'), 'changed ownership blocks removal')
                command_output(run(prefix + ['rm', '--force', cid], timeout=15))
                record['container_removed'] = True
            except BaseException as exc:
                record['status'] = 'failed'
                record['container_removed'] = False
                record['cleanup_errors'].append('remove: ' + type(exc).__name__ + ': ' + str(exc))
        else:
            record.update(unverified_container_untouched=True, container_removed=False)
        record['ownership_verified'] = ownership_verified
        try:
            q.write_receipt(output.parent / (which + '.final.json'), record)
        except BaseException as exc:
            record['status'] = 'failed'
            record['final_receipt_error'] = type(exc).__name__ + ': ' + str(exc)
    return record


def requirements(files):
    # Full pinned dependency resolver input. Never --no-deps at install time.
    rows = []
    for x in sorted(files, key=lambda x: x['name']):
        extras = x.get('install_extras', [])
        q.require(all(__import__('re').fullmatch(r'[A-Za-z0-9_-]+', e) for e in extras), 'invalid install extras')
        name = x['name'] + ('[' + ','.join(extras) + ']' if extras else '')
        rows.append(f"{name}=={x['version']} --hash=sha256:{x['sha256']}\n")
    return ''.join(rows)


def normalized_requirement(value):
    import re
    value = re.sub(r'\s+', '', value).replace("'", '"')
    match = re.match(r'([A-Za-z0-9_.-]+)(.*)', value)
    q.require(match is not None, 'invalid requirement text')
    return re.sub(r'[-_.]+', '-', match.group(1)).lower() + match.group(2)


def validate_built_wheels(directory, plan, limits):
    wheels = sorted(Path(directory).glob('*.whl'))
    q.require(len(wheels) == len(plan['sources']), 'wrong built wheel count')
    result = []
    import email.parser
    import zipfile
    for source in plan['sources']:
        matches = [p for p in wheels if p.name.startswith(source['wheel_filename_prefix'])]
        q.require(len(matches) == 1, 'built wheel identity is ambiguous')
        path = matches[0]
        audit = q.inspect_archive(path, limits['archive_expanded_bytes'], limits['archive_members'])
        names = set(audit['payloads'])
        allowed = set(source['expected_static_payload_sha256']) | set(source['required_wheel_members']) | set(source['required_native_members']) | set(source['allowed_generated_members'])
        q.require(names <= allowed, 'unexpected built wheel payload member')
        q.require(set(source['required_wheel_members']).issubset(names), 'built wheel missing reviewed member')
        native = sorted(n for n in names if n.endswith(('.so', '.pyd', '.a', '.o', '.dll', '.dylib')))
        q.require(native == source['required_native_members'], 'unexpected/missing built native payload; pure fallback rejected')
        with zipfile.ZipFile(path) as archive:
            metadata = [n for n in names if n.count('/') == 1 and n.endswith('.dist-info/METADATA')]
            q.require(len(metadata) == 1, 'wheel METADATA count mismatch')
            parsed = email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
            q.require(parsed['Name'].lower().replace('_', '-') == source['name'].lower().replace('_', '-') and
                      parsed['Version'] == source['version'], 'built wheel METADATA identity mismatch')
            q.require(sorted(map(normalized_requirement, parsed.get_all('Requires-Dist', []))) == sorted(map(normalized_requirement, source['expected_requires_dist'])), 'built wheel dependency metadata differs from reviewed source plan')
            for member, expected_hash in source['expected_static_payload_sha256'].items():
                q.require(audit['payloads'].get(member, {}).get('sha256') == expected_hash, 'built static/license payload differs from reviewed source')
            for member in native:
                with archive.open(member) as native_stream:
                    header = native_stream.read(20)
                q.require(header[:6] == b'\x7fELF\x02\x01' and int.from_bytes(header[18:20], 'little') == 62, 'built native member is not ELF64 x86_64')
            wheelmeta = [n for n in names if n.count('/') == 1 and n.endswith('.dist-info/WHEEL')]
            q.require(len(wheelmeta) == 1, 'wheel WHEEL count mismatch')
            wheel = email.parser.BytesParser().parsebytes(archive.read(wheelmeta[0]))
            q.require((wheel['Root-Is-Purelib'] == 'false') == bool(native), 'built wheel platform/purelib mismatch')
        result.append({'name': source['name'], 'version': source['version'], 'filename': path.name,
                       'bytes': path.stat().st_size, 'sha256': q.sha256_file(path),
                       'payload_sha256': audit['payload_sha256']})
    return result


def validate_root_policy(files, policy):
    versions = {x['name'].lower().replace('_', '-'): x['version'] for x in files}
    for name, version in policy['exact_versions'].items():
        q.require(versions.get(name) == version, 'recipe/root version constraint failed: ' + name)
    q.require(policy['target_python'] == '3.12.14' and policy['target_glibc_maximum'] == '2.36', 'container target policy mismatch')


def validate_recipe(inventory, source_bytes):
    expected_hash = 'd174a8a84b5613060d03d55507543e31b9e3726dc976160c62f47fecf101bd04'
    expected_url = 'https://raw.githubusercontent.com/QwenAudio/SenseVoice/586bbdb2b052f4fe19322581825228b6810a9fb3/requirements.txt'
    q.require(hashlib.sha256(source_bytes).hexdigest() == expected_hash and len(source_bytes) == 273, 'model recipe source bytes changed')
    recipe = inventory.get('model_recipe')
    q.require(isinstance(recipe, dict), 'model recipe evidence missing')
    source = recipe.get('source', {})
    q.require(source.get('sha256') == expected_hash and source.get('source_url') == expected_url and
              source.get('source_revision') == '586bbdb2b052f4fe19322581825228b6810a9fb3', 'model recipe source identity missing/changed')
    q.require(recipe.get('applied_numeric_constraints') == ['torch>=2.12.1', 'numpy<=1.26.4'], 'model numerical constraints missing/changed')
    source_requirements = [line.strip() for line in source_bytes.decode().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    q.require(recipe.get('all_default_requirements') == source_requirements, 'model recipe requirements differ from fixed source')
    q.require(inventory.get('runtime_extras', {}).get('funasr') == ['knf'], 'locked FunASR frontend changed')


def validate_timing_policy(limits):
    keys = ('wall_seconds', 'qualification_wall_seconds', 'github_job_timeout_minutes',
            'job_cleanup_reserve_seconds')
    q.require(all(type(limits.get(key)) is int and limits[key] > 0 for key in keys),
              'timing limits must be positive integer bounds')
    q.require(limits['github_job_timeout_minutes'] <= 360, 'GitHub hosted job exceeds six-hour platform limit')
    q.require(limits['job_cleanup_reserve_seconds'] >= 600, 'job cleanup reserve is insufficient')
    q.require(limits['qualification_wall_seconds'] + limits['job_cleanup_reserve_seconds'] ==
              limits['github_job_timeout_minutes'] * 60, 'qualification and job time budgets disagree')
    q.require(limits['wall_seconds'] <= limits['qualification_wall_seconds'],
              'container deadline exceeds whole qualification deadline')


def validate_admission(root):
    root = Path(root)
    admission = q.load_json(root / 'locks/admission.json')
    q.require(admission['execution_enabled'] is True, 'execution admission is disabled')
    validate_timing_policy(admission['limits'])
    invpath = root / 'locks/inventory.json'
    q.require(q.sha256_file(invpath) == admission['required_inventory_sha256'], 'inventory was not restored byte-identically')
    inventory = q.load_json(invpath)
    files = q.validate_inventory(inventory, admission['expected_input_files'], admission['expected_input_bytes'])
    validate_recipe(inventory, (root / 'locks/sensevoice-recipe-requirements.txt').read_bytes())
    validate_root_policy(files, admission['runtime_root_policy'])
    for filename, key in (('container-target-validation.json', 'target_validation_sha256'), ('build-tools.json', 'build_tools_sha256')):
        q.require(q.sha256_file(root / 'locks' / filename) == admission[key], 'metadata validation identity mismatch')
    target_validation = q.load_json(root / 'locks/container-target-validation.json')
    q.require(target_validation['errors'] == [] and target_validation['packages_checked'] == len(files), 'container target metadata incomplete')
    plan = q.load_json(root / 'locks/build-plan.json')
    q.require(plan.get('independently_reviewed') is True and len(plan['sources']) == 5 and plan['required_native_imports'], 'source/payload build plan not reviewed')
    q.require(plan.get('source_date_epoch') == 1704067200 and plan.get('two_builds_required') is True and plan.get('original_sources_modified') is False, 'source reproducibility policy changed')
    q.require(len(plan['runtime_names']) == admission['expected_runtime_packages'] and len(set(plan['runtime_names'])) == len(plan['runtime_names']), 'runtime closure count mismatch')
    q.validate_image_metadata(admission['image'], (root / 'locks/image.manifest.json').read_bytes(), (root / 'locks/image.config.json').read_bytes())
    return admission, files, plan


def validate_pulled_image(observed, image_policy, manifest):
    digest = image_policy['manifest_digest']
    accepted = {name + '@' + digest for name in ('python', 'library/python', 'docker.io/library/python')}
    q.require(observed.get('Os') == 'linux' and observed.get('Architecture') == 'amd64' and
              type(observed.get('Size')) is int and observed['Size'] <= image_policy['maximum_expanded_bytes'] and
              bool(accepted.intersection(observed.get('RepoDigests', []))), 'pulled image identity/expanded budget mismatch')
    q.require(observed.get('Id') == manifest['config']['digest'], 'pulled image ID differs from reviewed config digest')
    return observed['Id']


def _qualify_admitted(root, destination):
    """Complete admitted sequence; deliberately not connected to CLI/workflow yet."""
    root, destination = Path(root).resolve(strict=True), Path(destination)
    admission, files, plan = validate_admission(root)
    q.require(not destination.exists() and not destination.is_symlink(), 'destination must be new')
    destination.mkdir(mode=0o700)
    limits = admission['limits']
    overall_deadline = time.monotonic() + limits['qualification_wall_seconds']
    q.require(shutil.disk_usage(destination).free >= limits['minimum_free_disk_bytes'], 'insufficient fresh disk space')
    daemon = q.docker_preflight()
    q.write_receipt(destination / 'docker-preflight.json', daemon)
    docker = daemon['docker']
    image = admission['image']['repository'] + '@' + admission['image']['manifest_digest']
    prefix = [docker, '--host=unix:///var/run/docker.sock']
    inputs = destination / 'inputs'
    inputs.mkdir(mode=0o755)
    artifacts = inputs / 'artifacts'
    artifacts.mkdir(mode=0o755)
    for name in ('container_stage.py',):
        shutil.copyfile(root / name, inputs / name)
    for name in ('admission.json', 'inventory.json', 'build-plan.json'):
        shutil.copyfile(root / 'locks' / name, inputs / name)
    # Pull is separate and explicit, never implicit in create/run. Layer bytes are
    # manifest-budgeted; Docker's transport is not claimed as a hard network quota.
    command_output(q.bounded_command(prefix + ['pull', '--platform=linux/amd64', image], timeout=600))
    observed = json.loads(command_output(q.bounded_command(prefix + ['image', 'inspect', image])))[0]
    manifest = json.loads((root / 'locks/image.manifest.json').read_bytes())
    image_id = validate_pulled_image(observed, admission['image'], manifest)
    receipts = []

    def execute(which):
        name = 'kws-asr-' + uuid.uuid4().hex[:20] + '-' + which
        remaining = int(overall_deadline - time.monotonic())
        q.require(remaining > 0, 'qualification wall deadline exceeded')
        stage_limits = dict(limits, wall_seconds=min(limits['wall_seconds'], remaining))
        record = stage(docker, image, image_id, name, inputs, destination / which, stage_limits, which)
        receipts.append(record)
        q.require(record['status'] == 'pass' and record.get('container_removed'), 'container stage failed: ' + which)
        return record

    # Tiny stdlib containment probe comes before all package downloads/builds.
    execute('preflight')
    expanded, members = 0, 0
    download_deadline = min(time.monotonic() + 900, overall_deadline)
    for item in files:
        path = artifacts / item['filename']
        q.download_one(item, path, deadline=download_deadline)
        audit = q.inspect_archive(path, limits['archive_expanded_bytes'] - expanded, limits['archive_members'] - members, plan['reviewed_pth_sha256'])
        expanded += audit['expanded_bytes']
        members += audit['members']
        if item['kind'] == 'sdist':
            reviewed = next(x for x in plan['sources'] if x['name'] == item['name'])
            q.require(audit['payload_sha256'] == reviewed['source_payload_manifest_sha256'], 'source member manifest differs from reviewed payload')
        if item['kind'] == 'wheel':
            metadata = [v for n, v in audit['payloads'].items() if n.count('/') == 1 and n.endswith('.dist-info/METADATA')]
            q.require(len(metadata) == 1 and metadata[0]['sha256'] == item['metadata_sha256'] and metadata[0]['bytes'] == item['metadata_bytes'], 'wheel body METADATA differs from reviewed PEP658 identity')
    q.require(expanded + limits['runtime_install_headroom_bytes'] <= limits['work_disk_bytes'], 'actual expanded payload leaves insufficient install scratch space')
    name_to_file = {x['name']: x for x in files}
    (inputs / 'build-requirements.txt').write_text(requirements([name_to_file[n] for n in plan['build_tool_names']]))
    execute('build-a')
    a = validate_built_wheels(destination / 'build-a', plan, limits)
    execute('build-b')
    b = validate_built_wheels(destination / 'build-b', plan, limits)
    q.require(a == b, 'two independent source builds are not byte-identical')
    for item in a:
        shutil.copyfile(destination / 'build-a' / item['filename'], artifacts / item['filename'])
        name_to_file[item['name']] = item
    runtime = [dict(name_to_file[name]) for name in plan['runtime_names']]
    for item in runtime:
        item['install_extras'] = admission['runtime_root_policy']['install_extras'].get(item['name'], [])
    q.require(all(x['filename'].endswith('.whl') for x in runtime), 'runtime contains unbuilt source')
    (inputs / 'runtime-requirements.txt').write_text(requirements(runtime))
    (inputs / 'runtime-versions.json').write_text(json.dumps({x['name']: x['version'] for x in runtime}, sort_keys=True))
    execute('runtime')
    q.require(time.monotonic() < overall_deadline, 'qualification wall deadline exceeded')
    result = {'schema': q.SCHEMA, 'status': 'pass', 'evidence_class': 'hosted_no_model_dependency_cpu_qualification',
              'docker_preflight': daemon, 'image_id': image_id, 'image_expanded_bytes': observed['Size'], 'artifact_expanded_bytes': expanded,
              'dependency_input_bytes': sum(x['bytes'] for x in files), 'stages': receipts,
              'recipe_sha256': {name: q.sha256_file(root / name) for name in
                  ('qualify.py', 'controller.py', 'container_stage.py', 'locks/admission.json', 'locks/inventory.json',
                   'locks/build-plan.json', 'locks/image.manifest.json', 'locks/image.config.json',
                   'locks/container-target-validation.json', 'locks/build-tools.json', 'locks/sensevoice-recipe-requirements.txt')},
              'target_speech_model_weights_loaded': 0, 'user_audio_loaded': 0, 'interpretation': 'Does not establish ASR model compatibility, quality, latency, or KWS device performance.'}
    q.write_receipt(destination / 'qualification.json', result)
    return result


def qualify(root, destination):
    before = q.host_runtime_snapshot()
    old_term = signal.getsignal(signal.SIGTERM)
    def terminate(_signal, _frame):
        raise InterruptedError('qualification received SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    try:
        return _qualify_admitted(root, destination)
    except BaseException as exc:
        result = {'schema': q.SCHEMA, 'status': 'failed', 'evidence_class': 'hosted_no_model_dependency_cpu_qualification',
                  'error': type(exc).__name__ + ': ' + str(exc), 'target_speech_model_weights_loaded': 0, 'user_audio_loaded': 0}
        destination = Path(destination)
        if destination.is_dir() and not (destination / 'failure.json').exists():
            q.write_receipt(destination / 'failure.json', result)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        after = q.host_runtime_snapshot()
        unchanged = before == after
        destination = Path(destination)
        if destination.is_dir():
            q.write_receipt(destination / 'host-runtime-identity.json', {
                'scope': 'parent Python executable and system package metadata; excludes expected Docker image/container state changes',
                'before': before, 'after': after, 'unchanged': unchanged})
        q.require(unchanged, 'host Python/package metadata changed')
