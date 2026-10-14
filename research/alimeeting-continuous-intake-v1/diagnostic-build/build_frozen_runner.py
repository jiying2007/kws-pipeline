#!/usr/bin/env python3
"""Build only the frozen kws_wav target and retain verifiable inputs. Never run it."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

BASE = Path('/workspace/scratch/6c2ef8b5a46e')
SNAPSHOT = BASE / 'kws-asr-registration-p4-20261014'
SOURCE = SNAPSHOT / 'candidate'
ROOT = BASE / 'kws-alimeeting-eval-prep-20261014'
OUT = ROOT / 'diagnostic-build'
BUILD = OUT / 'build'
TOOLCHAIN = ROOT / 'toolchain'
REVISION = '11b93abe92427c3ece2a01f9a497cdb74bab0062'
TREE = 'b8ae29e82097c38e92dbb1bb1cd4db10ade6cdc6'
MODEL_HASH = 'ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402'
PACK_HASH = '370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723'
MISSING_PATH = 'research/source-screen32-whisper-v1/evidence/run-38126849922/chunks/terminal.zip.part-000'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'sha256': sha(data), 'bytes': len(data)}


def write(name, data):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
    return record(path)


whole = json.loads((SNAPSHOT / 'WHOLE-TREE.json').read_bytes())
manifest = json.loads((SNAPSHOT / 'CANDIDATE-MANIFEST.json').read_bytes())
require(whole['tree'] == TREE == manifest['candidate_tree'], 'candidate tree mismatch')
for artifact in manifest['artifacts']:
    actual = record(SNAPSHOT / artifact['path'])
    require(actual['sha256'] == artifact['sha256'] and actual['bytes'] == artifact['bytes'],
            'candidate artifact mismatch: ' + artifact['path'])
merge = json.loads((SNAPSHOT / 'root-publication/merge.wrapper.json').read_bytes())
require(merge['structuredContent']['merged'] is True and
        merge['structuredContent']['sha'] == REVISION, 'retained merge receipt mismatch')
declared = {item['path']: item for item in whole['files']}
require(len(declared) == len(whole['files']), 'duplicate candidate leaves')


def verify_tree():
    verified = []
    missing = []
    for item in whole['files']:
        path = SOURCE / item['path']
        require(path.resolve().is_relative_to(SOURCE.resolve()), 'unsafe leaf path')
        if not path.exists():
            require(item['path'] == MISSING_PATH and item['sha256'] is None,
                    'unexpected absent source leaf: ' + item['path'])
            missing.append(item)
            continue
        require(path.is_file() and not path.is_symlink(), 'unexpected source type')
        data = path.read_bytes()
        git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        require(len(data) == item['bytes'] and sha(data) == item['sha256'] and
                git_sha == item['git_blob_sha1'], 'source mismatch: ' + item['path'])
        verified.append(dict(item))
    require(len(missing) == 1 and len(verified) == manifest['local_candidate_body_count'],
            'unexpected source coverage')
    # The unavailable immutable research archive leaf is represented by its
    # already-retained Git blob ID. Its body is NOT claimed as verified.
    nested = {}
    for item in whole['files']:
        parts = item['path'].split('/')
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = (item['mode'], item['git_blob_sha1'])

    def git_tree(node):
        body = bytearray()
        for name, value in sorted(node.items(), key=lambda kv: (kv[0] + ('/' if isinstance(kv[1], dict) else '')).encode()):
            if isinstance(value, dict):
                mode, digest = '40000', git_tree(value)
            else:
                mode, digest = value
            body.extend(mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(digest))
        return hashlib.sha1(b'tree ' + str(len(body)).encode() + b'\0' + body).hexdigest()

    computed = git_tree(nested)
    require(computed == TREE, 'computed Git tree mismatch')
    expected = set(declared) - {MISSING_PATH}
    actual = {str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}
    return {'schema': 'frozen-source-verification-v1', 'source_root': str(SOURCE),
            'declared_tree': TREE, 'recomputed_tree_using_retained_missing_blob_id': computed,
            'available_leaf_count': len(verified), 'verified_leaves': verified,
            'missing_leaves': missing, 'full_tree_body_verification': False,
            'all_available_declared_leaf_hashes_verified': True,
            'undeclared_local_files': sorted(actual - expected)}


before = verify_tree()
before_ref = write('source-verification-before.json', before)
print('Verified', before['available_leaf_count'], 'available source leaves; one inherited archive body absent.', flush=True)
install = json.loads((TOOLCHAIN / 'cmake-install-receipt.json').read_bytes())
CMAKE = install['cmake_path']
PYTHON = sys.executable
for path in [TOOLCHAIN / 'home', TOOLCHAIN / 'tmp']:
    path.mkdir(exist_ok=True)
env = {'PATH': str(Path(PYTHON).parent) + ':/usr/bin:/bin', 'HOME': str(TOOLCHAIN / 'home'),
       'TMPDIR': str(TOOLCHAIN / 'tmp'), 'LANG': 'C', 'LC_ALL': 'C',
       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}


def capture(argv):
    return subprocess.check_output(argv, env=env, cwd=OUT, text=True).strip()


def command_text(argv):
    return 'env -i ' + ' '.join(shlex.quote(k + '=' + v) for k, v in env.items()) + ' ' + shlex.join(argv)


configure = [CMAKE, '-S', str(SOURCE), '-B', str(BUILD), '-G', 'Unix Makefiles',
             '-DCMAKE_BUILD_TYPE=Release', '-DKWS_STRICT=ON',
             '-DKWS_SOURCE_REVISION=' + REVISION, '-DCMAKE_C_COMPILER=/usr/bin/gcc',
             '-DCMAKE_MAKE_PROGRAM=/usr/bin/make', '-DPython3_EXECUTABLE=' + PYTHON,
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DKWS_BUILD_TESTS=OFF',
             '-DKWS_BUILD_BENCH=OFF', '-DKWS_BUILD_TOOLS=ON', '-DKWS_BUILD_FUZZ=OFF',
             '-DCMAKE_INSTALL_PREFIX=' + str(BUILD / 'install')]
build = [CMAKE, '--build', str(BUILD), '--target', 'kws_wav', '--parallel', '2', '--verbose']
commands = {'configure_argv': configure, 'build_argv': build, 'environment': env,
            'working_directory': str(OUT), 'configure_command': command_text(configure),
            'build_command': command_text(build)}
write('commands.json', commands)
(OUT / 'build-commands.sh').write_text('#!/bin/sh\nset -eu\ncd ' + shlex.quote(str(OUT)) + '\n' +
    commands['configure_command'] + '\n' + commands['build_command'] + '\n')
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name, argv in [('configure', configure), ('build', build)]:
    prior_log = OUT / (name + '.log')
    first_log = OUT / ('initial-' + name + '.log')
    if prior_log.is_file() and not first_log.exists():
        first_log.write_bytes(prior_log.read_bytes())
    with (OUT / (name + '.log')).open('w') as stream:
        result = subprocess.run(argv, env=env, cwd=OUT, stdout=stream, stderr=subprocess.STDOUT)
    (OUT / (name + '.exitcode')).write_text(str(result.returncode) + '\n')
    require(result.returncode == 0, name + ' failed; see ' + str(OUT / (name + '.log')))
    print(name + ' passed', flush=True)
after = verify_tree()
require(before == after, 'source snapshot changed during build')
after_ref = write('source-verification-after.json', after)
model = SOURCE / 'models/registry/model-749187ec1d66/xiaowo-model.kwm'
pack = SOURCE / 'models/registry/model-749187ec1d66/xiaowo-keywords.kwk'
require(record(model)['sha256'] == MODEL_HASH and record(pack)['sha256'] == PACK_HASH,
        'frozen model or pack mismatch')

dependency_paths = set()
dependency_files = sorted(BUILD.rglob('*.o.d'))
for path in dependency_files:
    data = path.read_text().replace('\\\n', ' ')
    for token in shlex.split(data.split(':', 1)[1]):
        p = Path(token)
        if p.is_file():
            dependency_paths.add(p.resolve())
expected_translation_units = {
    'src/kws.c', 'src/build_info.c', 'src/frontend.c', 'src/model.c',
    'src/decoder.c', 'src/keyword_pack.c', 'tools/tool_io.c',
    'tools/sha256.c', 'tools/kws_wav.c'}
actual_translation_units = {str(p.relative_to(SOURCE)) for p in dependency_paths
                            if p.is_relative_to(SOURCE) and p.suffix == '.c'}
require(actual_translation_units == expected_translation_units and
        len(dependency_files) == len(expected_translation_units),
        'unexpected built translation units')
build_contract_paths = [SOURCE / 'CMakeLists.txt', SOURCE / 'configs/parameter-contract.json',
                        SOURCE / 'tools/gen_parameter_limits.py'] + sorted((SOURCE / 'cmake').glob('*'))
source_paths = sorted({p for p in dependency_paths if p.is_relative_to(SOURCE)} |
                      {p for p in build_contract_paths if p.is_file()})
source_inputs = []
for path in source_paths:
    rel = str(path.relative_to(SOURCE))
    require(rel in declared and record(path)['sha256'] == declared[rel]['sha256'],
            'unverified build input: ' + rel)
    source_inputs.append({**record(path), 'repository_path': rel,
                          'git_blob_sha1': declared[rel]['git_blob_sha1']})
headers = [record(p) for p in sorted((BUILD / 'generated').rglob('*.h'))]
header_text = '\n'.join(Path(h['path']).read_text() for h in headers)
require('#define KWS_BUILD_SOURCE_REVISION "' + REVISION + '"' in header_text,
        'generated source revision mismatch')
require('#define KWS_BUILD_TYPE "Release"' in header_text, 'not Release')
config_digest = re.search(r'#define KWS_BUILD_CONFIG_DIGEST "([0-9a-f]{64})"', header_text).group(1)
identity_object = BUILD / 'CMakeFiles/kws_pipeline.dir/src/build_info.c.o'
identity_bytes = identity_object.read_bytes()
require(REVISION.encode() in identity_bytes and config_digest.encode() in identity_bytes,
        'compiled build identity mismatch')
runner_bytes = (BUILD / 'kws_wav').read_bytes()
compile_entries = json.loads((BUILD / 'compile_commands.json').read_bytes())
built_entries = [entry for entry in compile_entries
                 if str(Path(entry['file']).relative_to(SOURCE)) in expected_translation_units]
required_flags = {'-O3', '-DNDEBUG', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                  '-Wconversion', '-Wshadow', '-Wcast-qual'}
require(len(built_entries) == 9 and all(required_flags.issubset(set(shlex.split(entry['command'])))
                                      for entry in built_entries),
        'built target compile flags are not strict Release')
gcc = Path('/usr/bin/gcc').resolve()
compiler = {**record(gcc), 'invoked_path': '/usr/bin/gcc', 'version': capture(['/usr/bin/gcc', '--version']),
            'version_number': capture(['/usr/bin/gcc', '-dumpfullversion']),
            'target_triple': capture(['/usr/bin/gcc', '-dumpmachine'])}
compiler_components = []
for arg in ['-print-prog-name=cc1', '-print-prog-name=as', '-print-prog-name=ld', '-print-prog-name=collect2']:
    value = capture(['/usr/bin/gcc', arg])
    candidate = Path(value) if '/' in value else Path('/usr/bin') / value
    if candidate.is_file():
        compiler_components.append({**record(candidate.resolve()), 'compiler_query': arg})
receipt = {
    'schema': 'kws-frozen-diagnostic-build-receipt-v1', 'status': 'built-not-executed',
    'started_at_utc': start, 'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_revision': REVISION, 'source_tree': TREE, 'tree': TREE, 'source_root': str(SOURCE),
    'source_revision_basis': 'retained successful merge response plus authorized fixed merge SHA; generated and compiled override is a label, not an independent Git checkout assertion',
    'source_tree_basis': 'all 2382 available manifest leaves verified by size, SHA-256 and Git blob SHA-1; exact tree recomputed with one inherited unavailable research archive blob ID',
    'source_unchanged': True, 'model_executed': False, 'audio_executed': False,
    'runner': record(BUILD / 'kws_wav'), 'model': record(model), 'keyword_pack': record(pack),
    'build_type': 'Release', 'strict': True, 'target': 'kws_wav',
    'build_configuration_digest': config_digest, 'compiler': compiler,
    'build_identity_embedding': {
        'generated_headers_and_compiled_identity_object_verified': True,
        'source_revision_literal_in_runner': REVISION.encode() in runner_bytes,
        'config_digest_literal_in_runner': config_digest.encode() in runner_bytes,
        'note': 'Unmodified kws_wav does not call kws_build_info(); the static linker omits the unreferenced build_info object from the final executable. Exact runner provenance is supplied by this build receipt and executable hash.',
        'compiled_identity_object': record(identity_object),
        'core_archive': record(BUILD / 'libkws_pipeline.a'),
        'tool_io_archive': record(BUILD / 'libkws_tool_io.a'),
    },
    'compiler_components': compiler_components, 'make': {**record('/usr/bin/make'), 'version': capture(['/usr/bin/make', '--version'])},
    'cmake': {**record(CMAKE), 'version': capture([CMAKE, '--version']),
              'install_receipt': record(TOOLCHAIN / 'cmake-install-receipt.json')},
    'python': {**record(Path(PYTHON).resolve()), 'invoked_path': PYTHON, 'version': capture([PYTHON, '--version'])},
    **commands,
    'configure_exit_code': 0, 'build_exit_code': 0,
    'source_inputs': source_inputs, 'built_translation_unit_count': len(dependency_files),
    'built_translation_units': sorted(actual_translation_units),
    'built_translation_units_strict_release_verified': True,
    'compiler_dependency_files': [record(p) for p in dependency_files],
    'system_header_inputs': [record(p) for p in sorted(dependency_paths) if not p.is_relative_to(SOURCE) and not p.is_relative_to(BUILD)],
    'generated_headers': headers, 'compile_commands': record(BUILD / 'compile_commands.json'),
    'compile_commands_scope': 'CMake configured tools; only kws_wav and required libraries were built',
    'cmake_cache': record(BUILD / 'CMakeCache.txt'),
    'link_command': record(BUILD / 'CMakeFiles/kws_wav.dir/link.txt'),
    'logs': [record(p) for p in [OUT / 'configure.log', OUT / 'build.log',
                               OUT / 'initial-configure.log', OUT / 'initial-build.log'] if p.is_file()],
    'verification_before': before_ref, 'verification_after': after_ref,
    'manifest_inputs': [record(SNAPSHOT / n) for n in ['WHOLE-TREE.json', 'CANDIDATE-MANIFEST.json', 'SOURCE-ALLOWLIST.json', 'GIT-NATIVE-CROSSCHECK.json', 'root-publication/merge.wrapper.json', 'root-publication/tree.wrapper.json']],
    'missing_unrelated_archive_leaf': manifest['inherited_leaf'],
    'tests_not_run': 'Scope is dedicated build only. No kws_wav execution, CTest, model, audio, tuning, training or benchmark invocation.',
    'build_script': record(Path(__file__)),
}
write('build-receipt.json', receipt)
print(json.dumps({'receipt': str(OUT / 'build-receipt.json'), 'runner': receipt['runner'],
                  'source_inputs': len(source_inputs), 'generated_headers': headers}, indent=2), flush=True)
