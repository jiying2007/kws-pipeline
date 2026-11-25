#!/usr/bin/env python3
"""Prepare exact B27 delivery metadata without selecting, packaging or installing.

The manifest mode copies only a reviewed completion report and verifier into a
new review directory; it never executes that verifier. The entry mode requires
an already reviewed, committed pipeline selection and captures its ten exact
Git files. Neither mode changes this checkout or invokes a model or compiler.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location('b27_delivery_binding_authority', HERE / 'once_b27.py')
runner = importlib.util.module_from_spec(definition)
definition.loader.exec_module(runner)
original, d = runner.original, runner.d


def fresh(path):
    path = Path(path)
    d.need(path.is_absolute() and path == path.resolve() and path.parent.is_dir() and not path.exists()
           and all(not part.is_symlink() for part in path.parents), 'fresh absolute review directory')
    return path


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)


def manifest(report_path, verifier_path, output):
    output = fresh(output)
    current = d.selection()
    d.need(current['selected'] is None and current['manifest'] is None,
           'prepare only while the actual pipeline selection remains unbound')
    saved = {'completion.json': d.read(Path(report_path), d.META_LIMIT),
             'readback.py': d.read(Path(verifier_path), d.META_LIMIT)}
    report = d.document(saved['completion.json'])
    value = dict(schema='heym43-software-delivery-v1',
        **{key: report[key] for key in ('candidate_id', 'pipeline_commit', 'candidate_pack', 'files',
                                       'engineering', 'quality_locks', 'scope')},
        readback=dict(report=original.pin(saved['completion.json']), verifier=original.pin(saved['readback.py'])))
    d.manifest_contract(value)
    runner.validate_reviewed_readback(value, saved)
    raw = d.encoded(value); digest = d.sha(raw)
    chosen = dict(current, selected='B27', manifest=dict(path='manifests/' + digest + '.json', **original.pin(raw)))
    output.mkdir()
    write(output / 'manifests' / (digest + '.json'), raw)
    write(output / 'readbacks' / value['readback']['report']['sha256'] / 'completion.json', saved['completion.json'])
    write(output / 'readbacks' / (value['readback']['verifier']['sha256'] + '.py'), saved['readback.py'])
    write(output / 'selection.json', d.encoded(chosen))
    result = dict(schema='english-v6-b27-delivery-review-draft-v1', status='READY_FOR_REVIEW_ONLY',
        manifest_sha256=digest, actual_selection_changed=False, saved_verifier_executed=False,
        package_created=False, installed=False, new_model_calls=0, compiler_calls=0)
    write(output / 'BINDING.json', d.encoded(result))
    return result


def entry(commit, output):
    output = fresh(output)
    d.need(type(commit) is str and d.re.fullmatch(r'[0-9a-f]{40}', commit), 'one exact reviewed pipeline commit')
    value, raw = d.trusted_manifest()
    runner.reviewed_readback(value)
    d.need(original.git(runner.ROOT, 'rev-parse', 'HEAD').decode().strip() == commit
           and not original.git(runner.ROOT, 'status', '--porcelain', '--untracked-files=no'),
           'complete reviewed selection is already committed and current tracked bytes are clean')
    names = original.expected_pipeline_files(value, raw)
    entries = {}
    for line in original.git(runner.ROOT, 'ls-tree', '-z', 'HEAD', '--', *sorted(names)).split(b'\0'):
        if not line:
            continue
        fields, name = line.split(b'\t', 1)
        mode, kind, digest = fields.decode().split()
        d.need(kind == 'blob' and mode in ('100644', '100755'), 'one exact regular reviewed Git file')
        entries[name.decode()] = dict(mode=mode, git_blob_sha1=digest)
    d.need(set(entries) == names and len(names) == 10, 'complete original ten-file delivery producer')
    files = {}
    for name in sorted(names):
        body = d.read(runner.ROOT / name, d.META_LIMIT)
        blob = original.hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        d.need(blob == entries[name]['git_blob_sha1'], 'actual source still equals committed Git blob ' + name)
        files[name] = dict(entries[name], **original.pin(body))
    binding = dict(schema=runner.ENTRY_SCHEMA, enabled=True, pipeline_repository=original.PIPELINE_REPO,
        pipeline_commit=commit, manifest_sha256=d.sha(raw), pipeline_files=files)
    original.entry_contract(binding)
    original.pipeline_contract(binding, value, raw)
    output.mkdir(); write(output / 'engineering_delivery_inputs.json', d.encoded(binding))
    result = dict(schema='english-v6-b27-delivery-entry-review-draft-v1', status='READY_FOR_REVIEW_ONLY',
        pipeline_commit=commit, binding=original.pin(d.encoded(binding)), actual_data_entry_changed=False,
        new_model_calls=0, compiler_calls=0, package_created=False, installed=False)
    write(output / 'BINDING.json', d.encoded(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    item = modes.add_parser('manifest')
    for name in ('report', 'verifier', 'output'):
        item.add_argument('--' + name, required=True, type=Path)
    item = modes.add_parser('entry')
    item.add_argument('--commit', required=True); item.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = manifest(args.report, args.verifier, args.output) if args.mode == 'manifest' else entry(args.commit, args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
