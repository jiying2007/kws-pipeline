"""Read only train/calibration indices; never manufacture runner identities."""
import argparse
import json
import hashlib
import os
import stat
from pathlib import Path

FIELDS = ('voice_id', 'source_id', 'speaker_id', 'session_id', 'source_family_id')
UNKNOWN = frozenset(('', 'unknown', 'none', 'null', 'pending', 'n/a', 'na',
                     'unset', 'unspecified', 'missing', 'tbd', '-', '?'))


def known(v):
    if not isinstance(v, str):
        return False
    normalized = v.strip().casefold()
    return normalized not in UNKNOWN and not any(
        normalized.startswith(word + separator) for word in UNKNOWN if len(word) > 1
        for separator in ('_', '-', ':', ' '))


def check(selected):
    if set(selected) != {'train', 'calibration'}:
        raise ValueError('exact original train/calibration inputs required')
    coverage, identities = {}, {}
    for split, rows in selected.items():
        coverage[split] = {'rows': len(rows), 'missing': {f: 0 for f in FIELDS}}
        identities[split] = {f: set() for f in FIELDS}
        for row in rows:
            if row.get('split') != split:
                raise ValueError('original split mismatch')
            provenance = row.get('speech_like_provenance', {})
            for field in FIELDS:
                a, b = row.get(field), provenance.get(field)
                if known(a) and known(b) and a != b:
                    raise ValueError('conflicting original identity metadata')
                value = a if known(a) else b
                value = value.strip() if isinstance(value, str) else value
                if known(value):
                    identities[split][field].add(value)
                else:
                    coverage[split]['missing'][field] += 1
    overlap = {f: sorted(identities['train'][f] & identities['calibration'][f]) for f in FIELDS}
    overlap['source_namespace'] = sorted(
        (identities['train']['source_id'] | identities['train']['source_family_id']) &
        (identities['calibration']['source_id'] | identities['calibration']['source_family_id']))
    for split in selected:
        coverage[split]['unique_known'] = {field: len(values) for field, values in identities[split].items()}
    complete = all(selected.values()) and all(not count for v in coverage.values() for count in v['missing'].values())
    return {'schema': 'historical-base-selected-metadata-audit-v1', 'coverage': coverage, 'overlap': overlap,
            'identity_metadata_complete_and_disjoint': complete and not any(overlap.values()),
            'waveform_bytes_verified': False, 'label_admission_verified': False,
            'runner_ready': False,
            'runner_blockers': ['selection manifest is not supplied; no held-out test/qualification recycling',
                                'waveform and original source lineage must be verified',
                                'reviewed training admission must be verified or explicitly diagnostic'],
            'note': 'Missing speaker/session is unknown; a voice slot is not a person/session.'}


def selected_metadata_bytes(corpus_root):
    """Open only real selected directories/files, with no-follow at every hop.

    Directory-relative descriptors keep the boundary intact if a pathname is
    replaced after checking. No metadata payload is read before all four files
    have passed the boundary check.
    """
    paths = {}
    descriptors = []
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        corpus = Path(os.path.abspath(corpus_root))
        fd = os.open(corpus.anchor, directory_flags)
        descriptors.append(fd)
        for part in corpus.parts[1:]:
            fd = os.open(part, directory_flags, dir_fd=fd)
            descriptors.append(fd)
        bundle_fd = os.open('bundle', directory_flags, dir_fd=fd)
        descriptors.append(bundle_fd)
        for split in ('train', 'calibration'):
            split_fd = os.open(split, directory_flags, dir_fd=bundle_fd)
            descriptors.append(split_fd)
            split_path = corpus / 'bundle' / split
            # Structural containment is checked in addition to no-follow opens.
            if split_path.resolve(strict=True) != split_path:
                raise ValueError('metadata boundary: symlinked split directory')
            for name in ('dataset-index.jsonl', 'dataset-summary.json'):
                path = split_path / name
                if path.is_symlink() or path.resolve(strict=True).parent != split_path:
                    raise ValueError('metadata boundary: symlink or escaped metadata')
                file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=split_fd)
                descriptors.append(file_fd)
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise ValueError('metadata boundary: expected regular metadata file')
                paths[(split, name)] = file_fd
        payloads = {}
        for key, file_fd in paths.items():
            with os.fdopen(os.dup(file_fd), 'rb') as stream:
                payloads[key] = stream.read()
        return payloads
    except OSError as exc:
        raise ValueError('metadata boundary: missing, symlinked or invalid selected path') from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def audit_corpus(corpus_root):
    payloads = selected_metadata_bytes(corpus_root)
    selected, bindings = {}, {}
    for split in ('train', 'calibration'):
        index_bytes = payloads[(split, 'dataset-index.jsonl')]
        summary_bytes = payloads[(split, 'dataset-summary.json')]
        selected[split] = [json.loads(x) for x in index_bytes.decode('utf-8').splitlines() if x.strip()]
        summary = json.loads(summary_bytes)
        bindings[split] = {'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
                           'summary_sha256': hashlib.sha256(summary_bytes).hexdigest(),
                           'admission_mode': summary.get('admission_mode'),
                           'ctc_training_allowed': summary.get('ctc_training_allowed'),
                           'recordings_declared': summary.get('recordings')} 
    report = check(selected)
    report['bindings'] = bindings
    report['expected_recordings'] = {'train': 128, 'calibration': 64}
    report['expected_counts_match'] = all(len(selected[k]) == v for k, v in report['expected_recordings'].items())
    report['protected_splits_opened'] = []
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = audit_corpus(args.corpus_root)
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
