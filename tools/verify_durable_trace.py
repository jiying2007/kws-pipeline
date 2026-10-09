#!/usr/bin/env python3
"""Offline PR450 pointer regression checks, not an archive restoration or replay.

Pins were cross-checked against kws-data 528cdc880256ad417fcb839cdeba89ba59c2d331.
They intentionally identify this historical artifact, not every future source.
CI reads two small local JSON files only; it neither fetches nor executes evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = 'research/diagnostic-readiness-2026-10-09'
MAX_JSON_BYTES = 65536
SOURCE_REPOSITORY = 'jiying2007/kws-data'
SOURCE_COMMIT = '528cdc880256ad417fcb839cdeba89ba59c2d331'
SOURCE_PATH = 'research/2026-10-09-pr450-saved-trace-retention'
MEMBER_IDENTITIES_SHA256 = '5d1d3929a5c346a58c25bf1ef9a0c811083697fd711e77626597cf058af14d77'
ORIGINAL_ARTIFACT = {'run_id': 36689641752,
 'artifact_id': 11085476672,
 'bytes': 23590873,
 'sha256': 'dafaa468c0d9d6ffb323dca2845df74128ea128f0fc3870ce2d4224fcfe26672',
 'members': 44,
 'kwtr_files': 13,
 'synthetic_wav_files': 6}
WRAPPER = {'archive_bytes': 23591115,
 'archive_sha256': '8a3a6829fb01ccf3c04e1a69fb1b567f51721d794210fd7dd13c50f59c3ff2b9',
 'members': 1,
 'parts': [{'bytes': 8388608,
            'path': 'objects.zip.part-000',
            'sha256': '34398aad4e75a26ed1d378ef8c4db1fc1eb93ec6a4f9546da71797a1e6007a12'},
           {'bytes': 8388608,
            'path': 'objects.zip.part-001',
            'sha256': '56041dbde3c12eb78d2ff2eaf95836abaa0a42d2039a52ff6475a11df8ca099e'},
           {'bytes': 6813899,
            'path': 'objects.zip.part-002',
            'sha256': 'aa87f82fb018735e011389a08c7c990c286cbba37caa8aaa34abf8746c59d7cc'}],
 'schema_version': 1,
 'unique_objects': 1}
VERIFICATION_INPUTS = [{'path': 'research/2026-10-09-pr450-saved-trace-retention/ARCHIVE.json',
  'bytes': 673,
  'sha256': '7d319b89141346bfdef82a5dac0fd6938c0b8f739f73778c6643077eaecc82f6'},
 {'path': 'research/2026-10-09-pr450-saved-trace-retention/CATALOG.json',
  'bytes': 492,
  'sha256': '0a1e3574f2627631cc7efa6a2f6c8a6debda5a469772f39776f5092b7f55c245'},
 {'path': 'research/2026-10-09-pr450-saved-trace-retention/SOURCE-MEMBERS.json',
  'bytes': 9492,
  'sha256': '7f1ffaf86890fad4a986ab0c4ff84775fb5871c7352240de4ba4f6adc1e06c93'},
 {'path': 'research/2026-10-09-pr450-saved-trace-retention/verify.py',
  'bytes': 2799,
  'sha256': 'a184454f7f1bec53c8128e77975e436577937a666ec92c0fe4d59d0e544a515f'}]
SCOPE = {'new_model_calls': 0,
 'new_replays': 0,
 'new_tts_asr_calls': 0,
 'new_acoustic_qualification': False,
 'historical_initial_state_equivalence_established': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    # JSON serialization also distinguishes bool from int (True must not equal 1).
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    with path.open('rb') as source:
        raw = source.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, 'JSON input exceeds byte cap')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object,
                      parse_constant=lambda _: require(False, 'non-finite JSON number'))


def verify(pointer, readiness):
    require(isinstance(pointer, dict) and isinstance(readiness, dict), 'JSON object required')
    require(pointer.get('schema') == 'kws-pr450-durable-evidence-pointer-v1', 'pointer schema drift')
    require(pointer.get('source_repository') == SOURCE_REPOSITORY, 'source repository drift')
    require(pointer.get('source_commit') == SOURCE_COMMIT, 'fixed source commit drift')
    require(pointer.get('source_path') == SOURCE_PATH, 'source path drift')
    require(pointer.get('source_url') == 'https://github.com/' + SOURCE_REPOSITORY + '/tree/' +
            SOURCE_COMMIT + '/' + SOURCE_PATH, 'fixed source URL drift')
    require(canonical(pointer.get('original_artifact')) == canonical(ORIGINAL_ARTIFACT),
            'original ZIP identity drift')
    wrapper = pointer.get('wrapper')
    require(isinstance(wrapper, dict), 'wrapper object required')
    parts = wrapper.get('parts')
    require(isinstance(parts, list) and len(parts) == 3, 'exactly three wrapper parts required')
    require(all(isinstance(part, dict) and type(part.get('bytes')) is int and
                part['bytes'] > 0 for part in parts), 'invalid wrapper part byte count')
    require(sum(part['bytes'] for part in parts) == wrapper.get('archive_bytes'),
            'wrapper part byte sum mismatch')
    require(canonical(wrapper) == canonical(WRAPPER), 'wrapper ZIP or part identity drift')
    require(wrapper['archive_sha256'] != pointer['original_artifact']['sha256'],
            'wrapper and original ZIP identities are distinct')
    require(canonical(pointer.get('verification_inputs')) == canonical(VERIFICATION_INPUTS),
            'verification input identity drift')
    require(canonical(pointer.get('scope')) == canonical(SCOPE), 'saved-only scope drift')
    require(readiness.get('schema') == 'kws-diagnostic-readiness-public-summary-v1',
            'readiness schema drift')
    require(readiness.get('execution_ready') is False, 'execution must remain unready')
    require(readiness.get('D20_original_status') == 'FAIL' and
            readiness.get('D90_status') == 'NOT_RUN', 'historical outcome drift')
    for name in ('new_model_calls', 'new_operator_calls', 'new_decoder_replays'):
        require(type(readiness.get(name)) is int and readiness[name] == 0, 'new execution recorded')
    saved = readiness.get('shipping_saved_evidence')
    require(isinstance(saved, dict), 'saved evidence object required')
    for name in ('run_id', 'artifact_id'):
        require(type(saved.get(name)) is int and saved[name] == ORIGINAL_ARTIFACT[name],
                'readiness artifact identity drift')
    require(saved.get('artifact_sha256') == ORIGINAL_ARTIFACT['sha256'], 'readiness ZIP hash drift')
    members = saved.get('member_identities')
    require(isinstance(members, dict) and len(members) == ORIGINAL_ARTIFACT['members'],
            'exactly 44 original member identities required')
    require(sum(name.endswith('.kwtr') for name in members) == ORIGINAL_ARTIFACT['kwtr_files'] and
            sum(name.endswith('.wav') for name in members) == ORIGINAL_ARTIFACT['synthetic_wav_files'],
            'saved member kind counts drift')
    require(hashlib.sha256(canonical(members)).hexdigest() == MEMBER_IDENTITIES_SHA256,
            'saved member identity inventory drift')
    return {'status': 'PASS_DURABLE_POINTER_IDENTITIES_ONLY', 'original_members': len(members),
            'wrapper_parts': len(parts), 'archive_bytes_read': 0, 'restored_bytes_verified': False,
            'execution_ready': False, 'model_calls': 0, 'replay_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    directory = args.root / DIRECTORY
    result = verify(read_json(directory / 'DURABLE-TRACE.json'),
                    read_json(directory / 'PUBLIC-READINESS-SUMMARY.json'))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
