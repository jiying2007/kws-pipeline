#!/usr/bin/env python3
"""Measure existing public/synthetic audio only. No generation or inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from effect_dataset_split import inspect_wav, sha256_file, plan, write_outputs


def inventory(repo, far_root=None, k1_root=None, r5_root=None):
    rows, invalid = [], []
    def measure(path, root, root_id, meta):
        relative = path.relative_to(root).as_posix()
        row = {'record_id': root_id + ':' + relative, 'path': relative,
               'audio_root_id': root_id, **meta}
        try:
            frames, seconds, pcm = inspect_wav(path)
            row.update(file_sha256=sha256_file(path), pcm_sha256=pcm,
                       frames=frames, duration_s=seconds, audio_status='verified_pcm16')
        except ValueError:
            # Native float WAV is retained as source bytes, not silently converted.
            invalid.append({'path': relative, 'audio_root_id': root_id,
                            'file_sha256': sha256_file(path), 'reason': 'not_mono_16khz_pcm16'})
            return
        rows.append(row)
    base = repo / 'research/source-screen32-v1/evidence/run-38018029787'
    job = json.loads((base / 'blind/job.json').read_text())
    by_hash = {r['wav_sha256']: r['audio_id'] for r in job['clips']}
    raw = json.loads((base / 'asr/primary-raw.json').read_text())
    for path in sorted(repo.rglob('*.wav')):
        meta = {'development_only': True, 'source_collection': 'retained-repository-audio'}
        if base in path.parents:
            digest = sha256_file(path)
            audio_id = by_hash.get(digest)
            meta.update(source_collection='screen32-run-38018029787',
                        derivation_family_id=path.name.replace('.raw-float', '').removesuffix('.wav'))
            if audio_id:
                meta['asr_evidence'] = [{'family': family, **next(r for r in records if r['audio_id'] == audio_id)}
                                        for family, records in sorted(raw.items())]
                meta['asr_evidence_source'] = (base / 'asr/primary-raw.json').relative_to(repo).as_posix()
        measure(path, repo, 'repository', meta)
    frozen_path = repo / 'research/known-far-diagnostic-v1/original-domain-index.jsonl.gz'
    frozen = [json.loads(x) for x in gzip.open(frozen_path, 'rt')]
    evidence = {'frozen_domain_index_sha256': sha256_file(frozen_path),
                'historical_domain_rows': len(frozen), 'historical_audio_verified_here': False}
    if far_root:
        current = [json.loads(x) for x in (far_root / 'domain-index.jsonl').read_text().splitlines()]
        if len(current) != len(frozen):
            raise ValueError('historical FAR domain row count changed')
        metadata = {}
        for before, now in zip(frozen, current):
            if {k:v for k,v in before.items() if k not in ('path', 'source_path')} != {
                    k:v for k,v in now.items() if k not in ('path', 'source_path')}:
                raise ValueError('historical FAR domain metadata changed')
            for key, digest in (('path', 'wav_sha256'), ('source_path', 'source_wav_sha256')):
                if sha256_file(Path(now[key])) != before[digest]:
                    raise ValueError('historical FAR waveform hash changed')
            _, _, source_pcm = inspect_wav(Path(now['source_path']))
            if source_pcm != now['source_pcm_sha256']:
                raise ValueError('FAR original source PCM mismatch')
            shared = {'development_only': True, 'source_collection': 'known-far-run-38482297930',
                      'source_family_id': now['family_id'], 'kind': now['kind']}
            metadata[Path(now['path']).resolve()] = {**shared, 'source_pcm_sha256': source_pcm,
                                                   'source_path': Path(now['source_path']).relative_to(far_root).as_posix()}
            metadata[Path(now['source_path']).resolve()] = shared
        for path in sorted(far_root.rglob('*.wav')):
            measure(path, far_root, 'known_far_restored', metadata.get(path.resolve(), {
                'development_only': True, 'source_collection': 'known-far-restoration-extra'}))
        evidence['historical_audio_verified_here'] = True
        evidence['verified_domain_rows'] = len(current)
    if bool(k1_root) != bool(r5_root):
        raise ValueError('--k1-root and --r5-root must be supplied together')
    if k1_root:
        freeze = json.loads((r5_root / 'input-freeze.json').read_text())
        blind = {r['wav_sha256']: r['audio_id'] for r in freeze['actual_blind_job']['clips']}
        cells = {r['cell_id']: r for r in freeze['actual_new_cells']}
        terminal = json.loads((r5_root / 'terminal-outcomes.json').read_text())
        slots = {r['id']: r['status'] for r in terminal['slots']}
        if terminal['identity']['run_id'] != 38518609834:
            raise ValueError('expected original r5 run 38518609834')
        for path in sorted(k1_root.rglob('*.wav')):
            cell = cells[path.stem]
            meta = {'development_only': True, 'source_collection': 'k1-six-original-r5',
                    'derivation_family_id': path.stem,
                    'input_freeze_sha256': sha256_file(r5_root / 'input-freeze.json')}
            if path.parent.name == 'derived':
                frames, _, pcm = inspect_wav(path)
                if (frames, pcm, sha256_file(path)) != (cell['frames'], cell['pcm_sha256'], cell['wav_sha256']):
                    raise ValueError('k1 original audio differs from r5 frozen input')
                audio_id = blind[cell['wav_sha256']]
                meta['asr_evidence'] = []
                for family in ('qwen06', 'sensevoice', 'whisper'):
                    transcript = r5_root / 'output' / family / audio_id / 'raw-transcript.json'
                    record = json.loads(transcript.read_text())
                    meta['asr_evidence'].append({'family': family,
                        'status': slots[path.stem + '-' + family], 'raw_text': record['actual_text'],
                        'source_run_id': 38518609834, 'raw_transcript_sha256': sha256_file(transcript)})
            else:
                if sha256_file(path) != cell['raw_wav_sha256']:
                    raise ValueError('k1 native WAV differs from original frozen input')
            measure(path, k1_root, 'k1_original', meta)
    report = plan(rows)
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[row['pcm_sha256']].append(row['record_id'])
    pools = {}
    for seed in (1103, 3301):
        source = repo / f'research/known-far-diagnostic-v1/original-injections-{seed}.jsonl.gz'
        entries = [json.loads(x) for x in gzip.open(source, 'rt')]
        pools[seed] = {'entries': len(entries), 'hashes': {r['source_sha256'] for r in entries}}
    contract_path = repo / 'configs/training/product-speech-like-base-v1.json'
    contract = json.loads(contract_path.read_text())
    report['remote_declared_not_materialized'] = [{
        'source': contract_path.relative_to(repo).as_posix(), 'source_sha256': sha256_file(contract_path),
        'status': 'declared_public_release_not_materialized_by_this_inventory',
        **{key: contract[key] for key in ('release_tag', 'release_archive', 'release_archive_sha256',
            'recordings', 'voice_slots', 'source_sample_rate_hz', 'normalized_sample_rate_hz', 'splits')},
        'recovery_entrypoint': 'training/materialize_governed_product_base.sh',
        'train_scope': 'retain original train-only scope; never recycle calibration/test/qualification'}]
    provenance_path = repo / 'models/registry/model-749187ec1d66/xiaowo-model-provenance.json'
    provenance = json.loads(provenance_path.read_text())['training']['corpus_identity']['recordings']
    declared_pcm = {r['pcm_sha256'] for r in provenance}
    measured_pcm = {r['pcm_sha256'] for r in rows}
    report['historical_training_metadata'] = {
        'source': provenance_path.relative_to(repo).as_posix(), 'source_sha256': sha256_file(provenance_path),
        'declared_records': len(provenance), 'declared_unique_pcm': len(declared_pcm),
        'exact_pcm_overlap_with_local_measured': len(declared_pcm & measured_pcm),
        'status': 'retained_provenance_not_live_audio_reverification',
        'independent_acceptance_hours_added': 0}
    report['inventory'] = {'scope': 'repository WAV files plus explicitly supplied restored FAR tree; not a claim of every external corpus',
                           'valid_pcm16_files': len(rows), 'noncanonical_source_files': invalid,
                           'observed_audio_hours': sum(r['duration_s'] for r in rows) / 3600,
                           'unique_pcm_count': len(duplicates),
                           'unique_pcm_hours': sum(next(r['duration_s'] for r in rows if r['record_id'] == ids[0])
                                                   for ids in duplicates.values()) / 3600,
                           'duplicate_pcm_groups': [ids for ids in duplicates.values() if len(ids) > 1],
                           'collections': dict(Counter(r['source_collection'] for r in rows)),
                           'independent_acceptance_hours': 0,
                           'far_source_verification': evidence,
                           'known_fa_pools': [1103, 3301],
                           'known_fa_schedule_source_overlap': {
                               'shared_source_wav_hashes': len(pools[1103]['hashes'] & pools[3301]['hashes']),
                               'pool_1103_unique_source_hashes': len(pools[1103]['hashes']),
                               'pool_3301_unique_source_hashes': len(pools[3301]['hashes']),
                               'pool_1103_schedule_entries': pools[1103]['entries'],
                               'pool_3301_schedule_entries': pools[3301]['entries'],
                               'basis': 'frozen original 7200-second injection schedules; not independent corpora'},
                           'known_fa_replay_seconds_not_new_exposure': 9569,
                           'historical_fa_result_unchanged': '2 FA / 8 h failed',
                           'new_tts_calls': 0, 'new_asr_calls': 0, 'training_runs': 0}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, default=ROOT)
    p.add_argument('--far-root', type=Path)
    p.add_argument('--k1-root', type=Path)
    p.add_argument('--r5-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    write_outputs(inventory(args.repository.resolve(), args.far_root.resolve() if args.far_root else None,
                            args.k1_root.resolve() if args.k1_root else None,
                            args.r5_root.resolve() if args.r5_root else None), args.output)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
