"""Finite weak research audio assembly after authenticated source acquisition.

This module does not download or authorize anything. The worker must first bind
the existing 517 LibriSpeech FLAC/transcript identities and the new 144-cell
artifact. These functions then verify decoded PCM identities and split speakers
before selecting at most 16 seconds per speaker. All joins are disclosed splices.
"""
from __future__ import annotations
from array import array
import sys

from causal_embedding import CausalFeatureStream, windows_at_ends
from weak_embedding_contract import (EMBED_HOP, HEAD_HISTORY_SAMPLES, RATE,
    clip_envelope_instance_ends, digest, mel_supported_clip_ends, need,
    positive_prefix_samples, speaker_roles, verified_pcm)

MAX_BACKGROUND_SAMPLES_PER_SPEAKER = 16 * RATE
REAL_CONTEXT_SAMPLES = 4 * EMBED_HOP


def samples_from_pcm(pcm):
    need(type(pcm) is bytes and len(pcm) % 2 == 0, 'PCM16LE_BYTES')
    values = array('h')
    values.frombytes(pcm)
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def background_tracks(source_rows):
    """Input: 517 already authenticated rows, including decoded pcm16le bytes.

Each row has id (original FLAC member), speaker, frames and pcm_sha256. Original
archive, FLAC and transcript checks remain the caller's responsibility. The
returned 40 tracks are at most 16 seconds each, with exact source fragment spans.
"""
    need(type(source_rows) is list and len(source_rows) == 517 and
         len({r['id'] for r in source_rows}) == 517, 'EXACT_EXPOSED_517_ROWS')
    roles = speaker_roles(sorted({r['speaker'] for r in source_rows}))
    grouped = {speaker: [] for speaker in roles}
    for row in source_rows:
        need(type(row['id']) is str and type(row['frames']) is int and row['frames'] > 0
             and type(row['pcm16le']) is bytes and len(row['pcm16le']) == 2 * row['frames']
             and digest(row['pcm16le']) == row['pcm_sha256'], 'BACKGROUND_PCM_IDENTITY')
        grouped[row['speaker']].append(row)
    tracks = []
    for speaker, rows in sorted(grouped.items()):
        pieces, fragments, used = [], [], 0
        for row in sorted(rows, key=lambda r: r['id']):
            take = min(row['frames'], MAX_BACKGROUND_SAMPLES_PER_SPEAKER - used)
            if take == 0:
                break
            pieces.append(row['pcm16le'][:2 * take])
            fragments.append({'source_id': row['id'], 'source_pcm_sha256': row['pcm_sha256'],
                              'source_start_sample': 0, 'samples': take,
                              'track_start_sample': used})
            used += take
        need(used >= HEAD_HISTORY_SAMPLES + EMBED_HOP, 'INSUFFICIENT_REAL_BACKGROUND')
        pcm = b''.join(pieces)
        tracks.append({'id': 'libri-speaker-' + speaker, 'speaker': speaker,
                       'role': roles[speaker], 'pcm16le': pcm, 'frames': used,
                       'pcm_sha256': digest(pcm), 'fragments': fragments,
                       'kind': 'exposed-read-speech-splice', 'trusted_negative': False})
    return tracks


def assemble_audio_cases(rows, wav_by_id, tracks, excluded=None):
    """Observed real prefix + original clip + real context; no silence or fades.

rows must be the output of weak_rows(), and wav_by_id must contain original WAV
bytes. Prefix source choice and offset are deterministic within the row's split.
Positive candidates contain the original clip in their PCM input envelope. The
worker must inspect actual mel convolution metadata before selecting final bags.
This is weak clip supervision and does not provide a phonetic or word-end label.
An explicit excluded list records unsupported long clips without trimming or
replacement. Without that accounting sink, unsupported clips raise.
"""
    need(len(tracks) == 40 and len({t['speaker'] for t in tracks}) == 40,
         'FORTY_SPLIT_BACKGROUND_TRACKS')
    roles = speaker_roles([t['speaker'] for t in tracks])
    for track in tracks:
        need(track['role'] == roles[track['speaker']] and
             digest(track['pcm16le']) == track['pcm_sha256'], 'BACKGROUND_ROLE_OR_IDENTITY')
    cases = []
    for track in tracks:
        ends = list(range(HEAD_HISTORY_SAMPLES, track['frames'] + 1, EMBED_HOP))
        cases.append({'id': track['id'], 'role': track['role'], 'weak_class': 0,
                      'pcm16le': track['pcm16le'], 'ends': ends,
                      'evidence': {k: v for k, v in track.items() if k != 'pcm16le'}})
    for row in rows:
        pcm = verified_pcm(wav_by_id[row['cell_id']], row)
        count = row['frames']
        if row['weak_class'] and count > HEAD_HISTORY_SAMPLES:
            need(type(excluded) is list, 'LONG_CLIP_EXCLUSION_ACCOUNTING_REQUIRED')
            excluded.append({'id': row['cell_id'], 'role': row['role'], 'weak_class': row['weak_class'],
                             'frames': count, 'reason': 'ORIGINAL_CLIP_EXCEEDS_FIXED_INPUT_HISTORY'})
            continue
        prefix_count = (positive_prefix_samples(count) if row['weak_class'] else
                        HEAD_HISTORY_SAMPLES + (-count) % EMBED_HOP)
        eligible = sorted([t for t in tracks if t['role'] == row['role']], key=lambda t: t['id'])
        need(eligible, 'SAME_ROLE_REAL_BACKGROUND_REQUIRED')
        identity = digest(row['cell_id'].encode())
        background = eligible[int(identity[:16], 16) % len(eligible)]
        need(background['frames'] >= prefix_count, 'BACKGROUND_PREFIX_TOO_SHORT')
        start = int(identity[16:32], 16) % (background['frames'] - prefix_count + 1)
        prefix = background['pcm16le'][2 * start:2 * (start + prefix_count)]
        context_start = int(identity[32:48], 16) % (background['frames'] - REAL_CONTEXT_SAMPLES + 1)
        context = background['pcm16le'][2 * context_start:2 * (context_start + REAL_CONTEXT_SAMPLES)]
        combined = prefix + pcm + context
        observed = len(combined) // 2
        if row['weak_class']:
            ends = clip_envelope_instance_ends(prefix_count, count, observed)
        else:
            # Background intent does not require a word boundary; include each
            # head window with at least 400 actual candidate samples observed.
            ends = [t for t in range(HEAD_HISTORY_SAMPLES, observed + 1, EMBED_HOP)
                    if t >= prefix_count + 400]
        need(ends, 'NO_OBSERVED_WEAK_INSTANCE')
        cases.append({'id': row['cell_id'], 'role': row['role'], 'weak_class': row['weak_class'],
                      'pcm16le': combined, 'ends': ends,
                      'evidence': {'source_pcm_sha256': row['pcm_sha256'],
                          'source_wav_sha256': row['wav_sha256'], 'source_frames': count,
                          'prefix_track': background['id'], 'prefix_role': background['role'],
                          'prefix_start_sample': start, 'prefix_samples': prefix_count,
                          'clip_start_sample': prefix_count, 'original_clip_eof_sample': prefix_count + count,
                          'context_track': background['id'], 'context_role': background['role'],
                          'context_start_sample': context_start, 'context_samples': REAL_CONTEXT_SAMPLES,
                          'combined_pcm_sha256': digest(combined), 'observed_samples': observed,
                          'silence_or_invented_samples': 0, 'target_basis': row['target_basis'],
                          'research_selection_sha256': row.get('research_selection_sha256'),
                          'source_observation': row.get('source_observation'),
                          'original_eof_view': 'REPLAY_PREFIX_UNTIL_ORIGINAL_EOF_SEPARATELY_CENSORED',
                          'phone_review_status': row['phone_review_status']}})
    need(len({case['id'] for case in cases}) == len(cases), 'DUPLICATE_AUDIO_CASE')
    return cases


def extract_feature_bags(cases, embedding_callback, excluded=None):
    """Future authorized worker: run frozen embedding once per observed clock.

No model is imported or invoked by importing this module. Return actual cached
bags, per-case observed embeddings for saved-weight/C replay, and exact splice
evidence. Nothing here grants independent FAR/FRR or pronunciation status.
An explicit excluded list records geometry-ineligible cases. The worker must
retain those cases in the source denominator and must not regenerate them.
"""
    import numpy as np
    bags, streams, evidence = [], [], []
    for case in cases:
        ends = case['ends']
        if case['weak_class']:
            source = case['evidence']
            ends = mel_supported_clip_ends(source['clip_start_sample'], source['source_frames'],
                source['observed_samples'], getattr(embedding_callback, 'geometry', {}))
            if not ends:
                need(type(excluded) is list, 'MEL_SUPPORT_EXCLUSION_ACCOUNTING_REQUIRED')
                excluded.append({'id': case['id'], 'role': case['role'], 'weak_class': case['weak_class'],
                                 'frames': source['source_frames'], 'reason': 'NO_OBSERVED_WHOLE_CLIP_MEL_SUPPORT'})
                continue
        stream = CausalFeatureStream(embedding_callback)
        values = samples_from_pcm(case['pcm16le'])
        records = []
        for offset in range(0, len(values), 160):
            records.extend(stream.accept(values[offset:offset + 160]))
        need(stream.finish() == [], 'NO_EOF_OUTPUT')
        windows = np.asarray(windows_at_ends(records, ends), dtype=np.float32)
        bags.append({'id': case['id'], 'role': case['role'], 'weak_class': case['weak_class'],
                     'phone_review_status': case['evidence'].get('phone_review_status', 'NOT_APPLICABLE_REAL_READ_SPEECH'),
                     'source_observation': case['evidence'].get('source_observation'),
                     'research_selection_sha256': case['evidence'].get('research_selection_sha256'),
                     'windows': windows})
        streams.append({'id': case['id'], 'records': records,
                        'original_clip_eof_sample': case['evidence'].get('original_clip_eof_sample')})
        evidence.append({'id': case['id'], 'role': case['role'], 'weak_class': case['weak_class'],
                         'embedding_calls': stream.embedding_calls, 'head_instance_ends': ends,
                         'source': case['evidence']})
    return bags, streams, evidence
