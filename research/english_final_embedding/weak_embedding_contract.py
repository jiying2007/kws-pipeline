"""Small data/geometry contract for a new, unqualified weak-intent experiment.

This module grants no execution or formal training admission. It is independent
of the immutable English63 experiment and does not modify its labels or hashes.
"""
from __future__ import annotations
import hashlib
import io
import json
import re
import wave

RATE = 16000
MEL_SAMPLES = 16000
EMBED_HOP = 1280
MEL_FRAMES = 76
MEL_BANDS = 32
EMBED_WIDTH = 96
HEAD_HISTORY = 16
HEAD_HISTORY_SAMPLES = MEL_SAMPLES + (HEAD_HISTORY - 1) * EMBED_HOP
CLASSES = {'background': 0, 'Hey Vosen': 1, 'Hey Mero': 2}
TARGET_IPAS = {'Hey Vosen': '/heɪ ˈvoʊzən/', 'Hey Mero': '/heɪ ˈmiːroʊ/'}
VOICE_ROLES = {'aiden': 'train', 'dylan': 'train', 'eric': 'train', 'serena': 'train',
               'ryan': 'development', 'vivian': 'development'}
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold',
               'trusted_negative', 'pronunciation_verified',
               'model_promotion_allowed', 'qualification_allowed')
ASSETS = {
    'melspectrogram.onnx': (1087958, 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f'),
    'embedding_model.onnx': (1326578, '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'),
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def positive_prefix_samples(clip_frames):
    """Real background prefix; phase-align EOF, without appending any samples."""
    need(type(clip_frames) is int and 400 <= clip_frames <= HEAD_HISTORY_SAMPLES,
         'POSITIVE_FULL_CLIP_MUST_FIT_FIXED_HISTORY')
    return HEAD_HISTORY_SAMPLES + (-clip_frames) % EMBED_HOP


def clip_envelope_instance_ends(clip_start, clip_frames, observed_samples):
    """Candidate windows whose PCM input envelope contains the original clip.

This is not an assertion about acoustic receptive coverage. The real worker
must first inspect the fixed mel convolutions and then select supported windows.
"""
    need(all(type(x) is int and x >= 0 for x in (clip_start, clip_frames, observed_samples))
         and clip_frames >= 400 and observed_samples >= clip_start + clip_frames,
         'INSTANCE_GEOMETRY')
    end = clip_start + clip_frames
    return [t for t in range(HEAD_HISTORY_SAMPLES, observed_samples + 1, EMBED_HOP)
            if t >= end and t - HEAD_HISTORY_SAMPLES <= clip_start]


def mel_supported_clip_ends(clip_start, clip_frames, observed_samples, geometry):
    """Use actual ONNX metadata to locate observed whole-clip mel support.

This remains geometric weak clip supervision, not acoustic transcription,
pronunciation verification, a word-end label or independent event timing.
"""
    need(geometry.get('status') == 'FIXED_ONNX_CONV_GEOMETRY_VERIFIED' and
         type(geometry.get('head_support_start_before_available')) is int and
         type(geometry.get('support_end_before_available')) is int, 'MEL_PREFLIGHT_REQUIRED')
    history = geometry['head_support_start_before_available']
    lag = geometry['support_end_before_available']
    need(0 <= lag < history <= HEAD_HISTORY_SAMPLES, 'MEL_SUPPORT_GEOMETRY')
    return [t for t in clip_envelope_instance_ends(clip_start, clip_frames, observed_samples)
            if t - history <= clip_start and t - lag >= clip_start + clip_frames]


def speaker_roles(speakers):
    """Predeclared 30/10 speaker split of the already exposed 40-speaker pool."""
    need(len(speakers) == len(set(speakers)) == 40 and
         all(type(s) is str and re.fullmatch(r'[0-9]+', s) for s in speakers),
         'EXACT_LIBRISPEECH_SPEAKERS')
    order = sorted(speakers, key=lambda s: digest(('en-final-embedding-v1/' + s).encode()))
    return {s: ('development' if i < 10 else 'train') for i, s in enumerate(order)}


def verified_pcm(wav_bytes, row):
    need(type(wav_bytes) is bytes and digest(wav_bytes) == row['wav_sha256'], 'WAV_IDENTITY')
    with wave.open(io.BytesIO(wav_bytes), 'rb') as source:
        need((source.getnchannels(), source.getsampwidth(), source.getframerate(),
              source.getcomptype()) == (1, 2, RATE, 'NONE'), 'PCM_GEOMETRY')
        frames = source.getnframes()
        pcm = source.readframes(frames)
        need(not source.readframes(1), 'EXTRA_PCM')
    need(frames == row['frames'] and 400 <= frames <= 192000 and len(pcm) == 2 * frames
         and digest(pcm) == row['pcm_sha256'], 'PCM_IDENTITY')
    return pcm


def weak_rows(recipe, source_rows, *, eligible_cell_ids, observations, excluded):
    """Select explicitly reviewed source intents for this one weak experiment.

The caller must authenticate the exact source artifact and original per-cell
receipts first, and call verified_pcm on every selected WAV before extraction.
It must explicitly supply eligible IDs and PCM-bound observation identities.
Generation alone is never eligibility. STRONG_CONFLICT and UNKNOWN observations
cannot be overridden by the ID list. Original receipts are not relabeled; all
excluded candidates remain in the original 144-cell denominator, without retry.
"""
    cells = recipe.get('cells')
    need(type(cells) is list and len(cells) == 144 and
         len({c['cell_id'] for c in cells}) == 144, 'RECIPE_144')
    need(type(source_rows) is list and len(source_rows) == 144 and
         len({r['cell_id'] for r in source_rows}) == 144 and
         {r['cell_id'] for r in source_rows} == {c['cell_id'] for c in cells}, 'SOURCE_144')
    roster = {c['cell_id'] for c in cells}
    need(type(eligible_cell_ids) is list and eligible_cell_ids and
         len(eligible_cell_ids) == len(set(eligible_cell_ids)) and set(eligible_cell_ids) <= roster,
         'EXPLICIT_INTENT_RESEARCH_SELECTION_REQUIRED')
    need(type(observations) is dict and set(observations) <= roster and
         all(type(x) is dict for x in observations.values()) and type(excluded) is list,
         'OBSERVATION_AND_EXCLUSION_ACCOUNTING_REQUIRED')
    selection_sha = digest(json.dumps({'purpose': 'requested-intent-weak-research-only',
        'eligible_cell_ids': sorted(eligible_cell_ids), 'observations': observations},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    selected = set(eligible_cell_ids)
    actual = {r['cell_id']: r for r in source_rows}
    rows, seen_pcm = [], {}
    for cell in cells:
        need(cell.get('voice_id') in VOICE_ROLES and
             cell.get('role') == VOICE_ROLES[cell['voice_id']] and
             cell.get('source_family_id') == 'qwen3-preset-' + cell['voice_id'] and
             all(cell.get(k) is False for k in FALSE_FLAGS), 'RECIPE_ROLE_OR_AUTHORITY')
        if cell.get('intent') == 'wake-candidate':
            label = CLASSES.get(cell.get('canonical_phrase'))
            need(label in (1, 2) and cell.get('requested_keyword_index') == label and
                 cell.get('target_ipa') == TARGET_IPAS[cell['canonical_phrase']],
                 'WEAK_WAKE_IDENTITY')
        else:
            need(cell.get('intent') == 'ordinary-negative-candidate' and
                 cell.get('canonical_phrase') is None and cell.get('requested_keyword_index') is None,
                 'WEAK_BACKGROUND_IDENTITY')
            label = 0
        source = actual[cell['cell_id']]
        need(source.get('status') in ('GENERATED_UNADMITTED', 'FAILED_NO_RETRY',
                                     'NOT_ATTEMPTED_AFTER_FAILURE', 'STARTED_UNKNOWN_NO_RETRY',
                                     'NOT_STARTED', 'INVALID_SOURCE_RECEIPT'), 'SOURCE_NOT_TERMINAL')
        exclusion = {'id': cell['cell_id'], 'role': cell['role'], 'weak_class': label,
                     'source_status': source['status'], 'research_selection_sha256': selection_sha}
        if source['status'] != 'GENERATED_UNADMITTED':
            excluded.append({**exclusion, 'reason': 'SOURCE_UNAVAILABLE_NO_TRAINING_TARGET'})
            continue
        need(all(type(source.get(k)) is str and re.fullmatch(r'[0-9a-f]{64}', source[k])
                 for k in ('wav_sha256', 'pcm_sha256')) and
             type(source.get('frames')) is int and 400 <= source['frames'] <= 192000,
             'SOURCE_AUDIO_IDENTITY')
        observation = observations.get(cell['cell_id'])
        if observation is None:
            excluded.append({**exclusion, 'reason': 'UNKNOWN_OBSERVATION'})
            continue
        need(observation.get('source_pcm_sha256') == source['pcm_sha256'] and
             all(type(observation.get(k)) is str and re.fullmatch(r'[0-9a-f]{64}', observation[k])
                 for k in ('observer_artifact_sha256', 'observation_sha256')) and
             observation.get('evidence_status') in ('USEFUL_FOR_INTENT_RESEARCH', 'STRONG_CONFLICT', 'UNKNOWN') and
             type(observation.get('phone_review_status')) is str and
             bool(observation['phone_review_status'].strip()), 'PCM_BOUND_OBSERVATION_REQUIRED')
        if observation['evidence_status'] != 'USEFUL_FOR_INTENT_RESEARCH':
            excluded.append({**exclusion, 'reason': observation['evidence_status'],
                             'phone_review_status': observation['phone_review_status'],
                             'observation': dict(observation)})
            continue
        if cell['cell_id'] not in selected:
            excluded.append({**exclusion, 'reason': 'NOT_SELECTED_FOR_THIS_INTENT_EXPERIMENT'})
            continue
        if source['pcm_sha256'] in seen_pcm:
            need(seen_pcm[source['pcm_sha256']][:2] == (cell['role'], label),
                 'DUPLICATE_PCM_CROSS_ROLE_OR_LABEL')
            excluded.append({**exclusion, 'reason': 'DUPLICATE_PCM_SAME_ROLE_AND_INTENT',
                             'duplicate_of': seen_pcm[source['pcm_sha256']][2]})
            continue
        seen_pcm[source['pcm_sha256']] = (cell['role'], label, cell['cell_id'])
        rows.append({**source, 'role': cell['role'], 'source_family_id': cell['source_family_id'],
                     'weak_class': label, 'target_basis': 'requested-intent-not-acoustic-truth',
                     'research_selection_sha256': selection_sha, 'source_observation': dict(observation),
                     'phone_review_status': observation['phone_review_status'],
                     'independent_evaluation': False, **dict.fromkeys(FALSE_FLAGS, False)})
    need({(r['role'], r['weak_class']) for r in rows} ==
         {(role, label) for role in ('train', 'development') for label in range(3)},
         'MISSING_WEAK_CLASS_OR_ROLE')
    return rows
