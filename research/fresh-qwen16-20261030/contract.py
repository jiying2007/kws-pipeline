"""Pure contracts for a fresh, unadmitted audio batch. No inference imports."""
from pathlib import Path
import hashlib
import json
import math
import os
import re

CAMPAIGN = 'fresh-qwen16-20261030-v1'
BRANCH = 'exec-data/' + CAMPAIGN
REPOSITORY = 'jiying2007/kws-pipeline'
RETAINED_COMMIT = '983e8082f712bd5099feb32bfb099e9f6df22d3a'
MODEL_ID = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
MODEL_REVISION = '85e237c12c027371202489a0ec509ded67b5e4b5'
VOICES = ('Vivian', 'Uncle_Fu', 'Dylan', 'Serena')
TEXTS = ('你好小窝', '小窝小窝')
RECIPE = dict(language='Chinese', instruct='', do_sample=True, top_k=50,
              top_p=1.0, temperature=0.9, repetition_penalty=1.05,
              subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0,
              subtalker_temperature=0.9, max_new_tokens=120,
              non_streaming_mode=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as target:
        target.write(canonical(value) + b'\n')
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def plan():
    result = []
    for voice in VOICES:
        for keyword, text in enumerate(TEXTS, 1):
            for repeat in range(2):
                index = len(result)
                development = voice == 'Serena'
                result.append(dict(cell_id=f'fresh-qwen16-20261030-{index:02d}',
                    speaker=voice, intended_text=text, intended_keyword_id=keyword,
                    seed=2026103000 + index, repeat=repeat,
                    voice_group='qwen3-preset-' + voice,
                    source_family_id='qwen3-preset-' + voice,
                    intended_split='development-a' if development else 'train',
                    source_role='development' if development else 'train',
                    source_type='preset-tts', synthetic=True,
                    label_status='PENDING_ACTUAL_AUDIO_REVIEW', actual_text=None,
                    ctc_training_allowed=False, evaluation_gold_allowed=False,
                    qualification_allowed=False, human_gold=False))
    return result


def verify_environment(env, event):
    expected = dict(GITHUB_REPOSITORY=REPOSITORY, GITHUB_REF='refs/heads/' + BRANCH,
                    GITHUB_EVENT_NAME='push', GITHUB_RUN_ATTEMPT='1',
                    RUNNER_ENVIRONMENT='github-hosted', RUNNER_OS='Linux')
    require(all(env.get(key) == value for key, value in expected.items()),
            'exact first-attempt public hosted execution required')
    require(re.fullmatch(r'[0-9a-f]{40}', env.get('GITHUB_SHA', '')) is not None,
            'immutable source head required')
    require(re.fullmatch(r'[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')) is not None,
            'run identity required')
    require(type(event) is dict and event.get('created') is True and
            event.get('deleted') is False and event.get('before') == '0' * 40 and
            event.get('after') == env['GITHUB_SHA'] and
            event.get('repository', {}).get('full_name') == REPOSITORY and
            event.get('repository', {}).get('private') is False,
            'first creation of the dedicated public execution ref required')


def signal_metrics(samples, rate, ended_with_eos):
    require(type(rate) is int and rate == 24000 and type(ended_with_eos) is bool,
            'native rate and observed termination required')
    require(0 < len(samples) <= 12 * rate, 'bounded nonempty native audio required')
    values = [float(sample) for sample in samples]
    require(all(math.isfinite(sample) for sample in values), 'nonfinite native audio')
    peak = max(map(abs, values))
    rms = math.sqrt(sum(sample * sample for sample in values) / len(values))
    flags = []
    if peak >= 1.0:
        flags.append('source_peak_ge_one')
    if rms < 1e-4:
        flags.append('near_silent')
    if not ended_with_eos:
        flags.append('native_eos_not_observed')
    return dict(frames=len(values), sample_rate_hz=rate,
                duration_seconds=len(values) / rate, peak=peak, rms=rms,
                clipping_fraction=sum(abs(sample) >= 1.0 for sample in values) / len(values),
                flags=flags, technically_valid=not flags,
                ctc_training_allowed=False, qualification_allowed=False)
