"""Fixed, dependency-free contract for saved FP32 -> native C research replay."""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'english63-saved-fp32-native-c-stream-parity-v1'
STATUS = 'MEASURED_NATIVE_C_STREAM_PARITY_NOT_KWS_QUALIFICATION'
REPLAY_SCHEMA = 'english63-native-c-saved-evidence-verification-v1'
REPLAY_STATUS = 'VERIFIED_SAVED_C_COMPARISON_ARTIFACTS'
TOLERANCES = {'features': 0.0002, 'logits': 0.0002, 'probabilities': 0.0002}
KINDS = ('gru', 'dsc_tcn')
PLANS = ['160', 'mixed']
SOURCE_SUMMARY_SHA = '09db16f5947d39c1824565b20defc16a2349f09fea5621b8a935989f65555295'
SOURCE_WEIGHTS = {'gru': '71f731155a19538b533e4253d2e4b19d5c03328bd26da5263542159df066e623',
                  'dsc_tcn': '167282c1037a28152bed4afaf49e53a4307bd8d5576a486b41c0a2291fdca935'}
TTS_SHA = 'f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc'
REVIEW_SHA = '89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12'
UNMEASURED = ('independent_english_frr', 'continuous_negative_far_per_hour',
              'word_end_latency', 'ssc305_board_profile')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'C_REPLAY_DUPLICATE_JSON')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('C_REPLAY_NONFINITE_JSON:'+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=reject)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')


def validate_summary(s):
    require(s.get('schema') == SCHEMA and s.get('status') == STATUS
            and s.get('source_audio_clips') == 63 and s.get('source_pcm_samples') == 2101760
            and s.get('source_training_run_id') == 43595338774 and s.get('optimizer_updates') == 0
            and s.get('new_tts_asr_ipa_calls') == 0 and s.get('numeric_tolerances') == TOLERANCES
            and all(s.get(k) is False for k in ('formal_training_admitted', 'human_gold',
                    'model_promotion_allowed', 'model_selection_made', 'threshold_selection'))
            and all(s.get(k) == 'NOT_RUN' for k in UNMEASURED), 'C_REPLAY_SUMMARY_BOUNDARY')
    rows = s.get('models')
    require(type(rows) is list and len(rows) == 2 and {r.get('kind') for r in rows} == set(KINDS), 'C_REPLAY_MODEL_ROSTER')
    for row in rows:
        require(row.get('source_weights_sha256') == SOURCE_WEIGHTS[row['kind']]
                and row.get('isolated_clips') == 63 and row.get('isolated_frames') == 13010
                and row.get('continuous_frames') == 13134 and row.get('chunk_plans') == PLANS
                and row.get('reference_forward_sequences') == 64
                and all(row.get(k) is True for k in ('chunk_probabilities_bit_identical',
                         'chunk_events_identical', 'python_c_events_identical'))
                and all(row.get(k) == 0 for k in ('eof_padding_samples', 'eof_events', 'continuous_state_resets'))
                and row.get('resource_profile_scope') == 'native-linux-host-not-ssc305', 'C_REPLAY_MODEL_BOUNDARY')
        for key, bound in (('max_feature_abs_error', TOLERANCES['features']),
                           ('max_logit_abs_error', TOLERANCES['logits']),
                           ('max_probability_abs_error', TOLERANCES['probabilities'])):
            value = row.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= bound,
                    'C_REPLAY_NUMERIC_BOUNDARY:'+key)
    return True


def saved_manifest(directory):
    root = Path(directory)
    files = sorted(p for p in root.rglob('*') if p.is_file() and p != root/'REPLAY.json')
    require(not (root/'FAILED.json').exists() and all(not p.is_symlink() for p in root.rglob('*')),
            'C_REPLAY_FAILED_OR_SYMLINK')
    require(len(files) <= 18 and sum(p.stat().st_size for p in files) < 60*1024*1024,
            'C_REPLAY_OUTPUT_BUDGET')
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in files}


def verification_receipt(directory):
    root = Path(directory)
    validate_summary(read_json(root/'SUMMARY.json'))
    return {'schema': REPLAY_SCHEMA, 'status': REPLAY_STATUS,
            'summary_sha256': digest((root/'SUMMARY.json').read_bytes()),
            'files_sha256': saved_manifest(root), 'optimizer_updates': 0,
            'model_forward_calls': 0, 'new_audio_asr_calls': 0,
            'formal_training_admitted': False, 'model_promotion_allowed': False}


def verify_saved(directory):
    receipt = verification_receipt(directory)
    require(read_json(Path(directory)/'REPLAY.json') == receipt, 'C_REPLAY_SAVED_MANIFEST_CHANGED')
    return receipt
