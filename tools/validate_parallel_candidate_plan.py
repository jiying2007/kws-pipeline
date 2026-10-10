#!/usr/bin/env python3
"""Validate disabled candidate-plan metadata, never run or admit an experiment.

Only this JSON is read. Referenced files, sources, hashes, licenses, resources and
measurements are NOT authenticated. A valid shape is not evidence or permission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / 'configs/research/parallel-candidate-screen-2026-10-10.json'
MAX_PLAN_BYTES = 65536
CANDIDATES = {
    'rnn64_control': ('matched-control', 'tanh-rnn-32x64x5'),
    'causal_ds_tcn_ctc': ('primary-challenger', 'reduced-causal-depthwise-temporal-ctc'),
    'causal_fsmn_ctc': ('conditional-challenger', 'bounded-zero-lookahead-fsmn-ctc'),
}
SOURCE_REFERENCES = {
    'rnn64_control': ('https://github.com/jiying2007/kws-pipeline',
        '5144df99a9273e28d33677d856bce10d813c8be0',
        ['training/model.py', 'training/train_ctc.py', 'LICENSE'], 'Apache-2.0'),
    'causal_ds_tcn_ctc': ('https://github.com/wenet-e2e/wekws',
        '6a45aeb994dd81c0969ff877a5a7c46d60ed0c86',
        ['wekws/model/tcn.py', 'examples/hi_xiaowen/s0/conf/ds_tcn_ctc.yaml', 'LICENSE'], 'Apache-2.0'),
    'causal_fsmn_ctc': ('https://github.com/modelscope/FunASR',
        'e7e61293d308f4342ab9307895e710e4b51b949c',
        ['funasr/models/fsmn_kws/encoder.py', 'funasr/models/fsmn_kws/model.py', 'LICENSE'], 'MIT'),
}
CANDIDATE_SPECIFICATION = [
    'exact-layer-dimensions-kernels-dilations-strides',
    'frontend-vocabulary-frame-hop-output-stride-identities',
    'per-layer-cache-shape-dtype-reset-and-discontinuity-semantics',
    'weight-activation-accumulator-dtypes-and-quantization',
    'weights-bias-scales-state-scratch-and-frontend-ring-bytes',
    'macs-per-frame-and-audio-second-plus-non-mac-costs',
    'framewise-step-logits-state-interface-and-export-format',
    'measured-cpu-latency-memory-null-until-target-evidence',
]
AUTHORITY_PATHS = [
    'configs/shipping.xiaowo.json', 'commercial/real-human-qualification.policy.json',
    'commercial/target-qualification.policy.json',
    'configs/training/kws-v2-efficient-encoder-closure-v1.json',
    'research/d20-diagnostic-admission-v1/PLAN-SCHEMA.md',
    'research/experiment_quality_guards/admit_dataset.py', 'training/training_admission.py',
]
IDENTITIES = ('executable_source_sha training_manifest_sha256 actual_label_receipts_sha256 '
              'decoded_pcm_manifest_sha256 group_split_sha256 vocabulary_sha256 frontend_sha256 '
              'decoder_policy_sha256 seed_consumption_history_sha256 budget_policy_sha256 '
              'metric_policy_sha256 sample_exposure_policy_sha256 checkpoint_selection_policy_sha256 '
              'calibration_policy_sha256 development_calibration_manifest_sha256 '
              'development_selection_manifest_sha256').split()
FIXED_AXES = [
    'admitted-data-and-group-splits', 'vocabulary-and-frontend', 'training-objective',
    'paired-seed-sample-order', 'optimizer-and-checkpoint-selection-policy',
    'predeclared-budget-and-tuning-allowance', 'quantization-policy',
    'decoder-and-endpoint-policy', 'metric-scoring-and-exposure-rules',
    'paired-augmentation-and-data-exposure', 'chunk-context-lookahead-and-callback-timestamp-semantics',
    'development-calibration-data-rule-grid-budget-and-target-far',
]
METRICS = ['per-keyword-frr-with-counts', 'continuous-negative-far-and-exposure-and-upper95',
           'each-predeclared-slice-frr-with-support', 'worst-predeclared-slice-frr',
           'trigger-latency-distribution', 'peak-memory-and-compute-cost']
EXCLUDED = ['source-generation', 'training', 'inference', 'asr', 'tts', 'numerical-benchmark',
            'resource-probe', 'model-runtime-implementation', 'recalibration', 'qualification',
            'workflow-dispatch', 'shipping-modification']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names, where):
    expected = set(names.split() if isinstance(names, str) else names)
    require(type(value) is dict and set(value) == expected, where + ': missing or extra fields')


def equal(value, expected, where):
    # bool == int and int == float must never make a malformed plan valid.
    require(type(value) is type(expected) and value == expected, where + ': unexpected value/type')


def nonempty(value, where):
    require(type(value) is str and bool(value.strip()) and value == value.strip(), where + ': invalid text')


def digest(value, length, where):
    require(type(value) is str and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None,
            where + ': invalid digest')


def relative_path(value):
    nonempty(value, 'path')
    path = PurePosixPath(value)
    require('\\' not in value and '\0' not in value and not path.is_absolute()
            and '..' not in path.parts and str(path) == value and value != '.',
            'path: noncanonical relative path')


def strings(value, where):
    require(type(value) is list and bool(value), where + ': expected nonempty list')
    for item in value:
        nonempty(item, where)
    require(len(set(value)) == len(value), where + ': duplicates')


def frozen(obj, expected, where):
    for key, value in expected.items():
        equal(obj[key], value, where + '.' + key)


def validate(plan):
    fields(plan, 'schema_version plan_id as_of scope source_sha source_tree supersedes execution_enabled '
           'automatic_phase_advancement automatic_shipping_approval authority_paths retained_plan_path '
           'historical_anchor candidates comparison_contract data_contract seed_policy budget selection '
           'later_trigger_verifier readiness excluded_execution stop_conditions', 'plan')
    frozen(plan, dict(schema_version=1, plan_id='parallel-candidate-screen-2026-10-10', as_of='2026-10-10',
                     scope='research-plan-metadata-only',
                     source_sha='5144df99a9273e28d33677d856bce10d813c8be0',
                     source_tree='b6f8400bf6e8c15f961089bf5072e152507a2555',
                     supersedes=None, execution_enabled=False,
                     automatic_phase_advancement=False, automatic_shipping_approval=False,
                     retained_plan_path='configs/research/effect-chain-iteration-2026-10-10.json',
                     authority_paths=AUTHORITY_PATHS, excluded_execution=EXCLUDED), 'plan')
    for key in ('source_sha', 'source_tree'):
        digest(plan[key], 40, key)
    for path in plan['authority_paths'] + [plan['retained_plan_path']]:
        relative_path(path)
    anchor = plan['historical_anchor']
    expected_anchor = dict(model_release='model-749187ec1d66', architecture='tanh-rnn-32x64x5',
        role='frozen-regression-anchor-only', matched_control=False, d20='FAIL',
        d20_scope='existing-native-a20-route-only', d90='NOT_RUN', original_fa_causal_claim=False,
        missing_original_state_blocks='original-fa-causal-attribution-only', shipping_approved=False)
    fields(anchor, expected_anchor, 'historical_anchor')
    frozen(anchor, expected_anchor, 'historical_anchor')
    candidates = plan['candidates']
    require(type(candidates) is list and len(candidates) == len(CANDIDATES), 'exactly three candidate slots required')
    seen = set()
    for candidate in candidates:
        fields(candidate, 'id role architecture execution_enabled train_from_scratch pretrained_weights_imported '
               'shared_frontend_required right_context_frames architecture_config_sha256 implementation_source_sha '
               'training_recipe_sha256 resource_plan_sha256 per_seed_initialization_sha256 '
               'per_seed_thresholds_sha256 source gates blockers', 'candidate')
        cid = candidate['id']
        require(type(cid) is str and cid in CANDIDATES and cid not in seen, 'unknown or duplicate candidate ID')
        seen.add(cid)
        role, architecture = CANDIDATES[cid]
        frozen(candidate, dict(role=role, architecture=architecture, execution_enabled=False,
            train_from_scratch=True, pretrained_weights_imported=False, shared_frontend_required=True,
            right_context_frames=0, architecture_config_sha256=None, implementation_source_sha=None,
            training_recipe_sha256=None, resource_plan_sha256=None,
            per_seed_initialization_sha256=None, per_seed_thresholds_sha256=None), cid)
        source = candidate['source']
        fields(source, 'repository commit paths code_license code_license_basis weights_license training_data_license', cid + '.source')
        digest(source['commit'], 40, cid + '.source.commit')
        strings(source['paths'], cid + '.source.paths')
        for path in source['paths']:
            relative_path(path)
        repository, commit, paths, license_id = SOURCE_REFERENCES[cid]
        frozen(source, dict(repository=repository, commit=commit, paths=paths, code_license=license_id), cid + '.source')
        frozen(source, dict(code_license_basis='repository-LICENSE-notice',
            weights_license='NOT_APPLICABLE_NO_IMPORT', training_data_license='PENDING'), cid + '.source')
        gates = candidate['gates']
        fields(gates, 'design_review actual_label_and_pcm_admission causal_cache_and_chunk_equivalence '
               'candidate_c_export_parity resource_admission paired_development_comparison', cid + '.gates')
        for key, value in gates.items():
            require(type(value) is str and value in ('PENDING', 'NOT_RUN', 'BLOCKED', 'UNKNOWN', 'FAIL'),
                    cid + '.' + key + ': disabled plan cannot record a passing execution gate')
        strings(candidate['blockers'], cid + '.blockers')
    comparison = plan['comparison_contract']
    expected_comparison = dict(changed_axis='architecture-only', factorial_sweep_allowed=False,
        original_fa_reconstruction_required=False, pretrained_system_references_in_matched_ranking=False,
        fixed_between_arms=FIXED_AXES, architecture_specific_initialization=
        'paired-seed-deterministic-construction-record-shape-specific-state-hashes-no-identical-weight-claim',
        final_preregistration='PENDING', calibration_policy=
        'same-independent-development-data-method-grid-budget-target-far-per-arm-threshold-frozen-before-selection',
        identical_numeric_threshold_required=False, required_candidate_specification=CANDIDATE_SPECIFICATION)
    fields(comparison, [*expected_comparison, 'frozen_identities'], 'comparison_contract')
    frozen(comparison, expected_comparison, 'comparison_contract')
    fields(comparison['frozen_identities'], IDENTITIES, 'frozen_identities')
    for key, value in comparison['frozen_identities'].items():
        equal(value, None, key + ' awaits a separately reviewed executable preregistration')
    data = plan['data_contract']
    expected_data = dict(actual_labels_and_decoded_pcm_required=True, uncertain_labels='quarantine',
        speech_as_blank_negative_allowed=False, required_lineage=['source-file-and-decoded-pcm-sha256',
        'parent-source-and-transform-parameters-and-seed', 'actual-pronunciation-and-event-label-review',
        'sample-rate-real-sample-count-duration', 'speaker-voice-session-source-family-groups',
        'rights-consent-code-weights-data-licenses'], selection_role='independent-development-selection',
        calibration_role='independent-development-calibration', group_isolation_required=True,
        protected_final_heldout_used_for_selection=False, protected_final_heldout_shared_across_candidates=False,
        final_evaluation='fresh-unobserved-human-final-afe-after-single-candidate-freeze',
        policy_dependent_playback_labels='PENDING-user-product-decision',
        oov_nearphone_role='reviewed-event-no-wake-scoring-only-no-blank-or-wo1-training-target',
        vocabulary_extension_in_screen=False, development_afe='proxy-not-final-afe-qualification')
    fields(data, expected_data, 'data_contract')
    frozen(data, expected_data, 'data_contract')
    expected_seeds = dict(namespace='development-only', screen_seed_ids=None, confirmation_seed_ids=None,
        paired_across_arms=True, formal_qualification_seeds_allowed=False, reuse_consumed_seeds_allowed=False)
    fields(plan['seed_policy'], expected_seeds, 'seed_policy')
    frozen(plan['seed_policy'], expected_seeds, 'seed_policy')
    expected_budget = dict(status='PROPOSED_CEILING_NOT_FAIR_CONVERGENCE_PROOF', max_screen_arms=3,
        max_screen_seeds=2, max_screen_fits=6, max_confirmation_challengers=1, max_confirmation_seeds=1,
        max_confirmation_fits=2, max_total_fits=8, max_optimizer_updates_per_fit=1000, max_concurrent_fits=2,
        wall_clock_limit_s=None, peak_rss_limit_mib=None, budget_and_convergence_review='PENDING',
        blocked_arm_action='retain-blocker-no-substitution-no-expansion', sample_exposure_limit=None,
        compute_limit_per_fit=None, per_candidate_schedule_sha256=None, later_verifier_fits_included=False)
    fields(plan['budget'], expected_budget, 'budget')
    frozen(plan['budget'], expected_budget, 'budget')
    expected_selection = dict(verdict='NOT_RUN', selected_candidate_id=None, claim='development-comparison-only',
        required_metrics=METRICS, each_paired_seed_must_pass=True, noninferiority_policy=
        'predeclare-each-keyword-slice-far-latency-and-resource-limits-before-execution',
        strict_improvement_required=True, insufficient_support_verdict='INCONCLUSIVE',
        unknown_evidence_verdict='UNKNOWN', tie_or_tradeoff_action='retain-original-baseline-no-automatic-winner',
        multiple_comparisons='screen-at-most-two-challengers-confirm-at-most-one-on-fresh-third-seed',
        no_checkpoint_cherry_picking=True, original_baseline_retained=True,
        latency_semantics='word-end-to-callback-on-continuous-stream-not-clip-presence-or-eof-flush')
    fields(plan['selection'], expected_selection, 'selection')
    frozen(plan['selection'], expected_selection, 'selection')
    expected_verifier = dict(in_first_screen=False, execution_enabled=False, orthogonal_after_encoder_selection=True,
        requires=['separate-frozen-plan-and-budget', 'trigger-conditioned-frr-and-far-accounting',
                  'bounded-cache-and-worst-case-trigger-rate-cost', 'own-c-export-and-target-parity'])
    fields(plan['later_trigger_verifier'], expected_verifier, 'later_trigger_verifier')
    frozen(plan['later_trigger_verifier'], expected_verifier, 'later_trigger_verifier')
    result = dict(execution_ready=False, input_authenticity_verified=False, numerical_admission=False,
                  product_qualified=False, shipping_approved=False)
    fields(plan['readiness'], result, 'readiness')
    frozen(plan['readiness'], result, 'readiness')
    equal(plan['stop_conditions'], ['identity-or-group-split-drift', 'label-conflict-or-insufficient-support',
        'candidate-resource-or-parity-failure', 'any-predeclared-keyword-slice-far-latency-regression',
        'budget-exhausted', 'no-information-gain'], 'stop_conditions')
    return dict(metadata_shape_valid=True, evidence_status='NOT_RUN', **result)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('nonfinite JSON number: ' + value)


def load_plan(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_PLAN_BYTES + 1)
    require(len(raw) <= MAX_PLAN_BYTES, 'plan exceeds metadata byte limit')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object, parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path, nargs='?', default=DEFAULT_PLAN)
    args = parser.parse_args()
    try:
        result = validate(load_plan(args.plan))
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        parser.exit(1, 'INVALID disabled plan metadata: ' + str(error) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
