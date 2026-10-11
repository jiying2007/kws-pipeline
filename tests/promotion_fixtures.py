"""Invented metadata and opaque bytes for offline promotion-consumer tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'training')]
from corpus_identity import corpus_digest
from training_admission import initial_dataset_ancestry, POLICY
from feature_cached_trainer import cache_contract
from ctc_label_prior_contract import canonical_sha256

HEAD = 'a' * 40
TREE = 'b' * 40


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode()


def documents(*, product=False, label_prior=False):
    # No model/checkpoint/audio parser is invoked. These deliberately invented
    # bytes exercise metadata admission, not acoustics or usable model weights.
    manifests = [dict(name=name, sha256=digest * 64, lineage_sha256='c' * 64)
                 for name, digest in [('adversarial-hard-negatives.tsv', 'd'),
                                      ('development-failure-replay.tsv', 'e')]]
    rows = [dict(recording=f'manifest-{i}:1', manifest=row['name'], path=f'invented-{i}.wav',
                 file_sha256=str(i + 1) * 64, pcm_sha256=str(i + 3) * 64,
                 frames=160, duration_s=0.01) for i, row in enumerate(manifests)]
    corpus = dict(schema_version=1, recordings=rows, corpus_sha256=corpus_digest(rows))
    admission = dict(policy=POLICY, purpose='reviewed-ctc-training', promotion_allowed=True,
                     listener_authenticity_verified=False, reviewed_rows=2, manifests=manifests,
                     dataset_ancestry=initial_dataset_ancestry(manifests, corpus, 2))
    zero = lambda n: dict(expected=n, matched=n, false_rejects=0, false_accepts=0)
    common = dict(development_manifest_sha256='f' * 64, development_selected_round=1,
                  development_selected_frontend='logmel')
    anchors = [['ni3'], ['ni3', 'hao3'], ['ni3', 'hao3', 'xiao3'],
               ['xiao3'], ['xiao3', 'wo1'], ['xiao3', 'wo1', 'xiao3']]
    adversarial = dict(evidence_class='development-only-adversarial-lexicon',
        formal_qualification_used=False, data_augmentation_policy='train-only-balanced-mining-v1',
        selection_policy='balanced-strict-prefix-anchor-topk-v2', max_length=5,
        enumerated_sequences=1330, top_k=64, probes_per_sequence=2,
        replay_examples_per_sequence=8, replay_examples=512, min_per_keyword=24,
        per_keyword_selected={'1': 32, '2': 32}, strict_prefix_anchors=anchors, manifest_sha256='d' * 64)
    failure = dict(evidence_class='development-only-failure-resynthesis',
        policy='development-failure-resynthesis-v1', enabled=True,
        formal_qualification_used=False, development_source_wav_bytes_copied=False,
        source_splits=['calibration', 'test'], manifest_sha256='e' * 64,
        examples=1, observed_unique_failures=1, selected_unique_failures=1)
    cohort = dict(common, expected_wakes=256, qualification_seed=123,
        retired_exposed_qualification_seeds=[], generated_after_training=True, seed_disjoint=True,
        strict_development_candidate_required=True, development_candidate_policy='latest-strict-gate-passing-round',
        overlapping_development_active_wav_sha256=0, overlapping_exposed_active_wav_sha256=0,
        overlapping_retired_active_wav_sha256=0, formal_preflight_policy='shadow-adversarial-failure-formal-preflight-v2',
        shadow_qualification_required=True, shadow_qualification_qualified=True,
        adversarial_refinement_required=True, model_provenance_adversarial_manifest_verified=True,
        adversarial_overlap_guard_included=True, failure_replay_required=True,
        model_provenance_failure_replay_manifest_verified=True, adversarial_formal_qualification_used=False,
        failure_replay_formal_qualification_used=False, failure_replay_development_source_wav_bytes_copied=False,
        failure_replay_overlap_guard_included=True, shadow_seed_count=8, shadow_min_surrogate_separation=0.1)
    for field in ('enumerated_sequences', 'top_k', 'probes_per_sequence', 'replay_examples_per_sequence',
                  'replay_examples', 'min_per_keyword', 'per_keyword_selected', 'strict_prefix_anchors',
                  'selection_policy', 'data_augmentation_policy', 'manifest_sha256'):
        cohort['adversarial_' + field] = copy.deepcopy(adversarial[field])
    for field in ('manifest_sha256', 'examples', 'observed_unique_failures', 'selected_unique_failures'):
        cohort['failure_replay_' + field] = failure[field]
    result = {
        'best/model-provenance.json': dict(model={}, checkpoint={}, training={
            'admission': admission, 'manifests': copy.deepcopy(manifests),
            'environment': {'repository_sha': HEAD}, 'sample_weighting': {
                'wake_example_weight': 1.0,
                'wake_example_weight_semantics': 'exact-configured-keyword-target-with-per-keyword-override-v2',
                'wake_keyword_weights': {'1': 1.0, '2': 1.0}}}),
        'training-run-summary.json': dict(qualified=True, best_round=1, best_frontend='logmel'),
        'best/qualification/summary.json': dict(zero(256), per_keyword={'1': zero(128), '2': zero(128)}),
        'robustness-summary.json': dict(qualified=True, failures=[]),
        'hard-negative-stream/summary.json': dict(qualified=True, false_accepts=0,
            far_per_hour=0.0, full_negative_manifest_coverage=True),
        'hard-negative-stream/stream-contract.json': dict(full_negative_manifest_coverage=True,
            observed_min_payload_gap_seconds=2.0, decoder_retention_proof_window_seconds=1.6),
        'qualification-dataset/qualification-cohort.json': cohort,
        'shadow-qualification/summary.json': dict(common, schema_version=1,
            evidence_class='development-only-shadow-qualification', qualified=True,
            formal_qualification_seed_consumed=False, min_surrogate_separation=0.1,
            results=[dict(qualified=True, runtime_qualified=True, surrogate_separation_qualified=True,
                          qualification=zero(256), surrogate={'minimum_separation': 0.2}) for _ in range(8)]),
        'adversarial-refinement/summary.json': dict(policy='post-domain-adversarial-refinement-v1',
            qualified=True, formal_qualification_used=False, output_development_manifest_sha256='f' * 64,
            refinement_round=1, frontend='logmel', wake_balance={
                'schema_version': 3, 'policy': 'per-keyword-provenance-pressure-balance-v3',
                'pressure_assignment': 'explicit-replay-focus-then-token-edit-distance-v2',
                'default_wake_example_weight': 1.0, 'explicit_focus_nonwake_rows': 1,
                'fallback_edit_distance_nonwake_rows': 0, 'tokenized_nonwake_rows': 1,
                'empty_nonwake_rows': 0, 'wake_keyword_weights': {'1': 1.0, '2': 1.0}}),
        'best/adversarial-lexicon.json': adversarial,
        'best/development-failure-replay.json': failure,
        'formal-preflight.json': dict(common, policy='shadow-adversarial-failure-formal-preflight-v2',
            adversarial_manifest_sha256='d' * 64, failure_replay_manifest_sha256='e' * 64,
            failure_replay_examples=1, model_provenance_adversarial_manifest_verified=True,
            model_provenance_failure_replay_manifest_verified=True),
    }
    if product:
        base = dict(policy='product-speech-like-base-v1', release_tag='invented-base',
                    external_base_bundle_sha256='1' * 64, provider_identity_sha256='2' * 64)
        result['best/product-speech-like-base-contract.json'] = base
        result['best/effective-training-config.json'] = dict(
            product_candidate_data=dict(policy='external-speech-like-product-base-v1',
                tone_fallback_allowed=False, protected_evidence_used=False,
                **{key: value for key, value in base.items() if key != 'policy'}),
            train=dict(feature_cache_max_items=8192, feature_cache_max_bytes=1048576),
            generator={'external_base_dataset': {split: 'invented-' + split for split in
                                               ('train', 'calibration', 'test', 'qualification')}})
        result['best/training-invocation.json'] = dict(schema_version=1,
            evidence_class='governed-model-training-invocation-v1', head_sha=HEAD,
            ref='refs/heads/main', source_policy='exact-current-main',
            event_name='workflow_dispatch', request=None, request_sha256=None)
        training = result['best/model-provenance.json']['training']
        training['environment']['feature_cache'] = cache_contract(8192, 1048576)
        training['sample_weighting'].update(
            ordered_token_sample_weighting='training-sample-weights-v2-population-normalized',
            normalization=dict(schema_version=1, policy='dataset-mean-sample-weight-v1',
                               rows=2, nonempty_rows=2, all_mean_weight=1.0, nonempty_mean_weight=1.0))
        if label_prior:
            result['best/effective-training-config.json']['train']['ctc_primary_policy'] = 'label-prior-v1'
            training['ctc_primary_policy'] = 'label-prior-v1'
            training['corpus_identity'] = corpus
            values = [0.8, 0.1, 0.1]
            training['ctc_label_prior'] = dict(schema_version=1,
                policy='initial-model-training-corpus-posterior-marginal-v1',
                loss_policy='posterior-marginal-label-prior-ctc-v1',
                posterior_source='raw-initial-model-before-vad-alignment-v1',
                alpha=0.3, frame_count=123, training_corpus_sha256=corpus['corpus_sha256'],
                source_float_state_sha256='b' * 64, values=values,
                values_sha256=canonical_sha256(values))
    return result


def materialize(root, docs):
    docs = copy.deepcopy(docs)
    raw = {f'best/{name}': ('invented metadata fixture ' + name).encode()
           for name in ('model.kwm', 'model.pt', 'keywords.kwk')}
    raw['best/keywords.tsv'] = ('1\t你好小窝\t0.55\tni3 hao3 xiao3 wo1\n'
                                '2\t小窝小窝\t0.55\txiao3 wo1 xiao3 wo1\n').encode()
    model_hash = sha(raw['best/model.kwm'])
    docs['best/model-provenance.json']['model']['sha256'] = model_hash
    docs['best/model-provenance.json']['checkpoint']['sha256'] = sha(raw['best/model.pt'])
    if 'best/effective-training-config.json' in docs:
        docs['best/effective-training-config.json']['product_candidate_data']['base_contract_sha256'] = sha(
            encoded(docs['best/product-speech-like-base-contract.json']))
    for name in ('hard-negative-stream/summary.json', 'hard-negative-stream/stream-contract.json'):
        docs[name]['model_sha256'] = model_hash
    bindings = {'shadow_summary_sha256': 'shadow-qualification/summary.json',
        'adversarial_refinement_summary_sha256': 'adversarial-refinement/summary.json',
        'adversarial_lexicon_evidence_sha256': 'best/adversarial-lexicon.json',
        'failure_replay_evidence_sha256': 'best/development-failure-replay.json'}
    for key, path in bindings.items():
        digest = sha(encoded(docs[path]))
        docs['formal-preflight.json'][key] = digest
        docs['qualification-dataset/qualification-cohort.json'][key] = digest
    docs['qualification-dataset/qualification-cohort.json']['formal_preflight_sha256'] = sha(encoded(docs['formal-preflight.json']))
    raw.update({name: encoded(value) for name, value in docs.items()})
    artifact_names = ['model.kwm', 'model.pt', 'model-provenance.json', 'keywords.kwk', 'keywords.tsv']
    artifact_names += [name for name in ('effective-training-config.json',
        'product-speech-like-base-contract.json', 'training-invocation.json') if 'best/' + name in docs]
    docs['training-run-summary.json']['artifacts'] = {name: sha(raw['best/' + name]) for name in artifact_names}
    raw['training-run-summary.json'] = encoded(docs['training-run-summary.json'])
    for name, content in raw.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return raw
