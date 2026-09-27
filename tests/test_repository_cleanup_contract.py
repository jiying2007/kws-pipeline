#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'repository-cleanup.yml'


RETIRED_PATH_GLOBS = (
    '.github/workflows/development-generalization-*.yml',
    '.github/workflows/gru-*.yml',
    '.github/workflows/rnn-development-curriculum.yml',
    '.github/workflows/rnn-frozen-candidate-qualification.yml',
    '.github/workflows/resumable-development-curriculum.yml',
    '.github/workflows/development-round-segment.yml',
    'configs/qualification.evidence.example.json',
    'configs/training/development-generalization-v1.json',
    'configs/training/xiaowo.gru-development-*.json',
    'configs/training/xiaowo.rnn-development-*.json',
    'training/iterate_gru_development.py',
    'training/run_gru_development*.py',
    'training/train_gru_ctc.py',
    'training/export_gru_model.py',
    'training/gru_model.py',
    'training/qualify_frozen_gru_formal.py',
    'training/validate_frozen_gru_candidate.py',
    'training/iterate_rnn_development.py',
    'training/run_rnn_development.py',
    'training/qualify_frozen_rnn_formal.py',
    'training/validate_frozen_rnn_candidate.py',
    'training/development_resume.py',
    'training/development_loss_controller.py',
    'training/frozen_speech_ablation.py',
    'training/startup_context_*.py',
    'training/preceding_context_*.py',
    'training/select_context_checkpoint.py',
    'training/iterate.py',
    'training/prototype_model.py',
    'training/build_domain_prototype.py',
    'training/surrogate_score.py',
    'tools/build_kws_v2_generalization_plan.py',
    'tools/*development_generalization*.py',
    'tools/run_development_generalization_*.py',
    'tools/*gru*development*.py',
    'tools/*gru*frozen*.py',
    'tools/kws_wav_gru.c',
    'tools/*rnn*development*.py',
    'tools/*rnn*frozen*.py',
    'tools/diagnose_decoder_retention_curve.py',
    'tools/report_frozen_qualification_gate.py',
    'tools/build_vocab.py',
    'tests/test_gru_*.py',
    'tests/test_frozen_speech_ablation.py',
    'tests/test_startup_context.py',
    'tests/test_preceding_context*.py',
    'tests/test_context_checkpoint_selection.py',
    'tests/test_decoder_retention_curve.py',
    'tests/test_report_frozen_qualification_gate.py',
    'tests/test_development_loss_controller.py',
    'experiments/model_family/fresh_validation_registry.json',
    'experiments/model_family/shadow_arena_registry.json',
    'docs/KWS_RESEARCH_RESET.md',
)

REQUIRED_AUTHORITY_PATHS = (
    'configs/shipping.xiaowo.json',
    'configs/training/xiaowo.torch-domain.json',
    'configs/training/kws-v2-efficient-encoder-closure-v1.json',
    '.github/workflows/model-training.yml',
    '.github/workflows/model-training-preflight.yml',
    '.github/workflows/product-development-experiment.yml',
    '.github/workflows/product-training-data-contract.yml',
    '.github/workflows/real-human-qualification.yml',
    '.github/workflows/target-dut-qualification.yml',
    '.github/workflows/shipping-approval.yml',
)


REQUIRED_RETAINED_PATHS = (
    'docs/research/KWS_RESEARCH_RESET.md',
)


REQUIRED_ACTIVE_PATHS = (
    'training/adversarial_refinement.py',
    'training/qualification_failure_replay.py',
)


def main() -> int:
    source = WORKFLOW.read_text(encoding='utf-8')

    for pattern in RETIRED_PATH_GLOBS:
        matches = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern))
        assert not matches, f'retired execution path returned: {pattern}: {matches}'

    for relative in REQUIRED_AUTHORITY_PATHS:
        path = ROOT / relative
        assert path.is_file(), f'canonical product authority missing: {relative}'

    for relative in REQUIRED_RETAINED_PATHS:
        path = ROOT / relative
        assert path.is_file(), f'retained historical evidence missing: {relative}'

    for relative in REQUIRED_ACTIVE_PATHS:
        path = ROOT / relative
        assert path.is_file(), f'active execution path missing: {relative}'

    refinement = (ROOT / 'training' / 'adversarial_refinement.py').read_text(encoding='utf-8')
    assert 'from qualification_failure_replay import (' in refinement
    assert 'render_qualification_failure_replay(' in refinement

    for prefix in (
        'feature/', 'feat/', 'fix/', 'audit/', 'cleanup/', 'eval/',
        'exp/', 'experiment/', 'infra/', 'obs/', 'perf/', 'research/',
        'training/',
    ):
        assert prefix in source, f'repository cleanup safe prefix missing: {prefix}'

    assert 'CLOSED_PR_RETENTION_DAYS: "7"' in source

    for needle in (
        '[[ "${branch}" == "${default_branch}" ]] && continue',
        'if [[ "${protected}" == "true" ]]',
        '-f state=open -f head="${owner}:${branch}"',
        'skip open PR: ${branch}',
        'latest_sha=',
        'latest_open_prs=',
        'latest_active_runs=',
        '"${latest_sha}" != "${sha}"',
        '"${latest_open_prs}" != "0"',
        '"${latest_active_runs}" != "0"',
        'skip race: ${branch} changed while cleanup was running',
    ):
        assert needle in source, f'repository cleanup safety guard missing: {needle}'

    for needle in (
        '-f state=closed -f head="${owner}:${branch}"',
        'select(.merged_at != null and .head.sha == $sha)',
        'select(.merged_at == null and .head.sha == $sha)',
        '.closed_at',
        'retention_seconds="$((CLOSED_PR_RETENTION_DAYS * 86400))"',
        'age_seconds="$((now_epoch - closed_epoch))"',
        'closed-unmerged PR head older than ${CLOSED_PR_RETENTION_DAYS}d',
        'skip recently closed PR head: ${branch}',
    ):
        assert needle in source, f'closed-PR reclamation contract missing: {needle}'

    assert 'is_explicitly_retired "${branch}"' in source
    delete_line = 'git -C "${graph}" push --quiet origin ":refs/heads/${branch}"'
    assert source.count(delete_line) == 1, 'branch deletion must have one guarded owner'

    print('test_repository_cleanup_contract: ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
