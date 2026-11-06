#!/usr/bin/env python3
"""Select existing research checks without an API call or a path-filtered gate."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess

EXACT = frozenset({
    'configs/runtime/research-runtime.json', 'tools/research_runtime.py',
    'tools/research_python_entry.py', 'tests/test_research_python_entry.py',
    'training/setup_core_runtime.sh', '.github/actions/setup-research-python/action.yml',
    'tests/test_research_runtime.py',
    'tools/setup_asr_runtime.py', 'tools/active_asr_batch.py',
    'tools/active_asr_worker.py', 'tools/summarize_active_asr_batch.py',
    'tests/test_active_asr_batch.py', 'tests/test_active_asr_supervision.py', 'docs/ACTIVE_ASR_BATCH.md',
    'tools/active_asr_selective.py', 'tests/test_active_asr_selective.py',
    'configs/research/selective-asr-v2.json', 'docs/ACTIVE_ASR_SELECTIVE_V2.md',
    'tools/research_job_lock.py', 'tests/test_research_job_lock.py',
    'tools/run_zipformer_system.py', 'tests/test_zipformer_system.py',
    'tools/run_d1_zipformer_system.py', 'tests/test_d1_zipformer_system.py',
    'docs/research-d1-zipformer-native.md',
    'tools/run_d1_zipformer_extension.py', 'tests/test_d1_zipformer_extension.py',
    'tools/run_d1_zipformer_continuation.py',
    'tools/research_d1_native_coverage.py', 'tests/test_d1_native_coverage.py',
    'tools/research_d1_previous_native.py', 'docs/research-d1-zipformer-extension-v2.md',
    'docs/research-d1-zipformer-accounted-v3.md', 'docs/research-d1-partial-observation-v3.md',
    'configs/research/zipformer-system-20261105.lock.json', 'docs/ZIPFORMER_SYSTEM.md',
    'tools/research_queue.py', 'tests/test_research_queue.py', 'docs/LOCAL_RESEARCH_QUEUE.md',
    'tools/research_queue_supervisor.py', 'tests/test_research_queue_supervisor.py',
    'docs/templates/kws-research-queue.service',
    'tools/research_d2_generate.py', 'tests/test_research_d2_generate.py',
    'tools/research_d2_campaign.py', 'tests/test_research_d2_campaign.py',
    'tools/research_d2_checkpoints.py', 'tools/research_d2_recovery.py',
    'tests/test_research_d2_recovery.py', 'docs/D2_DURABLE_RECOVERY.md',
    'docs/D2_CAMPAIGN_DRIVER.md', 'docs/templates/kws-d2-campaign.service',
    'training/research_d2_inputs.py', 'training/research_d2_consume.py',
    'tests/test_research_d2_consume.py', 'docs/research-d2-consumption-probe.md',
    'tools/research_context_generate.py', 'tests/test_research_context_generate.py',
    'training/research_context_inputs.py', 'tests/test_research_context_inputs.py',
    'docs/CONTEXT_FULLTEXT_INPUTS.md',
    'tools/research_context_v2_generate.py', 'tests/test_research_context_v2_generate.py',
    'tools/research_context_v3_generate.py', 'tests/test_research_context_v3_generate.py',
    'docs/CONTEXT_V3_GENERATION.md',
    'tools/research_context_v3_campaign.py', 'tests/test_research_context_v3_campaign.py',
    'docs/CONTEXT_V3_CAMPAIGN.md',
    'training/research_context_v3_inputs.py', 'tests/test_research_context_v3_inputs.py',
    'docs/research-context-v3-inputs.md',
    'tools/research_context_partial.py', 'tests/test_research_context_partial.py',
    'docs/research-context-partial-observation.md',
    'training/research_context_v2_inputs.py', 'tests/test_research_context_v2_inputs.py',
    'docs/CONTEXT_V2_FULLTEXT_INPUTS.md',
    'training/research_d1_fit_diagnostic.py', 'tests/test_research_d1_fit_diagnostic.py',
    'tools/research_fit_budget_compare.py', 'tests/test_research_fit_budget_compare.py',
    'docs/research-fit-exposure-comparison.md',
    'training/research_acoustic_augment.py', 'tests/test_research_acoustic_augment.py',
    'docs/research-d1-tiny-fit-diagnostic.md', 'docs/KWS_D2_IMPLEMENTATION_20261105.md',
    'tools/research_no_network.py', 'tests/test_research_no_network.py',
    'tools/setup_tts_runtime.py', 'tests/test_active_tts_runtime.py',
    'configs/runtime/qwen-base-tts-model-lock.json',
    'tools/active_tts_smoke.py', 'tests/test_active_tts_smoke.py',
    'tools/active_tts_pair.py', 'tests/test_active_tts_pair.py', 'docs/active-tts-smoke.md',
    'tools/active_tts_expand.py', 'tests/test_active_tts_expand.py', 'docs/research-tts-continuation.md',
    'tools/active_tts_continuation.py', 'tools/research_tts_partial.py', 'tools/research_tts_handoff.py',
    'tests/test_active_tts_continuation.py', 'docs/tts-accounted-checkpoint-v2.md',
    'training/research_weak_recipe.py', 'training/research_weak_train.py',
    'training/research_fsmn_weak.py', 'configs/runtime/fsmn-fulltext-sources.json',
    'tests/test_research_weak_training.py', 'tests/test_research_fsmn_weak.py',
    'docs/research-weak-training.md', 'configs/research/active-plan-20261105.json',
    'docs/KWS_ACTIVE_EXECUTION_20261105.md',
    'training/research_d1_vocab.py', 'training/research_d1_inputs.py',
    'training/research_d1_recipe.py', 'training/research_d1_models.py',
    'training/research_d1_train.py',
    'training/research_d1_evaluate.py', 'tests/test_research_d1_evaluate.py',
    'tests/test_research_d1.py', 'tests/fixtures/d1-fulltext/fsmn-official-tokens.txt',
    'tests/fixtures/d1-fulltext/research_weak_sampling_v1.py',
    'tests/fixtures/d1-fulltext/README.md', 'docs/research-d1-fulltext-v2.md',
    'docs/research-d1-development-v2.md',
    'tests/test_causal_research_contract.py',
    'tests/test_research_three_window_c.py',
    'tools/verify_research_sources.py', 'tools/verify_research_publication.py',
    'tools/run_research_source_checks.py', 'tools/check_workflow_path_filters.py',
    'tools/research_ci_changes.py', 'tests/test_research_ci_changes.py',
    'tools/speech_data_queue.py', 'tools/import_saved_speech_result.py',
    'training/weak_speech_dataset.py', 'tests/test_speech_data_queue.py',
    'docs/speech-data-queue.md',
    'docs/language-conditioned-weak-import.md',
    'tests/test_language_conditioned_queue.py',
    'tools/import_language_conditioned_result.py',
    'tools/language_conditioned_vote.py',
    'tools/speech_history_evidence.py',
    'docs/batch-weak-import.md',
    'tests/test_batch_weak_import.py',
    'tests/test_batch_weak_import_integration.py',
    'tools/batch_weak_import.py',
    'tools/batch-source-contract.json',
})


def relevant(paths: list[str]) -> bool:
    return any(path.startswith(('research/', '.github/workflows/', 'host-tts-verifier-source/')) or path in EXACT
               for path in paths)


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], text=True)


def changed_paths(event: str, payload: dict, ref: str) -> list[str]:
    if event == 'pull_request':
        # Default actions/checkout checks out GitHub's PR merge commit. Its first
        # parent is the current base, so this includes the complete merged PR
        # delta, not just the most recent head commit or unrelated base changes.
        parents = git('show', '-s', '--format=%P', 'HEAD').split()
        if len(parents) != 2:
            raise ValueError('expected the GitHub pull-request merge commit')
        base = parents[0]
    elif event == 'push' and ref == 'refs/heads/main':
        base = payload['before']
        if not isinstance(base, str) or len(base) != 40 or any(c not in '0123456789abcdef' for c in base):
            raise ValueError('invalid push before SHA')
        if base == '0' * 40:
            return git('ls-files', '-z').split('\0')
        exists = subprocess.run(['git', 'cat-file', '-e', base + '^{commit}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if exists.returncode:
            # A multi-commit/force push may exceed checkout's two-commit depth.
            # Fetch only its exact before SHA once. Unavailable history fails
            # closed; it is never converted to "no relevant changes".
            subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', 'origin', base], check=True)
    else:
        # Preserve the previous source workflow's event scope. Feature/codex
        # pushes are covered by their PR; consolidate pushes retain their own
        # path-filtered source workflow trigger.
        return []
    return git('diff', '--name-only', '--no-renames', '-z', base, 'HEAD', '--').split('\0')


def main() -> None:
    event = os.environ['GITHUB_EVENT_NAME']
    payload = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    needed = relevant(changed_paths(event, payload, os.environ['GITHUB_REF']))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'required={str(needed).lower()}\n')
    print(f'research source checks required: {needed}')


if __name__ == '__main__':
    main()
