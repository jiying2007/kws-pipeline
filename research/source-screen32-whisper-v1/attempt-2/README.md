> Current status, 2026-10-11: the first and second campaign attempts have both
> finished with setup failures and zero audio attempts. Their retained freezes,
> releases, evidence, and run identities are historical and must not be refreshed
> or reactivated. The shared setup now contains dormant dependency-first source
> changes, so the old freezes intentionally reject it as source drift. The
> preparation/activation instructions below describe historical preparation only.
> Do not run either old `prepare_freeze.py` to refresh those records.
> The separately authorized third attempt has not run: its bounded official-pip transport
> is source-prepared under the revised acquisition contract. See
> [the current transport review](../attempt-3/README.md).

# Separately authorized second campaign attempt

This is a new, single campaign attempt after run `38105471896` failed in setup.
It is not a rerun of that workflow or a revision of its result. The first run,
branch, workflow template, execution freeze/release, and eight-file evidence
archive remain unchanged. Its 16 audio rows had zero attempts. The user separately
approved one more same-scope attempt on 2026-10-11, after diagnostics and CI review.

## Fixed identity and admission

- Campaign attempt: **2**; prior run: **38105471896**.
- New branch: `research/qwen16-whisper-small-third-attempt-2-v1`.
- New activation path: `.github/workflows/source-screen32-whisper-third-attempt-2-v1.yml`.
- New artifact: `qwen16-whisper-small-third-attempt-2-terminal-v1`.
- The template is dormant here. `execution-release.json` remains `approved=false`.
- A separately reviewed activation copies this exact template to the new path
  and approves this exact second source-freeze hash on the new branch's first
  public creation. GitHub `run_number=1` and `run_attempt=1` are still required.
  Deleting/recreating the old branch, increasing either counter, dispatching,
  updating the branch, or automatically retrying is not an alternative entrance.
- Only the new workflow is activated. The original release stays false and the
  original active workflow path must be absent. The first workflow and its history
  are not removed from GitHub; the original template remains retained in source.

Admission verifies every new frozen source byte and the exact old provenance plus
all seven original artifact members. It independently checks the failed setup
receipt and all 16 zero-attempt/not-run rows against the same anonymous retained
job. Only then does it call the shared execution function. The receipt records the
new experiment identity, `campaign_attempt=2`, prior run, original source freeze,
prior provenance and terminal hashes, and `automatic_retry=false`.

## Unchanged experiment

The original anonymous sixteen WAVs, native multilingual Whisper small `.pt`,
source/runtime/model locks, URLs, host allowlist, decoding policy and all resource
bounds are unchanged. No new TTS, Qwen/SenseVoice rerun, training, model switch,
result selection or automatic retry is allowed. Job hard limit is 45 minutes,
run step 42 minutes, total host work budget 40 minutes from entry. Failures stop
and retain real partial observations under the same evidence contract. Setup's
new bounded diagnostics were reviewed separately; the first inner failure remains
unknown, and a later HEAD 403 observation does not justify a source/host change.

The only shared runtime change extracts the old execution body into an explicit
function receiving the admitted experiment identity, source-freeze identity, and
optional campaign receipt. The original entry still calls its original admission.
The second entry calls its own admission before the same execution core. Model,
worker, setup, container-tail, terminal validation and collection paths are shared.
The blind mount remains the exact original eight files: no continuation metadata,
prior output, target text, or other ASR output reaches the model.

## Source-only preparation

Run `python3 -S -B research/source-screen32-whisper-v1/attempt-2/prepare_freeze.py`
after any reviewed source edit. This freezes the current bytes of the entire old
closure, old freeze/release, host-only prior evidence and all second-attempt source.
It never rewrites the old freeze/release or activates either gate. The original
freeze describes its old source revision; the changed shared host is intentionally
valid only under the new freeze. Offline coverage remains in the existing
`../test_contract.py`, already wired into normal, `-O` and `-OO` CI. No CI changes
or model installation/inference are needed for source preparation.

All outputs remain `source_screen_only`; training admission, product shipping and
human-gold flags stay false. This is exploratory third-route source screening,
not evidence that three recognizers had been chosen before the original two ran.
