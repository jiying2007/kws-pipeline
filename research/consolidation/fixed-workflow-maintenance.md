# Fixed research workflow dependency maintenance

Reviewed against main `a38d0262eafe4e6111c559026aad71d781423701` on 2026-10-19.

## Scope and reason

Dependabot PR #612 changes only upload-artifact pins in four one-time research
workflows. Their byte identities are part of reviewed source/evidence contracts,
not ordinary continuously maintained CI. The retention check correctly rejects
an unrecorded change. Updating only its current workflow hashes is insufficient:
the ASR activation tests also compare the registered workflow with its reviewed
template and historical source hash.

The github-actions update configuration excludes exactly these files:

- `.github/workflows/kw1-preset-supplement-20261014-v1.yml`
- `.github/workflows/tts-pilot-asr-20261014-v1.yml`
- `.github/workflows/tts-quality-probe-20261014-v1.yml`
- `.github/workflows/tts-quality-remaining-20261014-v1.yml`

No dependency-wide ignore, wildcard, group exclusion, new workflow, trigger,
activation, runner change, or execution authority is introduced. Ordinary CI,
observation workflows, and future independently named campaigns remain in the
existing weekly update scan. This also leaves the existing schedule and group
unchanged. Exclusion is dependency-maintenance policy, not a claim that the old
action is permanently secure or supported.

## Retention is not execution

These files remain in `.github/workflows` because current inventory checks and
reviewed template/source bindings require those exact paths and bytes. Removing,
renaming, or editing them is a separate retirement/migration project, not needed
to stop a dependency bot from rewriting their fixed identities. Keeping a file
registered does not grant permission or establish eligibility to run it again.
They target exact research branches and require first-created-branch and first
workflow-lifetime run conditions. No workflow is dispatched by this change.

Do not trust old comments saying "disabled" as a current admission verdict:
main's ASR and KW1 admission records include `enabled: true` historical approvals;
source bindings, first-run conditions and existing durable execution evidence
must still be evaluated. This change neither renews nor modifies those approvals.
We have not asserted all four workflows are disabled in the GitHub service.

## Security and future upgrades

GitHub's documented `exclude-paths` option accepts file paths relative to the
update entry's `directory`; here it is `/`. The four literal paths exclude only
those manifests. Official reference:
https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference#exclude-paths

If a vulnerability, runtime retirement or newly approved campaign makes an upgrade
necessary, review it independently. Preserve historical workflow/template bytes,
source manifests, activation receipts and artifact identities. Add a separately
reviewed current revision or new campaign with explicit provenance and offline
regressions; do not rewrite historical pins to make a new upload action look like
the old approved execution. Never reset workflow lifetime identity or rerun a
consumed campaign as part of dependency maintenance. A security finding requires
manual triage even when Dependabot excludes these fixed files.

## Verification and PR #612 disposition

Six focused tests in `source_tests/test_source_retention.py` verify exact scoped
configuration, preserved workflow hashes, rejection of missing/broad/duplicate/
misplaced exclusions and global dependency ignores, continued coverage of ordinary/future workflows, and acceptance of unrelated
schedule changes and reordered exclusions. The stdlib helper reads only the
existing block-YAML ecosystem/directory/exclude-paths shape; it is not a full
YAML parser or schema validation. Independent PyYAML semantic validation is also
performed when preparing this change. Existing CI already runs this file normally, with
`-O`, and with `-OO`; no CI workflow edit is needed.

After this maintenance change is integrated and its exact-head CI passes, review
#612 again and close it as superseded if it still consists solely of the four
fixed-workflow pin replacements. Closing it is a separate authorized action;
this patch does not comment on, alter, close or merge #612. Do not merge #612
merely to silence its failing check or add an ignore-all dependency command.
