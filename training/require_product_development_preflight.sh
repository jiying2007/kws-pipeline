#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"

[[ "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "governed training SHA must be canonical 40-hex" >&2
  exit 2
}

associated="$(gh api \
  -H 'Accept: application/vnd.github+json' \
  "repos/$GITHUB_REPOSITORY/commits/$GITHUB_SHA/pulls")"

pull="$(jq -c --arg sha "$GITHUB_SHA" '
  [.[] | select(.merged_at != null and .merge_commit_sha == $sha)]
  | if length == 1 then .[0] else empty end
' <<<"$associated")"

test -n "$pull" || {
  echo "governed training push is not bound to exactly one merged pull request" >&2
  exit 2
}

pr_number="$(jq -r '.number' <<<"$pull")"
pr_head_sha="$(jq -r '.head.sha' <<<"$pull")"
pr_base_sha="$(jq -r '.base.sha' <<<"$pull")"
for value in "$pr_head_sha" "$pr_base_sha"; do
  [[ "$value" =~ ^[0-9a-f]{40}$ ]] || {
    echo "governed training pull request head/base SHA is invalid" >&2
    exit 2
  }
done

runs="$(gh api \
  -H 'Accept: application/vnd.github+json' \
  "repos/$GITHUB_REPOSITORY/actions/runs?head_sha=$pr_head_sha&event=pull_request&per_page=100")"

readarray -t candidates < <(
  jq -r --arg head "$pr_head_sha" '
    [
      .workflow_runs[]
      | select(
          .name == "model-training-preflight"
          and .path == ".github/workflows/model-training-preflight.yml"
          and .event == "pull_request"
          and .head_sha == $head
          and .status == "completed"
          and .conclusion == "success"
        )
    ]
    | sort_by(.run_number)
    | reverse
    | .[].id
  ' <<<"$runs"
)

expected_artifact="xiaowo-product-development-preflight-$pr_base_sha"
matched_run=""
for run_id in "${candidates[@]}"; do
  artifacts="$(gh api \
    -H 'Accept: application/vnd.github+json' \
    "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/artifacts?per_page=100")"
  if jq -e --arg name "$expected_artifact" '
    any(.artifacts[]?; .name == $name and .expired == false)
  ' <<<"$artifacts" >/dev/null; then
    matched_run="$run_id"
    break
  fi
done

test -n "$matched_run" || {
  echo "no successful product development preflight artifact found for PR #$pr_number head=$pr_head_sha base=$pr_base_sha" >&2
  exit 2
}

echo "product development preflight verified: PR #$pr_number head=$pr_head_sha base=$pr_base_sha run=$matched_run"
