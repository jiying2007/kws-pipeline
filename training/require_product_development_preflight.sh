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
[[ "$pr_head_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "governed training pull request head SHA is invalid" >&2
  exit 2
}

runs="$(gh api \
  -H 'Accept: application/vnd.github+json' \
  "repos/$GITHUB_REPOSITORY/actions/runs?head_sha=$pr_head_sha&event=pull_request&per_page=100")"

latest="$(jq -c --arg head "$pr_head_sha" --argjson pr "$pr_number" '
  [
    .workflow_runs[]
    | select(
        .name == "model-training-preflight"
        and .event == "pull_request"
        and .head_sha == $head
        and any(.pull_requests[]?; .number == $pr)
      )
  ]
  | sort_by(.run_number)
  | last // empty
' <<<"$runs")"

test -n "$latest" || {
  echo "no product development preflight run found for PR #$pr_number head $pr_head_sha" >&2
  exit 2
}

status="$(jq -r '.status' <<<"$latest")"
conclusion="$(jq -r '.conclusion // ""' <<<"$latest")"
run_id="$(jq -r '.id' <<<"$latest")"

if [ "$status" != "completed" ] || [ "$conclusion" != "success" ]; then
  echo "latest product development preflight is not successful: PR #$pr_number run=$run_id status=$status conclusion=$conclusion" >&2
  exit 2
fi

echo "product development preflight verified: PR #$pr_number head=$pr_head_sha run=$run_id"
