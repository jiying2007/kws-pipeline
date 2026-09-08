#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"

if [[ "$GITHUB_REF" != "refs/heads/main" ]]; then
  echo "manual qualification/promotion must run from refs/heads/main; got $GITHUB_REF" >&2
  exit 2
fi

main_json="$(gh api "repos/${GITHUB_REPOSITORY}/branches/main")"
main_sha="$(jq -r '.commit.sha' <<<"$main_json")"
protected="$(jq -r '.protected' <<<"$main_json")"
if [[ "$GITHUB_SHA" != "$main_sha" ]]; then
  echo "manual qualification/promotion must run from current main: workflow=$GITHUB_SHA main=$main_sha" >&2
  exit 2
fi
if [[ "$protected" != true ]]; then
  echo "main is not protected" >&2
  exit 2
fi

ruleset_name="$(jq -r '.name' governance/main-ruleset-target.json)"
rulesets_json="$(gh api "repos/${GITHUB_REPOSITORY}/rulesets")"
count="$(jq --arg name "$ruleset_name" '[.[] | select(.name == $name and .target == "branch" and .enforcement == "active")] | length' <<<"$rulesets_json")"
if [[ "$count" != 1 ]]; then
  echo "expected exactly one active branch ruleset named $ruleset_name; got $count" >&2
  exit 2
fi
ruleset_id="$(jq -r --arg name "$ruleset_name" '[.[] | select(.name == $name and .target == "branch" and .enforcement == "active")][0].id' <<<"$rulesets_json")"
mkdir -p build/control-plane
gh api "repos/${GITHUB_REPOSITORY}/rulesets/${ruleset_id}" > build/control-plane/live-main-ruleset.json
python3 governance/verify_live_main_ruleset.py \
  --target governance/main-ruleset-target.json \
  --live build/control-plane/live-main-ruleset.json \
  > build/control-plane/verification.json

echo "qualified current-main control plane: $main_sha"
