#!/usr/bin/env bash
# Invoked only by successful release cleanup jobs. Bind deletion to the triggering
# commit, never the branch tip observed by a later API read or another run.
set -euo pipefail

case "${1:-}" in
  sdk) [[ "${GITHUB_REF_NAME:-}" =~ ^release/v[0-9]+\.[0-9]+\.[0-9]+$ ]] ;;
  deployment) [[ "${GITHUB_REF_NAME:-}" == deployment/commercial-candidate ]] ;;
  *) echo 'expected sdk or deployment bootstrap cleanup mode' >&2; exit 2 ;;
esac
[[ "${GITHUB_REF:-}" == "refs/heads/${GITHUB_REF_NAME}" ]]
[[ "${GITHUB_SHA:-}" =~ ^[0-9a-f]{40}$ ]]
[[ "${GITHUB_REPOSITORY:-}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]

gh auth setup-git >/dev/null
graph="$(mktemp -d "${RUNNER_TEMP:?}/bootstrap-cleanup.XXXXXX")"
trap 'rm -rf -- "$graph"' EXIT
git init -q "$graph"
git -C "$graph" remote add origin "https://github.com/${GITHUB_REPOSITORY}.git"
# Git's atomic compare-and-delete lease protects B when this is an older A run,
# including an update after any preflight check. API DELETE has no such lease.
git -C "$graph" push --quiet \
  --force-with-lease="refs/heads/${GITHUB_REF_NAME}:${GITHUB_SHA}" \
  origin ":refs/heads/${GITHUB_REF_NAME}"
