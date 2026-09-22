#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

contract="${KWS_PRODUCT_BASE_CONTRACT:-configs/training/product-speech-like-base-v1.json}"
effective_config="${KWS_EFFECTIVE_TRAINING_CONFIG:-.generated/xiaowo.product-effective.json}"
base_root="${KWS_PRODUCT_BASE_ROOT:-.generated/product-speech-like-base}"
provider_root="${KWS_REPLAY_PROVIDER_ROOT:-.generated/product-replay-provider}"
release_root="${KWS_PRODUCT_RELEASE_ROOT:-.generated/product-speech-like-release}"
provider_cache="${KWS_REPLAY_PROVIDER_CACHE:-.generated/product-replay-provider-cache}"
source_config="${KWS_SOURCE_TRAINING_CONFIG:-configs/training/xiaowo.torch-domain.json}"
retry_attempts="${KWS_DOWNLOAD_RETRY_ATTEMPTS:-4}"
retry_delay_seconds="${KWS_DOWNLOAD_RETRY_DELAY_SECONDS:-3}"

case "$retry_attempts" in
  ''|*[!0-9]*) echo "KWS_DOWNLOAD_RETRY_ATTEMPTS must be a positive integer" >&2; exit 2 ;;
esac
case "$retry_delay_seconds" in
  ''|*[!0-9]*) echo "KWS_DOWNLOAD_RETRY_DELAY_SECONDS must be a non-negative integer" >&2; exit 2 ;;
esac
if [ "$retry_attempts" -lt 1 ]; then
  echo "KWS_DOWNLOAD_RETRY_ATTEMPTS must be >= 1" >&2
  exit 2
fi

retry_to_file() {
  out=$1
  shift
  attempt=1
  while :; do
    tmp="$out.tmp"
    rm -f "$tmp"
    if "$@" >"$tmp"; then
      mv "$tmp" "$out"
      return 0
    fi
    rm -f "$tmp"
    if [ "$attempt" -ge "$retry_attempts" ]; then
      echo "command failed after $attempt attempt(s): $*" >&2
      return 1
    fi
    echo "transient command failure; retrying ($attempt/$retry_attempts): $*" >&2
    sleep "$retry_delay_seconds"
    attempt=$((attempt + 1))
  done
}

retry_release_download() {
  attempt=1
  while :; do
    rm -f "$release_root/$archive" "$release_root/SPEECH_LIKE_BASE_SHA256SUMS"
    if gh release download "$release_tag"       --repo "$GITHUB_REPOSITORY"       --dir "$release_root"       --clobber; then
      return 0
    fi
    if [ "$attempt" -ge "$retry_attempts" ]; then
      echo "release download failed after $attempt attempt(s): $release_tag" >&2
      return 1
    fi
    echo "transient release download failure; retrying ($attempt/$retry_attempts): $release_tag" >&2
    sleep "$retry_delay_seconds"
    attempt=$((attempt + 1))
  done
}

for path in "$contract" "$source_config"; do
  test -s "$path" || {
    echo "required product training input is missing: $path" >&2
    exit 2
  }
done

release_tag="$(jq -r '.release_tag' "$contract")"
release_target="$(jq -r '.source_repro_head_sha' "$contract")"
release_id="$(jq -r '.release_id' "$contract")"
archive="$(jq -r '.release_archive' "$contract")"
archive_sha="$(jq -r '.release_archive_sha256' "$contract")"

rm -rf "$release_root" "$base_root" "$provider_root" "$provider_cache"
mkdir -p "$release_root" "$base_root"

retry_to_file "$release_root/release.json"   gh api "repos/$GITHUB_REPOSITORY/releases/tags/$release_tag"
jq -e --arg target "$release_target" --argjson release_id "$release_id" '
  .id == $release_id
  and .draft == false
  and .prerelease == false
  and .immutable == true
  and .target_commitish == $target
' "$release_root/release.json" >/dev/null
jq -e --arg name "$archive" --arg digest "sha256:$archive_sha" '
  [.assets[] | select(.name == $name and .digest == $digest)] | length == 1
' "$release_root/release.json" >/dev/null

retry_release_download

(
  cd "$release_root"
  sha256sum -c SPEECH_LIKE_BASE_SHA256SUMS
)

tar -xzf "$release_root/$archive" -C "$base_root"

python3 tools/bootstrap_speech_like_stage_a.py \
  --provider-only \
  --cache-dir "$provider_cache" \
  --work-dir "$provider_root"

python3 training/materialize_product_training_config.py \
  --source-config "$source_config" \
  --base-contract "$contract" \
  --bundle-root "$base_root/corpus" \
  --replay-provider "$provider_root/corpus/provider/provider.json" \
  --voice-inventory "$provider_root/corpus/provider/voice-inventory.jsonl" \
  --output "$effective_config"

python3 training/verify_training_entry_contract.py \
  --config "$effective_config" \
  --require-product-speech-like-base

echo "governed product speech-like base materialized: $effective_config"
