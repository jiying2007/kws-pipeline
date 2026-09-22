#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

contract="${KWS_PRODUCT_BASE_CONTRACT:-configs/training/product-speech-like-base-v1.json}"
effective_config="${KWS_EFFECTIVE_TRAINING_CONFIG:-.generated/xiaowo.product-effective.json}"
base_root="${KWS_PRODUCT_BASE_ROOT:-.generated/product-speech-like-base}"
provider_root="${KWS_REPLAY_PROVIDER_ROOT:-.generated/product-replay-provider}"
provider_contract="${KWS_REPLAY_PROVIDER_CONTRACT:-configs/training/product-replay-provider-semantic-v1.json}"
release_root="${KWS_PRODUCT_RELEASE_ROOT:-.generated/product-speech-like-release}"
provider_cache="${KWS_REPLAY_PROVIDER_CACHE:-.generated/product-replay-provider-cache}"
source_config="${KWS_SOURCE_TRAINING_CONFIG:-configs/training/xiaowo.torch-domain.json}"

for path in "$contract" "$provider_contract" "$source_config"; do
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

gh api "repos/$GITHUB_REPOSITORY/releases/tags/$release_tag" > "$release_root/release.json"
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

gh release download "$release_tag" \
  --repo "$GITHUB_REPOSITORY" \
  --dir "$release_root"

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
  --replay-provider-summary "$provider_root/corpus/provider/provider-summary.json" \
  --replay-provider-contract "$provider_contract" \
  --voice-inventory "$provider_root/corpus/provider/voice-inventory.jsonl" \
  --output "$effective_config"

python3 training/verify_training_entry_contract.py \
  --config "$effective_config" \
  --require-product-speech-like-base

echo "governed product speech-like base materialized: $effective_config"
