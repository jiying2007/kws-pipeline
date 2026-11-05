#!/usr/bin/env bash
set -euo pipefail

contract="${KWS_PRODUCT_BASE_CONTRACT:-configs/training/product-speech-like-base-v1.json}"
effective_config="${KWS_EFFECTIVE_TRAINING_CONFIG:-.generated/xiaowo.product-effective.json}"
base_root="${KWS_PRODUCT_BASE_ROOT:-.generated/product-speech-like-base}"
provider_root="${KWS_REPLAY_PROVIDER_ROOT:-.generated/product-replay-provider}"
provider_contract="${KWS_REPLAY_PROVIDER_CONTRACT:-configs/training/product-replay-provider-semantic-v1.json}"
stage_root="${KWS_RESEARCH_BASE_STAGE_ROOT:-.generated/research-speech-like-base}"
provider_cache="${KWS_REPLAY_PROVIDER_CACHE:-.generated/product-replay-provider-cache}"
source_config="${KWS_SOURCE_TRAINING_CONFIG:-configs/training/xiaowo.torch-domain.json}"
for path in "$contract" "$provider_contract" "$source_config"; do
  test -s "$path" || {
    echo "required product training input is missing: $path" >&2
    exit 2
  }
done

# The original synthetic reference is in Git source with exact SHA256 and
# an explicit non-admission research receipt; no Release lookup is allowed.
archive="$(jq -er '.release_archive' "$contract")"
archive_sha="$(jq -er '.release_archive_sha256' "$contract")"
research_archive="${KWS_RESEARCH_BASE_ARCHIVE:-research/legacy-source-bases/$archive}"
test -f "$research_archive" || {
  echo "required frozen research base archive is missing: $research_archive" >&2
  exit 2
}
research_archive="$(realpath -- "$research_archive")"
for generated in "$stage_root" "$base_root" "$provider_root" "$provider_cache"; do
  generated="$(realpath -m -- "$generated")"
  case "$research_archive" in
    "$generated"/*) echo "research base archive must not be inside generated output: $research_archive" >&2; exit 2 ;;
  esac
done
# Validate before deleting prior generated output; never replace bad input.
printf '%s  %s\n' "$archive_sha" "$research_archive" | sha256sum -c --status || {
  echo "frozen research archive SHA256 mismatch" >&2
  exit 2
}
rm -rf -- "$stage_root" "$base_root" "$provider_root" "$provider_cache"
mkdir -p -- "$stage_root" "$base_root"
cp -- "$research_archive" "$stage_root/$archive"
printf '%s  %s\n' "$archive_sha" "$archive" > "$stage_root/SPEECH_LIKE_BASE_SHA256SUMS"

(
  cd "$stage_root"
  sha256sum -c SPEECH_LIKE_BASE_SHA256SUMS
)

tar -xzf "$stage_root/$archive" -C "$base_root"

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

echo "frozen synthetic research base materialized: $effective_config (not real-speech admission)"
