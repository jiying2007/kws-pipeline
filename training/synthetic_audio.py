from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import struct
import subprocess
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens  # noqa: E402

from frontend_spec import FFT_SIZE, SAMPLE_RATE_HZ, mel_bins

SPLITS = ("train", "calibration", "test", "qualification")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", -1)) != 1:
        raise ValueError("synthetic config schema_version must be 1")
    return value


def parse_keywords(path: pathlib.Path, token_map: dict[str, int]) -> list[dict]:
    items: list[dict] = []
    seen_ids: set[int] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise ValueError(
                f"{path}:{line_no}: production synthetic input requires 4 TSV columns"
            )
        keyword_id = int(cols[0])
        text = cols[1].strip()
        tokens = cols[3].split()
        if keyword_id in seen_ids or not text or not tokens:
            raise ValueError(f"{path}:{line_no}: invalid/duplicate keyword")
        missing = [token for token in tokens if token not in token_map]
        if missing:
            raise ValueError(f"{path}:{line_no}: unknown tokens: {', '.join(missing)}")
        items.append(
            {
                "id": keyword_id,
                "text": text,
                "tokens": tokens,
                "token_ids": [token_map[token] for token in tokens],
            }
        )
        seen_ids.add(keyword_id)
    if not items:
        raise ValueError("keyword TSV contains no keywords")
    return items


def token_carriers(keywords: list[dict], feature_dim: int) -> dict[str, dict]:
    active: list[str] = []
    for keyword in keywords:
        for token in keyword["tokens"]:
            if token not in active:
                active.append(token)
    if len(active) > min(feature_dim - 4, 24):
        raise ValueError("prototype tone backend supports at most 24 active tokens")
    bins = mel_bins(feature_dim)
    low = 3
    high = feature_dim - 4
    if len(active) == 1:
        feature_indices = [(low + high) // 2]
    else:
        feature_indices = [
            int(round(low + index * (high - low) / (len(active) - 1)))
            for index in range(len(active))
        ]
    result: dict[str, dict] = {}
    for token, feature_index in zip(active, feature_indices):
        center_bin = bins[feature_index + 1]
        result[token] = {
            "feature_index": feature_index,
            "frequency_hz": center_bin * SAMPLE_RATE_HZ / FFT_SIZE,
        }
    return result


def clamp16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def silence(seconds: float) -> list[int]:
    return [0] * max(0, int(round(seconds * SAMPLE_RATE_HZ)))


def tone_segment(
    frequency_hz: float,
    seconds: float,
    amplitude: float,
    phase: float,
) -> list[int]:
    count = max(1, int(round(seconds * SAMPLE_RATE_HZ)))
    fade = max(1, min(count // 4, int(0.012 * SAMPLE_RATE_HZ)))
    result: list[int] = []
    for index in range(count):
        envelope = 1.0
        if index < fade:
            envelope = index / fade
        elif index >= count - fade:
            envelope = (count - 1 - index) / fade
        sample = amplitude * envelope * math.sin(
            2.0 * math.pi * frequency_hz * index / SAMPLE_RATE_HZ + phase
        )
        result.append(clamp16(sample))
    return result


def noise_profile(profile: str, count: int, rng: random.Random) -> list[float]:
    values: list[float] = []
    pink = 0.0
    for index in range(count):
        white = rng.uniform(-1.0, 1.0)
        pink = 0.94 * pink + 0.06 * white
        t = index / SAMPLE_RATE_HZ
        if profile == "fan":
            value = 0.55 * pink + 0.45 * math.sin(2.0 * math.pi * 118.0 * t)
        elif profile == "motor":
            value = (
                0.35 * pink
                + 0.40 * math.sin(2.0 * math.pi * 183.0 * t)
                + 0.25 * math.sin(2.0 * math.pi * 366.0 * t)
            )
        elif profile == "media":
            value = (
                0.35 * pink
                + 0.25 * math.sin(2.0 * math.pi * 260.0 * t)
                + 0.20 * math.sin(2.0 * math.pi * 530.0 * t)
                + 0.20 * math.sin(2.0 * math.pi * 910.0 * t)
            )
        else:
            value = white
        values.append(value)
    return values


def augment(samples: list[int], rng: random.Random, cfg: dict) -> list[int]:
    if not samples:
        return samples
    gain_db = rng.uniform(
        float(cfg.get("gain_db_min", -4.0)), float(cfg.get("gain_db_max", 3.0))
    )
    gain = 10.0 ** (gain_db / 20.0)
    result = [float(sample) * gain for sample in samples]

    echo_max_ms = float(cfg.get("echo_max_ms", 45.0))
    echo_gain_max = float(cfg.get("echo_gain_max", 0.20))
    if echo_max_ms > 0.0 and echo_gain_max > 0.0:
        delay = int(rng.uniform(8.0, echo_max_ms) * SAMPLE_RATE_HZ / 1000.0)
        echo_gain = rng.uniform(0.0, echo_gain_max)
        dry = list(result)
        for index in range(delay, len(result)):
            result[index] += dry[index - delay] * echo_gain

    snr_min = float(cfg.get("snr_db_min", 16.0))
    snr_max = float(cfg.get("snr_db_max", 36.0))
    profiles = cfg.get("noise_profiles", ["white", "fan", "motor", "media"])
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("augment.noise_profiles must be a non-empty list")
    noise = noise_profile(str(rng.choice(profiles)), len(result), rng)
    signal_rms = math.sqrt(sum(value * value for value in result) / len(result) + 1.0e-9)
    noise_rms = math.sqrt(sum(value * value for value in noise) / len(noise) + 1.0e-9)
    snr_db = rng.uniform(snr_min, snr_max)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise_scale = target_noise_rms / noise_rms
    return [clamp16(value + n * noise_scale) for value, n in zip(result, noise)]


def render_tone_tokens(
    token_names: list[str],
    carriers: dict[str, dict],
    rng: random.Random,
    cfg: dict,
) -> list[int]:
    token_ms = float(cfg.get("token_ms", 170.0))
    token_jitter = float(cfg.get("token_ms_jitter", 25.0))
    gap_ms = float(cfg.get("gap_ms", 70.0))
    lead_ms = float(cfg.get("lead_ms", 160.0))
    tail_ms = float(cfg.get("tail_ms", 180.0))
    amplitude = float(cfg.get("amplitude", 12000.0))
    pitch_jitter = float(cfg.get("pitch_jitter", 0.018))

    samples = silence(lead_ms / 1000.0)
    for index, token in enumerate(token_names):
        carrier = carriers[token]["frequency_hz"]
        frequency = carrier * (1.0 + rng.uniform(-pitch_jitter, pitch_jitter))
        seconds = max(
            0.09, (token_ms + rng.uniform(-token_jitter, token_jitter)) / 1000.0
        )
        samples.extend(
            tone_segment(frequency, seconds, amplitude, rng.uniform(0.0, math.pi))
        )
        if index + 1 != len(token_names):
            samples.extend(silence(gap_ms / 1000.0))
    samples.extend(silence(tail_ms / 1000.0))
    return samples


def render_command_tts(text: str, output: pathlib.Path, cfg: dict) -> list[int]:
    command = cfg.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError("command TTS backend requires generator.tts.command argv list")
    argv = [str(part).format(text=text, output=str(output)) for part in command]
    subprocess.run(argv, check=True)
    with wave.open(str(output), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError("command TTS must emit mono 16-kHz PCM16 WAV")
        raw = reader.readframes(reader.getnframes())
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def write_wav(path: pathlib.Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def contains_subsequence(sequence: list[str], pattern: list[str]) -> bool:
    if not pattern:
        return True
    pattern_index = 0
    for token in sequence:
        if token == pattern[pattern_index]:
            pattern_index += 1
            if pattern_index == len(pattern):
                return True
    return False


def safe_negative(sequence: list[str], forbidden: list[list[str]]) -> bool:
    return not any(contains_subsequence(sequence, pattern) for pattern in forbidden)


def confusable_sequence(
    tokens: list[str],
    active: list[str],
    forbidden: list[list[str]],
    rng: random.Random,
) -> list[str]:
    if not tokens:
        return []
    for _ in range(128):
        mode = rng.choice(("drop", "substitute", "swap", "partial"))
        result = list(tokens)
        if mode == "drop" and len(result) > 1:
            del result[rng.randrange(len(result))]
        elif mode == "substitute":
            index = rng.randrange(len(result))
            choices = [token for token in active if token != result[index]]
            if choices:
                result[index] = rng.choice(choices)
        elif mode == "swap" and len(result) > 1:
            index = rng.randrange(len(result) - 1)
            result[index], result[index + 1] = result[index + 1], result[index]
        elif len(result) > 1:
            result = result[: rng.randrange(1, len(result))]
        if result != tokens and safe_negative(result, forbidden):
            return result
    return []


def random_negative(
    active: list[str], forbidden: list[list[str]], rng: random.Random
) -> list[str]:
    for _ in range(256):
        length = rng.randint(1, max(1, min(5, len(active) + 1)))
        result = [rng.choice(active) for _ in range(length)]
        if safe_negative(result, forbidden):
            return result
    return []


def generate_dataset(config_path: pathlib.Path, output: pathlib.Path) -> dict:
    config = load_config(config_path)
    tokens_path = pathlib.Path(config["tokens"])
    keywords_path = pathlib.Path(config["keywords"])
    if not tokens_path.is_absolute():
        tokens_path = (ROOT / tokens_path).resolve()
    if not keywords_path.is_absolute():
        keywords_path = (ROOT / keywords_path).resolve()
    token_map = load_tokens(tokens_path)
    keywords = parse_keywords(keywords_path, token_map)
    feature_dim = int(config.get("model", {}).get("feature_dim", 32))
    carriers = token_carriers(keywords, feature_dim)
    active = list(carriers)
    forbidden = [list(keyword["tokens"]) for keyword in keywords]
    seed = int(config.get("seed", 1337))
    generator_cfg = config.get("generator", {})
    tts_cfg = generator_cfg.get("tts", {"backend": "tone"})
    backend = str(tts_cfg.get("backend", "tone"))
    augment_cfg = generator_cfg.get("augment", {})
    split_cfg = config.get("dataset", {})
    continuous_gap_ms = float(generator_cfg.get("continuous_gap_ms", 1600.0))
    if continuous_gap_ms < 1300.0:
        raise ValueError(
            "generator.continuous_gap_ms must be >= 1300 to isolate the default "
            "1200-ms refractory and decoder carry-over between synthetic clips"
        )

    output.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    split_rows: dict[str, list[tuple[pathlib.Path, list[int], dict]]] = {
        name: [] for name in SPLITS
    }

    for split_index, split in enumerate(SPLITS):
        cfg = split_cfg.get(split)
        if not isinstance(cfg, dict):
            raise ValueError(f"dataset.{split} must be an object")
        positive = int(cfg.get("positive_families_per_keyword", 0))
        confusable = int(cfg.get("confusable_families_per_keyword", 0))
        random_count = int(cfg.get("random_negative_families", 0))
        variants = int(cfg.get("variants_per_family", 1))
        if min(positive, confusable, random_count, variants) < 0 or variants <= 0:
            raise ValueError(f"dataset.{split}: invalid counts")

        def add_family(
            kind: str,
            keyword: dict | None,
            family_index: int,
            token_names: list[str],
            target_ids: list[int],
        ) -> None:
            if kind != "positive" and not safe_negative(token_names, forbidden):
                raise ValueError(f"{kind} family accidentally contains a wake path")
            family_id = (
                f"{split}-{kind}-{keyword['id'] if keyword else 0}-{family_index}"
            )
            for variant in range(variants):
                family_seed = (
                    seed
                    + split_index * 1_000_003
                    + family_index * 997
                    + variant * 37
                )
                if keyword is not None:
                    family_seed += int(keyword["id"]) * 7919 + (
                        0 if kind == "positive" else 313
                    )
                rng = random.Random(family_seed)
                clip = output / "clips" / split / f"{family_id}-v{variant}.wav"
                if backend == "tone":
                    samples = render_tone_tokens(token_names, carriers, rng, tts_cfg)
                elif backend == "command":
                    if keyword is None or kind != "positive":
                        samples = render_tone_tokens(token_names, carriers, rng, tts_cfg)
                    else:
                        samples = render_command_tts(keyword["text"], clip, tts_cfg)
                else:
                    raise ValueError(f"unsupported TTS backend: {backend}")
                samples = augment(samples, rng, augment_cfg)
                write_wav(clip, samples)
                meta = {
                    "split": split,
                    "kind": kind,
                    "family_id": family_id,
                    "variant": variant,
                    "keyword_id": keyword["id"] if keyword else None,
                    "tokens": token_names,
                    "target_ids": target_ids,
                    "wav_sha256": sha256_file(clip),
                    "frames": len(samples),
                }
                index_rows.append(meta | {"path": str(clip.resolve())})
                split_rows[split].append((clip, target_ids, meta))

        for keyword in keywords:
            for family in range(positive):
                add_family(
                    "positive",
                    keyword,
                    family,
                    list(keyword["tokens"]),
                    list(keyword["token_ids"]),
                )
            for family in range(confusable):
                rng = random.Random(
                    seed + split_index * 100_003 + keyword["id"] * 1009 + family
                )
                sequence = confusable_sequence(
                    list(keyword["tokens"]), active, forbidden, rng
                )
                add_family("confusable", keyword, family, sequence, [])
        for family in range(random_count):
            rng = random.Random(seed + split_index * 200_003 + family * 1237)
            add_family(
                "negative",
                None,
                family,
                random_negative(active, forbidden, rng),
                [],
            )

    manifest_paths: dict[str, pathlib.Path] = {}
    reference_paths: dict[str, pathlib.Path] = {}
    gap_samples = int(round(continuous_gap_ms * SAMPLE_RATE_HZ / 1000.0))
    for split in SPLITS:
        manifest = output / f"{split}.tsv"
        manifest.write_text(
            "".join(
                f"{path.resolve()}\t{' '.join(str(value) for value in targets)}\n"
                for path, targets, _ in split_rows[split]
            ),
            encoding="utf-8",
        )
        manifest_paths[split] = manifest

        if split == "train":
            continue
        continuous: list[int] = [0] * gap_samples
        expected: list[dict] = []
        for clip, _, meta in split_rows[split]:
            with wave.open(str(clip), "rb") as reader:
                raw = reader.readframes(reader.getnframes())
                samples = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
            start_sample = len(continuous)
            continuous.extend(samples)
            end_sample = len(continuous)
            if meta["kind"] == "positive":
                expected.append(
                    {
                        "keyword_id": int(meta["keyword_id"]),
                        "start_s": start_sample / SAMPLE_RATE_HZ,
                        "end_s": end_sample / SAMPLE_RATE_HZ,
                    }
                )
            continuous.extend([0] * gap_samples)
        wav_path = output / f"{split}.continuous.wav"
        write_wav(wav_path, continuous)
        references = output / f"{split}.references.jsonl"
        references.write_text(
            json.dumps(
                {
                    "recording": f"synthetic-{split}",
                    "path": wav_path.name,
                    "duration_s": len(continuous) / SAMPLE_RATE_HZ,
                    "expected": expected,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        reference_paths[split] = references

    index_path = output / "dataset-index.jsonl"
    index_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in index_rows
        )
        + "\n",
        encoding="utf-8",
    )
    mapping_path = output / "token-carriers.json"
    mapping_path.write_text(
        json.dumps(carriers, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "evidence_class": "synthetic-only",
        "seed": seed,
        "backend": backend,
        "continuous_gap_ms": continuous_gap_ms,
        "tokens_sha256": sha256_file(tokens_path),
        "keywords_sha256": sha256_file(keywords_path),
        "config_sha256": sha256_file(config_path),
        "mapping_sha256": sha256_file(mapping_path),
        "index_sha256": sha256_file(index_path),
        "splits": {
            split: {
                "examples": len(split_rows[split]),
                "manifest": str(manifest_paths[split]),
                "manifest_sha256": sha256_file(manifest_paths[split]),
                **(
                    {
                        "references": str(reference_paths[split]),
                        "references_sha256": sha256_file(reference_paths[split]),
                    }
                    if split in reference_paths
                    else {}
                ),
            }
            for split in SPLITS
        },
    }
    summary_path = output / "dataset-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    summary = generate_dataset(args.config.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
