from __future__ import annotations

import hashlib
import math
import pathlib

from synthetic_audio import UINT32_MAX

WAKE_BALANCE_POLICY = "per-keyword-provenance-pressure-balance-v3"
PRESSURE_ASSIGNMENT_POLICY = "explicit-replay-focus-then-token-edit-distance-v2"
DEFAULT_POSITIVE_EXAMPLE_WEIGHT = 2.0
MAX_WAKE_EXAMPLE_WEIGHT = 12.0


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_targets(path: pathlib.Path) -> list[tuple[int, ...]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"wake-balance manifest is missing/empty: {path}")
    rows: list[tuple[int, ...]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" not in raw:
            raise ValueError(f"{path}:{line_no}: expected WAV<TAB>token_ids")
        _, token_text = raw.split("\t", 1)
        try:
            targets = tuple(int(value) for value in token_text.split())
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid token id") from exc
        if any(value < 0 for value in targets):
            raise ValueError(f"{path}:{line_no}: token ids must be non-negative")
        rows.append(targets)
    if not rows:
        raise ValueError(f"wake-balance manifest has no examples: {path}")
    return rows


def keyword_target_sequences(
    tokens: pathlib.Path,
    keywords: pathlib.Path,
) -> dict[int, tuple[int, ...]]:
    token_map: dict[str, int] = {}
    for line_no, raw in enumerate(tokens.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        cols = value.split()
        if len(cols) == 1:
            token = cols[0]
            token_id = len(token_map)
        elif len(cols) == 2:
            token, raw_id = cols
            token_id = int(raw_id)
        else:
            raise ValueError(f"{tokens}:{line_no}: invalid token row")
        if token in token_map:
            raise ValueError(f"{tokens}:{line_no}: duplicate token")
        token_map[token] = token_id

    sequences: dict[int, tuple[int, ...]] = {}
    seen_sequences: set[tuple[int, ...]] = set()
    for line_no, raw in enumerate(keywords.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise ValueError(f"{keywords}:{line_no}: expected four columns")
        keyword_id = int(cols[0])
        if keyword_id < 0 or keyword_id > UINT32_MAX or keyword_id in sequences:
            raise ValueError(
                f"{keywords}:{line_no}: keyword id must be unique and fit uint32"
            )
        names = cols[3].split()
        try:
            sequence = tuple(token_map[name] for name in names)
        except KeyError as exc:
            raise ValueError(f"{keywords}:{line_no}: unknown token {exc.args[0]}") from exc
        if not sequence:
            raise ValueError(f"{keywords}:{line_no}: empty wake sequence")
        if sequence in seen_sequences:
            raise ValueError(f"{keywords}:{line_no}: duplicate wake sequence")
        sequences[keyword_id] = sequence
        seen_sequences.add(sequence)
    if not sequences:
        raise ValueError("keyword TSV contains no wake sequences")
    return sequences


def sequence_edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            substitution = previous[right_index - 1] + int(left_value != right_value)
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def nearest_keyword_ids(
    targets: tuple[int, ...],
    wake_sequences: dict[int, tuple[int, ...]],
) -> list[int]:
    keyword_ids = sorted(wake_sequences)
    if not targets:
        return keyword_ids
    distances = {
        keyword_id: sequence_edit_distance(targets, wake_sequences[keyword_id])
        for keyword_id in keyword_ids
    }
    minimum = min(distances.values())
    return [
        keyword_id
        for keyword_id in keyword_ids
        if distances[keyword_id] == minimum
    ]


def repeat_focus_rows(
    output: list[tuple[int, ...]],
    raw_focus: object,
    count: int,
    *,
    label: str,
) -> None:
    if count < 0:
        raise ValueError(f"{label} example count must be non-negative")
    if raw_focus is None:
        focus: tuple[int, ...] = ()
    elif isinstance(raw_focus, (list, tuple)):
        values = [int(value) for value in raw_focus]
        if any(value < 0 or value > UINT32_MAX for value in values):
            raise ValueError(f"{label} focus keyword ids must fit uint32")
        focus = tuple(sorted(set(values)))
    else:
        value = int(raw_focus)
        if value < 0 or value > UINT32_MAX:
            raise ValueError(f"{label} focus keyword id must fit uint32")
        focus = (value,)
    output.extend([focus] * count)


def static_replay_focus_rows(static: dict) -> dict[pathlib.Path, list[tuple[int, ...]]]:
    rows: list[tuple[int, ...]] = []
    for index, item in enumerate(static.get("sequences", [])):
        if not isinstance(item, dict):
            raise ValueError("hard-negative sequence evidence must be objects")
        repeat_focus_rows(
            rows,
            item.get("focus_keyword_id"),
            int(item.get("examples", 0)),
            label=f"hard-negative sequence {index}",
        )
    for index, item in enumerate(static.get("positive_stress", [])):
        if not isinstance(item, dict):
            raise ValueError("positive-stress evidence must be objects")
        repeat_focus_rows(
            rows,
            item.get("keyword_id"),
            int(item.get("examples", 0)),
            label=f"positive-stress keyword {index}",
        )
    if len(rows) != int(static.get("examples", -1)):
        raise ValueError("hard-negative replay focus evidence row count drifted")
    manifest = pathlib.Path(str(static.get("manifest") or "")).resolve()
    return {manifest: rows}


def derive_wake_pressure_balance(
    *,
    manifests: list[pathlib.Path],
    tokens: pathlib.Path,
    keywords: pathlib.Path,
    positive_example_weight: float,
    focus_rows_by_manifest: dict[pathlib.Path, list[tuple[int, ...]]] | None = None,
) -> dict:
    if (
        not math.isfinite(positive_example_weight)
        or positive_example_weight <= 0.0
    ):
        raise ValueError("wake-balance positive example weight must be finite and > 0")
    wake_sequences = keyword_target_sequences(tokens, keywords)
    sequence_to_keyword = {
        sequence: keyword_id for keyword_id, sequence in wake_sequences.items()
    }
    wake_rows_by_keyword = {keyword_id: 0 for keyword_id in wake_sequences}
    assigned_nonwake_mass = {keyword_id: 0.0 for keyword_id in wake_sequences}
    assigned_nonwake_rows = {keyword_id: 0.0 for keyword_id in wake_sequences}
    wake_rows = tokenized_nonwake_rows = empty_nonwake_rows = 0
    explicit_focus_nonwake_rows = fallback_edit_distance_nonwake_rows = 0
    per_manifest: list[dict] = []

    manifest_keys = {manifest.resolve() for manifest in manifests}
    normalized_focus = {
        pathlib.Path(path).resolve(): list(rows)
        for path, rows in (focus_rows_by_manifest or {}).items()
    }
    unknown_focus_manifests = sorted(
        str(path) for path in set(normalized_focus) - manifest_keys
    )
    if unknown_focus_manifests:
        raise ValueError(
            "wake-balance focus evidence contains unknown manifest(s): "
            + ", ".join(unknown_focus_manifests)
        )

    for manifest in manifests:
        targets = manifest_targets(manifest)
        focus_rows = normalized_focus.get(manifest.resolve())
        if focus_rows is not None and len(focus_rows) != len(targets):
            raise ValueError(
                f"wake-balance focus evidence row count differs from manifest: {manifest}"
            )
        local_wake = {keyword_id: 0 for keyword_id in wake_sequences}
        local_nonwake_mass = {keyword_id: 0.0 for keyword_id in wake_sequences}
        local_nonwake_rows = {keyword_id: 0.0 for keyword_id in wake_sequences}
        local_tokenized = local_empty = 0
        local_explicit_focus = local_fallback_focus = 0
        for row_index, row in enumerate(targets):
            keyword_id = sequence_to_keyword.get(row)
            if keyword_id is not None:
                wake_rows += 1
                wake_rows_by_keyword[keyword_id] += 1
                local_wake[keyword_id] += 1
                continue

            mass = positive_example_weight if row else 1.0
            if row:
                tokenized_nonwake_rows += 1
                local_tokenized += 1
            else:
                empty_nonwake_rows += 1
                local_empty += 1
            raw_focus = focus_rows[row_index] if focus_rows is not None else ()
            if raw_focus:
                nearest = sorted({int(value) for value in raw_focus})
                unknown = [value for value in nearest if value not in wake_sequences]
                if unknown:
                    raise ValueError(
                        "wake-balance focus evidence references unknown keyword id(s): "
                        + ", ".join(str(value) for value in unknown)
                    )
                explicit_focus_nonwake_rows += 1
                local_explicit_focus += 1
            else:
                nearest = nearest_keyword_ids(row, wake_sequences)
                fallback_edit_distance_nonwake_rows += 1
                local_fallback_focus += 1
            share = 1.0 / float(len(nearest))
            for nearest_id in nearest:
                assigned_nonwake_rows[nearest_id] += share
                assigned_nonwake_mass[nearest_id] += mass * share
                local_nonwake_rows[nearest_id] += share
                local_nonwake_mass[nearest_id] += mass * share

        per_manifest.append(
            {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "rows": len(targets),
                "wake_rows": sum(local_wake.values()),
                "wake_rows_by_keyword": {
                    str(keyword_id): int(local_wake[keyword_id])
                    for keyword_id in sorted(local_wake)
                },
                "tokenized_nonwake_rows": local_tokenized,
                "empty_nonwake_rows": local_empty,
                "explicit_focus_nonwake_rows": local_explicit_focus,
                "fallback_edit_distance_nonwake_rows": local_fallback_focus,
                "assigned_nonwake_rows_by_keyword": {
                    str(keyword_id): local_nonwake_rows[keyword_id]
                    for keyword_id in sorted(local_nonwake_rows)
                },
                "assigned_nonwake_mass_by_keyword": {
                    str(keyword_id): local_nonwake_mass[keyword_id]
                    for keyword_id in sorted(local_nonwake_mass)
                },
            }
        )

    if wake_rows <= 0:
        raise ValueError("wake-balance manifests contain no exact configured wake examples")

    keyword_balance: dict[str, dict] = {}
    wake_keyword_weights: dict[str, float] = {}
    effective_wake_mass = 0.0
    any_bounded = False
    for keyword_id in sorted(wake_sequences):
        keyword_wake_rows = wake_rows_by_keyword[keyword_id]
        if keyword_wake_rows <= 0:
            raise ValueError(
                f"wake-balance manifests contain no exact wake examples for keyword {keyword_id}"
            )
        wake_base_mass = keyword_wake_rows * positive_example_weight
        target_mass = assigned_nonwake_mass[keyword_id]
        raw_weight = target_mass / wake_base_mass
        wake_weight = min(MAX_WAKE_EXAMPLE_WEIGHT, max(1.0, raw_weight))
        bounded = wake_weight != raw_weight
        effective_mass = wake_base_mass * wake_weight
        any_bounded = any_bounded or bounded
        effective_wake_mass += effective_mass
        wake_keyword_weights[str(keyword_id)] = wake_weight
        keyword_balance[str(keyword_id)] = {
            "wake_rows": keyword_wake_rows,
            "wake_base_mass": wake_base_mass,
            "assigned_nonwake_rows": assigned_nonwake_rows[keyword_id],
            "assigned_nonwake_mass": target_mass,
            "raw_wake_example_weight": raw_weight,
            "wake_example_weight": wake_weight,
            "effective_wake_mass": effective_mass,
            "bounded": bounded,
        }

    wake_base_mass = wake_rows * positive_example_weight
    nonwake_mass = (
        tokenized_nonwake_rows * positive_example_weight + empty_nonwake_rows
    )
    assigned_mass_total = sum(assigned_nonwake_mass.values())
    if not math.isclose(assigned_mass_total, nonwake_mass, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("per-keyword non-wake pressure does not conserve effective mass")

    return {
        "schema_version": 3,
        "policy": WAKE_BALANCE_POLICY,
        "positive_example_weight": positive_example_weight,
        "default_wake_example_weight": 1.0,
        "wake_keyword_weights": wake_keyword_weights,
        "max_wake_example_weight": MAX_WAKE_EXAMPLE_WEIGHT,
        "bounded": any_bounded,
        "wake_rows": wake_rows,
        "wake_rows_by_keyword": {
            str(keyword_id): wake_rows_by_keyword[keyword_id]
            for keyword_id in sorted(wake_rows_by_keyword)
        },
        "tokenized_nonwake_rows": tokenized_nonwake_rows,
        "empty_nonwake_rows": empty_nonwake_rows,
        "explicit_focus_nonwake_rows": explicit_focus_nonwake_rows,
        "fallback_edit_distance_nonwake_rows": fallback_edit_distance_nonwake_rows,
        "wake_base_mass": wake_base_mass,
        "nonwake_mass": nonwake_mass,
        "effective_wake_mass": effective_wake_mass,
        "keyword_balance": keyword_balance,
        "pressure_assignment": PRESSURE_ASSIGNMENT_POLICY,
        "manifests": per_manifest,
    }
