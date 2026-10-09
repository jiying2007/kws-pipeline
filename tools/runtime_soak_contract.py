"""Hard-cut CPU measurement contract shared by acquisition and qualification.

100% is one fully occupied logical core. Process CPU time sums all threads;
CPU capacity is descriptive metadata, never the gate denominator.
"""
from __future__ import annotations

from qualification_common import close_enough, finite, json_int

CPU_MEASUREMENT_CONTRACT_ID = "process-cpu-one-core-v1"
CPU_PERCENT_SEMANTICS = "process_cpu_seconds / wall_seconds * 100"
RUNTIME_SOAK_SCHEMA_VERSION = 3
TARGET_EVIDENCE_SCHEMA_VERSION = 3


def require_cpu_contract(value: dict, label: str) -> None:
    if value.get("measurement_contract_id") != CPU_MEASUREMENT_CONTRACT_ID:
        raise ValueError(f"{label} measurement_contract_id must be {CPU_MEASUREMENT_CONTRACT_ID}")


def cpu_percent(process_cpu_seconds: float, wall_seconds: float) -> float:
    cpu = finite(process_cpu_seconds, "process_cpu_seconds", 0.0)
    wall = finite(wall_seconds, "wall_seconds", 0.0)
    if wall <= 0.0:
        raise ValueError("wall_seconds must be > 0")
    # Deliberately neither capacity-normalized nor clipped at 100%.
    return finite(cpu / wall * 100.0, "one-core CPU percent", 0.0)


def validate_cpu_metrics(value: dict, label: str) -> dict:
    require_cpu_contract(value, label)
    if value.get("cpu_percent_semantics") != CPU_PERCENT_SEMANTICS:
        raise ValueError(f"{label} CPU percentage semantics are unsupported")
    result = {
        "measurement_contract_id": CPU_MEASUREMENT_CONTRACT_ID,
        "cpu_percent_semantics": CPU_PERCENT_SEMANTICS,
        "process_cpu_seconds": finite(value.get("process_cpu_seconds"), f"{label} process_cpu_seconds", 0.0),
        "wall_seconds": finite(value.get("wall_seconds"), f"{label} wall_seconds", 0.0),
        "cpu_capacity_count": json_int(value.get("cpu_capacity_count"), f"{label} cpu_capacity_count", 1),
        "max_thread_count": json_int(value.get("max_thread_count"), f"{label} max_thread_count", 1),
        "cpu_percent": finite(value.get("cpu_percent"), f"{label} cpu_percent", 0.0),
        # The process sampler cannot measure audio exposure. A board-benchmark
        # input WAV duration must never be substituted for this soak's audio.
        "audio_seconds": None,
        "cpu_seconds_per_audio_second": None,
    }
    for key in ("audio_seconds", "cpu_seconds_per_audio_second"):
        if key not in value or value[key] is not None:
            raise ValueError(f"{label} {key} must be null: soak audio exposure is not measured")
    close_enough(result["cpu_percent"], cpu_percent(result["process_cpu_seconds"], result["wall_seconds"]), f"{label} cpu_percent", 1e-9, 1e-9)
    return result


def validate_runtime_soak(value: dict) -> dict:
    if json_int(value.get("schema_version"), "runtime soak schema_version") != RUNTIME_SOAK_SCHEMA_VERSION:
        raise ValueError("runtime soak schema_version must be 3; legacy capacity-normalized evidence is unsupported")
    require_cpu_contract(value, "runtime soak")
    if value.get("cpu_percent_semantics") != CPU_PERCENT_SEMANTICS:
        raise ValueError("runtime soak CPU percentage semantics are unsupported")
    if value.get("completed_requested_duration") is not True:
        raise ValueError("runtime soak did not complete the requested duration")
    requested_hours = finite(value.get("requested_hours"), "runtime soak requested_hours", 0.0)
    elapsed = finite(value.get("elapsed_seconds"), "runtime soak elapsed_seconds", 0.0)
    hours = finite(value.get("elapsed_hours"), "runtime soak elapsed_hours", 0.0)
    if requested_hours <= 0.0 or elapsed <= 0.0:
        raise ValueError("runtime soak requested/elapsed duration must be > 0")
    close_enough(hours, elapsed / 3600.0, "runtime soak elapsed_hours", 1e-9, 1e-12)
    if hours + 1e-6 < requested_hours:
        raise ValueError("runtime soak elapsed_hours is shorter than requested_hours")
    initial_cpu = finite(value.get("initial_cpu_seconds"), "runtime soak initial_cpu_seconds", 0.0)
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("runtime soak samples must be non-empty")
    rss_values, temp_values, threads = [], [], []
    previous_elapsed = -1.0
    previous_cpu = initial_cpu
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"runtime soak samples[{index}] must be an object")
        label = f"runtime soak samples[{index}]"
        sample_elapsed = finite(sample.get("elapsed_s"), f"{label}.elapsed_s", 0.0)
        sample_cpu = finite(sample.get("cpu_seconds"), f"{label}.cpu_seconds", 0.0)
        if sample_elapsed <= previous_elapsed or sample_elapsed > elapsed:
            raise ValueError("runtime soak sample times must increase within the measured wall interval")
        if sample_cpu < previous_cpu:
            raise ValueError("runtime soak CPU time regressed")
        threads.append(json_int(sample.get("thread_count"), f"{label}.thread_count", 1))
        if sample.get("rss_kib") is not None:
            rss_values.append(finite(sample["rss_kib"], f"{label}.rss_kib", 0.0))
        if sample.get("temp_c") is not None:
            temp_values.append(finite(sample["temp_c"], f"{label}.temp_c", -273.15))
        previous_elapsed, previous_cpu = sample_elapsed, sample_cpu
    close_enough(previous_elapsed, elapsed, "runtime soak final sample elapsed_s", 1e-9, 1e-9)
    if not rss_values or not temp_values:
        raise ValueError("runtime soak must retain RSS, CPU, thread and thermal samples")
    cpu_seconds = previous_cpu - initial_cpu
    metrics = validate_cpu_metrics({**value, "cpu_percent": value.get("average_cpu_percent")}, "runtime soak")
    expected = {
        "process_cpu_seconds": cpu_seconds,
        "wall_seconds": elapsed,
        "max_thread_count": max(threads),
        "cpu_percent": cpu_percent(cpu_seconds, elapsed),
    }
    for key, number in expected.items():
        close_enough(metrics[key], number, f"runtime soak {key}", 1e-9, 1e-9)
    rss, temp = max(rss_values), max(temp_values)
    close_enough(finite(value.get("max_rss_kib"), "runtime soak max_rss_kib", 0.0), rss, "runtime soak max_rss_kib", 1e-9, 1e-9)
    close_enough(finite(value.get("max_temp_c"), "runtime soak max_temp_c", -273.15), temp, "runtime soak max_temp_c", 1e-9, 1e-9)
    return {**metrics, "soak_hours": hours, "rss_kib": rss, "max_temp_c": temp}
