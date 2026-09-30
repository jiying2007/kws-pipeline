#!/usr/bin/env python3
"""Synthetic-only actual-fbank + pinned accept_wave PCM adapter tests.

Freeze source goldens before producing C outputs:
  python test_pcm_adapter.py --generate-goldens --baseline /path/to/baseline \
      --goldens /new/goldens
  python test_pcm_adapter.py --library /path/to/libdonor_pcm.so \
      --goldens /new/goldens --report /new/report.json

No model, decoder, corpus, download, training, or new numerical variant is used.
The unchanged frontend logfbank absolute gate is1e-3 (rtol0); splice is an exact
copy of those features. This checks PCM composition, not CMVN/network parity.
"""

import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import sys
import unittest


BASELINE_SHA = "498af9c026d6172fc1fe92616d2edc2c0cc848dae2f410bcef3822912eb8cbd0"
SOURCE_SHA = "2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53"
FRONTEND_CONTRACT_SHA = "3bf49445ae77bef6397673379c892885d7d8a406d66b34bf76831a1b1b919019"
LENGTHS = (0, 1, 399, 400, 799, 800, 959, 960, 4799, 4800, 4801,
           5279, 5280, 5439, 5440, 5599, 5600, 5759, 5760,
           9599, 9600, 9601, 14400, 15199, 15200, 34240)
PATTERNS = ("silence", "ramp", "impulses", "noise")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_goldens(args):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    baseline = args.baseline.resolve()
    driver = baseline / "run_baseline.py"
    source = baseline / "upstream/wekws/bin/stream_kws_ctc.py"
    if sha(driver) != BASELINE_SHA or sha(source) != SOURCE_SHA:
        raise ValueError("Pinned source identity changed")
    args.goldens.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("pcm_pinned_baseline", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frontend, namespace, sources = module.audited_sources()
    # make_spotter binds ns['kaldi'] globally for its source-defined methods.
    # A shallow dictionary copy would still share those methods' __globals__.
    # Execute the audited loader again to give tagged and actual calls wholly
    # independent globals; never let the tagged oracle replace actual fbank.
    _, tagged_namespace, tagged_sources = module.audited_sources()
    assert sources == tagged_sources
    tokens = (baseline / "resources/tokens_2599.txt").read_text().splitlines()
    files, cases = {}, {}

    def save(name, data):
        path = args.goldens / name
        with path.open("xb") as handle:
            handle.write(data)
        files[name] = {"bytes": len(data), "sha256": sha(path)}
        return name

    class CapturedFrontend:
        def __init__(self, tagged=False):
            self.tagged, self.index, self.rows = tagged, 0, None

        def fbank(self, waveform, **kwargs):
            assert kwargs == dict(window_type="hamming", num_mel_bins=80,
                                  frame_length=25, frame_shift=10, dither=0,
                                  energy_floor=0.0, sample_frequency=16000)
            count = 1 + (waveform.shape[1] - 400) // 160
            if self.tagged:
                self.rows = (torch.arange(self.index, self.index + count,
                                         dtype=torch.float32)[:, None] * 128 +
                             torch.arange(80, dtype=torch.float32)[None, :])
            else:
                self.rows = frontend.fbank(waveform, **kwargs)
            self.index += count
            return self.rows

    maximum = max(LENGTHS)
    index = np.arange(maximum, dtype=np.int64)
    signals = {
        "silence": np.zeros(maximum, dtype="<i2"),
        "ramp": (((index * 17 + 123) % 8192) - 4096).astype("<i2"),
        "impulses": np.zeros(maximum, dtype="<i2"),
        "noise": np.random.Generator(np.random.PCG64(735)).integers(
            -4096, 4096, size=maximum, dtype=np.int16).astype("<i2"),
    }
    for number, location in enumerate((0, 159, 399, 400, 799, 800, 4799,
                                       4800, 5199, 9599, 9600, 14401, 34001)):
        signals["impulses"][location] = 32767 if number % 2 == 0 else -32768
    with torch.inference_mode():
        for pattern in PATTERNS:
            for length in LENGTHS:
                name = f"{pattern}_{length}"
                pcm = signals[pattern][:length].tobytes()
                actual, tagged = CapturedFrontend(), CapturedFrontend(True)
                spotter = module.make_spotter("hamming", None, actual, namespace, tokens)
                tagger = module.make_spotter("hamming", None, tagged, tagged_namespace, tokens)
                calls = []
                for number, start in enumerate(range(0, length, 4800)):
                    chunk = pcm[start * 2:min(start + 4800, length) * 2]
                    actual.rows, tagged.rows = None, None
                    trace = {}

                    def capture(frame, event, arg):
                        if (event == "return" and frame.f_code.co_name == "accept_wave"
                                and frame.f_code.co_filename == str(source)):
                            trace["splice_rows"] = frame.f_locals.get("ctx_frm", 0)
                        return capture

                    before = len(spotter.wave_remained)
                    output = spotter.accept_wave(chunk)
                    sys.settrace(capture)
                    try:
                        tags = tagger.accept_wave(chunk)
                    finally:
                        sys.settrace(None)
                    assert (actual.rows is None) == (tagged.rows is None)
                    assert actual.index == tagged.index
                    if tags is not None:
                        assert output.shape == tags.shape
                        assert torch.isfinite(output).all()
                        if number == 0:
                            assert tags[0, 160].item() == 0
                    prefix = f"{name}-{number:02d}"
                    fbank = b"" if actual.rows is None else actual.rows.numpy().astype("<f4").tobytes()
                    rows = b"" if output is None else output.numpy().astype("<f4").tobytes()
                    features = (b"" if spotter.feature_remained is None else
                                spotter.feature_remained.numpy().astype("<f4").tobytes())
                    wave = np.asarray(spotter.wave_remained, dtype="<i2").tobytes()
                    calls.append({
                        "call_index": number, "available_samples": start + len(chunk) // 2,
                        "call_samples": len(chunk) // 2,
                        "is_final_short": int(len(chunk) != 9600),
                        "waveform_samples": before + len(chunk) // 2,
                        "fbank_rows": len(fbank) // 320,
                        "splice_rows": trace["splice_rows"],
                        "selected_rows": len(rows) // 1600,
                        "centers": [] if tags is None else [int(x) // 128 for x in tags[:, 160].tolist()],
                        "fbank": save(prefix + ".fbank.f32le", fbank),
                        "rows": save(prefix + ".splice.f32le", rows),
                        "features": save(prefix + ".features.f32le", features),
                        "wave": save(prefix + ".wave.pcm16le", wave),
                        "feature_count": len(features) // 320,
                        "wave_samples": len(wave) // 2,
                        "offset": spotter.feats_ctx_offset,
                        "total_fbank": actual.index,
                    })
                cases[name] = {"pattern": pattern, "samples": length,
                               "pcm": save(name + ".pcm16le", pcm), "calls": calls}
    save("generator.py", pathlib.Path(__file__).read_bytes())
    write_json(args.goldens / "receipt.json", {
        "schema": "donor-pcm-source-goldens-v1",
        "scope": "synthetic PCM, actual pinned fbank and accept_wave; no model/corpus",
        "generator_policy": "immutable exact input bytes; frozen before comparison against this receipt",
        "supersedes_invalid_receipt_sha256": (
            sha(args.supersedes_invalid_goldens / "receipt.json")
            if args.supersedes_invalid_goldens else None),
        "supersedes_reason": (
            "Initial synthetic generator shared actual/tagged source globals; corrected by independent audited namespaces. Prior goldens and failed C comparison retained. No C arithmetic or gate changed."
            if args.supersedes_invalid_goldens else None),
        "baseline_sha256": BASELINE_SHA, "accept_wave_sha256": SOURCE_SHA,
        "frontend_contract_sha256": FRONTEND_CONTRACT_SHA,
        "audited_sources": sources,
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
        "gates": {"fbank_absolute": 1e-3, "splice_absolute": 1e-3,
                  "relative": 0, "discrete": "exact", "cmvn_network": "not tested"},
        "files": files, "cases": cases,
    })
    print(f"Frozen {len(cases)} cases/{sum(len(c['calls']) for c in cases.values())} "
          f"source calls; receipt {sha(args.goldens / 'receipt.json')}")


class SpliceState(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("initialized", "finished", "wave_samples", "feature_count", "offset")] + [
        ("total_samples", C.c_uint64), ("total_fbank", C.c_uint64),
        ("total_selected", C.c_uint64), ("feature_remained", (C.c_float * 80) * 4)]


class FrontendState(C.Structure):
    _fields_ = [("initialized", C.c_uint32), ("finished", C.c_int),
                ("used", C.c_size_t), ("total_samples", C.c_uint64),
                ("frame_index", C.c_uint64), ("pcm", C.c_int16 * 400),
                ("re", C.c_float * 512), ("im", C.c_float * 512)]


class Trace(C.Structure):
    _fields_ = [("dc", C.c_float * 400), ("preemphasis", C.c_float * 400),
                ("windowed", C.c_float * 512), ("power", C.c_float * 257),
                ("mel", C.c_float * 80), ("logfbank", C.c_float * 80)]


class State(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("initialized", "finished", "faulted", "busy")] + [
        ("ingress_samples", C.c_size_t), ("total_ingress", C.c_uint64),
        ("total_calls", C.c_uint64), ("ingress", C.c_int16 * 4800),
        ("waveform", C.c_int16 * 5600), ("splice", SpliceState),
        ("frontend", FrontendState), ("trace", Trace),
        ("fbank", (C.c_float * 80) * 33), ("rows", (C.c_float * 400) * 11),
        ("centers", C.c_uint64 * 11)]


class Batch(C.Structure):
    _fields_ = [("call_index", C.c_uint64), ("available_samples", C.c_uint64)] + [
        (name, C.c_size_t) for name in ("call_samples", "waveform_samples", "fbank_rows",
                                       "splice_rows", "selected_rows")] + [
        ("is_final_short", C.c_uint32), ("fbank", C.POINTER(C.c_float)),
        ("rows", C.POINTER(C.c_float)), ("centers", C.POINTER(C.c_uint64)),
        ("wave_samples", C.c_uint32), ("feature_count", C.c_uint32),
        ("offset", C.c_uint32)]


CALLBACK = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(Batch))
NULL_CALLBACK = CALLBACK()


def load_receipt(root):
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt["schema"] != "donor-pcm-source-goldens-v1" or
            receipt["baseline_sha256"] != BASELINE_SHA or
            receipt["accept_wave_sha256"] != SOURCE_SHA or
            receipt["frontend_contract_sha256"] != FRONTEND_CONTRACT_SHA or
            receipt["gates"] != {"fbank_absolute": 1e-3, "splice_absolute": 1e-3,
                                  "relative": 0, "discrete": "exact", "cmvn_network": "not tested"}):
        raise ValueError("Golden source identity or gate mismatch")
    if set(receipt["cases"]) != {f"{p}_{n}" for p in PATTERNS for n in LENGTHS}:
        raise ValueError("Golden case coverage changed")
    if {p.name for p in root.iterdir()} != {"receipt.json", *receipt["files"]}:
        raise ValueError("Golden folder membership changed")
    for name, info in receipt["files"].items():
        path = root / name
        if (path.name != name or path.is_symlink() or not path.is_file() or
                path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]):
            raise ValueError(f"Golden integrity failed: {name}")
    return receipt


class PcmAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.byteorder != "little":
            raise RuntimeError("Fixture comparison currently requires little-endian host")
        cls.lib = C.CDLL(str(cls.library))
        cls.lib.donor_pcm_state_bytes.restype = C.c_size_t
        if cls.lib.donor_pcm_state_bytes() != C.sizeof(State):
            raise ValueError("PCM state ABI mismatch")
        for name in ("init", "reset"):
            method = getattr(cls.lib, "donor_pcm_" + name)
            method.argtypes, method.restype = [C.POINTER(State)], C.c_int
        cls.lib.donor_pcm_feed.argtypes = [C.POINTER(State), C.POINTER(C.c_int16),
                                           C.c_size_t, CALLBACK, C.c_void_p]
        cls.lib.donor_pcm_feed.restype = C.c_int
        cls.lib.donor_pcm_finish.argtypes = [C.POINTER(State), CALLBACK, C.c_void_p]
        cls.lib.donor_pcm_finish.restype = C.c_int
        cls.metrics = {name: {"maximum_absolute": 0, "values": 0, "sum_squared": 0}
                       for name in ("fbank", "rows", "features")}

    def new_state(self):
        state = State()
        self.assertEqual(self.lib.donor_pcm_init(C.byref(state)), 0)
        return state

    def payload(self, name):
        return (self.goldens / name).read_bytes()

    def compare_floats(self, actual, expected, kind, location):
        self.assertEqual(len(actual), len(expected))
        a = struct.unpack(f"<{len(actual) // 4}f", actual)
        b = struct.unpack(f"<{len(expected) // 4}f", expected)
        metric = self.metrics[kind]
        for index, (got, want) in enumerate(zip(a, b)):
            self.assertTrue(math.isfinite(got))
            delta = abs(got - want)
            if delta > metric["maximum_absolute"]:
                metric.update(maximum_absolute=delta, location=f"{location}:{index}",
                              actual=got, expected=want)
            metric["values"] += 1
            metric["sum_squared"] += delta * delta
            self.assertLessEqual(delta, 1e-3, f"{kind}/{location}:{index}: {got} vs {want}")

    def receiver(self, state, results):
        @CALLBACK
        def receive(_, pointer):
            batch = pointer.contents
            result = {name: getattr(batch, name) for name, _ in Batch._fields_[:8]}
            result.update(
                fbank=C.string_at(batch.fbank, batch.fbank_rows * 320),
                rows=C.string_at(batch.rows, batch.selected_rows * 1600),
                centers=list(batch.centers[:batch.selected_rows]),
                features=C.string_at(state.splice.feature_remained, state.splice.feature_count * 320),
                wave=C.string_at(state.waveform, state.splice.wave_samples * 2),
                feature_count=state.splice.feature_count,
                wave_samples=state.splice.wave_samples, offset=state.splice.offset,
                total_fbank=state.splice.total_fbank,
                total_selected=state.splice.total_selected,
                ingress_samples=state.ingress_samples,
            )
            # Record the public batch's read-only state summary as the oracle
            # values; full state snapshots above independently verify its ABI.
            result.update(wave_samples=batch.wave_samples,
                          feature_count=batch.feature_count, offset=batch.offset)
            results.append(result)
        return receive

    def verify_results(self, results, case):
        self.assertEqual(len(results), len(case["calls"]))
        total_selected = 0
        for got, expected in zip(results, case["calls"]):
            location = f"{case['pattern']}_{case['samples']}/{expected['call_index']}"
            for key, value in expected.items():
                if key in ("fbank", "rows", "features"):
                    self.compare_floats(got[key], self.payload(value), key, location)
                elif key == "wave":
                    self.assertEqual(got[key], self.payload(value), location)
                else:
                    self.assertEqual(got[key], value, f"{location}/{key}")
            total_selected += expected["selected_rows"]
            self.assertEqual(got["total_selected"], total_selected)
            self.assertEqual(got["ingress_samples"], 0)

    def run_case(self, case, partition, state=None):
        state = self.new_state() if state is None else state
        results = []
        callback = self.receiver(state, results)
        pcm = self.payload(case["pcm"])
        samples = (C.c_int16 * (len(pcm) // 2)).from_buffer_copy(pcm)
        offset, part = 0, 0
        while offset < case["samples"]:
            take = min(partition[part % len(partition)], case["samples"] - offset)
            ptr = C.cast(C.byref(samples, offset * 2), C.POINTER(C.c_int16))
            self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), ptr, take, callback, None), 0)
            offset += take
            part += 1
            self.assertEqual(state.ingress_samples, offset % 4800)
            self.assertEqual(state.total_ingress, offset)
            self.assertEqual(state.splice.total_samples, offset - offset % 4800)
            self.assertEqual(len(results), offset // 4800)
        full_count = len(results)
        self.assertEqual(self.lib.donor_pcm_finish(C.byref(state), callback, None), 0)
        self.assertEqual(len(results) - full_count, int(case["samples"] % 4800 != 0))
        self.assertEqual(state.finished, 1)
        self.assertEqual(state.splice.finished, 1)
        self.assertEqual(state.frontend.used, 0)
        self.assertEqual(state.frontend.total_samples, 0)
        self.assertEqual(state.frontend.frame_index, 0)
        self.verify_results(results, case)
        return state, results

    def test_actual_source_and_arbitrary_partitioning(self):
        for name, case in self.receipt["cases"].items():
            with self.subTest(case=name):
                first, first_results = self.run_case(case, [4800])
                for partition in ([max(1, case["samples"])], [1, 159, 321, 7777, 13, 4096, 800, 3]):
                    other, results = self.run_case(case, partition)
                    self.assertEqual(results, first_results, "ingress partitions must not change C bytes")
                    self.assertEqual(bytes(other), bytes(first), "all final state/scratch bytes must agree")

    def test_single_sample_ingress_and_reset_replay(self):
        state = self.new_state()
        fresh = bytes(state)
        case = self.receipt["cases"]["noise_9601"]
        state, expected = self.run_case(case, [1], state)
        for partition in ([4800], [719, 3, 27], [9601]):
            self.assertEqual(self.lib.donor_pcm_reset(C.byref(state)), 0)
            self.assertEqual(bytes(state), fresh)
            _, results = self.run_case(case, partition, state)
            self.assertEqual(results, expected)

    def test_gate_short_tail_and_no_eof_flush(self):
        expected_counts = {799: [0], 800: [1], 959: [1], 960: [1],
                           4799: [9], 4800: [9], 4801: [9, 0],
                           5279: [9, 0], 5280: [9, 1], 9600: [9, 10]}
        for count, expected in expected_counts.items():
            state, outputs = self.run_case(self.receipt["cases"][f"silence_{count}"], [count])
            self.assertEqual([b["selected_rows"] for b in outputs], expected)
            before = bytes(state)
            calls = []
            callback = self.receiver(state, calls)
            self.assertEqual(self.lib.donor_pcm_finish(C.byref(state), callback, None), 2)
            self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), None, 0, callback, None), 2)
            self.assertEqual(bytes(state), before)
            self.assertEqual(calls, [])
        first = self.receipt["cases"]["silence_9600"]["calls"]
        self.assertEqual([c["centers"] for c in first], [list(range(0, 25, 3)), list(range(27, 55, 3))])
        self.assertEqual([c["wave_samples"] for c in first], [320, 320])
        self.assertEqual(self.receipt["cases"]["silence_5280"]["calls"][-1]["feature_count"], 3)

    def test_interleaved_independent_handles(self):
        cases = [self.receipt["cases"][n] for n in ("noise_14400", "ramp_15200", "impulses_9601")]
        states = [self.new_state() for _ in cases]
        results = [[] for _ in cases]
        callbacks = [self.receiver(s, out) for s, out in zip(states, results)]
        pcm = [self.payload(case["pcm"]) for case in cases]
        for start in range(0, max(c["samples"] for c in cases), 719):
            for i, case in enumerate(cases):
                chunk = pcm[i][start * 2:min(start + 719, case["samples"]) * 2]
                if not chunk:
                    continue
                untouched = [bytes(s) for s in states]
                array = (C.c_int16 * (len(chunk) // 2)).from_buffer_copy(chunk)
                self.assertEqual(self.lib.donor_pcm_feed(C.byref(states[i]), array,
                                 len(array), callbacks[i], None), 0)
                for j, state in enumerate(states):
                    if i != j:
                        self.assertEqual(bytes(state), untouched[j])
        for state, callback, out, case in zip(states, callbacks, results, cases):
            self.assertEqual(self.lib.donor_pcm_finish(C.byref(state), callback, None), 0)
            self.verify_results(out, case)

    def test_invalid_arguments_are_atomic(self):
        state, results = self.new_state(), []
        callback = self.receiver(state, results)
        value = (C.c_int16 * 1)(7)
        before = bytes(state)
        for pcm, count, cb in ((None, 1, callback), (value, 1, NULL_CALLBACK),
                              (None, 0, NULL_CALLBACK),
                              (value, C.c_size_t(-1).value, callback),
                              (C.cast(C.byref(state), C.POINTER(C.c_int16)), 1, callback)):
            self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), pcm, count, cb, None), 1)
            self.assertEqual(bytes(state), before)
            self.assertEqual(results, [])
        self.assertEqual(self.lib.donor_pcm_finish(C.byref(state), NULL_CALLBACK, None), 1)
        self.assertEqual(bytes(state), before)
        self.assertEqual(self.lib.donor_pcm_init(None), 1)
        self.assertEqual(self.lib.donor_pcm_reset(None), 1)
        self.assertEqual(self.lib.donor_pcm_feed(None, None, 0, callback, None), 1)
        self.assertEqual(self.lib.donor_pcm_finish(None, callback, None), 1)
        self.assertEqual(self.lib.donor_pcm_feed(C.byref(State()), value, 1, callback, None), 2)
        self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), None, 0, callback, None), 0)
        self.assertEqual(bytes(state), before)

    def test_nonfinite_fault_and_reset(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for action in ("feed", "finish"):
                state, results = self.new_state(), []
                callback = self.receiver(state, results)
                pcm = (C.c_int16 * 4800)()
                self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), pcm, 4800, callback, None), 0)
                state.splice.feature_remained[3][79] = bad
                if action == "feed":
                    status = self.lib.donor_pcm_feed(C.byref(state), None, 0, callback, None)
                else:
                    status = self.lib.donor_pcm_finish(C.byref(state), callback, None)
                self.assertEqual(status, 3)
                self.assertEqual(state.faulted, 1)
                self.assertEqual(len(results), 1)
                self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), pcm, 4800, callback, None), 2)
                self.assertEqual(self.lib.donor_pcm_reset(C.byref(state)), 0)
                self.assertEqual(bytes(state), bytes(self.new_state()))
                self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), pcm, 4800, callback, None), 0)
        state, results = self.new_state(), []
        callback = self.receiver(state, results)
        state.frontend.initialized = 0
        self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), (C.c_int16 * 4800)(),
                         4800, callback, None), 2)
        self.assertEqual(state.faulted, 1)
        self.assertEqual(results, [])

    def test_reentry_and_64bit_overflow(self):
        state = self.new_state()
        status = []

        @CALLBACK
        def callback(_, batch):
            status.append(self.lib.donor_pcm_feed(C.byref(state), None, 0, callback, None))
            status.append(self.lib.donor_pcm_finish(C.byref(state), callback, None))

        pcm = (C.c_int16 * 4800)()
        self.assertEqual(self.lib.donor_pcm_feed(C.byref(state), pcm, 4800, callback, None), 0)
        self.assertEqual(status, [2, 2])
        for calls in ((1 << 32) + 1, ((1 << 64) - 1) // 4800):
            state = self.new_state()
            state.total_calls = calls
            state.total_ingress = state.splice.total_samples = calls * 4800
            state.splice.total_fbank = calls * 30 - 2
            state.splice.total_selected = calls * 10 - 1
            state.splice.wave_samples = 320
            state.splice.feature_count = 4
            state.splice.offset = 1
            results = []
            callback = self.receiver(state, results)
            before = bytes(state)
            result = self.lib.donor_pcm_feed(C.byref(state), pcm, 4800, callback, None)
            if calls * 4800 + 4800 >= 1 << 64:
                self.assertEqual(result, 1)
                self.assertEqual(bytes(state), before)
                self.assertEqual(results, [])
            else:
                self.assertEqual(result, 0)
                self.assertEqual(results[0]["call_index"], calls)
                self.assertEqual(results[0]["available_samples"], (calls + 1) * 4800)
                self.assertEqual(results[0]["centers"], list(range(calls * 30 - 3, calls * 30 + 25, 3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldens", type=pathlib.Path, required=True)
    parser.add_argument("--generate-goldens", action="store_true")
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--library", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--supersedes-invalid-goldens", type=pathlib.Path)
    args = parser.parse_args()
    if args.generate_goldens:
        if args.baseline is None or args.library is not None or args.report is not None:
            parser.error("generation requires --baseline and forbids --library/--report")
        generate_goldens(args)
        return
    if args.library is None or args.baseline is not None or args.supersedes_invalid_goldens is not None:
        parser.error("comparison requires --library and forbids --baseline")
    PcmAdapterTests.receipt = load_receipt(args.goldens)
    PcmAdapterTests.goldens = args.goldens.resolve()
    PcmAdapterTests.library = args.library.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PcmAdapterTests))
    if args.report:
        metrics = PcmAdapterTests.metrics
        for value in metrics.values():
            value["rms"] = math.sqrt(value["sum_squared"] / max(1, value["values"]))
        write_json(args.report, {
            "scope": "synthetic PCM composition only; no CMVN/model/decoder/corpus parity claim",
            "passed": result.wasSuccessful(), "tests": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "golden_receipt_sha256": sha(args.goldens / "receipt.json"),
            "library_sha256": sha(args.library),
            "test_runner_sha256": sha(pathlib.Path(__file__)),
            "gates": PcmAdapterTests.receipt["gates"], "metrics": metrics,
            "cases": len(PcmAdapterTests.receipt["cases"]),
            "calls": sum(len(c["calls"]) for c in PcmAdapterTests.receipt["cases"].values()),
            "state_bytes": C.sizeof(State),
        })
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
