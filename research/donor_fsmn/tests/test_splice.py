#!/usr/bin/env python3
"""Freeze actual-source tagged-fbank goldens BEFORE loading the C library.

Oracle generation uses the existing, pinned baseline environment; it does not
load model tensors, calculate acoustic features, read audio, or download code.
Ordinary comparison uses only the standard library and a frozen golden folder.

  python test_splice.py --generate-goldens --baseline /path/to/baseline \
      --goldens /new/immutable/splice-goldens
  python test_splice.py --library /path/to/libdonor_fsmn.so \
      --goldens /new/immutable/splice-goldens --report /new/result.json

The frozen receipt binds exact synthetic PCM, fbank80, splice400, waveform and
feature remainders, source/generator hashes, runtime versions, phase and counts.
These are discrete copy/index tests, with zero numerical tolerance. The pending
waveform is externally buffered by the harness, as required by splice.h. This
is not an arbitrary-PCM-partition-invariance or real-fbank numerical test.
"""

import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import pathlib
import struct
import sys
import unittest


BASELINE_SHA = "498af9c026d6172fc1fe92616d2edc2c0cc848dae2f410bcef3822912eb8cbd0"
SOURCE_SHA = "2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53"
CONTRACT_SHA = "769bd419e0fea707b73b1968ddb3002769a0ea926a7fbfeed5b2dee2a663c022"
CASES = {
    **{f"edge_{n}": [n] for n in
       (0, 1, 399, 400, 719, 720, 799, 800, 959, 960, 4799, 4800, 4801)},
    **{f"full_tail_{n}": [4800, n] for n in
       (0, 1, 479, 480, 639, 640, 799, 800, 959, 960, 4799)},
    "many_full_calls": [4800] * 7,
    "three_new_frames": [800, 480, 480, 480, 480],
    "gate_accumulation": [400, 399, 1, 0, 100, 379, 1, 0, 640],
    "repartitioned_4800": [800] + [480] * 8 + [160],
    "max_supported_call": [16000],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_goldens(args):
    # This branch never loads a C library or runs C outputs.
    import numpy as np
    import torch

    baseline = args.baseline.resolve()
    driver = baseline / "run_baseline.py"
    source = baseline / "upstream/wekws/bin/stream_kws_ctc.py"
    if sha(driver) != BASELINE_SHA or sha(source) != SOURCE_SHA:
        raise ValueError("Pinned baseline or accept_wave source identity changed")
    args.goldens.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("splice_pinned_baseline", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, namespace, source_hashes = module.audited_sources()
    tokens = (baseline / "resources/tokens_2599.txt").read_text().splitlines()
    files = {}

    def save_bytes(name, payload):
        path = args.goldens / name
        with path.open("xb") as handle:
            handle.write(payload)
        files[name] = {"bytes": len(payload), "sha256": sha(path)}
        return name

    class TaggedFbank:
        def __init__(self):
            self.index = 0
            self.rows = None
            self.wave = None

        def fbank(self, waveform, **kwargs):
            assert kwargs == dict(window_type="hamming", num_mel_bins=80,
                                  frame_length=25, frame_shift=10, dither=0,
                                  energy_floor=0.0, sample_frequency=16000)
            assert waveform.dtype == torch.float32
            count = 1 + (waveform.shape[1] - 400) // 160
            assert count >= 3
            frames = torch.arange(self.index, self.index + count,
                                  dtype=torch.float32)[:, None]
            bins = torch.arange(80, dtype=torch.float32)[None, :]
            self.rows = frames * 128 + bins
            self.wave = waveform[0].numpy().astype("<i2").tobytes()
            self.index += count
            return self.rows

    all_cases = {}
    for name, sizes in CASES.items():
        tagged = TaggedFbank()
        spotter = module.make_spotter("hamming", None, tagged, namespace, tokens)
        calls = []
        sample_index = 0
        for number, sample_count in enumerate(sizes):
            prefix = f"{name}-{number:02d}"
            pcm = b"".join(struct.pack("<h", ((i * 977 + 123) % 65536) - 32768)
                           for i in range(sample_index, sample_index + sample_count))
            pending_before = np.asarray(spotter.wave_remained, dtype="<i2").tobytes()
            tagged.rows = None
            tagged.wave = None
            traced = {}

            def trace(frame, event, arg):
                if (event == "return" and frame.f_code.co_name == "accept_wave"
                        and frame.f_code.co_filename == str(source)):
                    traced["splice_rows"] = frame.f_locals.get("ctx_frm", 0)
                return trace

            # Count ctx_frm from the executing source, not a second oracle copy.
            sys.settrace(trace)
            try:
                selected = spotter.accept_wave(pcm)
            finally:
                sys.settrace(None)
            assert "splice_rows" in traced
            fbank = (b"" if tagged.rows is None else
                     tagged.rows.numpy().astype("<f4").tobytes())
            output = (b"" if selected is None else
                      selected.numpy().astype("<f4").tobytes())
            features = (b"" if spotter.feature_remained is None else
                        spotter.feature_remained.numpy().astype("<f4").tobytes())
            pending_after = np.asarray(spotter.wave_remained, dtype="<i2").tobytes()
            sample_index += sample_count
            assert tagged.wave is None or tagged.wave == pending_before + pcm
            calls.append({
                "call_samples": sample_count,
                "pcm": save_bytes(prefix + ".pcm16le", pcm),
                "new_fbank": save_bytes(prefix + ".fbank.f32le", fbank),
                "output": save_bytes(prefix + ".splice.f32le", output),
                "features": save_bytes(prefix + ".remained.f32le", features),
                "wave": save_bytes(prefix + ".wave.pcm16le", pending_after),
                "waveform_samples": len(pending_before) // 2 + sample_count,
                "fbank_rows": len(fbank) // (80 * 4),
                "consumed_samples": (len(pending_before) + len(pcm) - len(pending_after)) // 2,
                "retained_samples": len(pending_after) // 2,
                "splice_rows": traced["splice_rows"],
                "selected_rows": len(output) // (400 * 4),
                "next_offset": spotter.feats_ctx_offset,
                "feature_count": len(features) // (80 * 4),
                "total_fbank": tagged.index,
                "total_samples": sample_index,
                "source_returned_none": selected is None,
                "centers": ([] if selected is None else
                            [int(x) // 128 for x in selected[:, 160].tolist()]),
            })
        all_cases[name] = calls
    save_bytes("generator.py", pathlib.Path(__file__).read_bytes())
    receipt = {
        "schema": "donor-splice-source-goldens-v1",
        "scope": "synthetic PCM accounting and actual accept_wave with tagged fbank; no model/audio inference",
        "contract_sha256": CONTRACT_SHA,
        "baseline_driver_sha256": BASELINE_SHA,
        "accept_wave_source_sha256": SOURCE_SHA,
        "audited_source_hashes": source_hashes,
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
        "generator": "generator.py",
        "generator_policy": "frozen before any C output; receipt never overwritten",
        "numeric_gate": "exact little-endian FP32 bytes, counts, centers and state",
        "tag_formula": "fbank[j,b] = float32(global_frame_j * 128 + bin_b)",
        "files": files,
        "cases": all_cases,
    }
    write_json(args.goldens / "receipt.json", receipt)
    print(f"Frozen actual-source goldens: {len(all_cases)} cases, "
          f"{sum(map(len, all_cases.values()))} calls; receipt {sha(args.goldens / 'receipt.json')}")


class State(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("initialized", "finished", "wave_samples", "feature_count", "offset")] + [
        ("total_samples", C.c_uint64), ("total_fbank", C.c_uint64),
        ("total_selected", C.c_uint64), ("feature_remained", (C.c_float * 80) * 4)]


class Plan(C.Structure):
    _fields_ = [(name, C.c_size_t) for name in
                ("waveform_samples", "fbank_rows", "consumed_samples", "retained_samples",
                 "splice_rows", "selected_rows")] + [("next_offset", C.c_uint32)]


CALLBACK = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(C.c_float), C.c_uint64, C.c_uint64)
NULL_CALLBACK = CALLBACK()


def load_receipt(root):
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt["schema"] != "donor-splice-source-goldens-v1" or
            receipt["accept_wave_source_sha256"] != SOURCE_SHA or
            receipt["baseline_driver_sha256"] != BASELINE_SHA or
            receipt["contract_sha256"] != CONTRACT_SHA):
        raise ValueError("Golden source/contract identity mismatch")
    expected = {"receipt.json", *receipt["files"]}
    actual = {p.name for p in root.iterdir()}
    if actual != expected:
        raise ValueError("Golden directory membership changed")
    for name, info in receipt["files"].items():
        path = root / name
        if (path.name != name or path.is_symlink() or not path.is_file() or
                path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]):
            raise ValueError(f"Golden file integrity failure: {name}")
    if {k: [c["call_samples"] for c in v] for k, v in receipt["cases"].items()} != CASES:
        raise ValueError("Golden case coverage changed")
    return receipt


class SpliceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.byteorder != "little":
            raise RuntimeError("This fixture runner currently requires a little-endian host")
        # Integrity validation above happens BEFORE even loading the C library.
        cls.lib = C.CDLL(str(cls.library))
        lib = cls.lib
        lib.donor_splice_state_bytes.restype = C.c_size_t
        if lib.donor_splice_state_bytes() != C.sizeof(State):
            raise ValueError("C/Python state ABI mismatch")
        for name in ("init", "reset", "finish"):
            method = getattr(lib, "donor_splice_" + name)
            method.argtypes = [C.POINTER(State)]
            method.restype = C.c_int
        lib.donor_splice_plan_call.argtypes = [C.POINTER(State), C.c_size_t, C.POINTER(Plan)]
        lib.donor_splice_plan_call.restype = C.c_int
        lib.donor_splice_accept_call.argtypes = [C.POINTER(State), C.c_size_t,
                                               C.POINTER(C.c_float), C.c_size_t,
                                               CALLBACK, C.c_void_p]
        lib.donor_splice_accept_call.restype = C.c_int

    def new_state(self):
        state = State()
        self.assertEqual(self.lib.donor_splice_init(C.byref(state)), 0)
        return state

    def payload(self, name):
        return (self.goldens / name).read_bytes()

    def run_call(self, state, pending, call):
        plan = Plan()
        before = bytes(state)
        self.assertEqual(self.lib.donor_splice_plan_call(C.byref(state), call["call_samples"],
                                                       C.byref(plan)), 0)
        self.assertEqual(bytes(state), before, "plan must not mutate state")
        for name, _ in Plan._fields_:
            self.assertEqual(getattr(plan, name), call[name], name)
        pcm = self.payload(call["pcm"])
        wave = pending + pcm
        self.assertEqual(len(wave) // 2, plan.waveform_samples)
        remaining = wave[plan.consumed_samples * 2:]
        self.assertEqual(remaining, self.payload(call["wave"]))
        features = self.payload(call["new_fbank"])
        array = (C.c_float * (len(features) // 4)).from_buffer_copy(features) if features else None
        outputs, centers, available = [], [], []

        @CALLBACK
        def receive(_, row, center, samples):
            outputs.append(C.string_at(row, 400 * 4))
            centers.append(center)
            available.append(samples)

        self.assertEqual(self.lib.donor_splice_accept_call(C.byref(state), call["call_samples"],
                         array, call["fbank_rows"], receive, None), 0)
        self.assertEqual(b"".join(outputs), self.payload(call["output"]))
        self.assertEqual(centers, call["centers"])
        self.assertEqual(available, [call["total_samples"]] * call["selected_rows"])
        self.assertEqual(state.wave_samples, call["retained_samples"])
        self.assertEqual(state.feature_count, call["feature_count"])
        self.assertEqual(state.offset, call["next_offset"])
        self.assertEqual(state.total_fbank, call["total_fbank"])
        self.assertEqual(state.total_samples, call["total_samples"])
        self.assertEqual(state.total_selected,
                         State.from_buffer_copy(before).total_selected + call["selected_rows"])
        self.assertEqual(C.string_at(state.feature_remained, state.feature_count * 80 * 4),
                         self.payload(call["features"]))
        self.assertEqual(bytes(state.feature_remained)[state.feature_count * 80 * 4:],
                         bytes((4 - state.feature_count) * 80 * 4))
        return remaining

    def test_actual_source_goldens(self):
        for name, calls in self.receipt["cases"].items():
            with self.subTest(case=name):
                state, pending = self.new_state(), b""
                for call in calls:
                    pending = self.run_call(state, pending, call)
                before = bytes(state)
                self.assertEqual(self.lib.donor_splice_finish(C.byref(state)), 0)
                state.finished = 0
                self.assertEqual(bytes(state), before, "EOF only marks finished; no tail mutation")

    def test_first_second_call_and_left_edge(self):
        calls = self.receipt["cases"]["many_full_calls"]
        self.assertEqual(calls[0]["centers"], list(range(0, 25, 3)))
        self.assertEqual(calls[1]["centers"], list(range(27, 55, 3)))
        self.assertEqual([c["selected_rows"] for c in calls[:2]], [9, 10])
        self.assertEqual([c["next_offset"] for c in calls[:2]], [1, 1])
        self.assertEqual([c["retained_samples"] for c in calls[:2]], [320, 320])
        row = struct.unpack("<400f", self.payload(calls[0]["output"])[:1600])
        self.assertEqual(row[::80], (0, 0, 0, 128, 256))

    def test_three_new_rows_are_not_four_historical_rows(self):
        calls = self.receipt["cases"]["three_new_frames"]
        self.assertEqual([c["feature_count"] for c in calls], [3] * len(calls))
        self.assertTrue(any(c["selected_rows"] == 0 and not c["source_returned_none"]
                            for c in calls))
        whole = self.payload(self.receipt["cases"]["edge_4800"][0]["output"])
        repartitioned = b"".join(self.payload(c["output"]) for c in
                                self.receipt["cases"]["repartitioned_4800"])
        self.assertNotEqual(whole, repartitioned,
                            "Actual tiny-call source behavior is not partition invariant")

    def test_reset_and_repeated_file(self):
        state = self.new_state()
        fresh = bytes(state)
        for case in ("gate_accumulation", "many_full_calls", "three_new_frames", "many_full_calls"):
            pending = b""
            for call in self.receipt["cases"][case]:
                pending = self.run_call(state, pending, call)
            self.assertEqual(self.lib.donor_splice_finish(C.byref(state)), 0)
            self.assertEqual(self.lib.donor_splice_reset(C.byref(state)), 0)
            self.assertEqual(bytes(state), fresh)

    def test_interleaved_handles(self):
        cases = [self.receipt["cases"][name] for name in
                 ("many_full_calls", "three_new_frames", "gate_accumulation")]
        states = [self.new_state() for _ in cases]
        waves = [b""] * len(cases)
        for index in range(max(map(len, cases))):
            for stream, calls in enumerate(cases):
                if index < len(calls):
                    untouched = [bytes(s) for s in states]
                    waves[stream] = self.run_call(states[stream], waves[stream], calls[index])
                    for other, state in enumerate(states):
                        if other != stream:
                            self.assertEqual(bytes(state), untouched[other])

    def test_invalid_calls_are_atomic(self):
        state = self.new_state()
        output = []

        @CALLBACK
        def receive(*args):
            output.append(args)

        before = bytes(state)
        values = (C.c_float * 240)(*range(240))
        for samples, ptr, count, callback in (
                (16001, None, 0, receive), (800, None, 3, receive),
                (800, values, 2, receive), (800, values, 3, NULL_CALLBACK),
                (0, values, 3, receive), (800, values, C.c_size_t(-1).value, receive),
                (800, C.cast(C.byref(state), C.POINTER(C.c_float)), 3, receive)):
            self.assertEqual(self.lib.donor_splice_accept_call(C.byref(state), samples,
                                                             ptr, count, callback, None), 1)
            self.assertEqual(bytes(state), before)
            self.assertEqual(output, [])
        for value in (float("nan"), float("inf"), -float("inf")):
            values[-1] = value
            self.assertEqual(self.lib.donor_splice_accept_call(C.byref(state), 800,
                                                             values, 3, receive, None), 1)
            self.assertEqual(bytes(state), before)
            self.assertEqual(output, [])
        plan = Plan()
        self.assertEqual(self.lib.donor_splice_plan_call(None, 0, C.byref(plan)), 1)
        self.assertEqual(self.lib.donor_splice_plan_call(C.byref(state), 0, None), 1)
        self.assertEqual(self.lib.donor_splice_plan_call(C.byref(state), 0,
                         C.cast(C.byref(state), C.POINTER(Plan))), 1)
        self.assertEqual(bytes(state), before)
        self.assertEqual(self.lib.donor_splice_init(None), 1)
        self.assertEqual(self.lib.donor_splice_reset(None), 1)
        self.assertEqual(self.lib.donor_splice_finish(None), 1)

    def test_finished_and_uninitialized_states(self):
        state = State()
        plan = Plan()
        self.assertEqual(self.lib.donor_splice_plan_call(C.byref(state), 0, C.byref(plan)), 2)
        self.assertEqual(self.lib.donor_splice_finish(C.byref(state)), 2)
        self.assertEqual(self.lib.donor_splice_init(C.byref(state)), 0)
        self.assertEqual(self.lib.donor_splice_accept_call(C.byref(state), 799,
                                                         None, 0, NULL_CALLBACK, None), 0)
        self.assertEqual(state.wave_samples, 799)
        self.assertEqual(self.lib.donor_splice_finish(C.byref(state)), 0)
        before = bytes(state)
        self.assertEqual(self.lib.donor_splice_finish(C.byref(state)), 2)
        self.assertEqual(self.lib.donor_splice_plan_call(C.byref(state), 1, C.byref(plan)), 2)
        self.assertEqual(self.lib.donor_splice_accept_call(C.byref(state), 0,
                                                         None, 0, NULL_CALLBACK, None), 2)
        self.assertEqual(bytes(state), before)

    def test_64bit_timeline_and_overflow(self):
        for total_frames in ((1 << 31) - 3, (1 << 32) - 3, (1 << 32) + 1):
            state = self.new_state()
            state.feature_count = 4
            state.total_fbank = total_frames
            state.wave_samples = 320
            state.total_samples = total_frames * 160 + 320
            state.total_selected = total_frames // 3
            values = (C.c_float * (30 * 80))()
            centers, available = [], []

            @CALLBACK
            def receive(_, row, center, samples):
                centers.append(center)
                available.append(samples)

            self.assertEqual(self.lib.donor_splice_accept_call(C.byref(state), 4800,
                                                             values, 30, receive, None), 0)
            self.assertEqual(centers, list(range(total_frames - 2, total_frames + 28, 3)))
            self.assertEqual(available, [total_frames * 160 + 5120] * 10)
        state = self.new_state()
        state.feature_count = 4
        state.total_samples = (1 << 64) - 1
        state.total_fbank = (state.total_samples - 320) // 160
        state.wave_samples = state.total_samples - state.total_fbank * 160
        before = bytes(state)
        plan = Plan()
        self.assertEqual(self.lib.donor_splice_plan_call(C.byref(state), 1, C.byref(plan)), 1)
        self.assertEqual(self.lib.donor_splice_accept_call(C.byref(state), 1,
                                                         None, 0, NULL_CALLBACK, None), 1)
        self.assertEqual(bytes(state), before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldens", type=pathlib.Path, required=True)
    parser.add_argument("--generate-goldens", action="store_true")
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--library", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    if args.generate_goldens:
        if args.baseline is None or args.library is not None or args.report is not None:
            parser.error("generation requires --baseline, and forbids --library/--report")
        generate_goldens(args)
        return
    if args.library is None or args.baseline is not None:
        parser.error("comparison requires --library and forbids --baseline")
    SpliceContract.receipt = load_receipt(args.goldens)
    SpliceContract.goldens = args.goldens.resolve()
    SpliceContract.library = args.library.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SpliceContract))
    if args.report:
        write_json(args.report, {
            "scope": "synthetic tagged-fbank source parity; external PCM accounting only",
            "passed": result.wasSuccessful(), "tests": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "golden_receipt_sha256": sha(args.goldens / "receipt.json"),
            "library_sha256": sha(args.library),
            "test_runner_sha256": sha(pathlib.Path(__file__)),
            "cases": len(SpliceContract.receipt["cases"]),
            "calls": sum(map(len, SpliceContract.receipt["cases"].values())),
            "numeric_gate": "exact bytes and discrete state, no tolerance",
        })
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
