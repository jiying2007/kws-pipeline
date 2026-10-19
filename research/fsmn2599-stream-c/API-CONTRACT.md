# Continuous research API contract

## Input and identity

Use the immutable descriptor returned by `bsa_model()` with `bsa_create()`.
The current registered model has 400 inputs, 2599 outputs and a 3,027,732-byte
little-endian FP32 payload. The loader checks the registered exact payload hash;
matching shape alone is insufficient. Descriptor identity does not authorize
publishing trained data. The caller may release payload bytes after successful
create; decoded weights remain owned by the handle.

The input format is mono 16000Hz signed PCM16LE. WAV parsing and resampling are
outside the API. `bsa_begin` starts the first recording, while `bsa_reset` starts
subsequent generations. Identity strings must be nonempty NUL-terminated strings
of at most 255 bytes. A callback is mandatory.

Threshold must be finite and in [0,1]. Both signed zeros canonicalize to +0;
nonzero values are retained exactly. Reset preserves threshold. To change model
or threshold, create a new handle. Only the current B descriptor is registered;
there is no A-model routing or OR combiner.

## Feed, clocks, and immediate delivery

`bsa_feed` accepts arbitrary bytes; a trailing low byte is retained until its high
byte arrives. A zero-byte call is a no-op and may use NULL data. `consumed` is
mandatory and reports the processed prefix on failure. Buffers must not alias
handle storage or be mutated during calls.

Context row m requires source position `480*m+720` in the padded sample stream.
Each callback reports:

- `required_padded_sample`: the necessary source endpoint
- `available_real_sample`: floor(total transport bytes supplied / 2), the entire
  current block's endpoint, or total real samples at explicit EOF
- `processed_real_sample`: frontend processing progress at the callback
- `generation`, `valid_so_far`, and `eof_derived`

These are integer sample positions, not measured device alert latency. If a caller
requires finer acquisition timing, it must submit smaller blocks. Different block
sizes may produce different delivery endpoints while posterior and event contents
remain identical.

Events are delivered synchronously during feed and do not wait for EOF. A live
infinite stream can therefore deliver wakeups. A later fatal failure invalidates
the generation but cannot retract events already handed to the caller. No failed
row is automatically replayed. Offline scoring may require successful completion
of the whole recording; that scoring rule must not delay runtime event delivery.

Callback views expire when the callback returns. Copy needed event data before
return. No callback may reenter any handle API or modify the floating-point
mode, input, state or output. A callback error is fatal even if it received events.
Per-row event capacity is three, with no silent event drop or unbounded event list.

## EOF, reset, errors

Only explicit `bsa_finish` appends the canonical 4800 zero samples, exactly once.
There is no flush at a block, decoder window, I/O error or arbitrary timeout.
EOF-derived rows/events are flagged. An odd final byte fails before adding zeros.
An empty recording still produces synthetic-tail rows under this fixed policy.
Repeated finish and feed-after-finish return state errors without more work.

Reset clears partial bytes, frontend history, model cache, decoder windows,
deduplication/refractory state, per-generation counters and failure state. Reset
may explicitly abandon an open or failed generation; it does not undo delivery.
Destroy may abandon a recording and frees resources; NULL destroy is a no-op.

Errors distinguish argument/state/allocation/environment/restore/overflow/
callback/invariant/component failures. Component code and stage are retained.
Processing errors latch until reset. Reserved/completed counters distinguish
attempted operations from successful ones. No rollback or implicit retry occurs.

## Floating point, ownership, and bounded memory

Single-threaded and non-reentrant use is required. Numerical calls save the FP
environment, mask traps, validate RNE/gradual underflow, and restore the caller's
environment. No mode is silently repaired. A restoration error means the caller's
environment may be indeterminate. Compile without fast-math, contraction or excess
precision; probes cannot prove compiler compliance or all platform trap behavior.

Only create/destroy allocate/free. Feed and reset do not accumulate history or
allocate per block. `bsa_storage_bytes` reports persistent owned storage, excluding
caller payload, constants and stack; it is not a measured RSS or target-board
budget. Counters are integer/overflow-checked. Successful offline completion and
decoder support flags never override the frontend/network reference FAIL.

## Decoder diagnostics are separate from validity

`bsa_finish` requires real component success, exact counters, and decoder
`valid && finished`; `numerical_support_ok` is not a lifecycle gate. `bsa_get`
preserves `empty_beam_window_rows`, `zero_mass_window_rows` and the derived support
flag. Consumers should report a diagnostic warning when it is false, without
silently excluding a valid recording from diagnostic denominators. This does not
relax posterior/nonfinite, capacity, clock, FP, or component error handling and
does not change the independent frontend/network 1e-5 FAIL.

This is a contract-fidelity correction, not an observed negative-audio failure.
All-nontarget top3 rows retain the previous beam in the original decoder. A
private mock summary with valid/finished=true and support=false tests field
separation only; it is not proof that this state is reachable from valid rows.

## Capture discontinuities and absolute clocks

This PCM transport API does not detect missing device blocks, XRUN, route changes,
sequence gaps, suspend/resume or clock resets. The capture integration must detect
these conditions. Stop feeding the old generation, discard any incomplete PCM16LE
sample, resynchronise to a complete sample boundary, and call `bsa_reset` before
feeding the first post-gap sample. Check reset success. Do not call `bsa_finish`
at a gap: it appends synthetic zeros and may emit artificial tail events. Do not
silently concatenate post-gap bytes or invent missing PCM samples. An unknown
loss must remain unknown in capture diagnostics, not be recorded as zero loss.
Already delivered events belong to their old generation and cannot be retracted.
Reset clears frontend overlap, network cache and decoder refractory/dedup state;
a refractory period spanning generations, if desired, is an explicit product
policy outside this adapter, not something reset preserves.

Store a capture-side record per generation: recording identity, generation,
discontinuity reason, first complete sample's absolute device sample index (if
known), monotonic timestamp anchor, clock epoch, and known/unknown loss. With an
unbroken 16 kHz clock and origin A, a relative real sample endpoint r maps to A+r;
its timestamp is anchor_ns + r*62500. Check integer overflow. These are endpoint
coordinates: origin denotes the first sample's start, not an additional sample.
For an XRUN with unknown loss or a clock reset, establish a new epoch/anchor;
never infer a precise missing duration or compare incomparable device epochs.

`required_padded_sample` maps to a real acquisition endpoint only for non-EOF
rows. At EOF, it may exceed available real samples and is a synthetic coordinate;
label it accordingly rather than reporting a future hardware capture timestamp.
`available_real_sample` denotes the entire supplied block endpoint, and neither
clock measures wall-clock callback completion/alert latency. Record actual
callback monotonic time separately when profiling latency. `decision_frame` and
node frames are model/decoder coordinates, not sample-accurate acoustic boundaries.

No ABI fields or model arithmetic are changed by this contract. The bounded
`test_capture_contract.c` exercises the actual reset/transport with mocked
acoustic/decoder components, an odd byte at a gap, absolute-clock arithmetic,
repeated finish, and continuous arbitrary chunks. It is not a hardware XRUN test
or long-running acoustic qualification. `test_cpp_link.cpp` checks C++ linkage
against the actual C library at build time; it is not executed.

The optional header-only `bsa_capture.h` implements that reset and mapping pattern
without changing the DSO exports. `bsa_capture_reset` takes an explicit reason and
anchor, calls the real reset, and binds the mapping to the returned generation.
`bsa_capture_map_real` rejects stale generations and overflow, and preserves
unknown coordinates. Capture anchors must come from the caller's actual device
or acquisition clock, not from this helper. The helper neither detects a missing
block nor reconstructs byte alignment; callers must provide that integration.
For EOF rows, map available_real_sample only; required_padded_sample is synthetic.
