# Audio discontinuity handling

A streaming KWS engine must not bridge missing or unrelated audio. Product integrations should notify the engine whenever capture continuity is broken by XRUN, route change, clock reset, device suspend/resume or equivalent pipeline reset.

The discontinuity operation clears partial frontend samples, PCEN smoothing state, recurrent hidden state, decoder/pending prefix state and the active refractory window while preserving configured keywords, processed counters and cumulative telemetry. A discontinuity counter is retained for soak diagnostics.

Calling the discontinuity API from the same serialized owner context as `kws_engine_accept_pcm16()` preserves the single-owner runtime contract.
