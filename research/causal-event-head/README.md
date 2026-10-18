# Causal event-head research core

The unchanged event_head.py contains the 10,498-parameter lazy Torch constructor and a standard-library scalar oracle. Defaults use 140 input features, 32 channels and five causal depthwise/pointwise blocks. Declared per-frame MACs are 10,144, receptive field 63 frames and logical FP32 state 7,936 bytes. These exclude the frozen teacher/front-end and do not establish target-device performance.

Run `python3 -S -B test_event_head_synthetic.py` from this directory. Six artificial standard-library cases cover shape/budget, chunk/reset, causality, controlled order sensitivity and availability timing. This command neither imports Torch nor executes a learned model. CI's static/oracle pass is not a repeat of the numerical preflight.

anonymous-counts.json records prior actual bounded execution: the tiny float64 preflight used seven forwards, 95 input frames, one backward and no optimizer/cache/teacher calls, with atol=rtol=1e-10. Exact maximum error was not retained. The production FP32 head was constructed only, not numerically tested in that preflight.

The separate exposed two-example tiny-fit used 31 updates/backwards, 70 head forwards and 3,535 frames, loading one saved feature cache and making zero teacher calls. Its result is FLUSH_ONLY_MEMORIZATION_RISK: flush learned, pre-EOF did not. Legitimate frozen-backbone delay and padding/boundary effects are not causally separated. It is not complete-phrase, blind-generalization, FAR/FRR or device qualification. No weights, label audit, private sample identity, feature cache or real-example training script is included.

Torch execution is explicit opt-in through build_torch(torch, spec); the caller must provide an already installed runtime and a reviewed bounded experiment. Streaming inference requires torch.no_grad(). No runtime installation, teacher inference or training happens on import or in default CI.
