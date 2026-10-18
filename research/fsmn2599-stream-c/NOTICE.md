# Attribution and modifications

This Apache-2.0 research module combines new continuous-stream glue with retained
canonical frontend, full-alphabet FSMN C port, SHA loader and research softmax.
It is not a production model release. The full Apache-2.0 license is in LICENSE.

## FSMN and decoder lineage

WeKws commit 6a45aeb994dd81c0969ff877a5a7c46d60ed0c86,
Copyright 2022 Yueyue Nyy; Copyright 2023 Jing Du:
https://github.com/wenet-e2e/wekws/blob/6a45aeb994dd81c0969ff877a5a7c46d60ed0c86/wekws/model/fsmn.py

The C implementation derives from this repository's isolated native A20 FP32
model. The modified full-alphabet port has a 2599-class final affine, checked
byte loader, changed weight/bias offsets, capacities and API. The SHA256 memory
implementation retains the repository algorithm and removes its file-I/O helper.
The existing threshold-aware decoder is referenced at
research/fsmn2599-three-window-c/kws-fsmn2599-three-window-c-preparation-20261018;
its NOTICE and LICENSE apply and its sources are not copied here.

## Frontend and mathematical constants

The radix-2 FFT512 loop is adapted from the repository's Apache-2.0 donor frontend
skeleton. Existing research/donor_fbank/fft_twiddles.h is used byte-identically;
these are float32 mathematical roots of unity, not trained weights or proof of
Torch FFT agreement. Build-time include mapping avoids copying that file.

Torchaudio 2.11.0+cpu compliance/kaldi.py was the directly inspected algorithm
reference for feature-window, waveform-window, mel-bank and fbank semantics.
The BSD-2-Clause notice, Copyright 2017 Facebook, is retained in
LICENSE.torchaudio. canonical_tables.h contains frozen mathematical Povey-window
and dense 80x257 mel coefficients from that runtime; it contains no learned CMVN,
model parameters or audio. The donor Hamming/sparse-mel tables are not substituted.

The canonical frontend uses magnitude-then-square power and increasing-frequency
dense mel reduction. canonical_pcm implements frame-step transport with immediate
context callbacks and one explicit 4800-sample EOF tail. It does not apply CMVN.
These are researched contracts, not demonstrated reference numerical parity.

## Adapter and softmax

The new adapter replaces fixed diagnostic glue's recording-count, block-size,
old-model identity and threshold assumptions with a registered B descriptor,
explicit threshold, generations, checked arbitrary-byte transport and per-call
floating-point environment restoration. Numerical component algorithms remain
unchanged. softmax2599 uses its own documented mixed-precision research contract;
it is not a Torch bit-equivalence claim.

Public sources and shared dependency hashes are listed in source-binding.json.
No private path, recording, trained payload or binary is required for the synthetic
fixtures. The source license alone does not authorize distribution of checkpoints,
training material or recordings. The historical acoustic numerical FAIL remains.
