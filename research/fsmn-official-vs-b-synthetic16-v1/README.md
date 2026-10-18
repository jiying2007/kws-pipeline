# Original versus adapted head: fixed synthetic16 diagnostic

The existing B head improves fixed-threshold target detection on this small synthetic set when both heads use the same frozen encoder and current decoder. This is evidence of adaptation on this development source, not an independent acceptance test.

All16 inputs were retained:8 machine-weak targets,6 machine-weak near-homophones,2 ASR disagreements. At cumulative body+300ms, threshold0.1 gives original5/8 versus B8/8; threshold0.7 gives original0/8 versus B6/8. Neither triggers on the6 weak near clips. B has one event on an unknown-label clip at0.1, reported separately rather than silently treating it as correct or false.

The body analytical window gives only1/8 for each head at0.1, and0/8 versus1/8 at0.7. Both windows derive from one offline padded forward with right context. They are not measured real-time recall, separate body-only decoding, or guaranteed acoustic timing.

The four reference sources are train-only, obtained through a third-party mirror without official-release byte verification. Historical speaker exposure is unknown. Labels come from two blind ASR transcripts and are not human gold. No KWS-based label selection, training, threshold search or deployment promotion occurred. The prior augmentation coverage gate remains failed and unchanged.

The JSON contains anonymous aggregates, existing public model identities, exact operation counts and measured resources. No audio, weights, private clip identifiers, private paths or Library identifiers are published. Raw evidence is retained privately. This file grants no weight/data license; consult their original source terms.
