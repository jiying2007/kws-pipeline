# Corpus identity contract

Model and qualification provenance must bind the actual audio bytes, not only the manifest files that name those bytes.

For each mono 16-kHz PCM16 WAV retain:

- source file SHA256;
- decoded PCM SHA256;
- frame count and duration;
- stable recording/path identity;
- speaker/session/source/room/device metadata when available.

The canonical corpus digest is computed over the ordered recording identity records. A renamed or rewrapped file with identical decoded PCM is therefore visible as the same acoustic payload while still retaining distinct source-file identity.

Training checkpoints must capture the corpus identity at training time. Qualification execution must capture the exact WAV identities that produced detections. The final qualification manifest must verify and cross-link both chains.

The final held-out qualification corpus must remain independent from hard-negative and false-reject replay sources.
