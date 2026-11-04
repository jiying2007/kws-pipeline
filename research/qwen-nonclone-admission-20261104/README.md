# Read-only admission compatibility and prospective capture

The completed four-file pilot is not automatically training data. Its successful
blind three-recognizer run remains source-screen evidence. Public run IDs and
hashes are in ../qwen-nonclone-asr-20261104/public-run-metadata.json. The full ASR
archive is not published here; its publication was not approved. Do not upload
it through another route. These tools contain only source and invented tests.

compatibility.py reads an explicitly supplied local evidence directory and
mapping. It emits no raw transcripts and never emits admission=true. This gap checklist
does not independently authenticate the archive, decoder hashes or internal
field semantics; its field-presence flags refer to declarations only. Use the
separate exact archive/durable-terminal verification before interpreting it. Actual
input/processor/tokenizer/token/EOS observations remain reusable, without
relabelling legacy records as native-v2 or declaring an unreviewed train root.
Missing fields stay missing. Current rules allow reviewed machine weak labels
on train-only data, but require truthful native capture and a reviewed source.
The exact legacy run omitted Qwen's resolved runtime GenerationConfig bytes;
tokens, EOS and request arguments do not prove inherited beam/forced-token/
suppression/word/sequence-bias/stop-string settings. Unknown speech is never an
empty CTC target. The current transcript policy only represents 你/好/小/窝.

The prospective generation_capture.py observer closes that specific recording
gap for a future, separately admitted run. It intercepts the actual pinned
model's native _prepare_generation_config return, preserves its objects and all
JSON-compatible fields, validates required observed settings without defaults,
and persists the full immutable snapshot before native generation continues.
It does not recreate any past observation, alter decoding, load a model,
register trust, authorize training, or activate a workflow. Any failed resolution or persistence consumes the attempt and poisons the observer.
A future worker must attach to the actual GenerationMixin used by thinker.generate,
not assume the top-level wrapper is sufficient, and bind exactly one complete
capture to each claimed audio call. It has only been
tested with invented resolver objects; integration with the exact locked real
model still requires source review and a new bounded campaign.

Do not expand to 64–200 samples before the native capture/source-policy path
can deliver admissible evidence, or the same missing observation will be
replicated. A new expansion should prefreeze its weak-label train-only policy,
carry genuine speaker/source/derivation lineage and separate held-out data,
use representable positive/near-miss text or explicitly review a vocabulary
extension, preserve all failed slots, and never replace a failed old slot.

Both scripts are Python3.8-syntax compatible. Run each test file with normal,
-O and -OO. Tests use invented JSON and resolver objects, with zero ASR/TTS/
training calls, network access or dependency installation.
