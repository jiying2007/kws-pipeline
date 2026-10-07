# Offline keyword-set identity v2

`tools/keyword_set_identity.py` provides one current keyword semantic identity and
strict source contract verifier. It is an offline, standard-library-only tool;
it does not train, run inference, download models/audio, or change product policy.

## Use

All file arguments are canonical POSIX paths relative to an explicit existing
root. Absolute paths, traversal, symlinks (including symlinked parent components),
missing files, directories, and malformed input fail closed. Root components are
also opened without following symlinks. POSIX descriptor-relative `O_NOFOLLOW`
and directory opening are required; unsupported platforms fail closed. Each
root/parent directory and regular input file is pinned by an open descriptor and
checked with `fstat`, including the identity contract itself. Leaf/parent symlink
swaps cannot redirect a later read outside the selected root. Verification keeps
one root descriptor for its contract and all sources. There is no implicit
repository root or implicit parameter-contract selection.

From the repository root:

```sh
python3 -B tools/keyword_set_identity.py build \
  --root . \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --parameter-contract configs/parameter-contract.json \
  > /tmp/keyword-identity.json
```

The complete JSON output is itself the v2 identity contract. To retain and verify
a contract, place it inside your explicitly selected input root (for example, an
ignored `build/` directory in a working checkout), then run:

```sh
mkdir -p build
python3 -B tools/keyword_set_identity.py build \
  --root . \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --parameter-contract configs/parameter-contract.json \
  > build/keyword-identity.json
python3 -B tools/keyword_set_identity.py verify \
  --root . --contract build/keyword-identity.json
python3 -B tests/test_keyword_set_identity.py
```

Both successful commands write the complete identity JSON to stdout and return
zero. Invalid inputs/mismatches return nonzero with a diagnostic on stderr. The
tool only reads the selected root and uses temporary snapshots to call existing
parsers; the shell examples explicitly select any output file. Do not redirect
output onto an input file.

Python callers can import `build_keyword_set_identity` and
`verify_keyword_set_contract` from `tools/keyword_set_identity.py` with the `tools`
directory on their import path. Both require `root=`. The builder also requires
`tokens_path=`, `keywords_path=`, and `parameter_contract_path=` as root-relative
strings; the verifier takes a root-relative identity-contract path.

## What is bound

The schema is `schema_version: 2`, with policy
`kws-keyword-set-identity-v2`. There is one implementation and no legacy shim.

- `sources` records each exact source path, byte length, and SHA-256 for the token
  vocabulary, keyword TSV, and parameter contract
- `semantics.tokens` binds the entire vocabulary, including unused tokens, sorted
  by numeric ID. Explicit token-file row order does not matter; changed ID
  assignments, renamed unused tokens, or additions do
- `semantics.keywords` contains every field returned by the current
  `compile_keywords.parse_keywords`: ID, text, threshold, tokens, token IDs,
  trailing blanks, priority, prefix policy/name/ID, and grace frames. Keyword row
  order is retained because it affects the compiled record order and can matter
  to tie handling
- `semantics.parameter_policy` binds the keyword field types, defaults, bounds,
  bound exclusivity, and longest/grace fallback defaults. Even a currently unused
  keyword default or range change changes this identity
- `semantics.compiler_limits` binds the current keyword/token/vocabulary limits,
  pack version, and prefix-policy encoding
- `semantic_sha256` hashes canonical UTF-8 JSON of `semantics`, with sorted object
  keys, compact separators, and no non-finite numbers

The canonical vocabulary loader and compiler parser remain authoritative. The
new tool adds strict JSON/parameter validation, requires explicit TSV tokens, and
rejects empty keyword text. It never attempts implicit pinyin conversion or
installs/imports `pypinyin`. The existing parameter-header validator is also used
for range/default/wire-limit and algorithm-invariant checks. Unknown fields,
duplicate JSON keys, unsupported versions, boolean/string numeric substitutes,
non-finite values, and malformed policies are rejected rather than guessed.

Comments, whitespace, equivalent threshold notation, explicit values equal to
resolved defaults, and parameter JSON key order do not change the semantic hash.
They do change raw source hashes. Commentary fields do not affect semantic
identity. Runtime/L1 values are outside **keyword** semantic identity; they are
still validated and pinned by the complete parameter-contract raw hash.

Verification always requires **both** exact raw sources and recomputed semantics.
There is no permissive or semantic-only verification flag. Reformatting a source
therefore requires generating a new raw contract, even when semantic hashes match.
Identity-contract JSON formatting/key order itself may change. All claimed fields
are compared against rebuilt canonical JSON, so unrecognized nested fields and
Python's `True == 1` coercion cannot silently validate a contract.

This is a source/parsed-parameter identity, not a claim that different source
threshold values which round to the same float32 are interchangeable. In particular, source thresholds `1e-100` and `0.999999999` pass the current
source parser but round to float32 `0` and `1`, respectively; those emitted values
violate the current binary threshold range. The negative-claim regression records
this distinction deliberately: successful identity verification does not certify
binary/deployment validity. This tool does not change that shipping compiler. It
does not replace binary pack validation, calibration, model provenance, or shipping
qualification. It is not a signature or proof that whoever created a contract was
authorized to change the inputs.

## Historical source provenance and deliberate changes

Two preserved branch designs motivated this fresh, bounded implementation:

1. `training/keyword-set-contract-v1`, source commit
   `986d44c9e0efe15ce0656413a2f2112aeccaea96`:
   [training/keyword_set_contract.py](https://github.com/jiying2007/kws-pipeline/blob/986d44c9e0efe15ce0656413a2f2112aeccaea96/training/keyword_set_contract.py)
   compared expected keywords but duplicated token/TSV parsing and accepted policy
   columns without binding their effective values
2. `training/keyword-set-identity-v1`, source commit
   `2d52b58f36163fb93b9ba1b1356fa184c187fa2f`:
   [training/keyword_set_identity.py](https://github.com/jiying2007/kws-pipeline/blob/2d52b58f36163fb93b9ba1b1356fa184c187fa2f/training/keyword_set_identity.py)
   bound the complete vocabulary but omitted threshold and per-keyword policies
   from semantic rows, and did not constrain resolved paths to the root

The useful contract/identity ideas are absorbed together, not exposed as two
permanent APIs. Version 2 intentionally has different semantics and rejects v1;
no old identity digest is relabeled as equivalent. The current compiler/parser
at protected-main base `221dedd21d0892965aacda15e676fc97d29bd32b` is reused instead
of porting either incomplete parser.

Deferred: wiring identity into corpus generators, trainer/checkpoint/provenance,
replay, model promotion, and deployment would change their contracts and requires
separate design and tests. Historical trainers/replay/workflows are not imported.
No shipping/training config, model, keyword TSV, parameter contract, deployment,
or approval flag changes are made by this port. Current shipping remains the
immutable `model-749187ec1d66` two-keyword tuple (`你好小窝`, `小窝小窝`, threshold
0.55 each), with `shipping_approved=false`.

## Verification and CI source retention

`tests/test_keyword_set_identity.py` uses invented vocabulary/TSV fixtures and
local parameter copies. It tests all effective fields, unused tokens/ID maps,
defaults/ranges, raw versus semantic stability, canonical parser reuse, malformed
JSON/numbers/contracts, strict paths, snapshots, CLI round trips, and unchanged
current shipping/config bytes. Deterministic adversarial tests swap every input
leaf and parent immediately before opening, swap parents after descriptor capture,
and replace the root between identity-contract and source reads. FIFO and
unsupported-access checks fail closed. Float32 edge cases assert the binary-validity
claim is deliberately absent. It needs no optional Python dependencies.

CI adds this one test to the existing `Keyword compiler` step, retaining its
tracked-worktree guard. The exact `.github/workflows/ci.yml` baseline is updated
in `research/consolidation/source-retention-2026-10-07.json` with a maintenance
record containing the old/new byte identities:

- Before: 23,870 bytes; SHA-256
  `81fa55fe96fbc5b8008129a5442b95b219a4940dc4925b20eac13295d4f8fa0d`;
  Git blob `1a9d3eaa494922fe2481749741f94b7639ab277a`
- After: 23,935 bytes; SHA-256
  `457a7a4179f3bd2807d5d6e556669bbf21c111d770285ed1351b819b453c1bbf`;
  Git blob `7d02db422f20c68f2ffdc25b7ea0e98df3b6ad20`

The retained-source verifier is unchanged: historical files, provenance,
projections, original freezes, and the active-workflow inventory remain exact.
This updates an intentional active-workflow baseline; it does not relax retention.

The separate `core-2026-10-07.json` retains all 102 original source entries. Its
original PR 485 CI bytes are preserved unchanged at
`research/consolidation/maintained-sources/pr-485/ci.yml`, outside the active
workflow directory. Only that manifest row's retained `path` changes; explicit
`source_path` and `relocation_reason` retain the original CI location and explain
maintenance. Original source PR, commit, mode, byte count, and both hashes remain
unchanged. The active CI is still pinned separately as described above. Both
retention verifiers remain unchanged. Focused tests prove tampering archived CI
fails core retention and tampering active CI fails current source retention. The
offline historical projection copies the archived path as data and does not
substitute the changed active workflow.
