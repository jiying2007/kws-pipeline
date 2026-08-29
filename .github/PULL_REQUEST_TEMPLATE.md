## Scope

Describe the change and the product invariant it preserves or changes.

## Claim boundary

- [ ] This PR makes only software/synthetic/hosted claims, or real human/device evidence is explicitly attached and hashed.
- [ ] Hosted x86 metrics and Cortex-A32 cross-build results are not presented as physical target-board performance.
- [ ] Synthetic/rendered acoustic results are not presented as production Mandarin FAR/FRR.

## Contract impact

- [ ] No binary ABI change, or the affected ABI/version and migration policy are updated explicitly.
- [ ] Frontend/model/keyword-pack vocabulary identity remains consistent.
- [ ] Real-time data-plane invariants remain intact, or the architecture/tests/docs are updated in the same PR.
- [ ] No transitional compatibility branch or dead legacy path is introduced.

## Validation

- [ ] GCC and Clang strict builds/tests.
- [ ] ASan/UBSan for C/runtime changes.
- [ ] Parser fuzz smoke for `.kwm` / `.kwk` parser changes.
- [ ] Frontend parity when DSP/frontend behavior changes.
- [ ] Dataset/provenance/release-qualification tests when training/evaluation/release evidence changes.
- [ ] Cortex-A32 cross-build for target-facing changes.
- [ ] Clean SDK install/pkg-config/CMake consumer validation for public SDK changes.

## Evidence and provenance

List exact commands, CI run(s), artifact hashes, corpus/provenance identifiers and any intentionally deferred real-world qualification evidence.

## Documentation

- [ ] English and Simplified Chinese user-facing documentation are updated together when behavior changes.
- [ ] `CHANGELOG.md` and qualification documentation are updated when release-visible behavior or evidence boundaries change.
