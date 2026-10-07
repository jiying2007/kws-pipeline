# CosyVoice3 fixed six-pair CPU pilot

This draft contains one narrowly scoped exploratory pronunciation test. It is not a production model, training-data admission, speech-quality result, or request to merge.

## Frozen comparison

The phrases are 你好小窝, 小窝小窝, 你好小屋, 小窝小屋, 小屋小窝 and 小屋小屋. Each has a plain control and a pronunciation-inpainting variant replacing 窝 with [w][ō] and 屋 with [w][ū]. The mapping is a testable syntactic hypothesis, not an upstream-validated acoustic guarantee. Six seeds are fixed, with alternating condition order and the same already-public 3.44-second synthetic Serena reference. The first two clips form the resource smoke pair and count toward the absolute twelve-clip maximum. No quality-based retries, extra voices, seed searches or text changes are allowed.

## Execution contract

Ordinary PR events run stdlib/static tests only. The one-time job requires the owner-controlled fixed branch, exact head/source/budget approval line, one explicit label, run attempt one, serialized global concurrency, and no previously admitted pilot job. A failed admitted run consumes its admission. There is no workflow dispatch, schedule, private-repository route, paid/larger/GPU runner, cache upload, or merge.

The admitted ubuntu-24.04 job installs the exact isolated Python3.12 runtime sequentially, checks the entire installed dependency closure, tests CPU operators, then qualifies the pinned source and actual frontend-bound SoundFile loader. Models are acquired only after all gates pass and actual/effective disk free space is at least 6,635,020,288 bytes. Models, source and reference are exact allowlisted bytes, and subsequent inference is offline with local-only loaders and forced weights-only checkpoint loading.

The controller limits the process tree to12GiB RSS with at least2GiB host memory available,4CPU threads,1ORT thread, an effective14GB job ceiling and1GiB disk reserve. Startup is limited to900seconds, each clip to300seconds, each accepted audio file to20seconds, and the job to120minutes. Monitored peaks are sampled observations, not proof of every transient between samples. Failure stops the chain and records its stage; no fallback alters precision, model, mapping or hardware.

## Results and interpretation

One-day public Actions artifacts contain only validated synthetic outputs and bounded technical receipts. Reference audio, model files, dependency packages, source trees and credentials are excluded. Native float32,24kHzPCM16 and16kHzPCM16 variants preserve provenance; invalid or partial audio is excluded, with the failed/not-attempted outcomes still recorded. All twelve intended outcomes remain visible.

The later frozen three-ASR evaluation compares both arms independently: exact3/3 target agreement for both K1/K2 and no2/3-or3/3 target false positive among the four foils. Inpaint-only passing is a narrow paired benefit; both passing shows no additional inpainting benefit; control-only passing supports plain CosyVoice under this condition; neither passing is negative/inconclusive for this fixed setup. ASR agreement is not human phonetic ground truth or population accuracy.

See DEPENDENCIES.md, SECURITY_REVIEW.md, COST_AND_PUBLICATION.md and the frozen JSON manifests for the input, safety, cost and public-audience boundaries. The existing reference's partial target prefix, lexical rather than full human transcript qualification, and prior calibration exposure limit interpretation. No real-person reference or voice cloning is used.

## One infrastructure recovery

The original admitted run 37083858274 stopped during setuptools wheel inspection before model acquisition or any synthesis. Its verified artifact (ID 11259881038, ZIP SHA256 df5b34caf1b82f8593dd53545a12998726952357e44626ff71efcef15c4609be) records all twelve outcomes as not attempted. The original admission remains consumed. One explicitly rearmed infrastructure recovery corrects only top-level wheel/installed METADATA selection; dependencies, source/model/reference/phrase/seed inputs and the total twelve-clip maximum do not change. The new gate verifies that exact failed predecessor and artifact, rejects any altered/rerun predecessor, and rejects every other prior non-skipped pilot job. No pronunciation or audio-quality retry is permitted.
