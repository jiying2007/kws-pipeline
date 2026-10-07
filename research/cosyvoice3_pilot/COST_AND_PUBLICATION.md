# Bounded public pilot: cost and publication basis

Checked 2026-10-02 UTC against current official GitHub billing documentation:
https://docs.github.com/en/billing/concepts/product-billing/github-actions

The current page states that Actions usage in public repositories using standard hosted runners is free. It explicitly introduces minute, artifact-storage and cache quotas under private repositories. The accepted GitHub Community answer to the specific public-build-artifact question clarifies that those artifacts do not count toward the storage limit:
https://github.com/orgs/community/discussions/26438

The clarification is dated 2020 and was checked alongside the current documentation, not used alone. The later shared-storage paragraphs are ambiguous in isolation; no contradictory current rule specifically charging public build artifacts was found. This is the basis for this narrowly bounded public-only artifact. It is not a claim of unlimited artifact availability or of a verified account-wide allowance.

Execution is restricted to public jiying2007/kws-pipeline on standard ubuntu-24.04, with no larger/GPU/self-hosted runner, cache upload, registry package, paid service or paid fallback. Only the allowlisted at-most-twelve newly generated synthetic clips and technical receipts may be uploaded, at most 128 MiB, retention one day. No weights, packages, source archives, credentials, private audio, or real-person cloning reference are output artifacts.

The existing prompt is an already-public Qwen3-TTS Serena preset generation. The test scope is six fixed paired phrases, one seed per phrase and no output-quality retry. Public repository readers signed into GitHub may download the artifact. Retention expiry does not remove downloaded copies or public code/logs. The owner approved this public draft-PR and artifact scope on 2026-10-02. This approval does not authorize merge, training admission, another synthesis sweep or paid resources.
