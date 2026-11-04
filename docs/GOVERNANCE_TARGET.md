# Terminal governance target

The source-controlled workflows are the implementation of software gates; repository settings are the enforcement boundary.

The canonical machine target for `main` is `governance/main-ruleset-target.json`. It is intentionally shaped as a GitHub repository-ruleset request body so an administrator can apply the reviewed policy without translating prose into a second configuration.

The target ruleset is named `kws-main-terminal`, targets `~DEFAULT_BRANCH`, is `active`, has no bypass actors, and requires:

- all updates to `main` to arrive through a pull request;
- zero approving reviews while this remains a single-maintainer repository, avoiding self-deadlock;
- all review conversations to be resolved;
- squash as the allowed merge method;
- strict required checks `hosted (gcc)`, `hosted (clang)`, `coverage`, `sanitizers`, `fuzz`, and `armv7-cross`;
- deletion protection;
- non-fast-forward protection, blocking force pushes.

The reviewed target is currently applied to the live repository as the active
`kws-main-terminal` ruleset. A live audit on 2026-09-27 confirmed the rule
types, strict required status contexts, no bypass actors, pull-request
requirements and default-branch targeting match
`governance/main-ruleset-target.json`; the branch API reports
`protected=true`.

The command below is therefore a **recovery/migration example**, not an
instruction to create a second overlapping ruleset:

```bash
gh api --method POST \
  "repos/OWNER/REPO/rulesets" \
  --input governance/main-ruleset-target.json
```

If the live ruleset already exists, audit/update that rule instead of creating
another policy with the same purpose. Use the narrowest repository-rules
administration credential available and never commit an administration token.

For any research change, verify the live ruleset still protects `main` and that exact-head source, contract, compiler, sanitizer, and ARM checks are complete before a protected pull-request merge. The former `deployment/commercial-candidate` branch and deployment publication workflow are retired historical records, **not** branches or workflows to create or execute. See [research-only execution](RESEARCH_ONLY_EXECUTION.md).

The currently connected GitHub App may still be unable to perform direct
branch-protection administration; that endpoint can return
`403 Resource not accessible by integration`. This connector limitation does
not mean the repository ruleset is absent and is never permission to bypass the
platform gate.

Repository governance protects scientific source integrity, not product publication.
It does not establish real-speaker FRR/FAR, final microphone/enclosure/AFE
performance, or physical SSC305 CPU/RSS/latency/soak evidence. Those research
measurements require independent recorded inputs and truthful NOT_RUN/FAIL
states when evidence is absent. Neither a GitHub Release nor a commercial
candidate is required to carry out reproducible technical experiments.
