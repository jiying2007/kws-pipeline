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

An administrator with repository-rules administration permission may apply the reviewed target with an equivalent command such as:

```bash
gh api --method POST \
  "repos/OWNER/REPO/rulesets" \
  --input governance/main-ruleset-target.json
```

Do not run that command with a token broader than necessary, do not commit an administration token, and do not weaken the target merely to make deployment publication succeed. If a ruleset with the same purpose already exists, update/audit the live rule instead of creating overlapping policy blindly.

Before creating `deployment/commercial-candidate`, independently re-read live GitHub state and verify the enforcing ruleset targets `main` and contains at least the rules above. The deployment workflow separately requires the source SHA to equal current `main` and `main` to report `protected=true`; publication remains fail-closed otherwise.

The currently connected GitHub App cannot perform this administration operation: its direct branch-protection endpoint access returns `403 Resource not accessible by integration`. That connector limitation is not permission to bypass the platform gate.

Until the platform controls are enabled, the repository remains a software/commercial candidate with repository-side policy prepared but platform enforcement incomplete. Once enabled and the immutable deployment candidate is published, the remaining product evidence boundary is only real Mandarin through the final microphone/enclosure/AFE chain and physical target-board qualification.
