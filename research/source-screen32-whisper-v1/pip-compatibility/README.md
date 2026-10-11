# Source-only pip 25.0.1 compatibility policy

The compatibility module is integrated into the updated source worker on the new source branch. This is source-only integration: no activation is wired and no runtime, acquisition, inference, network, model, image, or dependency-upgrade authorization is granted. The historical executed commit and third freeze remain unchanged.

## Verified upstream identity

Official pypa/pip tag `25.0.1` resolves to commit [`bc7c88cb3de9c9af769c51517833ea48bbe70d9a`](https://github.com/pypa/pip/commit/bc7c88cb3de9c9af769c51517833ea48bbe70d9a). `upstream-source-manifest.json` records immutable URLs, Git blob SHA-1, and raw UTF-8 content SHA-256 for eleven reviewed source files. Each content blob was independently checked against its Git blob SHA-1 before generating the manifest.

The historical worker at KWS source commit `d54c72cce1623f2c60d236b36183cd75943ad605` requires seven flags. Official 25.0.1 supports `--retries`, `--require-hashes`, `--no-deps`, `--no-index`, `--only-binary`, and `--keyring-provider`, but has no `--resume-retries` option. See [option definitions](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_internal/cli/cmdoptions.py) and [download command setup](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_internal/commands/download.py).

The retained runtime evidence proves `pip_help_probe` / `pip_capability_probe` and pip version 25.0.1. It did not retain help exit status or the exact missing flags. The source incompatibility is verified; do not relabel it as a directly observed exact missing-option runtime result.

## Zero retry/resume semantics

The [25.0.1 downloader](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_internal/network/download.py) performs a streaming GET and writes a new file in `wb` mode. Neither Downloader nor BatchDownloader has a recovery/restart/resume loop. [Response chunk iteration](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_internal/network/utils.py) does not reconnect after interruption.

[Session construction](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_internal/cli/index_command.py) passes CLI retries into [PipSession](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_internal/network/session.py), which constructs Retry(total=retries). With zero, [vendored urllib3 retry increment](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_vendor/urllib3/util/retry.py) decrements the total to -1 and raises before another attempt. [The connection pool](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_vendor/urllib3/connectionpool.py) increments before reissuing connection/status failures. A `MaxRetryError` at zero does not mean a retry occurred.

[Requests' adapter](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_vendor/requests/adapters.py) passes the Retry object through and disables response preloading, so later streamed body errors are outside that retry loop. [Requests handles redirects separately](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_vendor/requests/sessions.py): zero failure retries is not a one-HTTP-request guarantee. Existing broker limits remain necessary. [Pip authentication](https://github.com/pypa/pip/blob/bc7c88cb3de9c9af769c51517833ea48bbe70d9a/src/pip/_internal/network/auth.py) returns 401 unchanged when prompting is off and no keyring credentials are obtained; retain `--no-input` and `--keyring-provider disabled`.

## Integration contract

- Obtain version and bounded download-help bytes with the same pinned worker interpreter used for acquisition.
- Resolve that interpreter's pip package root inside its validated immutable image. Do not substitute a host pip path or a caller-supplied version alone.
- Call `select_contract(version, help_exit_code, help_text, pip_root)` before acquisition. Exact 25.0.1 gets the no-resume exemption only when all eleven installed-source hashes match. Missing/changed files fail closed.
- Other versions retain all seven required options and explicit `--resume-retries 0`. This preserves the existing capability contract, without asserting any future version is otherwise compatible.
- Use returned `retry_argv` once, retaining hashes, direct locked URLs, no-index/no-deps/binary-only, isolated configuration, disabled keyring/cache/prompts/version check, and all original scope/deadline/byte limits.
- Retain only fixed sanitized capability evidence. Both accepted contracts and `CompatibilityFailure.evidence` include bounded help exit status and all missing names drawn only from the seven fixed options. In particular, absent `--resume-retries` remains recorded for accepted audited 25.0.1; only the gate requirement is exempted after source-hash verification. Never retain raw help, logs, paths, exceptions, environment variables, or arbitrary option text.
- The updated source worker binds the tested contract to its acquisition argv. Importing the module performs no probes or acquisition; source integration does not activate a runtime.

## Verification scope

Nine offline tests cover source-manifest/hash consistency, the audited exemption, source mismatch, unchanged non-audited policy, each missing option, nonzero help status, invalid/bounded input, exact option-token matching, and changed/missing/symlink source files. Tests use synthetic help and local text only. No pip invocation, installed pip import, wheel/model fetch, runtime execution, or inference was performed. Installed runtime hashes remain unverified until a separately authorized, bounded gate performs its preflight.
