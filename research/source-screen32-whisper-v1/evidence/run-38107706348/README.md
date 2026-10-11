# Second Whisper campaign attempt: retained setup failure

[Run 38107706348](https://github.com/jiying2007/kws-pipeline/actions/runs/38107706348)
was campaign attempt **2**, following run **38105471896**. GitHub run number and
run attempt were both **1**. The executed head was
`733fbb90a98b3d39242594cebea378f790bc5775`, tree
`44ca7a527542982823294eda6f8d189d98122873`.

Setup stopped without retry while downloading wheel **19**, **torch 2.12.1+cpu**,
with observed **HTTP 403**. The underlying reason for that response is unknown.
This observation does not establish the first run's inner failure cause; that
cause remains unknown because the first run did not retain it.

Before this failure, setup verified **41 assets**: 22 source files, the one native
Whisper small model and 18 wheels. The model response body was **483,617,219 bytes**
and matched SHA-256
`9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794`.
Selected setup response bodies totalled **580,174,803 bytes**. This is not total
network/wire traffic; image transfer is separately recorded and total wire bytes
are unknown. The failed wheel recorded zero response-body bytes.

All **16** audio rows remain **not_run**, with **0** audio/model attempts. There
is no native inference configuration, start, decoder trace, EOS or raw inference
output, and no transcription-quality conclusion follows from this setup failure.

## Durable public evidence

The nine JSON artifact members here are byte-identical copies of artifact
**11690174265**, `qwen16-whisper-small-third-attempt-2-terminal-v1`. Its ZIP was
**18,353 bytes**, SHA-256
`ffa2a7201133f6cb3f916538ddc96291806c43ad8b0f1a19090b6405999ee491`.
`artifact-freeze.json` verifies the other eight members; `provenance.json` records
all nine member hashes and sizes plus verified execution identifiers.

This archive adds source-only public evidence and this result note. The original
source files, locks, URLs, host allowlist, first evidence archive, execution
freezes/releases, workflow gates and branches are unchanged. The preparation
READMEs remain historical source-frozen instructions, not a claim that this
attempt is still pending. No further attempt is authorized or activated here.
All evidence remains `source_screen_only`; training admission, shipping approval
and human-gold flags remain false. No model bytes or arbitrary job logs are copied.
