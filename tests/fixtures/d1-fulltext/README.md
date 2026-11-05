# Original FSMN vocabulary fixture

`fsmn-official-tokens.txt` is the exact 24,528-byte token file identified by
`configs/runtime/fsmn-fulltext-sources.json`. SHA-256:
`5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa`.

Source: the `tokens.txt` file of `thuduj12/kws_wenwen_fsmn_ctc`, revision
`cd0c18d0dda6742c371fbb72ec66185906a9885d`, at the ModelScope URL retained in
that source lock. This is a vocabulary regression fixture, with no model
weights, private recordings, generated audio, or claims about recognition.

The original file includes blank aliases and decoder disambiguation symbols.
It deliberately does not include 灯 or 沃 among its 2,599 model outputs. D1's
separate derived mapping appends these characters; this file is never edited.

`research_weak_sampling_v1.py` is an unchanged test copy of kws-data's maintained
`automation/research_weak_sampling.py`, Git blob
`3eab60624017b71b062f151bb343aec8352deadc`, SHA-256
`5a6f5447b1f4e618f38e9f1bde0b79b1ba3b21db3fbab3fd0e237c622f39e288`.
Runtime preparation imports and verifies that same file from the declared data
source checkout. The fixture makes its real fixed-exposure behavior testable
without a second repository checkout or network access in offline CI.
