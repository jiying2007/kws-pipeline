"""Bind corrected semantics to captured raw IDs and the pinned SentencePiece.

No ASR/model import. The verifier's optional tokenizer load only detokenizes
retained IDs; it never invokes acoustic inference or changes token sequences.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
CONTRACT_SHA256 = '878278e46ae9af349a93e37dddffa83e916f447ef82fb6dd61fa8948bb8261d7'
TOKENIZER_SHA256 = 'aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8'
TOKENIZER_NAME = 'chn_jpn_yue_eng_ko_spectok.bpe.model'
TOKEN_SCOPE = 'Greedy CTC collapsed nonblank IDs passed to SentencePiece; not framewise logits'
VOCABULARY_SIZE = 25055
LEXICAL_UNKNOWN_ID = 0


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _contract():
    path = STAGE / 'sense_parser.py'
    raw = path.read_bytes()
    if _hash(raw) != CONTRACT_SHA256:
        raise ValueError('Frozen corrected SenseVoice contract changed')
    # Execute the exact bytes just hashed, not an unbound second file read.
    module = {'__name__': 'cosyvoice12_frozen_plan_contract'}
    exec(compile(raw, str(path), 'exec'), module)
    return module


def tokenizer_path():
    plan = json.loads((STAGE / 'model-locks.json').read_bytes())['sensevoice']
    row = next(r for r in plan['asset_lock']['files'] if r['filename'] == TOKENIZER_NAME)
    if row['sha256'] != TOKENIZER_SHA256:
        raise ValueError('Pinned SenseVoice tokenizer lock changed')
    return STAGE / 'runtime/models/sensevoice' / TOKENIZER_NAME


def verify_tokenizer(sp, path=None):
    path = tokenizer_path() if path is None else Path(path)
    if path.is_symlink() or not path.is_file() or _hash(path.read_bytes()) != TOKENIZER_SHA256:
        raise ValueError('Pinned SenseVoice tokenizer bytes changed')
    if (type(sp.vocab_size()) is not int or sp.vocab_size() != VOCABULARY_SIZE or
            type(sp.unk_id()) is not int or sp.unk_id() != LEXICAL_UNKNOWN_ID or
            sp.id_to_piece(LEXICAL_UNKNOWN_ID) != '<unk>' or
            not sp.is_unknown(LEXICAL_UNKNOWN_ID) or
            sp.id_to_piece(25009) != '<|EMO_UNKNOWN|>' or sp.is_unknown(25009)):
        raise ValueError('Pinned tokenizer vocabulary/unknown semantics mismatch')
    return {'sha256': TOKENIZER_SHA256, 'vocabulary_size': VOCABULARY_SIZE,
            'lexical_unknown_id': LEXICAL_UNKNOWN_ID, 'lexical_unknown_piece': '<unk>',
            'emo_unknown_id': 25009, 'emo_unknown_is_lexical_unknown': False}


def load_pinned_tokenizer():
    """Existing runtime only. No model package imports, download, or inference."""
    if importlib.metadata.version('sentencepiece') != '0.2.2':
        raise RuntimeError('Qualified SentencePiece version changed')
    path = tokenizer_path()
    if _hash(path.read_bytes()) != TOKENIZER_SHA256:
        raise ValueError('Pinned SenseVoice tokenizer bytes changed before load')
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(model_file=str(path))
    verify_tokenizer(sp, path)
    return sp


def bound_sense_transcript(decoded, capture, sp):
    """Re-decode exact raw IDs and verify metadata token boundary before slicing.

    Malformed rich metadata stays an unknown warning, never silently repaired.
    Metadata emotion EMO_UNKNOWN is not the lexical unk token (ID0).
    """
    from asr_stage.decoding import bounded_text, token_ids
    contract = _contract()
    identity = verify_tokenizer(sp)
    if (type(capture) is not dict or set(capture) != {'token_ids', 'decoded', 'token_scope'} or
            capture['decoded'] != decoded or capture['token_scope'] != TOKEN_SCOPE):
        raise ValueError('SenseVoice captured output association mismatch')
    decoded = bounded_text(decoded)
    ids = token_ids(capture['token_ids'], allow_empty=True)
    if any(i >= VOCABULARY_SIZE for i in ids):
        raise ValueError('SenseVoice raw token outside pinned vocabulary')
    passed_ids = list(ids)
    if bounded_text(sp.decode(passed_ids)) != decoded or passed_ids != ids:
        raise ValueError('Exact captured raw token re-decode mismatch')
    pieces = [sp.id_to_piece(i) for i in ids[:4]]
    groups = contract['SENSE_GROUPS']
    boundary = (len(pieces) == 4 and
        all(type(piece) is str and piece.startswith('<|') and piece.endswith('|>') and
            piece[2:-2] in group for piece, group in zip(pieces, groups)) and
        decoded.startswith(''.join(pieces)))
    lexical = ids[4:] if boundary else None
    if boundary:
        extracted = contract['corrected_sense_extract'](decoded, lexical)
        metadata = extracted.pop('metadata')
        row = extracted
    else:
        metadata = []
        row = {'raw_text': decoded, 'completeness': 'unknown', 'quality_flags': ['decoding_warning']}
    binding = {'schema': 'cosyvoice12-sense-token-binding-v1', 'tokenizer': identity,
        'parser_contract_sha256': CONTRACT_SHA256, 'raw_token_ids': ids,
        'raw_decoded_sha256': _hash(decoded.encode('utf-8')),
        'metadata_token_ids': ids[:4], 'metadata_pieces': pieces,
        'metadata_boundary_verified': boundary, 'lexical_token_ids': lexical,
        'lexical_unknown_present': (LEXICAL_UNKNOWN_ID in lexical) if boundary else None,
        'raw_redecode_matches_capture': True, 'token_scope': TOKEN_SCOPE}
    return row, metadata, binding
