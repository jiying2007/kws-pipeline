"""Exact source-extracted corrected SenseVoice semantics; no target labels."""
import re

SENSE_GROUPS = (
 frozenset(('zh','en','yue','ja','ko','nospeech')),
 frozenset(('HAPPY','SAD','ANGRY','NEUTRAL','FEARFUL','DISGUSTED','SURPRISED','EMO_UNKNOWN')),
 frozenset(('BGM','Speech','Applause','Laughter','Cry','Sneeze','Breath','Cough','Sing','Speech_Noise','GBG','Event_UNK')),
 frozenset(('withitn','woitn')),
)

def require(ok, message):
    if not ok: raise ValueError(message)

def corrected_sense_extract(decoded, lexical_token_ids):
    """Only token IDs after the four validated rich metadata positions are lexical.

    No posthoc blanket removal of an ambiguous flag. A future ASR adapter must bind
    the captured raw decode/token evidence to this parser and the reviewed tokenizer.
    """
    require(type(decoded) is str and len(decoded)<=32768,'bounded SenseVoice decode')
    require(type(lexical_token_ids) is list and len(lexical_token_ids)<=4096 and
            all(type(i) is int and i>=0 for i in lexical_token_ids),'lexical token IDs')
    pos=0;meta=[]
    for group in SENSE_GROUPS:
        match=re.match(r'<\|([^<>|]+)\|>',decoded[pos:])
        if not match or match[1] not in group:
            return {'raw_text':decoded,'metadata':meta,'quality_flags':['decoding_warning'],'completeness':'unknown'}
        meta.append(match[1]);pos+=len(match[0])
    text=decoded[pos:];flags=[]
    if meta[0]=='nospeech':flags.append('non_speech')
    # EMO_UNKNOWN is an emotion category, not lexical uncertainty.
    if meta[2]=='Event_UNK':flags.append('ambiguous')
    if meta[3]!='woitn' or '<|' in text:flags.append('decoding_warning')
    if 0 in lexical_token_ids or '<unk>' in text:flags.append('lexical_unknown')
    if not text.strip() or not lexical_token_ids:flags.append('empty_transcript')
    return {'raw_text':text,'metadata':meta,'quality_flags':sorted(set(flags)),'completeness':'complete'}
