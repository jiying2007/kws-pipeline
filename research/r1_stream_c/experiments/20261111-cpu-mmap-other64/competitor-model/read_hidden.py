#!/usr/bin/env python3
"""Portable NumPy reader for owner-only R1HD140 binary64 teacher records.

Header is 64 bytes; each subsequent record is exactly 1200 bytes. This module
only reads retained data and never executes a neural model.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import struct
import numpy as np

EXACT_IDS=(0,1027,1401,1462,2093)
HEADER=struct.pack('<8s6I5IIQ',b'R1HD140\0',1,64,1200,140,5,2601,*EXACT_IDS,1,0)
ROW_DTYPE=np.dtype([
    ('frame','<u8'),('source','<u8'),('delivery','<u8'),
    ('original_best_token','<u4'),('other_best_token','<u4'),
    ('hidden','<f8',(140,)),('exact_logits','<f8',(5,)),('max_other','<f8')])

def read_hidden(path,expected_sha256=None,pcm_samples=None):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>64+512*1200:
        raise ValueError('invalid hidden file')
    raw=path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('hidden SHA256 mismatch')
    if ROW_DTYPE.itemsize!=1200 or raw[:64]!=HEADER or len(raw)<=64 or (len(raw)-64)%1200:
        raise ValueError('invalid hidden header or record length')
    rows=np.frombuffer(raw,dtype=ROW_DTYPE,offset=64)
    n=len(rows)
    if not np.array_equal(rows['frame'],np.arange(n,dtype=np.uint64)):
        raise ValueError('frame sequence mismatch')
    if not np.array_equal(rows['source'],720+480*np.arange(n,dtype=np.uint64)):
        raise ValueError('source clock mismatch')
    if np.any(rows['source']>rows['delivery']):raise ValueError('invalid delivery clock')
    if pcm_samples is not None:
        want=np.minimum(((rows['source']+319)//320)*320,pcm_samples)
        if n!=max(0,(pcm_samples-720)//480+1) or not np.array_equal(rows['delivery'],want):
            raise ValueError('input exposure/frame/delivery mismatch')
    if any(not np.isfinite(rows[name]).all() for name in ('hidden','exact_logits','max_other')):
        raise ValueError('nonfinite teacher output')
    ids=np.asarray(EXACT_IDS,dtype=np.uint32)
    if (np.any(rows['original_best_token']>=2601) or np.any(rows['other_best_token']>=2601) or
            np.isin(rows['other_best_token'],ids).any()):
        raise ValueError('invalid original token identity')
    best_exact=np.argmax(rows['exact_logits'],axis=1)
    value=rows['exact_logits'][np.arange(n),best_exact]
    token=ids[best_exact]
    other=rows['max_other'];other_id=rows['other_best_token']
    reconstructed=np.where(value>other,token,np.where(value<other,other_id,np.minimum(token,other_id)))
    if not np.array_equal(reconstructed,rows['original_best_token']):
        raise ValueError('teacher max/tie reconstruction mismatch')
    return rows.copy()
