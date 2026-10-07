#!/usr/bin/env python3
"""Strict portable offline saved-evidence validator. Python standard library only.

This never opens model/audio payloads, imports a native library, starts a process,
or reads /proc. It checks the exact saved trace and its explicit metadata
projection, then applies the byte-identical frozen scorer to an in-memory view.
The archival trace is never rewritten. This is not an acoustic run.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zlib

RAW_SHA = '0ae1cf74a15ba2d14f8d8994ac4cf972f76153973cddb9666043757f18e11c86'
RAW_BYTES = 197933
ORIGINAL_MANIFEST = '75b322f33857152bc59d4700cba9b85dd818a36408d0e10a4fe837926b78ed58'
PROTOCOL_SHA = '0724e87b6e6d9128fa2bd1ef7d3af8e3425c6d29fa495adba59d9a65fc46e116'
FROZEN_SCORER = 'f77aac80b91cd5d4c8baf43cca2929e7fac2a66630ca28c365397021602055cb'

class Invalid(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise Invalid(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def unique(pairs):
    out={}
    for key,value in pairs:
        require(key not in out,'duplicate JSON key: '+key)
        out[key]=value
    return out

def parse(data):
    try:
        return json.loads(data,object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(Invalid('nonfinite JSON '+value)))
    except (UnicodeError,json.JSONDecodeError) as error:
        raise Invalid('invalid JSON') from error

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def safe_path(root,name):
    require(isinstance(name,str) and bool(name),'empty path')
    p=PurePosixPath(name)
    require(not p.is_absolute() and str(p)==name and not any(x in ('','.','..') for x in p.parts),'unsafe path')
    require('\\' not in name and '\0' not in name,'unsafe path characters')
    cur=root
    for part in p.parts:
        cur=cur/part
        require(not cur.is_symlink(),'symlink path')
    return cur

def verify_inventory(root, manifest_name='EVIDENCE-MANIFEST.json'):
    require(root.is_dir() and not root.is_symlink(),'invalid evidence root')
    manifest_path=safe_path(root,manifest_name)
    require(manifest_path.stat().st_size<=262144,'oversized inventory')
    manifest=parse(manifest_path.read_bytes())
    require(manifest.get('schema')=='qwen20-public-file-manifest-v1','wrong inventory schema')
    names=set()
    for row in manifest['files']:
        name=row['path'];require(name not in names and name!=manifest_name,'duplicate/self inventory')
        names.add(name);p=safe_path(root,name)
        require(p.is_file(),'missing file '+name)
        require(type(row['bytes']) is int and 0<=row['bytes']<=1048576,'invalid bounded file size')
        require(p.stat().st_size==row['bytes'],'size mismatch '+name)
        require(sha(p.read_bytes())==row['sha256'],'hash mismatch '+name)
    actual=set()
    for p in root.rglob('*'):
        require(not p.is_symlink(),'symlink in inventory')
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    require(actual==names|{manifest_name},'unexpected or absent file')
    require(sum(r['bytes'] for r in manifest['files'])==manifest['total_bytes'],'inventory byte total')
    return manifest

def raw_bytes(blob):
    # Refuse concatenated members, trailing bytes, bombs and incomplete streams.
    d=zlib.decompressobj(31)
    try:
        raw=d.decompress(blob,RAW_BYTES+1)
    except zlib.error as error:
        raise Invalid('invalid gzip') from error
    require(d.eof and not d.unused_data and not d.unconsumed_tail,'incomplete/extra gzip member')
    require(len(raw)==RAW_BYTES and sha(raw)==RAW_SHA,'raw original identity mismatch')
    require(raw.endswith(b'\n') and all(raw.splitlines()),'truncated/blank raw record')
    return raw

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def reconstruct_original_manifest(public_manifest, annotation_bytes):
    """Reconstruct exact old manifest from caller-supplied already-public annotations.

    No network access and no file write. Every technical field is retained, the
    omitted public receipt objects are restored, then the original byte hash is
    verified. This is separate from the default projection check.
    """
    require(len(annotation_bytes)<=65536,'oversized annotation file')
    require(sha(annotation_bytes)==public_manifest['metadata_sha256']['audio-review.jsonl'],'annotation identity mismatch')
    lines=annotation_bytes.splitlines(keepends=True)
    require(len(lines)==20,'annotation count')
    receipts={}
    for line in lines:
        receipt=parse(line);key=receipt['source_id']
        require(key not in receipts,'duplicate annotation source')
        receipts[key]=(receipt,sha(line))
    restored=copy.deepcopy(public_manifest)
    for row in restored['rows']:
        reference=row.pop('annotation_reference')
        receipt,line_hash=receipts[row['source_id']]
        require(line_hash==reference['line_sha256']==row['review_receipt_line_sha256'],'annotation line mismatch')
        require(receipt['file_sha256']==row['wav_sha256'],'annotation audio identity mismatch')
        row['review_receipt']=receipt
    data=(json.dumps(restored,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(sha(data)==ORIGINAL_MANIFEST,'original manifest reconstruction mismatch')
    return data

def validate(root, data_root, annotations=None):
    code=Path(root); root=Path(data_root)
    code_inventory=verify_inventory(code,'CODE-MANIFEST.json')
    inventory=verify_inventory(root)
    reference=parse((code/'DATA-REFERENCE.json').read_bytes())
    require(reference['repository']=='jiying2007/kws-data' and reference['path']=='research/2026-10-04-qwen20-descriptive','data repository scope')
    require(reference['manifest_sha256']==sha((root/'EVIDENCE-MANIFEST.json').read_bytes()),'pinned data manifest mismatch')
    if reference['publication_ready']:
        require(re.fullmatch(r'[0-9a-f]{40}',reference['commit']) is not None,'data commit must be immutable')
    else:
        require(reference['commit']=='PENDING_REVIEWED_DATA_PUBLICATION','invalid staging state')
    def read(name):return parse(safe_path(root,name).read_bytes())
    provenance=read('PROVENANCE.json')
    require(provenance['original_protocol_sha256']==PROTOCOL_SHA,'protocol reference mismatch')
    require(provenance['original_manifest_sha256']==ORIGINAL_MANIFEST,'manifest reference mismatch')
    mapping={r['public_path']:r for r in provenance['mapping']}
    require(len(mapping)==len(provenance['mapping']),'duplicate projection mapping')
    for name,row in mapping.items():
        location=code if row['repository']=='jiying2007/kws-pipeline' else root
        b=safe_path(location,name).read_bytes()
        require({'sha256':sha(b),'bytes':len(b)}==row['public'],'projection public identity mismatch')
        if row['transformation']=='byte-identical':require(row['original']==row['public'],'false byte-identical mapping')
    require(mapping['evidence/raw.jsonl.gz']['original']=={'bytes':RAW_BYTES,'sha256':RAW_SHA},'raw provenance mismatch')
    raw=raw_bytes((root/'evidence/raw.jsonl.gz').read_bytes())
    records=[parse(line) for line in raw.splitlines()]
    require(len(records)==317,'raw record count')
    original_header=copy.deepcopy(records[0])
    expected=read('evidence/descriptive-score.json')
    for key,value in expected['source_hashes'].items():
        if key!='raw_sha256':require(original_header.get(key)==value,'raw/scored identity mismatch '+key)
    require(expected['source_hashes']['raw_sha256']==RAW_SHA,'report/raw mismatch')
    require(original_header['protocol_sha256']==PROTOCOL_SHA and original_header['manifest_sha256']==ORIGINAL_MANIFEST,'original header identity')
    require(sha((code/'src/score_descriptive.py').read_bytes())==FROZEN_SCORER,'frozen scorer changed')
    scorer=load_module(code/'src/score_descriptive.py','qwen20_frozen_scorer')
    public_manifest=read('metadata/inputs.public.json')
    original_metadata_reconstructed=False
    if annotations is not None:
        reconstruct_original_manifest(public_manifest,annotations)
        original_metadata_reconstructed=True
    for row in public_manifest['rows']:
        require('review_receipt' not in row,'annotation identity unexpectedly duplicated')
        ref=row['annotation_reference']
        require(ref['recording']==row['recording'] and ref['line_sha256']==row['review_receipt_line_sha256'],'annotation link mismatch')
        require(ref['file_sha256']==public_manifest['metadata_sha256']['audio-review.jsonl'],'annotation file mismatch')
    require(sha((root/'metadata/geometry.json').read_bytes())==original_header['geometry_sha256'],'geometry identity')
    require(sha((root/'metadata/decoder-config.json').read_bytes())==original_header['decoder_config_sha256'],'decoder identity')
    # Explicit adapter, limited to one hash label in a separate in-memory view:
    # the original manifest's reviewer fields were replaced by public references.
    # Numerical records, input labels, shapes, order and gates remain unmodified.
    view=copy.deepcopy(records)
    public_canonical_sha=sha(canonical(public_manifest))
    view[0]['manifest_sha256']=public_canonical_sha
    bindings={k:original_header[k] for k in scorer.HASH_KEYS}
    bindings['manifest_sha256']=public_canonical_sha
    derived=scorer.score(view,public_manifest,root/'metadata/geometry.json',bindings)
    # Compare all substantive/scientific output under the original saved identity.
    # These restored identity labels are references, not recomputed old metadata.
    derived['source_hashes']['manifest_sha256']=ORIGINAL_MANIFEST
    derived['source_hashes']['raw_sha256']=RAW_SHA
    require(derived==expected,'saved-score derivation mismatch')
    require(records[0]==original_header,'raw header mutation')
    execution=read('evidence/execution-summary.json')
    resource=read('evidence/resource-record.json')
    require(execution['status']==resource['status']=='FAILED_NO_RETRY','failure classification changed')
    require(execution['collector_returncode']==resource['returncode']==0,'collector return code changed')
    require(execution['supervisor']=='FAIL' and execution['final_io']==execution['children_observation']=='UNKNOWN_NOT_CAPTURED','unknown/failure semantics changed')
    require(execution['attempts']==1 and execution['retries']==0 and resource['retries_allowed'] is False,'once-only changed')
    require(execution['guard_stop_reason']==resource['guard_stop_reason'] and "PermissionError" in resource['guard_stop_reason'] and '/proc/7/io' in resource['guard_stop_reason'],'failure evidence changed')
    require(len(resource['proc_samples'])==5 and all('children' not in r for r in resource['proc_samples']),'children availability changed')
    require(execution['probabilities']==execution['native_beam_paths']=='NOT_CAPTURED','unsaved observations invented')
    protocol=read('metadata/protocol.public.json')
    require(protocol['status']=='PUBLIC_TECHNICAL_PROJECTION_OF_FROZEN_PROTOCOL_NOT_EXECUTED','projection execution claim')
    require(protocol['numerical_qualification'] is False and protocol['independent_qualified_metrics_allowed'] is False,'qualification changed')
    audit=read('evidence/saved-audit.public.json')
    require(audit['aggregate']==expected['aggregate'] and audit['totals']['decoder_rows_decoded']==1111,'audit count mismatch')
    require(audit['run_stage_status']=='FAILED_NO_RETRY' and audit['supervisor_status']=='FAIL','audit failure changed')
    diagnostic=read('evidence/diagnostic.public.json')
    algorithm=load_module(code/'src/derive_saved.py','qwen20_saved_scalar_derivation')
    recomputed=algorithm.derive(records,read('metadata/geometry.json'),derived)
    for key in ('all20','totals','raw_tail_under800_examples','observed_event_score_ranking_ascending'):
        require(recomputed[key]==diagnostic[key],'diagnostic mismatch '+key)
    require(len(diagnostic['focus3'])==3,'focus count')
    for actual,saved in zip(recomputed['focus3_numeric'],diagnostic['focus3']):
        require(all(saved[k]==v for k,v in actual.items()),'focus scalar mismatch')
    future=parse((code/'future/REVIEW-SUMMARY.json').read_bytes())
    require(future['launcher_sha256']==sha((code/'future/src/launch_once.py').read_bytes()),'future launcher identity')
    require(future['execution_entry']=='UNCONDITIONALLY_DISABLED','future release claim')
    return {'status':'PASS_SAVED_EVIDENCE_ONLY','data_files':len(inventory['files'])+1,'code_files':len(code_inventory['files'])+1,'raw_records':len(records),'data_reference_publication_ready':reference['publication_ready'],'original_metadata_reconstructed':original_metadata_reconstructed,
            'callbacks':127,'model_rows':1144,'actual_decoder_search_rows':1111,'events':9,
            'original_run_status':'FAILED_NO_RETRY','supervisor':'FAIL','final_io':'UNKNOWN','children':'UNKNOWN',
            'acoustic_runs':0,'probability_or_beam_replays':0,
            'projection':'One in-memory manifest hash adaptation; archived raw unchanged; original omitted metadata identity referenced, not rederived.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--data-root',type=Path,required=True,help='Offline checkout path for the separately published numeric record')
    parser.add_argument('--annotations',type=Path,help='Optional existing immutable public audio-review.jsonl for exact original manifest reconstruction')
    args=parser.parse_args()
    try:
        annotations=None
        if args.annotations is not None:
            require(args.annotations.stat().st_size<=65536,'oversized annotations')
            annotations=args.annotations.read_bytes()
        print(json.dumps(validate(args.root,args.data_root,annotations),sort_keys=True))
    except (Invalid,ValueError,KeyError,OSError,AssertionError,TypeError) as error:
        print('INVALID: '+str(error),file=sys.stderr);return 2
    return 0

if __name__=='__main__':
    raise SystemExit(main())
