"""Bounded public setup diagnostics. No raw argv, paths, stderr or tracebacks."""
import hashlib
import json
import math
from pathlib import Path
import re

MAX_TAIL_BYTES=16384
STAGES={
    'bootstrap_pip':'bootstrap', 'install_build_tools':'install', 'install_runtime_wheels':'install',
    'check_build_requirements':'preflight', 'build_wheel':'build', 'verify_built_metadata':'metadata',
    'install_built_wheels':'install', 'verify_setup':'verification'}
EXCEPTIONS={'ValueError','RuntimeError','TimeoutError','TimeoutExpired','OSError','FileNotFoundError',
            'PermissionError','MetadataMismatch','ModuleNotFoundError','ImportError','MemoryError','CalledProcessError'}
MESSAGES={
    'MetadataMismatch: Built/installed normalized Requires-Dist differs from the saved lock':'requires_dist_mismatch',
    'ValueError: Built/installed normalized Requires-Dist differs from the saved lock':'requires_dist_mismatch',
    'ValueError: Exact package metadata name/version mismatch':'metadata_identity_mismatch',
    'ValueError: Wheel METADATA hash differs from the saved lock':'metadata_hash_mismatch',
    'ValueError: Build requirement is not satisfied by locked build tools':'build_requirement_not_locked',
    'ValueError: Installed build tool version drift':'build_tool_version_drift',
    'ERROR Missing dependencies:':'build_frontend_missing_dependencies'}

def safe_exception(error):
    name=type(error).__name__
    return name if name in EXCEPTIONS else 'OtherException'

def stderr_diagnostic(path):
    path=Path(path)
    if not path.exists(): return {'bytes':0,'tail_bytes':0,'codes':[],'exception_types':[],'stream_scope':'stderr_only','raw_text_disclosed':False}
    size=path.stat().st_size
    with path.open('rb') as f:
        f.seek(max(0,size-MAX_TAIL_BYTES));tail=f.read(MAX_TAIL_BYTES)
    lines=tail.decode('utf-8',errors='replace').splitlines();codes=set();types=set()
    for line in lines:
        value=line.strip()
        if value.startswith('<run_path>.'):value=value[len('<run_path>.'):]
        if value in MESSAGES:codes.add(MESSAGES[value])
        match=re.match(r'^(MetadataMismatch|[A-Za-z]+Error|TimeoutExpired|CalledProcessError):',value)
        if match and match[1] in EXCEPTIONS:types.add(match[1])
    return {'bytes':size,'tail_bytes':len(tail),'tail_sha256':hashlib.sha256(tail).hexdigest(),
            'tail_truncated':size>len(tail),'codes':sorted(codes),'exception_types':sorted(types),
            'raw_text_disclosed':False,'stream_scope':'stderr_only'}

def metadata_difference(expected,observed):
    """Only fixed-lock declarations appear as text; unexpected values are hashes."""
    from packaging.requirements import Requirement
    def canonical(text):
        r=Requirement(text)
        return (re.sub(r'[-_.]+','-',r.name).lower(),tuple(sorted(r.extras)),str(r.specifier),r.url or '',str(r.marker) if r.marker else '')
    known={canonical(text):text for text in expected};actual=[];unknown=[]
    for text in observed:
        try:key=canonical(text)
        except Exception:key=None
        if key in known:actual.append(key)
        else:unknown.append({'sha256':hashlib.sha256(text.encode()).hexdigest(),'bytes':len(text.encode())})
    return {'code':'requires_dist_mismatch','expected_requires_dist':list(expected),
        'observed_known_requires_dist':[known[k] for k in actual],
        'missing_expected_requires_dist':[text for key,text in known.items() if key not in actual],
        'unexpected_requires_dist':unknown,'observed_count':len(observed),
        'unexpected_text_disclosed':False}

def public_metadata(diagnostic,expected):
    allowed={'schema','metadata_sha256','metadata_bytes','status','code','exception_type','difference'}
    if set(diagnostic)-allowed or diagnostic.get('schema')!='fixed30-built-metadata-diagnostic-v1': raise ValueError('metadata diagnostic shape')
    if not re.fullmatch(r'[0-9a-f]{64}',diagnostic.get('metadata_sha256','')) or type(diagnostic.get('metadata_bytes')) is not int or not 0<=diagnostic['metadata_bytes']<=2*1024**2: raise ValueError('metadata diagnostic identity')
    if diagnostic.get('status') not in ('started','failed','success') or diagnostic.get('code') not in ('full_requires_dist_match','requires_dist_mismatch','metadata_validation_failed'): raise ValueError('metadata diagnostic status')
    if 'exception_type' in diagnostic and diagnostic['exception_type'] not in EXCEPTIONS|{'OtherException'}: raise ValueError('metadata exception class')
    result={k:diagnostic[k] for k in allowed-{'difference'} if k in diagnostic}
    if 'difference' in diagnostic:
        d=diagnostic['difference']
        if set(d)!={'code','expected_requires_dist','observed_known_requires_dist','missing_expected_requires_dist','unexpected_requires_dist','observed_count','unexpected_text_disclosed'}: raise ValueError('metadata difference shape')
        if d['expected_requires_dist']!=expected or d['code']!='requires_dist_mismatch' or d['unexpected_text_disclosed'] is not False: raise ValueError('metadata difference authority')
        for key in ('observed_known_requires_dist','missing_expected_requires_dist'):
            if type(d[key]) is not list or len(d[key])>512 or any(v not in expected for v in d[key]): raise ValueError('undeclared metadata text')
        if type(d['observed_count']) is not int or not 0<=d['observed_count']<=512 or type(d['unexpected_requires_dist']) is not list or len(d['unexpected_requires_dist'])>512: raise ValueError('metadata difference bounds')
        for row in d['unexpected_requires_dist']:
            if set(row)!={'sha256','bytes'} or not re.fullmatch(r'[0-9a-f]{64}',row['sha256']) or type(row['bytes']) is not int or not 0<=row['bytes']<=2*1024**2: raise ValueError('metadata unknown declaration')
        result['difference']=d
    return result

def public_commands(records,package_rows):
    out=[]
    for record in records:
        stage=record.get('stage');package=record.get('package')
        if stage not in STAGES or (package is not None and package not in set(package_rows)|{'pip'}):
            out.append({'status':'diagnostic_rejected','code':'unsafe_diagnostic_payload'});continue
        if record.get('command_role')!=STAGES[stage] or record.get('status') not in ('started','failed','success'):
            out.append({'status':'diagnostic_rejected','code':'unsafe_diagnostic_payload'});continue
        row={k:record[k] for k in ('stage','package','command_role','status','returncode','wall_seconds') if k in record}
        if ((row.get('returncode') is not None and (type(row['returncode']) is not int or not -128<=row['returncode']<=255))
            or ('wall_seconds' in row and (type(row['wall_seconds']) not in (int,float) or not math.isfinite(row['wall_seconds']) or row['wall_seconds']<0))):
            out.append({'status':'diagnostic_rejected','code':'unsafe_diagnostic_payload'});continue
        if 'exception_type' in record: row['exception_type']=record['exception_type'] if record['exception_type'] in EXCEPTIONS else 'OtherException'
        if 'stderr_diagnostic' in record:
            d=record['stderr_diagnostic'];allowed={'bytes','tail_bytes','tail_sha256','tail_truncated','codes','exception_types','raw_text_disclosed','stream_scope'}
            valid=(type(d) is dict and not (set(d)-allowed) and type(d.get('bytes')) is int and 0<=d['bytes']<2**63
                and type(d.get('tail_bytes')) is int and 0<=d['tail_bytes']<=min(MAX_TAIL_BYTES,d['bytes'])
                and type(d.get('codes')) is list and len(d['codes'])<=len(MESSAGES) and all(v in set(MESSAGES.values()) for v in d['codes'])
                and type(d.get('exception_types')) is list and len(d['exception_types'])<=len(EXCEPTIONS) and all(v in EXCEPTIONS for v in d['exception_types'])
                and ('tail_sha256' not in d or type(d['tail_sha256']) is str and re.fullmatch(r'[0-9a-f]{64}',d['tail_sha256']))
                and ('tail_truncated' not in d or type(d['tail_truncated']) is bool)
                and d.get('raw_text_disclosed') is False and d.get('stream_scope')=='stderr_only')
            row['stderr_diagnostic']=dict(d) if valid else {'codes':['unsafe_diagnostic_payload'],'raw_text_disclosed':False}
        if 'metadata_diagnostic' in record:
            try:row['metadata_diagnostic']=public_metadata(record['metadata_diagnostic'],package_rows[package]['requires_dist'])
            except (ValueError,KeyError,TypeError):row['metadata_diagnostic']={'code':'unsafe_diagnostic_payload','raw_text_disclosed':False}
        out.append(row)
    return out
