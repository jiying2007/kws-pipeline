"""Validate a trusted archive's Sense runtime-derived frontend dependency.

No trust discovery, model execution, source/asset relabeling, or admission.
The resolver must supply independently authenticated manifest and archive reads.
"""
import hashlib
import json
import math
import re
import struct

FIELDS = ('fs','window','n_mels','frame_length','frame_shift','filter_length_min',
          'filter_length_max','lfr_m','lfr_n','dither','snip_edges','upsacle_samples')
DEPS = {'frontend_source','fbank_source','backend_package_source','backend_binary',
        'cmvn_file','tokenizer_file','parser_source','backend_core_library'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def exact(value,fields,label):
    require(type(value) is dict and set(value)==set(fields),label+' exact fields')


def digest(value):
    require(type(value) is str and re.fullmatch('[a-f0-9]{64}',value) is not None,'SHA256 shape')


def parse(raw, maximum=1024**2):
    require(type(raw) is bytes and 0<len(raw)<=maximum,'bounded original JSON bytes')
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    def constant(value):raise ValueError('nonfinite JSON value')
    try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    except (UnicodeError,RecursionError) as exc:raise ValueError('invalid JSON encoding/depth') from exc


def reference(value):
    exact(value,{'archive','member'},'dependency reference')
    for key in ('archive','member'):
        v=value[key]
        require(type(v) is str and 0<len(v)<=1024 and '\\' not in v and '\x00' not in v and
                not v.startswith('/') and all(p not in ('','.','..') for p in v.split('/')),
                'safe exact dependency reference')


def packed(values):
    require(type(values) is list and 0<len(values)<=8192,'bounded CMVN row')
    require(all(type(v) is float and math.isfinite(v) for v in values),'actual finite float CMVN values')
    try:raw=struct.pack('<%df'%len(values),*values)
    except (OverflowError,struct.error) as exc:raise ValueError('float32 CMVN range') from exc
    observed=list(struct.unpack('<%df'%len(values),raw))
    require(observed==values,'CMVN values must be exact observed float32 values')
    return raw


def cmvn_from_file(raw):
    require(type(raw) is bytes and 0<len(raw)<=4*1024**2,'bounded original CMVN bytes')
    try:lines=raw.decode('utf-8').splitlines()
    except UnicodeError as exc:raise ValueError('CMVN UTF8') from exc
    rows={}
    for i,line in enumerate(lines):
        words=line.split()
        if words and words[0] in ('<AddShift>','<Rescale>'):
            key=words[0];require(key not in rows and i+1<len(lines),'single CMVN block')
            tokens=lines[i+1].split()
            require(len(tokens)>=5 and tokens[0]=='<LearnRateCoef>' and tokens[2]=='[' and tokens[-1]==']',
                    'exact CMVN vector syntax')
            try:values=[float(v) for v in tokens[3:-1]]
            except ValueError as exc:raise ValueError('CMVN numeric vector') from exc
            require(0<len(values)<=8192 and all(math.isfinite(v) for v in values),'bounded finite CMVN file vector')
            try:rows[key]=struct.pack('<%df'%len(values),*values)
            except (OverflowError,struct.error) as exc:raise ValueError('CMVN file float32 range') from exc
    require(set(rows)=={'<AddShift>','<Rescale>'} and len(rows['<AddShift>'])==len(rows['<Rescale>']),
            'both CMVN rows with identical shape')
    return rows['<AddShift>']+rows['<Rescale>']


def validate_snapshot(raw):
    value=parse(raw)
    exact(value,{'schema','frontend_class','attributes','attribute_types','attribute_source','dependencies',
        'cmvn','native_execution_mappings','observation_stage','dither_rng_reproducibility','training_admitted'},'frontend snapshot')
    expected={'schema':'observed-sense-frontend-v1','frontend_class':'funasr.frontends.wav_frontend.WavFrontend',
        'attribute_source':'actual_loaded_WavFrontend_attributes_no_defaults',
        'observation_stage':'loaded_object_observation','dither_rng_reproducibility':'not established'}
    require(all(type(value[k]) is str and value[k]==v for k,v in expected.items()),'frontend snapshot fixed semantics')
    require(value['training_admitted'] is False,'snapshot cannot admit training')
    attrs=value['attributes'];types=value['attribute_types']
    exact(attrs,FIELDS,'frontend attributes');exact(types,FIELDS,'attribute types')
    for name,v in attrs.items():
        allowed=(str,) if name=='window' else (bool,) if name in ('snip_edges','upsacle_samples') else (int,float) if name in ('frame_length','frame_shift','dither') else (int,)
        require(type(v) in allowed and types[name]==type(v).__name__,'exact observed attribute type: '+name)
        if type(v) is float:require(math.isfinite(v),'finite attribute')
    require(attrs['fs']==16000 and 1<=attrs['n_mels']<=8192 and 1<=attrs['lfr_m']<=64 and 1<=attrs['lfr_n']<=64,
            'frontend dimensions')
    require(0<attrs['frame_length']<=1000 and 0<attrs['frame_shift']<=1000 and 0<=attrs['dither']<=100,
            'frontend frame/dither bounds')
    require(0<len(attrs['window'])<=64 and all(-1<=attrs[k]<=1000000 for k in ('filter_length_min','filter_length_max')),
            'window/filter bounds')
    exact(value['native_execution_mappings'],{'backend_binary','backend_core_library'},'actual native mapping evidence')
    require(all(value['native_execution_mappings'][name] is True for name in ('backend_binary','backend_core_library')),
            'both native binaries must be observed executable mappings')
    exact(value['dependencies'],DEPS,'snapshot dependencies')
    for name,identity in value['dependencies'].items():
        exact(identity,{'sha256','bytes'},'snapshot dependency identity');digest(identity['sha256'])
        require(type(identity['bytes']) is int and 0<identity['bytes']<=(32*1024**2 if name in ('backend_binary','backend_core_library') else 4*1024**2),
                'dependency byte bounds')
    cmvn=value['cmvn'];exact(cmvn,{'device','dtype','shape','encoding','values','sha256'},'CMVN')
    require(cmvn['device']=='cpu' and cmvn['dtype']=='float32' and cmvn['encoding']=='ieee754-binary32-le','CMVN encoding')
    shape=cmvn['shape'];rows=cmvn['values']
    require(type(shape) is list and len(shape)==2 and all(type(v) is int for v in shape) and
            shape[0]==2 and 1<=shape[1]<=8192,'CMVN shape')
    require(type(rows) is list and len(rows)==2 and all(type(r) is list and len(r)==shape[1] for r in rows),'CMVN rows')
    tensor=packed(rows[0])+packed(rows[1]);digest(cmvn['sha256'])
    require(sha(tensor)==cmvn['sha256'],'CMVN tensor hash mismatch')
    return value,tensor


def verify_frontend_dependency(snapshot_raw, binding_raw, read, inventory):
    """read is the existing trusted-provenance resolver; no filesystem/network I/O."""
    snapshot,tensor=validate_snapshot(snapshot_raw)
    binding=parse(binding_raw)
    exact(binding,{'schema','snapshot_sha256','dependencies'},'reviewed frontend binding')
    require(binding['schema']=='reviewed-sense-frontend-dependencies-v1','reviewed frontend binding schema')
    digest(binding['snapshot_sha256']);require(binding['snapshot_sha256']==sha(snapshot_raw),'reviewed snapshot hash')
    exact(binding['dependencies'],DEPS,'reviewed frontend dependencies')
    require(type(inventory) is list and inventory,'actual loaded/executed inventory required')
    bodies={};seen=set()
    for name,item in binding['dependencies'].items():
        exact(item,{'reference','sha256','bytes'},'reviewed dependency entry');reference(item['reference'])
        identity={k:item[k] for k in ('sha256','bytes')}
        require(identity==snapshot['dependencies'][name] and type(item['bytes']) is int,'reviewed dependency identity mismatch')
        key=(item['reference']['archive'],item['reference']['member'])
        require(key not in seen,'dependency aliases');seen.add(key)
        matches=[r for r in inventory if type(r) is dict and r.get('path')==item['reference']['member'] and
                 type(r.get('bytes')) is int and r['bytes']==item['bytes'] and r.get('sha256')==item['sha256']]
        require(len(matches)==1,'original dependency must uniquely match actual loaded/executed inventory')
        raw=read(item['reference'])
        require(type(raw) is bytes and len(raw)==item['bytes'] and sha(raw)==item['sha256'],'original dependency raw bytes changed')
        bodies[name]=raw
    require(cmvn_from_file(bodies['cmvn_file'])==tensor,'loaded CMVN tensor differs from original CMVN bytes')
    return {'schema':'verified-sense-frontend-runtime-dependency-v1','snapshot_sha256':sha(snapshot_raw),
            'dependency_count':len(bodies),'training_admitted':False}
