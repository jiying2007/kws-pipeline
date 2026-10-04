"""Pure train49 feature identity and closed public-transport admission checks."""
from common import read_json,require,digest,safe
from candidate_control import TOKENS,validate_rows
OLD_COMMIT='d0d54cf635189bdc522c9d0f6135082e81250fd8'
OLD_INPUTS_SHA='b9fb5acc917c248041dff328d9c9ea3de0c640230d41de85f01326500e4d23ab'

def hex_digest(value,length=64):
    return type(value)is str and len(value)==length and all(c in '0123456789abcdef'for c in value)

def validate_new_pin(pin,rows):
    validate_rows(rows)
    require(pin.get('schema')=='a20-cosy17-prepared-inputs-v1'and pin.get('status')=='VERIFIED_PUBLIC_PIN','Cosy17 audited public pin required')
    require(pin.get('repository')=='jiying2007/kws-data'and hex_digest(pin.get('commit'),40),'Cosy17 immutable public commit')
    require(hex_digest(pin.get('audit_receipt_sha256')),'Cosy17 independent feature audit')
    require(hex_digest(pin.get('public_projection_review_sha256')),'Cosy17 clean public projection review')
    require(pin.get('source_count')==17 and pin.get('valid_rows')==693 and pin.get('raw_feature_bytes')==1108800,'Cosy17 exact geometry')
    require(pin.get('normalization')=='PRE_CMVN'and pin.get('dtype')=='<f4','Cosy17 pre-CMVN format')
    manifest=pin.get('feature_manifest');require(type(manifest)is dict and manifest.get('file')=='MANIFEST.json'and manifest.get('schema')=='cosy17-public-precmvn-features-v1'and hex_digest(manifest.get('sha256')),'Cosy17 frozen manifest')
    archive=pin.get('archive');require(type(archive)is dict and type(archive.get('bytes'))is int and 0<archive['bytes']<=1024**2 and hex_digest(archive.get('sha256')),'Cosy17 exact bounded ZIP identity')
    require(pin.get('transport')=='single-zip-closed-members'and archive.get('unpacked_bytes')==1131737,'Cosy17 exact ZIP transport')
    members=archive.get('members');require(type(members)is list and len(members)==19,'Cosy17 exact19 ZIP members')
    from pathlib import PurePosixPath
    prefix='research/2026-10-04-cosy17-precmvn'
    expected={'MANIFEST.json':(manifest['bytes'],manifest['sha256']),'notices/COSY17-README.md':(1799,'7a456203496e267383a59b69ae43b6c4b2ac712973ce528543c0966daf783120')}
    expected.update({r['feature']['file']:(r['feature']['bytes'],r['feature']['sha256'])for r in rows[32:]})
    require(len({e.get('destination')for e in members})==19 and {e.get('destination')for e in members}==set(expected),'Cosy17 ZIP exact destination allowlist')
    for e in members:
        require((e['bytes'],e['sha256'])==expected[e['destination']],'Cosy17 ZIP member frozen identity')
        name='MANIFEST.json'if e['destination']=='MANIFEST.json'else('README.md'if e['destination']=='notices/COSY17-README.md'else 'features/'+PurePosixPath(e['destination']).name)
        require(e['path']==prefix+'/'+name,'Cosy17 ZIP member path mapping')
    require([e['path']for e in members]==sorted(e['path']for e in members)and sum(e['bytes']for e in members)==archive['unpacked_bytes'],'Cosy17 ordered ZIP member byte bound')
    requests=pin.get('requests');require(type(requests)is list and len(requests)==1,'Cosy17 exactly one public ZIP request')
    e=requests[0];require(e.get('repository')==pin['repository']and e.get('commit')==pin['commit'],'Cosy17 request commit binding')
    path=e.get('path');require(type(path)is str and bool(path),'Cosy17 public asset path')
    parts=PurePosixPath(path);require(not parts.is_absolute()and '..'not in parts.parts and str(parts)==path and '?'not in path and '#'not in path and path.endswith('.zip'),'Cosy17 closed public ZIP path')
    require(e.get('raw_url')=='https://raw.githubusercontent.com/'+pin['repository']+'/'+pin['commit']+'/'+path,'Cosy17 exact immutable public URL')
    require(e.get('bytes')==archive['bytes']and e.get('sha256')==archive['sha256']and hex_digest(e.get('git_blob_sha1'),40),'Cosy17 public ZIP file identity')
    return pin

def validate_feature_entries(rows,features):
    validate_rows(rows);require(len(features)==49,'exact49 features')
    for r,f in zip(rows,features):
        require(f['recording']==r['recording']and f['pcm_sha256']==r['pcm_sha256']and f['wav_sha256']==r['wav_sha256'],'feature source identity and order')
        require(f['shape']==[r['model_rows'],400]and f['bytes']==r['model_rows']*1600 and f['normalization']=='PRE_CMVN','feature geometry and normalization')
        require(f['file']=='native/'+r['recording']+'.f32le','feature path')
        require(hex_digest(f['sha256'])and f['sha256']==r['feature']['sha256'],'frozen feature hash')
        require(all(f[k]==r['feature'][k]for k in ('file','bytes','normalization','shape')),'train49 feature binding')
    return features

def frozen_inputs(root):
    rows=read_json(root/'metadata/TRAIN49.json')['rows'];validate_rows(rows)
    require(digest(root/'metadata/PREPARED-INPUTS.json')==OLD_INPUTS_SHA,'unchanged old32 input manifest bytes')
    old=read_json(root/'metadata/PREPARED-INPUTS.json')
    require(old['status']=='VERIFIED_PUBLIC_PIN'and old['commit']==OLD_COMMIT,'unchanged old32 public pin')
    old_members={e['path']:e for e in old['archive']['members']}
    for r in rows[:32]:
        f=r['feature'];e=old_members[f['file']]
        require(f['sha256']==e['sha256']and f['bytes']==e['bytes'],'old32 immutable feature identities')
    new=validate_new_pin(read_json(root/'metadata/COSY17-INPUTS.json'),rows)
    return rows,old,new


def public_feature_entries(manifest,rows):
    """Explicit schema adapter, preserving raw bytes and fixed source order."""
    validate_rows(rows);expected=rows[32:]
    require(manifest.get('schema')=='cosy17-public-precmvn-features-v1','Cosy17 public manifest schema')
    require(manifest.get('source_count')==17 and manifest.get('selected_rows')==693 and manifest.get('raw_feature_bytes')==1108800,'Cosy17 public manifest geometry')
    require(manifest.get('dtype')=='<f4'and manifest.get('normalization')=='PRE_CMVN'and manifest.get('feature_width')==400,'Cosy17 public feature format')
    require(manifest.get('tokens')==list(TOKENS)and manifest.get('development_included')is False and manifest.get('inaudible_rows_included')is False and manifest.get('original_audio_included')is False,'Cosy17 public exclusion contract')
    require(manifest.get('source_order')==[r['recording']for r in expected]and len(manifest.get('rows',[]))==17,'Cosy17 public exact source order')
    result=[]
    for p,r in zip(manifest['rows'],expected):
        require(all(p[k]==r[k]for k in ('recording','cohort','target_ids','frames','wav_sha256','pcm_sha256','complete_wake_target')),'Cosy17 public row source and target binding')
        require(p['split']=='train'and p['actual_human_lexical_transcript']==r['text'],'Cosy17 public actual transcript')
        f=p['feature'];require(f['path']=='features/'+r['recording']+'.f32le','Cosy17 public relative feature path')
        require(all(f[k]==r['feature'][k]for k in ('bytes','sha256','shape','dtype','normalization')),'Cosy17 public feature identity')
        result.append(dict(recording=r['recording'],wav_sha256=r['wav_sha256'],pcm_sha256=r['pcm_sha256'],file=r['feature']['file'],**{k:f[k]for k in ('bytes','sha256','shape','dtype','normalization')}))
    return result
