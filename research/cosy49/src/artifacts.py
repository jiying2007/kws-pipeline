"""Bounded, exact output allowlist; failed partial weights never become candidates."""
import gzip,hashlib,io,json,tarfile,math,struct,os
from pathlib import Path
from common import require,read_json,write_bytes,write_json,digest
BASE={'initial-train-logits.f32le','input-reconstitution.json','initial-train-diagnostics.json','first-forward-binding.json','call-journal.jsonl','updates.jsonl','final-status.json','resources.json','environment.json','dependencies.json','source-identities.json','rights-and-provenance.json','safe-failure.json','saved-only-recovery.json','private-log-identities.json','validated-jsonl-prefixes.json'}
PHASES=('wheels','venv','install','verify_install','inputs','training')
LOGS={f'phase-{p}.{stream}.log'for p in PHASES for stream in ('stdout','stderr')}
TERMINAL={'terminal-fit.json','terminal-step300.pt','terminal-state-payload.f32le','terminal-tensors.json','optimizer-state.json','checkpoint.json','terminal-train-logits.f32le','terminal-train-diagnostics.json','training-result.json'}
CAP=20*1024**2

def journal_counts(path):
    started=dict(forward=0,backward=0,update=0);completed=started.copy();truncated=False
    if not path.exists():return dict(started=started,completed=completed,partial_final_record=False,exact_completed_count=True)
    raw=path.read_bytes();require(len(raw)<=256*1024,'journal byte cap')
    lines=raw.splitlines(keepends=True)
    for i,line in enumerate(lines):
        if not line.endswith(b'\n'):
            require(i==len(lines)-1,'journal middle truncation');truncated=True;break
        x=json.loads(line);require(set(x)=={'kind','status','step'}and x['kind']in started and x['status']in ('started','completed'),'journal schema')
        dst=started if x['status']=='started'else completed;dst[x['kind']]+=1;require(completed[x['kind']]<=started[x['kind']],'journal call ordering')
    return dict(started=started,completed=completed,partial_final_record=truncated,exact_completed_count=not truncated,interrupted_call_may_have_executed={k:started[k]>completed[k]for k in started})

def package(root,status):
    out=root/'work/artifact';pub=root/'publication'
    if pub.exists():
        require(not pub.is_symlink(),'publication symlink')
        require(not (root/'work/PUBLICATION-READY.json').exists(),'completed publication immutable')
        quarantine=root/'work/publication-incomplete'
        require(not quarantine.exists(),'single saved-only packaging recovery')
        os.replace(pub,quarantine)
    pub.mkdir(exist_ok=False)
    retain_validated_prefixes(out)
    found={p.name for p in out.iterdir()};require(all(p.is_file()and not p.is_symlink()for p in out.iterdir()),'flat regular artifacts only')
    require(found<=BASE|TERMINAL|LOGS,'unknown output rejected')
    chosen=set();issues=[];members=[];total=0;checkpoint_valid=False
    if (out/'checkpoint.json').exists():
        try:
            c=read_json(out/'checkpoint.json');p=out/'terminal-step300.pt'
            require(p.exists()and p.stat().st_size==c['bytes']and digest(p)==c['sha256'],'terminal checkpoint receipt')
            checkpoint_valid=True
        except (ValueError,KeyError,TypeError,OSError):pass
    for n in sorted(found):
        p=out/n;size=p.stat().st_size
        e=dict(path=n,bytes=size,sha256=digest(p))
        valid=True;reason=None
        try:
            require(size<=6*1024**2,'individual artifact cap')
            if n in LOGS:raise ValueError('raw process logs remain private')
            if n.endswith('.json'):read_json(p,1024**2)
            elif n.endswith('.jsonl'):
                require(size<=256*1024,'JSONL output cap')
                raw=p.read_bytes();require(not raw or raw.endswith(b'\n'),'partial JSONL')
                for line in raw.splitlines():
                    x=json.loads(line);require(type(x)is dict,'JSONL mapping')
                    if n=='updates.jsonl':
                        require(set(x)=={'step','loss_before_update','gradient_norm_before_clip'}and type(x['step'])is int and 1<=x['step']<=300,'numeric update schema')
                        require(all(type(x[k])in (int,float)and math.isfinite(x[k])for k in ('loss_before_update','gradient_norm_before_clip')),'finite numeric update')
                    else:require(set(x)=={'kind','status','step'}and x['kind']in ('forward','backward','update')and x['status']in ('started','completed')and type(x['step'])is int and 1<=x['step']<=300,'safe call journal schema')
            elif n.endswith('.f32le'):
                expected=111720 if n in ('initial-train-logits.f32le','terminal-train-logits.f32le')else 1565280
                require(size==expected and all(math.isfinite(v[0])for v in struct.iter_unpack('<f',p.read_bytes())),'finite complete raw numerical payload')
            elif n.endswith('.pt'):require(checkpoint_valid,'validated checkpoint receipt required')
        except (ValueError,KeyError,TypeError,OSError,UnicodeError):
            valid=False;reason='UNPUBLISHED_INVALID_OR_INCOMPLETE_OR_PRIVATE_RAW_BYTES';issues.append(dict(file=n,bytes=size,sha256=e['sha256'],status=reason,content_captured=False))
        if valid:chosen.add(n);members.append(e);total+=size
        elif status=='SUCCESS'and n not in LOGS:raise ValueError('invalid successful artifact')
    if status=='SUCCESS':require(TERMINAL|{'initial-train-logits.f32le','initial-train-diagnostics.json','first-forward-binding.json'}<=chosen and checkpoint_valid,'complete success outputs')
    require(total<=32*1024**2,'raw artifact cap')
    ignored=[dict(file=n,bytes=(out/n).stat().st_size,sha256=digest(out/n),status='UNPUBLISHED_CONTENT_UNKNOWN_OR_PRIVATE')for n in sorted(found-chosen)]
    manifest=dict(schema='a20-cosy49-artifact-manifest-v1',status=status,files=members,raw_bytes=total,unpublished=ignored,raw_integrity_issues=issues,checkpoint_receipt_valid=checkpoint_valid,qualification=False)
    body=io.BytesIO()
    with tarfile.open(fileobj=body,mode='w')as t:
        for e in members:
            b=(out/e['path']).read_bytes();m=tarfile.TarInfo(e['path']);m.size=len(b);m.mtime=0;m.mode=0o600;t.addfile(m,io.BytesIO(b))
        b=(json.dumps(manifest,sort_keys=True,allow_nan=False)+'\n').encode();m=tarfile.TarInfo('artifact-manifest.json');m.size=len(b);m.mtime=0;m.mode=0o600;t.addfile(m,io.BytesIO(b))
    raw=gzip.compress(body.getvalue(),mtime=0);require(len(raw)<=CAP,'compressed artifact cap')
    name='a20-cosy49-artifact.tar.gz';write_bytes(pub/name,raw,CAP);write_bytes(pub/(name+'.sha256'),(hashlib.sha256(raw).hexdigest()+'  '+name+'\n').encode(),256)
    receipt=dict(schema='a20-cosy49-publication-v1',status=status,archive=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),retention_days=1,manifest=manifest)
    write_json(root/'work/PUBLICATION-READY.json',receipt);return receipt

def collect_phase_logs(root):
    out=root/'work/artifact';logs=root/'work/logs';receipts=[]
    if not logs.exists():return
    total=0
    for p in sorted(logs.iterdir()):
        require(p.name in {phase+'.'+stream for phase in PHASES for stream in ('stdout','stderr')}and p.is_file()and not p.is_symlink(),'closed phase logs')
        total+=p.stat().st_size
        receipts.append(dict(phase_stream=p.name,bytes=p.stat().st_size,sha256=digest(p),content_publication='EXCLUDED_PRIVATE_RUN_LOCAL_ONLY'))
    target=out/'private-log-identities.json'
    if not target.exists():write_json(target,dict(schema='a20-private-log-identities-v1',logs=receipts,raw_content_captured=False,aggregate_bytes=total,operational_log_cap_exceeded=total>512*1024))

def retain_validated_prefixes(out):
    target=out/'validated-jsonl-prefixes.json'
    if target.exists():return
    derived=[]
    for name in ('updates.jsonl','call-journal.jsonl'):
        p=out/name
        if not p.exists():continue
        raw=p.read_bytes();require(len(raw)<=256*1024,'bounded source JSONL for prefix')
        if not raw or raw.endswith(b'\n'):continue
        rows=[];reason='TRUNCATED_FINAL_LINE';stopped=False
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b'\n'):break
            try:
                x=json.loads(line)
                if name=='updates.jsonl':
                    require(set(x)=={'step','loss_before_update','gradient_norm_before_clip'}and type(x['step'])is int and x['step']==len(rows)+1 and x['step']<=300,'complete numeric prefix step')
                    require(all(type(x[k])in (int,float)and math.isfinite(x[k])for k in ('loss_before_update','gradient_norm_before_clip')),'complete finite prefix')
                else:require(set(x)=={'kind','status','step'}and x['kind']in ('forward','backward','update')and x['status']in ('started','completed')and type(x['step'])is int and 1<=x['step']<=300,'safe complete journal prefix')
            except (ValueError,KeyError,TypeError):reason='INVALID_LINE_BEFORE_END';stopped=True;break
            rows.append(x)
        derived.append(dict(source_file=name,source_sha256=digest(p),source_bytes=len(raw),status=reason,complete_source=False,rows=rows,known_complete_rows=len(rows),remaining_content='UNKNOWN_NOT_RECONSTRUCTED',extra_calculation=False))
    if derived:write_json(target,dict(schema='a20-saved-validated-jsonl-prefix-v1',derivations=derived,complete_attempt=False),256*1024)

def publication_complete(root):
    ready=root/'work/PUBLICATION-READY.json'
    if not ready.exists():return False
    try:
        require(ready.is_file()and not ready.is_symlink(),'safe publication receipt')
        r=read_json(ready);name='a20-cosy49-artifact.tar.gz';p=root/'publication'/name
        require(r['schema']=='a20-cosy49-publication-v1'and r['archive']==name and r['status']in ('SUCCESS','FAILED_NO_RETRY'),'publication receipt schema')
        require(p.is_file()and not p.is_symlink()and p.stat().st_size==r['bytes']and r['bytes']<=CAP and digest(p)==r['sha256'],'publication archive verified')
        side=p.parent/(name+'.sha256');require(side.is_file()and not side.is_symlink()and side.stat().st_size<=256,'publication sidecar')
        require(side.read_text()==r['sha256']+'  '+name+'\n','publication checksum identity')
        return True
    except (ValueError,KeyError,TypeError,OSError,UnicodeError):return False
