"""Closed local paths, exact file identities and bounded JSON for preparation."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe(root, relative):
    p=PurePosixPath(relative)
    require(type(relative)is str and relative and not p.is_absolute() and '..'not in p.parts and str(p)==relative, 'unsafe relative path')
    result=Path(root)/p
    require(result.resolve().is_relative_to(Path(root).resolve()),'path escape')
    require(not any((Path(root)/Path(*p.parts[:i])).is_symlink()for i in range(1,len(p.parts)+1)),'symlink path')
    return result


def unique(pairs):
    result={}
    for k,v in pairs:
        require(k not in result,'duplicate JSON key');result[k]=v
    return result


def read_json(path, cap=1024*1024):
    path=Path(path);require(path.stat().st_size<=cap,'JSON byte cap')
    return json.loads(path.read_text(),object_pairs_hook=unique,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def write_bytes(path, raw, cap):
    require(len(raw)<=cap,'output file cap')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())


def write_json(path, obj, cap=1024*1024):
    write_bytes(path,(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),cap)


def verify_file(root, entry):
    p=safe(root,entry['path'])
    require(p.is_file()and p.stat().st_size==entry['bytes']and digest(p)==entry['sha256'],'file identity')
    return p


def check_context(root, context, phase):
    require(context.get('schema')=='a20-preparation-phase-context-v1'and context.get('phase')==phase,'phase context')
    require(context.get('source_freeze_sha256')==digest(Path(root)/'SOURCE-FREEZE.json'),'phase source freeze')
    require(context.get('training_authorized')is False,'training forbidden')
    ledger=read_json(Path(root)/'work/EXECUTION-LEDGER.json')
    require(ledger['status']=='START_RESERVED_NO_RETRY' and ledger['release_sha256']==context['release_sha256'],'exclusive attempt context')


def replace_json(path,obj,cap=1024*1024):
    """Atomic bounded progress for this attempt's already-owned JSON file."""
    path=Path(path);require(not path.is_symlink(),'progress symlink')
    tmp=path.with_name(path.name+'.pending')
    raw=(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    write_bytes(tmp,raw,cap);os.replace(tmp,path)
