"""Supervised wheel stage or installed-metadata verification. No Torch import."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from common import check_context,read_json,require,write_json,digest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/deps'))
import exact_wheels


def run(context,action):
    check_context(ROOT,context,action)
    if action=='wheels':
        r=exact_wheels.release_template();r['approved']=True
        result=exact_wheels.download_all(ROOT/'work/dependencies',r)
        write_json(ROOT/'work/wheel-inspection.json',result,1024*1024)
        print('Exact CPU wheel download and full safe ZIP inspection complete; no installation in this phase')
        return
    require(action=='verify_install','dependency stage')
    expected=exact_wheels.load_lock()['packages'];installed=[]
    for p in expected:
        d=importlib.metadata.distribution(p['project'])
        require(d.version==p['version'],'installed exact package version')
        md=d.read_text('METADATA');wh=d.read_text('WHEEL');record=d.read_text('RECORD')
        require(md is not None and wh is not None and record is not None,'installed distribution evidence')
        installed.append(dict(project=p['project'],version=d.version,metadata_sha256=hashlib.sha256(md.encode()).hexdigest(),
            wheel_metadata_sha256=hashlib.sha256(wh.encode()).hexdigest(),record_sha256=hashlib.sha256(record.encode()).hexdigest()))
    files=[p for p in (ROOT/'work/dependencies/venv').rglob('*')if p.is_file()]
    size=sum(p.stat().st_size for p in files)
    require(size<=2*1024**3,'installed venv exceeds proposed2GiB')
    require(sys.prefix!=sys.base_prefix and Path(sys.prefix).resolve()==(ROOT/'work/dependencies/venv').resolve(),'isolated interpreter')
    write_json(ROOT/'work/installed.json',dict(status='PASS_INSTALLED_METADATA_ONLY',packages=installed,
        file_count=len(files),installed_bytes=size,python_version=sys.version.split()[0],torch_imported=False))
    print('Installed dependency metadata and isolated environment verified; no model import or call')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-context',type=Path);p.add_argument('--action',choices=['wheels','verify_install'])
    a=p.parse_args()
    if a.execute_context is None or a.action is None:raise SystemExit('PREPARATION_ONLY: no exact dependency context')
    from safe_failure import save
    try:run(read_json(a.execute_context),a.action)
    except BaseException as error:
        save(ROOT,error,a.action,'CPU_WHEEL_DOWNLOAD_INSPECTION'if a.action=='wheels'else 'INSTALLED_METADATA_CHECK')
        raise SystemExit(1)
