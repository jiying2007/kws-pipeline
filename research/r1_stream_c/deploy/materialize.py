#!/usr/bin/env python3
"""Generate a standalone R1 reference64 research SDK; never launches inference."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent

def prepare(repo:Path,output:Path,model_package:Path|None=None)->dict:
    output=output.absolute();repo=repo.resolve()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('fresh non-symlink output is required')
    source=repo/'research/r1_stream_c'
    spec=importlib.util.spec_from_file_location('r1_sdk_numeric_source',source/'numerics.py')
    if spec is None or spec.loader is None:raise ValueError('missing reference64 source')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output.mkdir(parents=True)
    try:
        status=module.materialize(repo,output/'reference64','reference64',model_package)
        adapter=output/'adapter';adapter.mkdir()
        for name in ('r1_sdk.h','r1_sdk.c','r1_sdk_cli.c','test_sdk.c'):
            shutil.copyfile(HERE/name,adapter/name)
        shutil.copyfile(HERE/'CMakeLists.txt',output/'CMakeLists.txt')
        shutil.copyfile(HERE/'README.md',output/'README.md')
        receipt={'schema':'kws-r1-reference-sdk-source-v1','model_id':'R1-DEV-FSMN-20261107',
                 'precision':'reference64','legacy_fp32_gate':'FAIL_UNCHANGED',
                 'policy':'r1-original-greedy-span150-cooldown50-v1',
                 'status':'PREPARED_NOT_EXECUTED','model_forwards':0,'weights_present':model_package is not None,
                 'underlying_status':status,'physical_ssc305':False,'calibrated_detector':False,
                 'files':{p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(output.rglob('*')) if p.is_file()}}
        (output/'SDK_SOURCE_MANIFEST.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
        return {k:v for k,v in receipt.items() if k!='files'}
    except Exception:
        # Preserve the partial directory as a failed attempt, never overwrite it.
        (output/'PREPARATION_FAILED.txt').write_text('Partial source generation; no inference launched.\n')
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=HERE.parents[2]);p.add_argument('--output',type=Path,required=True);p.add_argument('--model-package',type=Path)
    a=p.parse_args()
    try:r=prepare(a.repo,a.output,a.model_package)
    except (ValueError,OSError,KeyError) as e:p.exit(2,str(e)+'\n')
    print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
