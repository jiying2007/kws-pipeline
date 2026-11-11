#!/usr/bin/env python3
"""Pack unchanged selected parameter bytes after a passing cached ACTIVE113 screen."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from active_format import (FRONTEND,FRONTEND_SHA,MODEL_SHA,PACKAGE_BYTES,POLICY,SELECTION_SHA,
    base_binding,digest,need,pack_bytes,save,screen_gate,sha,source_gate)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--screen-result',type=Path,required=True)
    p.add_argument('--screen-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve()
    need(not out.exists(),'fresh pack directory required')
    source_gate(a.plan_sha256); screen_gate(a.screen_result,a.screen_sha256); base_binding()
    data,sections=pack_bytes(); need(sha(FRONTEND)==FRONTEND_SHA,'frontend changed')
    frontend=FRONTEND.read_bytes(); os.umask(0o077);out.mkdir(parents=True)
    for name,payload in [('model-active113.r1a113',data),('frontend.f32le',frontend)]:
        with (out/name).open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    save(out/'PACK.json',{'schema':'r1-active113-compact-pack-v1','status':'PACKED_NO_MODEL_EXECUTION',
        'preparation_plan_sha256':a.plan_sha256,'screen_result_sha256':a.screen_sha256,
        'selection_sha256':SELECTION_SHA,'policy':POLICY,'original_model_sha256':MODEL_SHA,
        'model':{'path':'model-active113.r1a113','bytes':PACKAGE_BYTES,'sha256':digest(data)},
        'frontend':{'path':'frontend.f32le','bytes':len(frontend),'sha256':FRONTEND_SHA},
        'sections':sections,'numerical_scores':6,'precise_rows':5,'selected_other_rows':113,
        'other_winner_original_id_metadata':True,'full_2601_outputs':False,
        'parameter_values_modified':False,'training':0,'neural_model_calls':0,'pcm_reads':0,
        'default_source_or_model_modified':False,'model_promotion_authorized':False})
    print('PACKED_NO_MODEL_EXECUTION '+digest(data))


if __name__=='__main__':main()
