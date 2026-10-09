"""Plan-shape checker ONLY. It does not authenticate inputs or admit execution."""
import re

LIMITS = {'operator_rows': 798, 'wall_seconds': 120, 'rss_bytes': 536870912}
DIMS = [140,250,250] + [128,128,250,250]*4 + [140,6]
MEMORY_STAGES = {4,8,12,16}
FROZEN_SOURCES = {
 'torch_npz':'2bff23791b80147968fb7e5b4986f1dc612a088f7310d17f352092de2e69cdc3',
 'c_npz':'2b02fbb3179d3c25a65e6881539b1bf4bbb5d7048e73b8f6528e342bf8b73992',
 'export_manifest':'bd7b256f5def4ad2c211b47789ada0c8f07ced107c8add0afea79561ff96d6eb',
 'weights':'b1061525d82c2d1c9dfab30e382701a04e9861c8fda32cf1b8a7f539cdf185a1'}
GATES = {'raw':[1e-4,1e-5], 'probability':[1e-5,0], 'frontend':[1e-3,0], 'composed_cmvn':[2e-4,0]}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}', value) is not None,'invalid SHA256')

def validate(plan):
    """Check plan shape, never input authenticity, authorization or admission."""
    require(plan.get('schema') == 'd20-saved-single-op-plan-v1', 'wrong schema')
    require(plan.get('frozen_sources') == FROZEN_SOURCES, 'frozen source identity mismatch')
    require(plan.get('gates') == GATES, 'original gates changed')
    require(plan.get('required_backend') == {'torch':'2.11.0+cpu','numpy':'1.26.4'}, 'backend version changed')
    require(plan.get('limits') == LIMITS, 'limits must remain frozen')
    require(plan.get('anchor') == 'C_SAVED_OBSERVED', 'one common C saved anchor required')
    require(plan.get('input_chaining') is False, 'output chaining prohibited')
    require(plan.get('historical_chunk_equivalence') is False, 'single-row is not historical chunk dispatch')
    require(plan.get('original_status') == {'D20':'FAIL','D90':'NOT_RUN'},'original status changed')
    jobs = plan.get('jobs'); require(isinstance(jobs,list) and len(jobs)==798, 'exactly 798 planned rows')
    seen = set(); pairs = {}
    for j in jobs:
        r,s,b = j.get('row'),j.get('stage'),j.get('backend')
        require(type(r) is int and 0<=r<19 and type(s) is int and 0<=s<21 and b in ('C','Torch'),'invalid job')
        key=(r,s,b); require(key not in seen,'duplicate job'); seen.add(key)
        require(j.get('input_origin')=='frozen_saved_pre_input','input must be saved, never newly computed')
        require(j.get('dtype')=='<f4' and j.get('order')=='C','float32 little-endian C order required')
        require(j.get('input_shape')==[400 if s==0 else DIMS[s-1]],'input shape mismatch')
        require(j.get('output_shape')==[DIMS[s]],'output shape mismatch')
        digest(j.get('input_sha256'))
        require(j.get('input_bytes')==j['input_shape'][0]*4,'input byte count mismatch')
        if s in MEMORY_STAGES:
            require(j.get('cache_origin')=='C_SAVED_OBSERVED','memory requires actual C saved cache')
            require(j.get('cache_shape')==[128,11,4] and j.get('cache_bytes')==22528,'cache layout mismatch')
            digest(j.get('cache_sha256'))
        else:
            require(all(j.get(k) is None for k in ('cache_origin','cache_shape','cache_bytes','cache_sha256')),'unexpected cache')
        identity=(j['input_sha256'],j.get('cache_sha256'))
        previous=pairs.setdefault((r,s),identity)
        require(previous==identity,'backends must consume identical input and cache bytes')
    return {'planned_rows':len(seen),'model_calls':0,'operator_calls':0,'numerical_admission':False,'execution_ready':False,'input_authenticity_verified':False}
