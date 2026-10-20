#!/usr/bin/env python3
"""ABI/configuration mocks only. No posterior, model, or decoder-step calls."""
import ctypes as C, json, pathlib, subprocess, sys, time, resource
p=pathlib.Path(sys.argv[1]).resolve(); lib=C.CDLL(str(p/'research.so'))
checks=0
def check(x):
    global checks
    checks+=1
    assert x
lib.kr_create.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p)];lib.kr_create.restype=C.c_int
lib.kr_destroy.argtypes=[C.c_void_p];lib.kr_threshold.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.POINTER(C.c_uint64)]
for n in ['kr_storage_bytes','kr_event_bytes','kr_summary_bytes','kr_retention_bytes']:getattr(lib,n).restype=C.c_size_t
for name in ['kws3_create','kws3_create_with_threshold','kws3_reset','kws3_finish','kws3_step','kws3_step_f32']:
    try: getattr(lib,name)
    except AttributeError:check(True)
    else:check(False)
for mode in [None,b'',b'production',b'bounded-retention-research-v2',b'bounded-retention-research-v1 ']:
    h=C.c_void_p(123);check(lib.kr_create(mode,b'mock',C.byref(h))==1);check(h.value is None)
check(lib.kr_create(b'bounded-retention-research-v1',b'mock',None)==1)
for identity in [None,b'',b'x'*256]:
    h=C.c_void_p();check(lib.kr_create(b'bounded-retention-research-v1',identity,C.byref(h))==1);check(h.value is None)
h=C.c_void_p(); check(lib.kr_create(b'bounded-retention-research-v1',b'config-mock',C.byref(h))==0)
try:
    theta=C.c_double(); rejects=C.c_uint64();check(lib.kr_threshold(h,C.byref(theta),C.byref(rejects))==0);check(theta.value==0.7);check(rejects.value==0)
    check(lib.kr_event_bytes()==360);check(lib.kr_summary_bytes()==424);check(lib.kr_retention_bytes()==216);check(lib.kr_storage_bytes()==533032)
finally:lib.kr_destroy(h)
result={'status':'PASS_FACADE_CONFIG_ONLY','checks':checks,'scientific_replays':0,'model_calls':0,'posterior_reads':0,'decoder_step_calls':0}
(p/'FACADE-TEST.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
