#!/usr/bin/env python3
"""Compile and execute invented transport/decoder contracts, without model calls."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
# Mock core only needs the stable callback/header ABI, not any model source.
HEADER='''#include <stdint.h>\n#include <stddef.h>\n#define R1D_OUTPUTS 2601u\ntypedef struct r1d_stream r1d_stream;\ntypedef int (*r1d_sink)(void*,uint64_t,uint64_t,uint64_t,const double*,const double*);\nr1d_stream*r1d_create(const unsigned char*,size_t,const unsigned char*,size_t,r1d_sink,void*);\nint r1d_feed(r1d_stream*,const unsigned char*,size_t,size_t*);\nint r1d_finish(r1d_stream*);\nint r1d_reset(r1d_stream*);\nvoid r1d_destroy(r1d_stream*);\nuint64_t r1d_frames(const r1d_stream*);\nsize_t r1d_state_bytes(void);\nsize_t r1d_weight_bytes(void);\n'''
def run(cc='gcc',sanitize=False):
 env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
 with tempfile.TemporaryDirectory(prefix='r1-sdk-contracts-') as d:
  p=Path(d);(p/'r1d_stream.h').write_text(HEADER)
  cmd=[cc,'-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic','-fno-fast-math','-ffp-contract=off','-I'+str(p),'-I'+str(HERE),str(HERE/'test_sdk.c'),str(HERE/'r1_sdk.c'),'-Wl,--wrap=malloc,--wrap=calloc,--wrap=free','-lm','-o',str(p/'test')]
  if sanitize:cmd[2:2]=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
  subprocess.run(cmd,env=env,capture_output=True,check=True,timeout=30)
  result=subprocess.run([str(p/'test')],env=env,capture_output=True,text=True,check=True,timeout=20)
  data=json.loads(result.stdout)
  if data.get('groups')!=21 or data.get('status')!='PASS' or data.get('model_forwards')!=0:raise RuntimeError('incomplete contract run')
  return data
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cc',default='gcc');p.add_argument('--sanitize',action='store_true');a=p.parse_args()
 print(json.dumps(run(a.cc,a.sanitize),sort_keys=True))
