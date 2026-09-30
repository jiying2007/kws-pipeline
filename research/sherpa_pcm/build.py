#!/usr/bin/env python3
"""Build optional host research adapter with installed GCC; never fetch dependencies."""
import argparse,hashlib,json,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=pathlib.Path);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--stub',action='store_true');g.add_argument('--library',type=pathlib.Path);args=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(root/'verify_dependencies.py')],check=True)
cmd=['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-fPIC','-shared',str(root/'pcm_kws.cc'),'-I'+str(root/'vendor'),'-o',str(output/'libkws_sherpa_pcm.so')]
lock=json.loads((root/'dependencies.lock.json').read_text());libhashes=[]
if args.stub:cmd.append(str(root/'tests/runtime_stub.cc'))
else:
 lib=args.library.resolve()
 if lib.name!='libsherpa-onnx-c-api.so':p.error('expected reviewed libsherpa-onnx-c-api.so')
 for expected in lock['host_runtime']['runtime_libraries']:
  f=lib.parent/expected['filename'];actual=hashlib.sha256(f.read_bytes()).hexdigest()
  if actual!=expected['sha256']:p.error('runtime SHA mismatch: '+str(f))
  libhashes.append({'filename':f.name,'sha256':actual,'path':str(f)})
 cmd+=['-L'+str(lib.parent),'-Wl,-rpath,'+str(lib.parent),'-lsherpa-onnx-c-api']
subprocess.run(cmd,check=True)
commands=[cmd]
if args.stub:
 test=['gcc','-std=c11','-Wall','-Wextra','-Werror',str(root/'tests/lifecycle_test.c'),'-I'+str(root),'-L'+str(output),'-Wl,-rpath,'+str(output),'-lkws_sherpa_pcm','-o',str(output/'lifecycle-test')]
 subprocess.run(test,check=True);subprocess.run([str(output/'lifecycle-test')],check=True);commands.append(test)
receipt={'stub_only':args.stub,'speech_inference_tested':False,'commands':commands,'runtime_libraries':libhashes,'compiler':subprocess.check_output(['g++','--version'],text=True),'files':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'pcm_kws.cc',root/'pcm_kws.h',root/'dependencies.lock.json',root/'vendor/sherpa-onnx/c-api/c-api.h']},'library_sha256':hashlib.sha256((output/'libkws_sherpa_pcm.so').read_bytes()).hexdigest()}
(output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Built research adapter; '+('model-free lifecycle tests passed' if args.stub else 'real runtime SHA verified; acoustic test is separate'))
