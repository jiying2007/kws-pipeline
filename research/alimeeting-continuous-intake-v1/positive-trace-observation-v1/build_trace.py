import json,pathlib,hashlib,subprocess,time
D=pathlib.Path(__file__).resolve().parent; B=D.parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
save=lambda p,v:pathlib.Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
r=json.loads((B/'diagnostic-build/build-receipt.json').read_text()); assert sha(B/'diagnostic-build/build-receipt.json')=='78b2988e60a4bf6b875934222fd5e732633ca34de64e470c433e74289f57ac8a'
checks={x['path']:x['sha256'] for x in r['source_inputs']}
for k in ['model','keyword_pack','runner','cmake','compiler']:checks[r[k]['path']]=r[k]['sha256']
src=pathlib.Path(r['source_root'])
for f in ['tools/kws_posterior_dump.c','tools/kws_decoder_replay.c','tools/kws_decoder_path_replay.c','tools/kws_trace_io.c','tools/kws_trace_io.h','src/kws_debug.h']:
 checks[str(src/f)]=sha(src/f)
for p,h in checks.items():assert sha(p)==h,p
build=D/'build'; assert not build.exists()
a=[v.replace(str(B/'diagnostic-build/build'),str(build)) for v in r['configure_argv']]
record={'source_revision':r['source_revision'],'source_tree':r['source_tree'],'source_inputs':checks,'environment':r['environment'],'parent_build_receipt_sha256':sha(B/'diagnostic-build/build-receipt.json'),'commands':[]}
save(D/'build-receipt.json',record)
for name,argv in [('configure',a),('build',[r['cmake']['path'],'--build',str(build),'--target','kws_posterior_dump','kws_decoder_replay','kws_decoder_path_replay','--parallel','2','--verbose'])]:
 t=time.monotonic();p=subprocess.run(argv,env=r['environment'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
 (D/(name+'.stdout.log')).write_bytes(p.stdout);(D/(name+'.stderr.log')).write_bytes(p.stderr)
 record['commands'].append({'argv':argv,'exit_code':p.returncode,'elapsed_s':time.monotonic()-t});save(D/'build-receipt.json',record);assert p.returncode==0,p.stderr.decode()
record['outputs']={str(p.relative_to(D)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in build.rglob('*') if p.is_file() and (p.suffix in ['.o','.a','.h','.json','.d'] or p.name.startswith('kws_'))}
for p,h in checks.items():assert sha(p)==h,p
record['input_unchanged_after']=True;record['status']='pass';save(D/'build-receipt.json',record)
print('BUILD PASS')
