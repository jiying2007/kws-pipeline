#!/usr/bin/env python3
"""Opt-in offline research library; source reconstruction only, no scientific inputs."""
import argparse, hashlib, json, pathlib, shutil, subprocess, tempfile, time, resource, re
HERE=pathlib.Path(__file__).resolve().parent
FLAGS=['-std=c11','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard','-O2']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(c,m):
    if not c: raise ValueError(m)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['bounded-retention-trace-v1'])
    p.add_argument('--repo-root',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--cc',choices=['gcc','clang'],default='gcc')
    a=p.parse_args(); root=a.repo_root.resolve(); out=a.output.resolve()
    require(not out.exists(),'Refusing to overwrite output')
    package=root/'research/bounded-local-retention-2026-10-20'
    identity=json.loads((package/'BUILD-IDENTITY.json').read_text())
    base=root/identity['base_repo_directory']
    for path,digest in [(base/'kws3.c',identity['base_source_sha256']),(base/'kws3.h',identity['base_header_sha256'])]:
        require(sha(path)==digest,'Base identity mismatch: '+str(path))
    for name,digest in identity['support_sha256'].items():
        require(sha(package/name)==digest,'Support identity mismatch: '+name)
    wall=time.monotonic(); before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory(prefix='kws-research-build-') as td:
        work=pathlib.Path(td)
        for n in ['kws3.c','kws3.h']: shutil.copyfile(base/n,work/n)
        for h in (package/'headers').glob('*.h'): shutil.copyfile(h,work/h.name)
        subprocess.run(['patch','--batch','--forward','-p1','-d',str(work),'-i',str(package/'candidate.patch')],check=True,capture_output=True,timeout=10)
        require(sha(work/'kws3.c')==identity['reconstructed_candidate_sha256'],'Reconstructed source mismatch')
        cmd=[a.cc,*FLAGS,'-fPIC','-shared','-pthread','-I'+str(work),'-I'+str(HERE),str(work/'kws3.c'),str(HERE/'trace_mode.c'),'-Wl,--version-script='+str(HERE/'exports.map'),'-lm','-o',str(work/'trace.so')]
        subprocess.run(cmd,check=True,capture_output=True,timeout=30)
        exports=subprocess.check_output(['nm','-D','--defined-only',str(work/'trace.so')],text=True)
        expected={'kt_create','kt_begin','kt_step_f32','kt_drain','kt_status','kt_summary','kt_retention','kt_end','kt_destroy','kt_status_bytes','kt_record_bytes'}
        require({line.split()[-1] for line in exports.splitlines()}==expected,'Unexpected public exports')
        out.mkdir(parents=True)
        shutil.copyfile(HERE/'trace_mode.h',out/'trace_mode.h')
        shutil.copyfile(work/'trace.so',out/'trace.so')
        for n in ['kws3.h','kws3_retention.h','kws3_diagnostic_ledger.h']: shutil.copyfile(work/n,out/n)
        elf=subprocess.check_output(['readelf','-W','-l',str(out/'trace.so')],text=True)
        tls=[line.split() for line in elf.splitlines() if line.strip().startswith('TLS ')]
        require(len(tls)==1,'Expected one ELF TLS segment')
        tls_bytes=int(tls[0][5],16)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt={'status':'BUILT_EXPLICIT_OFFLINE_RESEARCH_MODE','mode':a.mode,'threshold':0.7,'retention_rows':5,'nominal_audio_ms':150,'product_extra_wait_cap_ms':500,'product_cap_enforced':False,'product_cap_validation':'PENDING','production_default_modified':False,'scientific_replays':0,'candidate_sha256':identity['reconstructed_candidate_sha256'],'library_sha256':sha(out/'trace.so'),'facade_sha256':sha(HERE/'trace_mode.c'),'trace_header_sha256':sha(HERE/'trace_mode.h'),'export_map_sha256':sha(HERE/'exports.map'),'compiler':subprocess.check_output([a.cc,'--version'],text=True).splitlines()[0],'flags':FLAGS,'exports':sorted(expected),'library_tls_memsz_bytes':tls_bytes,'diagnostic_capture_enabled':False,'diagnostic_capture_available':True,'aggregate_abi_modified':False,'diagnostic_mode':'explicit kt_begin; one decoder and thread; sidecar admission required','decoder_storage_bytes':533032,'build_wall_seconds':time.monotonic()-wall,'build_child_cpu_seconds':(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),'build_child_peak_rss_kib_process_highwater':after.ru_maxrss}
    (out/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__': main()
