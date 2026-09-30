#!/usr/bin/env python3
"""Local isolated C11 build, without dependency downloads or product changes."""
import argparse,hashlib,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();root=pathlib.Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=True);out=a.output.resolve()
cmd=['gcc','-O2','-std=c11','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Wcast-qual','-Werror','-ffp-contract=off','-fPIC','-shared',str(root/'donor_fbank.c'),'-lm','-o',str(out/'libdonor_fbank.so')];subprocess.run(cmd,check=True)
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
receipt={'scope':'host C11 research-only frontend; no model/target claim','command':cmd,'compiler':subprocess.check_output(['gcc','--version'],text=True),'library_sha256':sha(out/'libdonor_fbank.so'),'files':{n:sha(root/n) for n in ['donor_fbank.c','donor_fbank.h','frontend_tables.h','fft_twiddles.h','spec.json']}}
(out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
