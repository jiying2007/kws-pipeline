#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib
def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('inputs',nargs='+',type=pathlib.Path); ap.add_argument('--output',required=True,type=pathlib.Path); a=ap.parse_args(); tokens=set()
    for path in a.inputs:
        for raw in path.read_text(encoding='utf-8').splitlines():
            if not raw.strip() or raw.startswith('#'): continue
            tokens.update(raw.split('\t')[-1].split())
    ordered=['<blk>']+sorted(t for t in tokens if t!='<blk>'); a.output.write_text('\n'.join(f'{tok} {i}' for i,tok in enumerate(ordered))+'\n',encoding='utf-8')
if __name__=='__main__': main()
