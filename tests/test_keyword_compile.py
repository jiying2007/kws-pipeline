#!/usr/bin/env python3
from __future__ import annotations
import pathlib, subprocess, sys, tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main()->int:
    with tempfile.TemporaryDirectory() as td:
        out=pathlib.Path(td)/'keywords.h'; manifest=pathlib.Path(td)/'keywords.json'
        subprocess.check_call([sys.executable,str(ROOT/'tools'/'compile_keywords.py'),'--tokens',str(ROOT/'keywords'/'tokens.example.txt'),'--keywords',str(ROOT/'keywords'/'zh_cn_example.tsv'),'--out-header',str(out),'--out-json',str(manifest)])
        text=out.read_text(encoding='utf-8'); assert 'kws_generated_keyword_count' in text; assert '{1u, 2u, 3u, 4u}' in text; assert manifest.exists()
    print('test_keyword_compile: ok'); return 0
if __name__=='__main__': raise SystemExit(main())
