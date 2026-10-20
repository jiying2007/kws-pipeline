#!/usr/bin/env python3
"""Build the explicit research facade and test only invented inputs."""
import argparse, json, pathlib, subprocess, sys, tempfile
HERE=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root',type=pathlib.Path)
p.add_argument('--cc',choices=['gcc','clang'],default='gcc')
a=p.parse_args();repo=(a.repo_root or HERE.parents[2]).resolve()
with tempfile.TemporaryDirectory(prefix='kws-explicit-research-check-') as t:
    out=pathlib.Path(t)/'build'
    subprocess.run([sys.executable,'-B',str(HERE/'build_research.py'),'--mode','bounded-retention-research-v1','--repo-root',str(repo),'--output',str(out),'--cc',a.cc],check=True,timeout=45)
    subprocess.run([sys.executable,'-B',str(HERE/'test_facade.py'),str(out)],check=True,timeout=15)
    subprocess.run([a.cc,'-std=c11','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard','-O2','-I'+str(out),str(HERE/'api_synthetic.c'),str(out/'research.so'),'-lm','-o',str(out/'api_synthetic')],check=True,timeout=30)
    subprocess.run([str(out/'api_synthetic')],check=True,timeout=15)
print(json.dumps({'status':'PASS_EXPLICIT_RESEARCH_SYNTHETIC_ONLY','scientific_replays':0,'production_default_modified':False}))
