#!/usr/bin/env python3
"""Offline source closure generator; no model or remote operation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from admission import required_sources
from execution_common import read,sha,canonical

def build(repo,enabled=False):
    files={name:sha(read(Path(repo)/name)) for name in sorted(required_sources())}
    return dict(schema='kw1-execution-admission-v1',enabled=enabled,max_tts_calls=16,max_asr_calls=48,retry=False,source_files=files,
        revision='reconstructed-20261015',review_status='enabled_by_explicit_final_source_review' if enabled else 'pending_independent_review')

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--enable-after-review',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args(argv)
    here=Path(__file__).resolve().parent;target=here/'execution-admission.json';value=build(here.parents[1],a.enable_after_review)
    raw=canonical(value)+b'\n'
    if a.check:
        if read(target)!=raw:raise ValueError('source configuration is not freshly frozen')
    else:target.write_bytes(raw)
    print(json.dumps(dict(status='verified' if a.check else 'frozen',files=len(value['source_files']),enabled=value['enabled'])))
if __name__=='__main__':main()
