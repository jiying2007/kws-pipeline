#!/usr/bin/env python3
"""Offline source-only lock integrity checks; no dependency fetching."""
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent
lock=json.loads((root/'dependencies.lock.json').read_text())
assert lock['research_only'] is True and lock['default_product_runtime_unchanged'] is True
assert lock['sherpa_version']=='1.13.8'
assert hashlib.sha256((root/lock['header']['path']).read_bytes()).hexdigest()==lock['header']['sha256']
assert len(lock['models'])==4
assert len({x['path'] for x in lock['models']})==4
assert all(len(x['expected_sha256'])==64 and 'Revision=3787015f084cb241dfa0e4ba237703a2d4322d50' in x['url'] for x in lock['models'])
assert lock['decoder']==dict(sample_rate=16000,feature_dim=80,num_threads=1,max_active_paths=4,num_trailing_blanks=1,keywords_score=1.0,keywords_threshold=.25,feed_samples=320)
assert not list(root.rglob('*.onnx')) and not list(root.rglob('*.so'))
print('sherpa research dependency lock: passed; no dependencies downloaded')
