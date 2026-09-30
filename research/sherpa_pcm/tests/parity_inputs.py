"""Fail-closed local evidence guards, run before loading any native library."""
import hashlib,json,pathlib
REFERENCE_SHA256='68c46ed002d00868623f21f354b1b62ff47125b7fe24add9933502a2e6ef45ad'
SOURCE_MEMBERS={'pcm_kws.cc','pcm_kws.h','dependencies.lock.json','vendor/sherpa-onnx/c-api/c-api.h'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def contained_wav(data_root,row_path):
 rel=pathlib.Path(row_path)
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('recording path must be contained relative path')
 if data_root.is_symlink():raise ValueError('symlink data root rejected')
 root=data_root.resolve();p=root
 for part in rel.parts:
  p=p/part
  if p.is_symlink():raise ValueError('symlink recording path rejected')
 p.resolve().relative_to(root)
 if not p.is_file():raise ValueError('missing recording')
 return p

def validate_inputs(args,source_root):
 if args.output.exists():raise ValueError('refuse to overwrite parity output')
 if sha(args.reference)!=REFERENCE_SHA256:raise ValueError('reference digest is not the approved observed42 report')
 lock=json.loads((source_root/'dependencies.lock.json').read_text())
 receipt=json.loads(args.build_receipt.read_text())
 if receipt.get('stub_only') is not False:raise ValueError('real non-stub build receipt required')
 if set(receipt['files'])!=SOURCE_MEMBERS:raise ValueError('build source binding coverage mismatch')
 for name,h in receipt['files'].items():
  if sha(source_root/name)!=h:raise ValueError('build source digest mismatch: '+name)
 if sha(args.library)!=receipt['library_sha256']:raise ValueError('built library digest mismatch')
 expected={x['filename']:x['sha256'] for x in lock['host_runtime']['runtime_libraries']}
 actual=receipt['runtime_libraries']
 if len(actual)!=len(expected) or {x['filename'] for x in actual}!=set(expected):raise ValueError('runtime binding coverage mismatch')
 for x in actual:
  if x['sha256']!=expected[x['filename']] or sha(pathlib.Path(x['path']))!=x['sha256']:raise ValueError('runtime dependency digest mismatch')
 report=json.loads(args.reference.read_text())
 if len(report['recordings'])!=42 or len({x['recording'] for x in report['recordings']})!=42:raise ValueError('expected42 unique original clips')
 for row in report['recordings']:
  wav=contained_wav(args.data_root,row['path'])
  if sha(wav)!=row['file_sha256']:raise ValueError('recording digest mismatch')
 return report
