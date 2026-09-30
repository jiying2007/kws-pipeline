"""Strict local-only full donor manifest loader. No download or unsafe pickle."""
import hashlib,json,pathlib,struct
SOURCE='d02b09c34f4a8bbb06f0dd1bf5eb58db3395eb7f1fd15c3625fe09d3a2492233'
PAYLOAD_SHA256='c72bfaac14ad0d7e0cdb8f423c450e449fb059a091b77fe54fe86f68aa88b4ae'
def inventory():
 out=[('global_cmvn.mean',[400]),('global_cmvn.istd',[400])]
 def affine(name,ni,no):out.extend([(name+'.linear.weight',[no,ni]),(name+'.linear.bias',[no])])
 affine('backbone.in_linear1',400,140);affine('backbone.in_linear2',140,250)
 for i in range(4):
  p=f'backbone.fsmn.{i}';out.extend([(p+'.0.linear.weight',[128,250]),(p+'.1.conv_left.weight',[128,1,10,1]),(p+'.1.conv_right.weight',[128,1,2,1])]);affine(p+'.2',128,250)
 affine('backbone.out_linear1',250,140);affine('backbone.out_linear2',140,2599);return out
def load(manifest,payload):
 import math
 manifest=pathlib.Path(manifest);payload=pathlib.Path(payload)
 if manifest.stat().st_size>32768 or payload.stat().st_size!=3027732:raise ValueError('size')
 m=json.loads(manifest.read_text());raw=payload.read_bytes()
 if type(m.get('version'))is not int or m['version']!=1 or m.get('source_sha256')!=SOURCE:raise ValueError('identity')
 if m.get('payload_sha256')!=PAYLOAD_SHA256:raise ValueError('canonical donor payload')
 if type(m.get('payload_bytes'))is not int or m['payload_bytes']!=len(raw) or hashlib.sha256(raw).hexdigest()!=m.get('payload_sha256'):raise ValueError('payload')
 expected=inventory();ts=m.get('tensors');offset=0
 if not isinstance(ts,list) or len(ts)!=len(expected):raise ValueError('inventory')
 for t,(name,shape) in zip(ts,expected):
  n=math.prod(shape)*4
  if t.get('name')!=name or t.get('shape')!=shape or any(type(x)is not int for x in t['shape']):raise ValueError('shape/name')
  if type(t.get('offset'))is not int or t['offset']!=offset or type(t.get('bytes'))is not int or t['bytes']!=n:raise ValueError('layout')
  if hashlib.sha256(raw[offset:offset+n]).hexdigest()!=t.get('sha256'):raise ValueError('tensor hash')
  offset+=n
 if offset!=len(raw) or not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',raw)):raise ValueError('finite/layout')
 return raw
