"""Lossless bounded transport of the independently reviewed runtime lock."""
import argparse,base64,hashlib,json,lzma,pathlib
HERE=pathlib.Path(__file__).resolve().parent
RAW_SIZE=734574
RAW_SHA256='c9a2a0db05af4d90861cbab7f06837007a74c6a3b82d255b9fad231e0bc00406'
def read_bytes():
 encoded=(HERE/'runtime-lock.json.xz.b64').read_bytes()
 if len(encoded)>95000:raise ValueError('Encoded lock exceeds transport bound')
 packed=base64.b64decode(encoded.strip(),validate=True)
 decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=64*1024**2)
 raw=decoder.decompress(packed,max_length=RAW_SIZE+1)
 if not decoder.eof or decoder.unused_data or len(raw)!=RAW_SIZE:raise ValueError('Invalid runtime lock framing or size')
 if hashlib.sha256(raw).hexdigest()!=RAW_SHA256:raise ValueError('Reviewed runtime lock identity differs')
 return raw
def load():return json.loads(read_bytes())
def materialize(path):
 path=pathlib.Path(path)
 with path.open('xb') as output:output.write(read_bytes())
 return path
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True)
 materialize(parser.parse_args().output)
