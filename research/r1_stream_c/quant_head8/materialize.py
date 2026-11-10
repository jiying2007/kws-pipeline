#!/usr/bin/env python3
"""Construct an isolated, opt-in, R1 qhead8 SDK from byte-pinned original C source.

Never overwrites the original reference64 SDK. Uses an exact source identity and
requires the original model only when explicitly requested for owner-local packing.
No training, ASR, TTS, network or model inference is invoked.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import stat
import zipfile
from pathlib import PurePosixPath

HERE=Path(__file__).resolve().parent
SOURCES={
'core/r1dfsmn.c':'09c4fbd77f4ad12917877edd356b7bb31315bdd68d132fb1e5a91af3d13f1a00',
'core/r1dfsmn.h':'77da63e236bf72329af5aa0a89d27a1f83ca4b0fa635d2be34ddc935554bce03',
'core/r1d_stream.c':'661fddc4b0a06562e7d9b837679d9f21380e4b8d1f9e01de856a9d22467c58c4',
'core/model_identity.h':'6f267c817901c2a8b6bb7734a6684c4b5ae2fb65ded519e66df05521237e6678',
'include/fsmn_sdk.h':'a55af158ff1f43b72e3cc5ae438064a37159fb5b90774f110a406c2d87ab7628'
}
PACK_SHA='63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4'
ORIGINAL_SHA='e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb'
QBYTES=1960140
SOURCE_ARCHIVE_REL=('research/r1_stream_c/deploy/archive-20261110/'
                    'KWS_FSMN_R1_SOURCE_20261110.zip')
SOURCE_ARCHIVE_SHA='2785c8377c2cd23754673cc8cbbe09d0ceba637ce450c68a5262d7809f9ae102'
SOURCE_ARCHIVE_PREFIX='KWS_FSMN_R1_SOURCE_20261110/tree/research/r1_deploy_sdk/'

def exactly(s, old, new):
    if s.count(old)!=1:raise ValueError('pinned C source transform missing or ambiguous: '+old[:90])
    return s.replace(old,new,1)

def patch_model_header(s):
    s=exactly(s,'#define R1DFSMN_PAYLOAD_BYTES (R1DFSMN_FLOATS * 4u)', '''#define R1DFSMN_PAYLOAD_BYTES 1960140u
#define R1DFSMN_Q8_PACKAGE_BYTES 1960140u
#define R1DFSMN_Q8_PREFIX_FLOATS 390474u
#define R1DFSMN_Q8_HEADER_BYTES 88u
#define R1DFSMN_Q8_QUANT_OFFSET (R1DFSMN_Q8_HEADER_BYTES + R1DFSMN_Q8_PREFIX_FLOATS * 4u)
#define R1DFSMN_Q8_PAD_OFFSET (R1DFSMN_Q8_QUANT_OFFSET + 2601u * 140u)
#define R1DFSMN_Q8_SCALE_OFFSET (R1DFSMN_Q8_PAD_OFFSET + 4u)
#define R1DFSMN_Q8_BIAS_OFFSET (R1DFSMN_Q8_SCALE_OFFSET + 2601u * 8u)
#define R1DFSMN_Q8_PRECISE_OFFSET (R1DFSMN_Q8_BIAS_OFFSET + 2601u * 4u)''')
    s=exactly(s,'''const float *weights; /* caller-owned, immutable throughout model lifetime */''','''const float *weights; /* subset of verified compact immutable payload */
    const int8_t *qhead;
    const double *scales;
    const float *biases;
    const float *precise_head;''')
    return s

Q8_LOADER = r'''r1dfsmn_status r1dfsmn_load(r1dfsmn_model *m, float *w, size_t nw,
        const unsigned char *payload, size_t bytes, unsigned order, const char hash[65]) {
    if (!m || !w || !payload || !hash) return R1DFSMN_ARGUMENT;
    if (nw != R1DFSMN_Q8_PACKAGE_BYTES / 4u || bytes != R1DFSMN_Q8_PACKAGE_BYTES)
        return R1DFSMN_SIZE;
    if (order != R1DFSMN_PAYLOAD_LITTLE_ENDIAN || !r1dfsmn_environment_ok())
        return R1DFSMN_ENVIRONMENT;
    if (overlaps(m, sizeof(*m), w, R1DFSMN_Q8_PACKAGE_BYTES)
        || overlaps(m, sizeof(*m), payload, bytes)
        || overlaps(w, R1DFSMN_Q8_PACKAGE_BYTES, payload, bytes)
        || overlaps(m, sizeof(*m), hash, 65)
        || overlaps(w, R1DFSMN_Q8_PACKAGE_BYTES, hash, 65)) return R1DFSMN_OVERLAP;
    for (unsigned i = 0; i < 64; ++i)
        if (!((hash[i] >= '0' && hash[i] <= '9') || (hash[i] >= 'a' && hash[i] <= 'f')))
            return R1DFSMN_IDENTITY;
    if (hash[64] != '\0') return R1DFSMN_IDENTITY;
    char actual[65];
    r1d_sha256_memory_hex(payload, bytes, actual);
    if (memcmp(actual, hash, 65) != 0) return R1DFSMN_IDENTITY;
    /* Immutable source identity is bound inside the byte-hashed package. */
    static const unsigned char magic[8] = {'R','1','H','Q','8','V','1',0};
    static const char base_hash[65] =
        "e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb";
    if (memcmp(payload, magic, 8) || memcmp(payload + 24, base_hash, 64))
        return R1DFSMN_IDENTITY;
    static const uint32_t fields[4] = {1, 2601, 140, 5};
    for (unsigned i = 0; i < 4; i++) {
        unsigned n = (unsigned)payload[8+i*4] | ((unsigned)payload[9+i*4]<<8)
            | ((unsigned)payload[10+i*4]<<16) | ((unsigned)payload[11+i*4]<<24);
        if (n != fields[i]) return R1DFSMN_IDENTITY;
    }
    for (unsigned i = 0; i < 4; i++)
        if (payload[R1DFSMN_Q8_PAD_OFFSET+i] != 0) return R1DFSMN_IDENTITY;
    for (size_t i = 0; i < R1DFSMN_Q8_PREFIX_FLOATS; ++i) {
        float value;
        memcpy(&value, payload + R1DFSMN_Q8_HEADER_BYTES + i*4u, 4u);
        if (!isfinite(value)) return R1DFSMN_NONFINITE;
        if (i >= 400 && i < 800 && !(value > 0.0f)) return R1DFSMN_ARGUMENT;
    }
    for (size_t i = 0; i < 2601u*140u; ++i)
        if ((int8_t)payload[R1DFSMN_Q8_QUANT_OFFSET+i] == INT8_MIN)
            return R1DFSMN_ARGUMENT;
    for (size_t i = 0; i < 2601; ++i) {
        double scale;
        float bias;
        memcpy(&scale,payload+R1DFSMN_Q8_SCALE_OFFSET+i*8u,8u);
        memcpy(&bias,payload+R1DFSMN_Q8_BIAS_OFFSET+i*4u,4u);
        if (!(scale>0.0) || !isfinite(scale) || !isfinite(bias))
            return R1DFSMN_NONFINITE;
    }
    for (size_t i = 0; i < 5u*140u; ++i) {
        float value;
        memcpy(&value,payload+R1DFSMN_Q8_PRECISE_OFFSET+i*4u,4u);
        if (!isfinite(value)) return R1DFSMN_NONFINITE;
    }
    /* Allocate once outside the stream data path; all views are aligned. */
    memcpy(w,payload,bytes);
    unsigned char *raw=(unsigned char *)w;
    m->weights=(const float *)(raw+R1DFSMN_Q8_HEADER_BYTES);
    m->qhead=(const int8_t *)(raw+R1DFSMN_Q8_QUANT_OFFSET);
    m->scales=(const double *)(raw+R1DFSMN_Q8_SCALE_OFFSET);
    m->biases=(const float *)(raw+R1DFSMN_Q8_BIAS_OFFSET);
    m->precise_head=(const float *)(raw+R1DFSMN_Q8_PRECISE_OFFSET);
    memset(m->cache,0,sizeof(m->cache));
    m->magic=R1DFSMN_MAGIC;m->fault=0;
    return R1DFSMN_OK;
}
'''
Q8_HEAD = '''/* Five critical R1 blank/keyword classes retain exact FP32 head rows. */
static void affine_qhead(const r1dfsmn_model *m, const double *x, double *y) {
    static const unsigned precise_ids[5] = {0,1027,1401,1462,2093};
    for (unsigned o=0;o<R1DFSMN_OUTPUTS;o++) {
        unsigned exact=5;
        for (unsigned k=0;k<5;k++) if (precise_ids[k]==o) {exact=k;break;}
        double sum=0.0;
        if (exact<5) {
            const float *row=m->precise_head+(size_t)exact*140u;
            for (unsigned i=0;i<140;i++) sum+=x[i]*(double)row[i];
        } else {
            const int8_t *row=m->qhead+(size_t)o*140u;
            for (unsigned i=0;i<140;i++) sum+=x[i]*(double)row[i];
            sum*=m->scales[o];
        }
        y[o]=sum+(double)m->biases[o];
    }
}

'''
Q8_HEAD_PAR4 = Q8_HEAD.replace(
    """for (unsigned i=0;i<140;i++) sum+=x[i]*(double)row[i];
            sum*=m->scales[o];""",
    """double s0=0.0,s1=0.0,s2=0.0,s3=0.0;
            for (unsigned i=0;i<140;i+=4) {
                s0+=x[i]*(double)row[i];
                s1+=x[i+1]*(double)row[i+1];
                s2+=x[i+2]*(double)row[i+2];
                s3+=x[i+3]*(double)row[i+3];
            }
            sum=(s0+s1)+(s2+s3);
            sum*=m->scales[o];""")
assert Q8_HEAD_PAR4 != Q8_HEAD

def patch_model_source(s, kernel='scalar'):
    s=exactly(s,'#define R1DFSMN_MAGIC UINT32_C(0x5232444e)',
              '#define R1DFSMN_MAGIC UINT32_C(0x5232444e)\n'
              '_Static_assert(R1DFSMN_Q8_PRECISE_OFFSET + 5u*140u*4u == R1DFSMN_Q8_PACKAGE_BYTES,\n'
              '               "quant package byte layout mismatch");\n'
              '_Static_assert(R1DFSMN_Q8_SCALE_OFFSET%8u == 0, "unaligned scales");')
    s=exactly(s,'return m && m->magic == R1DFSMN_MAGIC && m->weights;',
              'return m && m->magic == R1DFSMN_MAGIC && m->weights && m->qhead && m->scales && m->biases && m->precise_head;')
    begin=s.index('r1dfsmn_status r1dfsmn_load(')
    end=s.index('r1dfsmn_status r1dfsmn_reset(',begin)
    s=s[:begin]+Q8_LOADER+'\n'+s[end:]
    s=exactly(s,'static int finite_values(const double *x, size_t count) {',
              (Q8_HEAD if kernel=='scalar' else Q8_HEAD_PAR4)+'static int finite_values(const double *x, size_t count) {')
    s=exactly(s,'''affine(a, y, w + R1DFSMN_HEAD_WEIGHT_OFFSET, w + R1DFSMN_HEAD_BIAS_OFFSET,
           140, R1DFSMN_OUTPUTS);''','''affine_qhead(m, a, y);''')
    return s

def patched_text(name,raw,kernel="scalar"):
    s=raw.decode('utf-8')
    if name=='core/r1dfsmn.c':return patch_model_source(s,kernel)
    if name=='core/r1dfsmn.h':return patch_model_header(s)
    if name=='core/r1d_stream.c':
        s=exactly(s,'s->weights,R1DFSMN_FLOATS,payload,bytes,1,R1D_MODEL_SHA',
                  's->weights,R1DFSMN_Q8_PACKAGE_BYTES/4u,payload,bytes,1,R1D_MODEL_SHA')
        return s
    if name=='core/model_identity.h':
        return exactly(s,ORIGINAL_SHA,PACK_SHA)
    if name=='include/fsmn_sdk.h':
        s=exactly(s,'#define FSMN_MODEL_BYTES 3028860u','#define FSMN_MODEL_BYTES 1960140u')
        return s
    raise ValueError('unexpected target')

def materialize(base:Path,dest:Path,model:Path|None=None,
                frontend:Path|None=None,kernel:str='scalar'):
    if kernel not in ('scalar','par4'):
        raise ValueError('unknown kernel must not be interpreted as default')
    if dest.exists() or any(p.is_symlink() for p in (dest,*dest.parents)):
        raise ValueError('fresh non-symlink output required')
    base=base.resolve()
    if base.is_symlink() or not base.is_dir():raise ValueError('invalid source root')
    for path,wanted in SOURCES.items():
        p=base/path
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:
            raise ValueError('source identity differs: '+path)
    if model is not None and frontend is None:
        raise ValueError('frontend required when supplying original weights')
    if model is None and frontend is not None:
        raise ValueError('model required when supplying frontend')
    if model is not None:
        if hashlib.sha256(frontend.read_bytes()).hexdigest() != '49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49':
            raise ValueError('frontend identity mismatch')
        spec=importlib.util.spec_from_file_location('r1_head8_pack',HERE/'pack_model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        packed=module.pack(model.read_bytes())
        if hashlib.sha256(packed).hexdigest()!=PACK_SHA:raise ValueError('unexpected quant package identity')
    else:
        packed=None
    # Source project is the retained byte-pinned standalone SDK, not runtime output.
    def ignore(dirname,names):
        p=Path(dirname).relative_to(base).as_posix()
        if p=='.':return ['bin','lib','weights','SHA256SUMS','FILE_MANIFEST.json','notes']
        return [n for n in names if n=='__pycache__']
    shutil.copytree(base,dest,ignore=ignore)
    for path,wanted in SOURCES.items():
        raw=(base/path).read_bytes();(dest/path).write_text(patched_text(path,raw,kernel))
    (dest/'README-QUANT8.md').write_text((HERE/'README.md').read_text())
    if packed is not None:
        p=dest/'weights';p.mkdir()
        (p/'model.r1hq8').write_bytes(packed)
        (p/'frontend.f32le').write_bytes(frontend.read_bytes())
    receipt={'schema':'r1-quant-head8-source-v1','source_model_id':'R1-DEV-FSMN-20261107',
             'source_fp32_sha256':ORIGINAL_SHA,'q8_model_sha256':PACK_SHA,
             'q8_model_bytes':QBYTES,'original_model_bytes':757215*4,
             'private_model_packed':packed is not None,'independent_acoustic_validation':'NOT_RUN',
             'physical_ssc305':'NOT_RUN','original_fp32_numeric_gate':'FAIL_UNCHANGED',
             'kernel':kernel,'source_archive_sha256':SOURCE_ARCHIVE_SHA,
             'source_identities':SOURCES}
    (dest/'Q8_PREPARATION.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt

def materialize_from_repository(repo:Path,dest:Path,model:Path|None=None,
                                frontend:Path|None=None,kernel:str='scalar'):
    """Read the existing exact public source archive; never download or infer files."""
    repo=Path(repo).resolve()
    archive=repo/SOURCE_ARCHIVE_REL
    if not archive.is_file() or archive.is_symlink():
        raise ValueError('missing regular public source archive')
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SOURCE_ARCHIVE_SHA:
        raise ValueError('original public source archive SHA256 mismatch')
    with tempfile.TemporaryDirectory(prefix='r1-qhead8-source-') as temporary:
        base=Path(temporary)/'source'
        base.mkdir()
        seen=set()
        total=0
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                name=entry.filename
                if not name.startswith(SOURCE_ARCHIVE_PREFIX):
                    continue
                relative=name[len(SOURCE_ARCHIVE_PREFIX):]
                if not relative or entry.is_dir():
                    continue
                path=PurePosixPath(relative)
                if path.is_absolute() or '..' in path.parts or str(path)!=relative or relative in seen:
                    raise ValueError('noncanonical or duplicated source ZIP member')
                mode=(entry.external_attr>>16)&0o170000
                if mode not in (0,stat.S_IFREG):
                    raise ValueError('nonregular ZIP source member')
                if entry.file_size>8*1024*1024:
                    raise ValueError('oversized single ZIP member')
                total+=entry.file_size
                if total>32*1024*1024:
                    raise ValueError('oversized source ZIP contents')
                seen.add(relative)
                dest_file=base.joinpath(*path.parts)
                dest_file.parent.mkdir(parents=True,exist_ok=True)
                content=z.read(entry)
                if len(content)!=entry.file_size:
                    raise ValueError('truncated source ZIP member')
                with dest_file.open('xb') as sink:
                    sink.write(content)
        if len(seen)<20:
            raise ValueError('incomplete source ZIP')
        return materialize(base,dest,model,frontend,kernel)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--repo',type=Path,default=HERE.parents[2],
                   help='complete pinned repository with retained source archive')
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--kernel',choices=('scalar','par4'),default='scalar')
    a.add_argument('--model',type=Path,
                   help='original owner-held model (optional, NEVER publish output weights)')
    a.add_argument('--frontend',type=Path,
                   help='original owner-held frontend (optional)')
    args=a.parse_args()
    report=materialize_from_repository(args.repo,args.output,args.model,args.frontend,args.kernel)
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
