#!/usr/bin/env python3
"""Opt-in R1 numerical implementations; never changes the default FP32 port.

wide32 improves reductions but retains the original parity FAIL. reference64
is a separate all-binary64 oracle implementation, not a relabeling of that FAIL.
Neither precision choice grants acoustic, INT8 or physical-device qualification.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TWIDDLE_SHA = 'ae77105d27a2ffa59ba2e95f4f403a651e7f461db8c261957caa0e352c9c519d'
TWIDDLE_FILE = HERE / 'twiddles64.b64'

FFT = '''/* Binary64 scratch FFT; input/output type follows the selected mode. */
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53, "IEEE binary64 required");
#include "vendor/fft_twiddles_f64.h"
static void fft512(kcf_fbank *s) {
    double re[512], im[512];
    for(unsigned i=0;i<512;i++){re[i]=s->re[i];im[i]=s->im[i];}
    unsigned j=0;
    for(unsigned i=1;i<512;i++) {
        unsigned bit=256;
        while((j&bit)!=0){j^=bit;bit>>=1;}
        j^=bit;
        if(i<j){double v=re[i];re[i]=re[j];re[j]=v;v=im[i];im[i]=im[j];im[j]=v;}
    }
    for(unsigned len=2;len<=512;len<<=1)
        for(unsigned i=0;i<512;i+=len)
            for(unsigned k=0;k<len/2;k++) {
                unsigned a=i+k,b=a+len/2,t=k*(512/len);
                double wr=r1_fft_cos64[t],wi=r1_fft_sin64[t];
                double vr=re[b]*wr-im[b]*wi,vi=re[b]*wi+im[b]*wr;
                double ur=re[a],ui=im[a];
                re[a]=ur+vr;im[a]=ui+vi;re[b]=ur-vr;im[b]=ui-vi;
            }
    for(unsigned i=0;i<512;i++){s->re[i]=(float)re[i];s->im[i]=(float)im[i];}
}'''
AFFINE = '''/* Wider accumulation; precision mode defines activation boundaries. */
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53, "IEEE binary64 required");
static void affine(const float *x, float *y, const float *w, const float *b, int ni, int no) {
    for(int o=0;o<no;o++) {
        double sum=0.0;
        for(int i=0;i<ni;i++)sum+=(double)x[i]*(double)w[o*ni+i];
        y[o]=(float)(sum+(b?(double)b[o]:0.0));
    }
}
'''

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def replace_once(text, old, new):
    require(text.count(old) == 1, 'upstream source shape changed: ' + old[:64])
    return text.replace(old, new)

def twiddles(prefix='r1'):
    with TWIDDLE_FILE.open('rb') as stream:
        text = stream.read(4097)
    require(len(text) <= 4096, 'bounded numerical constant source')
    encoded = base64.b64decode(text.strip(), validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(encoded, 4097)
    require(len(raw) == 4096 and decoder.eof and not decoder.unused_data
            and not decoder.unconsumed_tail, 'bounded numerical constant payload')
    require(hashlib.sha256(raw).hexdigest() == TWIDDLE_SHA, 'numerical constant identity')
    values = struct.unpack('<512d', raw)
    rows = ['/* SPDX-License-Identifier: Apache-2.0; fixed exact binary64 constants. */']
    for offset, name in ((0, 'cos'), (256, 'sin')):
        rows.append(f'static const double {prefix}_fft_{name}64[256] = {{')
        for i in range(offset, offset + 256, 4):
            rows.append('    ' + ', '.join(v.hex() for v in values[i:i+4]) + ',')
        rows.append('};')
    return '\n'.join(rows) + '\n'

def wide_texts(source):
    names = ('canonical_fbank.c','canonical_fbank.h','r1_pcm.c','r1_pcm.h',
             'r1_stream.c','r1_stream.h','r1fsmn.c','r1fsmn.h','r1_pcm_cli.c')
    texts = {name: (source / name).read_text() for name in names}
    text = texts['canonical_fbank.c']
    begin, end = text.index('static void fft512('), text.index('\nint kcf_fbank_init')
    text = text[:begin] + FFT + text[end:]
    texts['canonical_fbank.c'] = replace_once(text, '#include "vendor/fft_twiddles.h"',
                                             '/* Fixed binary64 twiddles below. */')
    text = texts['r1fsmn.c']
    begin, end = text.index('#ifdef KWS_AFFINE_BLOCK2'), text.index('\nstatic int finite_values')
    text = text[:begin] + AFFINE + text[end:]
    text = replace_once(text, 'float left = 0.0f;', 'double left_sum = 0.0;')
    text = replace_once(text,
        'left += w[offset + 32000u + (size_t)c * 10u + (size_t)k]\n                    * m->cache[',
        'left_sum += (double)w[offset + 32000u + (size_t)c * 10u + (size_t)k]\n                    * (double)m->cache[')
    text = replace_once(text, 'float right = w[offset + 33280u + (size_t)c * 2u]',
        'float left = (float)left_sum;\n        double right_sum = (double)w[offset + 33280u + (size_t)c * 2u]')
    text = replace_once(text,
        '* m->cache[((size_t)c * 11u + 10u) * 4u + (size_t)layer]\n                      + w[offset + 33281u + (size_t)c * 2u] * p[c];',
        '* (double)m->cache[((size_t)c * 11u + 10u) * 4u + (size_t)layer]\n                      + (double)w[offset + 33281u + (size_t)c * 2u] * (double)p[c];\n        float right = (float)right_sum;')
    texts['r1fsmn.c'] = text
    return texts

def reference_text(name, text):
    # Types change only in the separately generated ABI; packed weights stay FP32.
    text = re.sub(r'\bfloat\b', 'double', text).replace('#include <double.h>', '#include <float.h>')
    if name == 'canonical_fbank.c':
        text = text.replace('sizeof(double)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24',
                            'sizeof(double)==8 && FLT_RADIX==2 && DBL_MANT_DIG==53').replace('<=4608','<=9216')
        text = text.replace('400.0f','400.0').replace('0.97f','0.97')
        text = text.replace('hypotf(','hypot(').replace('logf(','log(').replace('fmaxf(','fmax(')
    if name == 'r1_pcm.c':
        text = text.replace('<= 8192','<= 16384')
    if name == 'r1_stream.c':
        text = text.replace('double *weights;', 'float *weights;').replace('s->weights=(double *)malloc','s->weights=(float *)malloc')
        text = replace_once(text, 'memcpy(s->table_data,tables,table_bytes);',
            'for(unsigned i=0;i<R1_TABLE_FLOATS;i++){float v;memcpy(&v,tables+4u*i,4u);s->table_data[i]=(double)v;}')
    if name == 'r1fsmn.h':
        text = text.replace('const double *weights;', 'const float *weights;').replace('double *decoded_weights', 'float *decoded_weights')
    if name == 'r1fsmn.c':
        text = text.replace('sizeof(double) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24',
                            'sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24')
        text = text.replace('volatile double tiny = FLT_MIN','volatile double tiny = DBL_MIN').replace('recovered == FLT_MIN','recovered == DBL_MIN')
        text = text.replace('r1fsmn_load(r1fsmn_model *m, double *w','r1fsmn_load(r1fsmn_model *m, float *w')
        text = replace_once(text, 'double value;\n        memcpy(&value, payload + i * 4u, 4u);',
                            'float value;\n        memcpy(&value, payload + i * 4u, 4u);')
        text = text.replace('const double *w','const float *w').replace('const double *b, int ni','const float *b, int ni')
    return rename_reference(text)

def rename_reference(text):
    for before, after in (('r1fsmn','r1dfsmn'),('R1FSMN','R1DFSMN'),
            ('r1_','r1d_'),('R1_','R1D_'),('canonical_fbank','reference64_fbank'),
            ('kcf_','r1d_fbank_'),('KCF_','R1DCF_'),('kws_sha256_','r1d_sha256_'),
            ('KWS_CANONICAL_PCM_H','R1D_REFERENCE_PCM_H'),
            ('KWS_CANONICAL_FBANK_H','R1D_REFERENCE_FBANK_H'),
            ('0x52315354','0x52324436'),('0x52313236','0x5232444e'),
            ('0x4b464231','0x4b463634'),('0x4b434631','0x4b433634')):
        text = text.replace(before, after)
    return text

def materialize(repo, output, mode, model_package=None):
    require(mode in ('wide32','reference64'), 'explicit supported precision mode required')
    repo, output = Path(repo).resolve(), Path(output).absolute()
    require(not output.exists() and not any(p.is_symlink() for p in (output,*output.parents)),
            'fresh real output directory required')
    spec = importlib.util.spec_from_file_location('r1_numeric_base_port', HERE / 'port.py')
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)  # Existing source exporter definitions only.
    with tempfile.TemporaryDirectory(prefix='r1-numeric-source-') as temporary:
        base = Path(temporary) / 'base'
        port.prepare_sources(repo, base)
        # Validate/export through the unchanged trusted owner-package gate, before publishing output.
        if model_package is not None:
            port.export_weights(Path(model_package), base)
        texts = wide_texts(base / 'source')
        output.mkdir(parents=True)
        source = output / 'source'
        source.mkdir(); (source / 'vendor').mkdir()
        reference = mode == 'reference64'
        for name, text in texts.items():
            target = rename_reference(name) if reference else name
            rendered = reference_text(name,text) if reference else text
            (source / target).write_text('/* Opt-in R1 numeric research variant; legacy FP32 gate unchanged. */\n' + rendered)
        for name in ('sha256.c','sha256.h','model_identity.h'):
            text = (base / 'source' / name).read_text()
            (source / name).write_text(rename_reference(text) if reference else text)
        for name in ('LICENSE','NOTICE.md','LICENSE.torchaudio'):
            shutil.copyfile(base / 'source' / name, source / name)
        (source / 'vendor/fft_twiddles_f64.h').write_text(twiddles('r1d' if reference else 'r1'))
        build = (base / 'build.sh').read_text()
        (output / 'build.sh').write_text(rename_reference(build) if reference else build)
        shutil.copytree(base / 'weights', output / 'weights')
    status = {'schema':'kws-r1-optin-numerics-v1','mode':mode,
              'legacy_fp32_gate':'FAIL_UNCHANGED','reference_gate':'NOT_RUN_FOR_THIS_BUILD',
              'acoustic_qualification':False,'device_qualification':False,'model_forwards':0,
              'weights_present':model_package is not None,'default_runtime_changed':False,
              'precision_scope':'separate binary64 reference' if reference else 'wider reductions with FP32 boundaries'}
    (output / 'NUMERICAL_STATUS.json').write_text(json.dumps(status,sort_keys=True,indent=2)+'\n')
    files = {p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)}
             for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'SOURCE-MANIFEST.json').write_text(json.dumps({'schema':'kws-r1-optin-numeric-source-v1',
          'mode':mode,'files':files,'twiddle_sha256':TWIDDLE_SHA},sort_keys=True,indent=2)+'\n')
    return status

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=HERE.parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('wide32','reference64'),required=True)
    parser.add_argument('--model-package',type=Path)
    args=parser.parse_args()
    try:
        result=materialize(args.repo,args.output,args.mode,args.model_package)
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(2,str(exc)+'\n')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':
    main()
