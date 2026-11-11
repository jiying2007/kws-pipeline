#!/usr/bin/env python3
"""Opt-in CPU output-blocking experiment on source-bound R1 qhead8/par4 view SDK.

No network, model execution, weights, audio generation, threshold sweep or promotion.
The source-bound borrowed SDK is materialized into a fresh staging directory.
Only the classification-head function is exchanged; reference paths remain intact.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BORROWED=HERE.parent/'borrowed_weights'
QROOT=HERE.parent/'quant_head8'

BLOCK4_HEAD=r'''/* Research-only output-blocked, double-precision par4 kernel.
 * Each individual output retains the original par4 accumulation tree.
 * Do not enable reassociation / fast math; full-model parity is NOT_RUN here.
 */
static void affine_qhead(const r1dfsmn_model *m, const double *x, double *y) {
    static const unsigned precise_ids[5]={0,1027,1401,1462,2093};
    unsigned o=0u, exact=0u;
    while (o<R1DFSMN_OUTPUTS) {
        if (exact<5u && o==precise_ids[exact]) {
            const float *row=m->precise_head+(size_t)exact*140u;
            double sum=0.0;
            for (unsigned i=0;i<140u;i++) sum+=x[i]*(double)row[i];
            y[o]=sum+(double)m->biases[o];
            ++exact;
            ++o;
            continue;
        }
        const unsigned stop=exact<5u ? precise_ids[exact] : R1DFSMN_OUTPUTS;
        while (o<stop && stop-o>=4u) {
        const int8_t *r0=m->qhead+(size_t)(o+0u)*140u;
        const int8_t *r1=m->qhead+(size_t)(o+1u)*140u;
        const int8_t *r2=m->qhead+(size_t)(o+2u)*140u;
        const int8_t *r3=m->qhead+(size_t)(o+3u)*140u;
        double s00=0.0,s01=0.0,s02=0.0,s03=0.0;
        double s10=0.0,s11=0.0,s12=0.0,s13=0.0;
        double s20=0.0,s21=0.0,s22=0.0,s23=0.0;
        double s30=0.0,s31=0.0,s32=0.0,s33=0.0;
            for (unsigned i=0;i<140u;i+=4u) {
                const double x0=x[i],x1=x[i+1u],x2=x[i+2u],x3=x[i+3u];
            s00+=x0*(double)r0[i];
            s01+=x1*(double)r0[i+1u];
            s02+=x2*(double)r0[i+2u];
            s03+=x3*(double)r0[i+3u];
            s10+=x0*(double)r1[i];
            s11+=x1*(double)r1[i+1u];
            s12+=x2*(double)r1[i+2u];
            s13+=x3*(double)r1[i+3u];
            s20+=x0*(double)r2[i];
            s21+=x1*(double)r2[i+1u];
            s22+=x2*(double)r2[i+2u];
            s23+=x3*(double)r2[i+3u];
            s30+=x0*(double)r3[i];
            s31+=x1*(double)r3[i+1u];
            s32+=x2*(double)r3[i+2u];
            s33+=x3*(double)r3[i+3u];
            }
        double z0=(s00+s01)+(s02+s03);
        z0*=m->scales[o+0u];
        y[o+0u]=z0+(double)m->biases[o+0u];
        double z1=(s10+s11)+(s12+s13);
        z1*=m->scales[o+1u];
        y[o+1u]=z1+(double)m->biases[o+1u];
        double z2=(s20+s21)+(s22+s23);
        z2*=m->scales[o+2u];
        y[o+2u]=z2+(double)m->biases[o+2u];
        double z3=(s30+s31)+(s32+s33);
        z3*=m->scales[o+3u];
        y[o+3u]=z3+(double)m->biases[o+3u];
            o+=4u;
        }
        while(o<stop) {
            const int8_t *row=m->qhead+(size_t)o*140u;
            double s0=0.0,s1=0.0,s2=0.0,s3=0.0;
            for (unsigned i=0;i<140u;i+=4u) {
                s0+=x[i]*(double)row[i];
                s1+=x[i+1u]*(double)row[i+1u];
                s2+=x[i+2u]*(double)row[i+2u];
                s3+=x[i+3u]*(double)row[i+3u];
            }
            double sum=(s0+s1)+(s2+s3);
            sum*=m->scales[o];
            y[o]=sum+(double)m->biases[o];
            ++o;
        }
    }
}

'''

BLOCK2_HEAD=r'''/* Research-only two-output blocked par4; same five exact rows and FP64 sum tree. */
static void affine_qhead(const r1dfsmn_model *m, const double *x, double *y) {
    static const unsigned precise_ids[5]={0,1027,1401,1462,2093};
    unsigned o=0u, exact=0u;
    while (o<R1DFSMN_OUTPUTS) {
        if (exact<5u && o==precise_ids[exact]) {
            const float *row=m->precise_head+(size_t)exact*140u;
            double sum=0.0;
            for (unsigned i=0;i<140u;i++) sum+=x[i]*(double)row[i];
            y[o]=sum+(double)m->biases[o];
            ++exact;
            ++o;
            continue;
        }
        const unsigned stop=exact<5u ? precise_ids[exact] : R1DFSMN_OUTPUTS;
        while (o<stop && stop-o>=2u) {
        const int8_t *r0=m->qhead+(size_t)(o+0u)*140u;
        const int8_t *r1=m->qhead+(size_t)(o+1u)*140u;
        double s00=0.0,s01=0.0,s02=0.0,s03=0.0;
        double s10=0.0,s11=0.0,s12=0.0,s13=0.0;
            for (unsigned i=0;i<140u;i+=4u) {
                const double x0=x[i],x1=x[i+1u],x2=x[i+2u],x3=x[i+3u];
                s00+=x0*(double)r0[i];
                s01+=x1*(double)r0[i+1u];
                s02+=x2*(double)r0[i+2u];
                s03+=x3*(double)r0[i+3u];
                s10+=x0*(double)r1[i];
                s11+=x1*(double)r1[i+1u];
                s12+=x2*(double)r1[i+2u];
                s13+=x3*(double)r1[i+3u];
            }
            double z0=(s00+s01)+(s02+s03);
            z0*=m->scales[o+0u];
            y[o+0u]=z0+(double)m->biases[o+0u];
            double z1=(s10+s11)+(s12+s13);
            z1*=m->scales[o+1u];
            y[o+1u]=z1+(double)m->biases[o+1u];
            o+=2u;
        }
        while(o<stop) {
            const int8_t *row=m->qhead+(size_t)o*140u;
            double s0=0.0,s1=0.0,s2=0.0,s3=0.0;
            for (unsigned i=0;i<140u;i+=4u) {
                s0+=x[i]*(double)row[i];
                s1+=x[i+1u]*(double)row[i+1u];
                s2+=x[i+2u]*(double)row[i+2u];
                s3+=x[i+3u]*(double)row[i+3u];
            }
            double sum=(s0+s1)+(s2+s3);
            sum*=m->scales[o];
            y[o]=sum+(double)m->biases[o];
            ++o;
        }
    }
}

'''

def digest(data):
    return hashlib.sha256(data).hexdigest()

def module(path,name):
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ValueError('missing source-bound dependency: '+str(path))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def materialize(repo, output, model=None, frontend=None, kernel='block4'):
    if kernel not in ('block2','block4'):
        raise ValueError('unsupported CPU kernel variant')
    candidate=BLOCK2_HEAD if kernel=='block2' else BLOCK4_HEAD
    repo=Path(repo).resolve()
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('fresh non-symlink output required')
    borrowed=module(BORROWED/'materialize.py','r1_cpu_bound_borrowed')
    quant=module(QROOT/'materialize.py','r1_cpu_bound_quant')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.r1-cpuopt-',dir=output.parent) as tmp:
        sdk=Path(tmp)/'sdk'
        prior=borrowed.materialize(repo,sdk,model,frontend)
        path=sdk/'core/r1dfsmn.c'
        old=path.read_text()
        original=quant.Q8_HEAD_PAR4
        if old.count(original)!=1:
            raise ValueError('source kernel drift or ambiguous reference; refusing CPU transform')
        if old.count('static void affine_qhead(')!=1:
            raise ValueError('unexpected classifier-head entry count')
        updated=old.replace(original,candidate,1)
        path.write_text(updated)
        old_report_path=sdk/'BORROWED_PREPARATION.json'
        old_report=json.loads(old_report_path.read_text())
        before=old_report['changed_source_sha256']['core/r1dfsmn.c']
        if before != digest(old.encode()):
            raise ValueError('borrowed source hash mismatch')
        old_report['changed_source_sha256']['core/r1dfsmn.c']=digest(updated.encode())
        old_report['kernel']='par4_output_'+kernel+'_experimental'
        old_report['arithmetic_changed']='output scheduling only; per-class par4 sum tree preserved'
        old_report['frame_event_clock_parity']='NOT_RUN_FOR_THIS_BUILD'
        old_report_path.write_text(json.dumps(old_report,indent=2,sort_keys=True)+'\n')
        receipt={
            'schema':'r1-cpu-output-block4-source-v1',
            'source_model_id':prior['source_model_id'],
            'model_sha256':prior['q8_model_sha256'],
            'model_bytes':prior['q8_model_bytes'],
            'input_kernel':'qhead8_par4_borrowed_immutable',
            'candidate_kernel':'qhead8_par4_output_'+kernel+'_experimental',
            'reference_head_sha256':digest(original.encode()),
            'candidate_head_sha256':digest(candidate.encode()),
            'source_r1dfsmn_sha256_before':before,
            'source_r1dfsmn_sha256_after':digest(updated.encode()),
            'private_model_packed':prior['private_model_packed'],
            'arithmetic_tree_preserved_per_output':True,
            'cpu_speedup':'NOT_MEASURED',
            'full_model_parity':'NOT_RUN_FOR_THIS_BUILD',
            'physical_ssc305':'NOT_RUN',
            'independent_far_frr':'NOT_RUN',
            'legacy_fp32_numeric_gate':'FAIL_UNCHANGED',
            'model_forwards':0,
        }
        (sdk/'CPU_OPT_PREPARATION.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        (sdk/'README-CPU-OPT.md').write_bytes((HERE/'README.md').read_bytes())
        if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
            raise ValueError('output appeared during staging')
        sdk.rename(output)
    return receipt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=ROOT)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',type=Path)
    ap.add_argument('--frontend',type=Path)
    ap.add_argument('--kernel',choices=('block2','block4'),default='block4')
    args=ap.parse_args()
    print(json.dumps(materialize(args.repo,args.output,args.model,args.frontend,args.kernel),sort_keys=True))

if __name__=='__main__':
    main()
