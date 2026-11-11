#!/usr/bin/env python3
"""Materialize independent six-score ACTIVE113 SDK; never execute a model."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from active_format import (BASE,BASE_NAMES,FRONTEND_SHA,HEADER_BYTES,HERE,MODEL_SHA,OFFSETS,
    OTHER_COUNT,PACKAGE_BYTES,POLICY,PRECISE,PREFIX_FLOATS,ROOT,SELECTION_SHA,
    base_binding,header,load,need,save,screen_gate,selection,sha,source_gate,validate_package)


def names(text):
    for old,new in (('r1dfsmn','r1active113'),('R1DFSMN','R1ACTIVE113'),
        ('r1d_','r1a113_'),('R1D_','R1A113_'),('R1DCF','R1A113CF'),('r1dcf','r1a113cf'),
        ('fsmn_','active113_'),('FSMN_','ACTIVE113_')):
        text=text.replace(old,new)
    return text


def replace_once(text, old, new):
    need(text.count(old)==1,'source adaptation anchor changed: '+old[:65])
    return text.replace(old,new)


def replace_between(text, start, end, replacement):
    need(text.count(start)==1 and text.count(end)==1,'source block anchors changed')
    a=text.index(start);b=text.index(end,a);return text[:a]+replacement+'\n'+text[b:]


def layout(ids):
    offsets={'IDS_OFFSET':'ids','TRUNK_OFFSET':'trunk','QUANT_OFFSET':'quant','SCALE_OFFSET':'scales',
             'OTHER_BIAS_OFFSET':'other_bias','PRECISE_OFFSET':'precise_weights','PRECISE_BIAS_OFFSET':'precise_bias'}
    h=['/* SPDX-License-Identifier: Apache-2.0; fixed ACTIVE113 format. */',
       '#ifndef ACTIVE113_LAYOUT_H','#define ACTIVE113_LAYOUT_H','#include <stdint.h>',
       f'#define R1ACTIVE113_HEADER_BYTES {HEADER_BYTES}u',f'#define R1ACTIVE113_PREFIX_FLOATS {PREFIX_FLOATS}u',
       f'#define R1ACTIVE113_OTHER_COUNT {OTHER_COUNT}u']
    h += [f'#define R1ACTIVE113_{k} {OFFSETS[v]}u' for k,v in offsets.items()]
    h += ['extern const unsigned char r1active113_expected_header[256];',
          'extern const uint32_t r1active113_precise_ids[5];',
          'extern const uint32_t r1active113_other_ids[113];',
          '#define R1ACTIVE113_EXPECTED_HEADER r1active113_expected_header',
          '#define R1ACTIVE113_PRECISE_IDS r1active113_precise_ids',
          '#define R1ACTIVE113_OTHER_IDS r1active113_other_ids','#endif','']
    c=['/* SPDX-License-Identifier: Apache-2.0; generated immutable format constants. */',
       '#include "active113_layout.h"',
       'const unsigned char r1active113_expected_header[256]={'+','.join(str(x) for x in header())+'};',
       'const uint32_t r1active113_precise_ids[5]={'+','.join(str(x)+'u' for x in PRECISE)+'};',
       'const uint32_t r1active113_other_ids[113]={'+','.join(str(x)+'u' for x in ids)+'};','']
    return '\n'.join(h),'\n'.join(c)


def core_source(original):
    s=names(original)
    s=s.replace('Full2599 output, verified byte loader, bounded trace, explicit cache contract.',
                'Independent ACTIVE113 six-score output, strict compact loader, explicit OTHER-ID metadata.')
    assertions='''#include "active113_layout.h"
#define R1ACTIVE113_MAGIC UINT32_C(0x41313133)
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,"IEEE binary32 required");
_Static_assert(FLT_EVAL_METHOD==0,"no excess evaluation precision");
_Static_assert(R1ACTIVE113_TRUNK_OFFSET%8u==0 && R1ACTIVE113_SCALE_OFFSET%8u==0,"aligned payload views");
_Static_assert(R1ACTIVE113_PRECISE_BIAS_OFFSET+20u==R1ACTIVE113_PAYLOAD_BYTES,"compact payload layout");
'''
    s=replace_between(s,'#define R1ACTIVE113_MAGIC','int r1active113_environment_ok',assertions)
    s=replace_between(s,'static int valid','/* Wider accumulation;', (HERE/'loader.c.inc').read_text())
    s=replace_between(s,'/* Five critical R1','static int finite_values',(HERE/'head.c.inc').read_text())
    s=replace_once(s,'    affine_qhead(m, a, y);','    m->other_winner_id=affine_qhead(m, a, y);')
    need(s.count('m->weights, R1ACTIVE113_PAYLOAD_BYTES')==3,'overlap anchors')
    s=s.replace('m->weights, R1ACTIVE113_PAYLOAD_BYTES','m->payload, R1ACTIVE113_PAYLOAD_BYTES')
    return s


def stream_header(original):
    s=names(original).replace('#define R1A113_OUTPUTS 2601u','#define R1A113_OUTPUTS 6u')
    s=replace_once(s,'const double *,const double *);','const double *,const double *,uint32_t other_original_id);')
    s=replace_once(s,'const double *feature,double *logits);','const double *feature,double *scores,uint32_t *other_original_id);')
    return '/* Independent six-score ACTIVE113 stream; original OTHER ID accompanies every frame. */\n'+s


def stream_source(original):
    s=names(original).replace('distinct2601 R1 payload','distinct six-score ACTIVE113 payload')
    s=replace_once(s,'    float *weights;','    void *weights;')
    s=replace_once(s,'s->weights=(float *)malloc(R1ACTIVE113_PAYLOAD_BYTES);','s->weights=malloc(R1ACTIVE113_PAYLOAD_BYTES);')
    s=replace_once(s,'R1ACTIVE113_Q8_PACKAGE_BYTES/4u','R1ACTIVE113_PAYLOAD_BYTES')
    s=replace_once(s,'source_end,s->delivery_end,feature,s->logits);',
                   'source_end,s->delivery_end,feature,s->logits,r1active113_other_winner(&s->net));')
    s=replace_once(s,'int r1a113_feature(r1a113_stream *s,const double *feature,double *logits){',
                   'int r1a113_feature(r1a113_stream *s,const double *feature,double *logits,uint32_t *other_original_id){\n    if(!other_original_id)return R1A113CF_ARGUMENT;')
    s=replace_once(s,'    if(!rc)s->frames++;',
                   '    if(!rc){s->frames++;*other_original_id=r1active113_other_winner(&s->net);}')
    return s


def sdk_source(original):
    s=names(original).replace('ACTIVE113_LOGIT_COUNT','ACTIVE113_SCORE_COUNT')
    s=replace_once(s,'#include "model_identity.h"','#include "model_identity.h"\n#include "active113_layout.h"')
    s=replace_once(s,'#define SDK_MAGIC UINT32_C(0x46534d31)','#define SDK_MAGIC UINT32_C(0x41313153)')
    s=replace_once(s,'static int emit(', (HERE/'sdk_winner.c.inc').read_text()+'\nstatic int emit(')
    s=replace_once(s,'const double *feature,const double *logits) {',
                   'const double *feature,const double *logits,uint32_t other_original_id) {')
    old='''    unsigned best=0;
    for (unsigned i=0;i<ACTIVE113_SCORE_COUNT;i++) {
        if (!isfinite(logits[i])) return ACTIVE113_BACKEND_ERROR;
        if (logits[i]>logits[best]) best=i;
    }'''
    new='''    uint32_t best=0;
    if(active113_sdk_pick_original_token(logits,ACTIVE113_SCORE_COUNT,other_original_id,&best))
        return ACTIVE113_BACKEND_ERROR;'''
    s=replace_once(s,old,new)
    s=replace_once(s,'s->origin+delivery,feature,logits))','s->origin+delivery,feature,logits,other_original_id))')
    s=replace_once(s,'"r1-greedy-complete-bounded-v1"','"r1-active113-original-id-competition-v1"')
    s+='\nconst char *active113_sdk_selection_sha256(void){return R1A113_SELECTION_SHA;}\n'
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--pack-manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();pack=a.pack.resolve()
    need(not out.exists(),'fresh materialized SDK directory required')
    source_gate(a.plan_sha256);before=base_binding();selected,ids=selection()
    need(sha(pack/'PACK.json')==a.pack_manifest_sha256,'compact pack manifest changed')
    receipt=load(pack/'PACK.json')
    need(receipt['status']=='PACKED_NO_MODEL_EXECUTION' and receipt['preparation_plan_sha256']==a.plan_sha256 and
         receipt['selection_sha256']==SELECTION_SHA and receipt['policy']==POLICY,'pack source/policy binding')
    screen_gate(ROOT/'competitor-active-v1/execution/AGGREGATE.json',receipt['screen_result_sha256'])
    model=pack/receipt['model']['path'];need(sha(model)==receipt['model']['sha256'],'compact model changed')
    validate_package(model.read_bytes(),ids)
    need(sha(pack/receipt['frontend']['path'])==FRONTEND_SHA,'frontend changed')
    os.umask(0o077);out.mkdir(parents=True)
    for directory in ('core','src','include'):(out/directory).mkdir()
    generated={}
    def write(name,text):
        path=out/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
        generated[name]={'bytes':path.stat().st_size,'sha256':sha(path)}
    special={'core/r1dfsmn.c','core/r1dfsmn.h','core/r1d_stream.c','core/r1d_stream.h',
             'core/model_identity.h','include/fsmn_sdk.h','src/fsmn_sdk.c'}
    for name in BASE_NAMES:
        if name not in special:write(names(name),names((BASE/name).read_text()))
    write('core/r1active113.h',(HERE/'r1active113.h').read_text())
    write('core/r1active113.c',core_source((BASE/'core/r1dfsmn.c').read_text()))
    write('core/r1a113_stream.h',stream_header((BASE/'core/r1d_stream.h').read_text()))
    write('core/r1a113_stream.c',stream_source((BASE/'core/r1d_stream.c').read_text()))
    write('include/active113_sdk.h',(HERE/'active113_sdk.h').read_text())
    write('src/active113_sdk.c',sdk_source((BASE/'src/fsmn_sdk.c').read_text()))
    lh,lc=layout(ids);write('core/active113_layout.h',lh);write('core/active113_layout.c',lc)
    write('core/model_identity.h','\n'.join(['#ifndef R1A113_MODEL_IDENTITY_H','#define R1A113_MODEL_IDENTITY_H',
        '#define R1A113_MODEL_SHA "'+receipt['model']['sha256']+'"',
        '#define R1A113_FRONTEND_SHA "'+FRONTEND_SHA+'"','#define R1A113_TABLE_FLOATS 20960u',
        '#define R1A113_SELECTION_SHA "'+SELECTION_SHA+'"','#endif','']))
    need(before==base_binding(),'base SDK drift during materialization')
    save(out/'MODEL_CARD.json',{'schema':'active113-independent-six-score-model-card-v1',
        'model_id':'R1-TRAIN-ACTIVE113-20261111','public_abi':'0xA1130001','policy':POLICY,
        'scores':{'count':6,'precise_original_ids':list(PRECISE),'other_score_index':5,
                  'other_id':'explicit per-frame original winner ID, never a sentinel'},
        'original_model_sha256':MODEL_SHA,'compact_model_sha256':receipt['model']['sha256'],
        'compact_model_bytes':PACKAGE_BYTES,'selection_sha256':SELECTION_SHA,'selected_other_rows':113,
        'full_vocabulary_outputs':False,'probability_or_ctc_equivalence':False,'runtime_evidence':'NOT_EXECUTED',
        'cached_screen_pass_is_not_pcm_or_target_evidence':True,'physical_ssc305':'NOT_RUN',
        'independent_far_frr':'NOT_RUN','default_model_changed':False,'model_promotion_authorized':False})
    save(out/'MATERIALIZATION.json',{'schema':'active113-sdk-materialization-v1','status':'MATERIALIZED_NO_MODEL_EXECUTION',
        'preparation_plan_sha256':a.plan_sha256,'pack_manifest_sha256':a.pack_manifest_sha256,
        'pack_path':str(pack),'compact_model_sha256':receipt['model']['sha256'],
        'screen_result_sha256':receipt['screen_result_sha256'],'selection_sha256':SELECTION_SHA,'policy':POLICY,
        'base_source_sha256':before,'generated_sources':generated,'model_card_sha256':sha(out/'MODEL_CARD.json'),
        'neural_model_calls':0,'pcm_reads':0,'namespace':'active113_sdk/r1active113/r1a113',
        'frontend_pcm_detector_changes':'identifier namespacing only; arithmetic and event policy unchanged',
        'backbone_changes':'identifier namespacing and payload pointer/bounds only; arithmetic unchanged'})
    print('MATERIALIZED_NO_MODEL_EXECUTION '+sha(out/'MATERIALIZATION.json'))


if __name__=='__main__':main()
