/* Public deterministic synthetic matrices only. No I/O of models or audio. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <time.h>
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128,
               "fixture requires IEEE binary32 geometry");
#include "affine_reference.inc"
#include "affine_block2.inc"
static uint32_t seed=0x2468abcd;
static float random_float(void) {seed=seed*1664525u+1013904223u;return ((int)(seed>>16)-32768)*0x1p-13f;}
#ifndef KWS_FIXTURE_CORRECTNESS_ONLY
static uint64_t now(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))exit(3);return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec;}
#endif
static volatile float consume;
int main(void){
 static float x[400],w[400*2599],bias[2599],ref[2599],got[2599];
 const int shape[][2]={{400,140},{140,250},{250,128},{128,250},{250,140},{140,2599},{1,1},{3,3},{5,5}};
 unsigned tested=0;
 if(fegetround()!=FE_TONEAREST)return 2;
 for(size_t sh=0;sh<sizeof(shape)/sizeof(shape[0]);++sh){
  int ni=shape[sh][0],no=shape[sh][1];
  for(int trial=0;trial<36;++trial){
   for(int i=0;i<ni;++i)x[i]=random_float();
   for(int i=0;i<ni*no;++i)w[i]=random_float();
   for(int i=0;i<no;++i)bias[i]=random_float();
   if(trial==32){for(int i=0;i<ni;++i)x[i]=(i%2)?-0.0f:0.0f;}
   if(trial==33){for(int i=0;i<ni;++i)x[i]=(i%2)?-FLT_TRUE_MIN:FLT_TRUE_MIN;}
   if(trial==34){for(int i=0;i<ni;++i)x[i]=(i%2)?-FLT_MAX:FLT_MAX;}
   if(trial==35){for(int i=0;i<ni;++i)x[i]=(i%2)?0x1p-120f:0x1p+100f;}
   for(int hasbias=0;hasbias<2;++hasbias){
    feclearexcept(FE_ALL_EXCEPT);affine_reference(x,ref,w,hasbias?bias:NULL,ni,no);int flags=fetestexcept(FE_ALL_EXCEPT);
    feclearexcept(FE_ALL_EXCEPT);affine_block2(x,got,w,hasbias?bias:NULL,ni,no);int flags2=fetestexcept(FE_ALL_EXCEPT);
    if(memcmp(ref,got,(size_t)no*sizeof(float)) || flags!=flags2){fprintf(stderr,"mismatch shape%zu trial%d bias%d flags%d/%d\n",sh,trial,hasbias,flags,flags2);return 1;}
    ++tested;
   }
  }
  /* Warm both, then alternate order; fixed 5 pairs, 20 calls each.
   * Toy timing only; mutable volatile output prevents dead-call elimination. */
  for(int i=0;i<ni;++i)x[i]=random_float();
  for(int i=0;i<ni*no;++i)w[i]=random_float();
#ifndef KWS_FIXTURE_CORRECTNESS_ONLY
  affine_reference(x,ref,w,bias,ni,no);affine_block2(x,got,w,bias,ni,no);
  for(int round=0;round<5;++round){uint64_t dt[2];
   for(int turn=0;turn<2;++turn){int mode=(turn+round)%2;uint64_t begin=now();
    for(int r=0;r<20;++r){if(mode)affine_block2(x,got,w,bias,ni,no);else affine_reference(x,got,w,bias,ni,no);consume=got[r%no];}
    dt[mode]=now()-begin;
   }
   printf("{\"ni\":%d,\"no\":%d,\"round\":%d,\"calls\":20,\"reference_ns\":%llu,\"block2_ns\":%llu}\n",ni,no,round,(unsigned long long)dt[0],(unsigned long long)dt[1]);
  }
 #endif
 }
 fprintf(stderr,"PASS %u bytewise output and FP sticky-flag comparisons; synthetic finite inputs including signed zero, subnormal and overflow.\n",tested);return consume==123456.0f;
}
