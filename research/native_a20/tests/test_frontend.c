/* Invented-only tests; no model includes, loading, initialization, or steps. */
#include "../src/donor_fft64.c"
#include "../src/pcm_fft64.h"
#include "../baseline/native/donor_fbank/donor_fbank.h"
#include <assert.h>
#include <stdio.h>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

static donor_fft64_state s, snapshot;
static donor_fft64_pcm_state p, q;
static donor_fbank_state original;
static uint32_t bits(float x) { uint32_t u;memcpy(&u,&x,4);return u; }
static unsigned callbacks;
static float saved_rows[400];
static void collect(void *u,const donor_fft64_pcm_batch *b) {
 (void)u;assert(b->call_index==0);assert(b->call_samples==960);
 assert(b->fbank_rows==4&&b->selected_rows==1&&b->is_final_short==1);
 assert(b->wave_samples==320);callbacks++;
 for(size_t i=0;i<b->fbank_rows*80;i++)saved_rows[i]=b->fbank[i];
}

int main(void) {
 assert(donor_fft64_environment_ok());assert(donor_fft64_init(&s)==0);
 float x[512]={0}, power[257];
 assert(donor_fft64_fft_power(&s,x,power)==0);
 for(int i=0;i<512;i++)assert(s.re[i]==0&&s.im[i]==0);
 x[0]=1;assert(donor_fft64_fft_power(&s,x,power)==0);
 for(int i=0;i<512;i++)assert(s.re[i]==1&&s.im[i]==0);
 for(int i=0;i<257;i++)assert(power[i]==1);
 x[0]=0;x[1]=1;assert(donor_fft64_fft_power(&s,x,power)==0);
 assert(s.re[0]==1&&s.im[0]==0&&s.re[128]==0&&s.im[128]==-1);
 assert(s.re[256]==-1&&s.im[256]==0&&s.re[384]==0&&s.im[384]==1);
 for(int i=0;i<512;i++)x[i]=(i%2)?-3:3;
 assert(donor_fft64_fft_power(&s,x,power)==0);
 for(int i=0;i<512;i++)assert(s.re[i]==(i==256?1536:0)&&s.im[i]==0);
 for(int i=0;i<512;i++)x[i]=7;
 assert(donor_fft64_fft_power(&s,x,power)==0);
 for(int i=0;i<512;i++)assert(s.re[i]==(i==0?3584:0)&&s.im[i]==0);
 /* Exact cancellation and a DC midpoint produced through the actual FFT. */
 memset(x,0,sizeof(x));x[0]=65536;x[256]=-65536;
 assert(donor_fft64_fft_power(&s,x,power)==0);
 for(int i=0;i<512;i++)assert(s.re[i]==(i%2?131072:0)&&s.im[i]==0);
 memset(x,0,sizeof(x));x[0]=1;x[1]=0x1p-24f;
 assert(donor_fft64_fft_power(&s,x,power)==0);
 assert(s.fft_re[0]==1.0+0x1p-24&&bits(s.re[0])==UINT32_C(0x3f800000));
 /* Same production conversion helper: ties, signs, subnormals,
  * adjacent normal exponent boundary and overflow. */
 assert(bits(interface_rne(1.0+0x1p-24))==UINT32_C(0x3f800000));
 assert(bits(interface_rne(1.0+3*0x1p-24))==UINT32_C(0x3f800002));
 assert(bits(interface_rne(-1.0-0x1p-24))==UINT32_C(0xbf800000));
 assert(bits(interface_rne(nextafter(1.0+0x1p-24,INFINITY)))==UINT32_C(0x3f800001));
 assert(bits(interface_rne(nextafter(1.0+0x1p-24,-INFINITY)))==UINT32_C(0x3f800000));
 assert(bits(interface_rne(0x1p-150))==0);
 assert(bits(interface_rne(-0x1p-150))==UINT32_C(0x80000000));
 assert(bits(interface_rne(3*0x1p-150))==2);
 assert(bits(interface_rne(0x1p-126-0x1p-150))==UINT32_C(0x00800000));
 assert(bits(interface_rne(2.0-0x1p-24))==UINT32_C(0x40000000));
 assert(bits(interface_rne(0x1p128-0x1p103))==UINT32_C(0x7f800000));
 /* Nonfinite/out-of-bound/null rejection is atomic. */
 for(int which=0;which<3;which++) {
  snapshot=s;for(int i=0;i<257;i++)power[i]=123;
  x[17]=which==0?NAN:which==1?INFINITY:65537;
  assert(donor_fft64_fft_power(&s,x,power)==DONOR_FFT64_ARGUMENT);
  assert(memcmp(&s,&snapshot,sizeof(s))==0);
  for(int i=0;i<257;i++)assert(power[i]==123);
 }
 assert(donor_fft64_fft_power(NULL,x,power)==DONOR_FFT64_ARGUMENT);
 assert(donor_fft64_fft_power(&s,NULL,power)==DONOR_FFT64_ARGUMENT);
 assert(donor_fft64_fft_power(&s,x,NULL)==DONOR_FFT64_ARGUMENT);
 snapshot=s;assert(fesetround(FE_DOWNWARD)==0);
 assert(!donor_fft64_environment_ok());assert(donor_fft64_reset(&s)==DONOR_FFT64_STATE);
 assert(memcmp(&s,&snapshot,sizeof(s))==0);assert(fesetround(FE_TONEAREST)==0);
#if defined(__SSE2__)
 unsigned mode=_mm_getcsr();
 _mm_setcsr(mode|UINT32_C(0x8000));assert(!donor_fft64_environment_ok());_mm_setcsr(mode);
 _mm_setcsr(mode|UINT32_C(0x0040));assert(!donor_fft64_environment_ok());_mm_setcsr(mode);
#endif
 assert(donor_fft64_reset(&s)==0);snapshot=s;
 for(unsigned i=0;i<512;i++)assert(s.fft_re[i]==0&&s.fft_im[i]==0);
 /* Invented integer waveform: original preprocessing must be bit exact. */
 int16_t pcm[960];for(unsigned i=0;i<960;i++)pcm[i]=(int16_t)((int)(i%17)-8);
 donor_fft64_trace t;donor_fbank_trace t0;
 assert(donor_fbank_init(&original)==0);
 assert(donor_fft64_analyze_frame(&s,pcm,&t)==0);
 assert(donor_fbank_analyze_frame(&original,pcm,&t0)==0);
 assert(memcmp(t.dc,t0.dc,sizeof(t.dc))==0);
 assert(memcmp(t.preemphasis,t0.preemphasis,sizeof(t.preemphasis))==0);
 assert(memcmp(t.windowed,t0.windowed,sizeof(t.windowed))==0);
 assert(s.total_samples==0&&s.frame_index==0&&s.used==0);
 memset(pcm,0,sizeof(pcm));assert(donor_fft64_analyze_frame(&s,pcm,&t)==0);
 for(int i=0;i<80;i++)assert(t.mel[i]==0&&t.logfbank[i]==logf(FLT_EPSILON));
 /* Small invented stream only: equal full/split state, actual short tail. */
 for(unsigned i=0;i<960;i++)pcm[i]=(int16_t)((int)(i%17)-8);
 assert(donor_fft64_pcm_init(&p)==0&&donor_fft64_pcm_init(&q)==0);
 callbacks=0;assert(donor_fft64_pcm_feed(&p,pcm,960,collect,NULL)==0);
 assert(donor_fft64_pcm_finish(&p,collect,NULL)==0&&callbacks==1);
 float expected[320];memcpy(expected,saved_rows,sizeof(expected));
 assert(donor_fft64_pcm_feed(&q,pcm,7,collect,NULL)==0);
 assert(donor_fft64_pcm_feed(&q,pcm+7,953,collect,NULL)==0);
 assert(donor_fft64_pcm_finish(&q,collect,NULL)==0&&callbacks==2);
 assert(memcmp(saved_rows,expected,sizeof(expected))==0&&memcmp(&p,&q,sizeof(p))==0);
 assert(donor_fft64_pcm_finish(&q,collect,NULL)==DONOR_FFT64_PCM_STATE);
 assert(donor_fft64_pcm_reset(&q)==0);assert(q.total_calls==0&&q.finished==0);
 puts("PASS invented FFT/sign/cardinal/cancellation/RNE/rejection/preprocessing/reset/partition units; frontend functions evaluated on invented inputs; A20 model steps=0; actual audio=0; holdout=0");
 return 0;
}
