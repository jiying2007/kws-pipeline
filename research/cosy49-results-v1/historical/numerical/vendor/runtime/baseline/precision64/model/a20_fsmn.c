#include "a20_fsmn.h"
#include <math.h>
#include <string.h>
static void affine(const float*x,float*y,const float*w,const float*b,int ni,int no){
 /* Research-only precision variant: FP32 operands, exact FP64 products,
  * sequential FP64 sum and bias, one final rounding to FP32. */
 for(int o=0;o<no;o++){
  double s=0.0;
  for(int i=0;i<ni;i++)s+=(double)x[i]*(double)w[o*ni+i];
  s+=b?(double)b[o]:0.0;
  y[o]=(float)s;
 }
}
static int finite_state(a20_model*m,const float*x,int n){for(int i=0;i<n;i++)if(!isfinite(x[i])){m->fault=1;return 0;}return 1;}
static void relu(float*x,int n){for(int i=0;i<n;i++)if(x[i]<0)x[i]=0;}
static void trace(float*t,int s,const float*x,int n){if(t)memcpy(t+(size_t)s*A20_TRACE_STRIDE,x,(size_t)n*sizeof(float));}
int a20_init(a20_model*m,const float*w,size_t n){
 if(!m)return -1;
 m->weights=0; m->fault=1;
 if(!w||n!=A20_FLOATS)return -1;
 for(size_t i=0;i<n;i++)if(!isfinite(w[i]))return -2;
 m->weights=w;a20_reset(m);return 0;
}
void a20_reset(a20_model*m){if(m){memset(m->cache,0,sizeof(m->cache));m->fault=0;}}
int a20_cmvn(const a20_model*m,const float*x,float*z){
 if(!m||!m->weights||!x||!z)return -1;
 for(int i=0;i<400;i++)if(!isfinite(x[i]))return -2;
 for(int i=0;i<400;i++){z[i]=(x[i]-m->weights[i])*m->weights[400+i];if(!isfinite(z[i]))return -3;}
 return 0;
}
static void memory(a20_model*m,int l,size_t off,const float*p,float*mem){
 const float*w=m->weights;
  for(int c=0;c<128;c++){
   double left=0.0;for(int k=0;k<10;k++)left+=(double)w[off+32000u+(size_t)c*10+k]*(double)m->cache[((size_t)c*11+k)*4+l];
   double right=(double)w[off+33280u+(size_t)c*2]*(double)m->cache[((size_t)c*11+10)*4+l]+(double)w[off+33281u+(size_t)c*2]*(double)p[c];
   mem[c]=(float)(((double)m->cache[((size_t)c*11+9)*4+l]+left)+right);
   for(int k=0;k<10;k++)m->cache[((size_t)c*11+k)*4+l]=m->cache[((size_t)c*11+k+1)*4+l];
   m->cache[((size_t)c*11+10)*4+l]=p[c];
  }
}
int a20_step(a20_model*m,const float*x,float*y,float*t){
 if(!m||!m->weights||!x||!y)return -1;
 if(m->fault)return -3;
 for(int i=0;i<400;i++)if(!isfinite(x[i]))return -2;
 const float*w=m->weights;float z[400],a[250],b[250],p[128],mem[128];
 if(a20_cmvn(m,x,z)){m->fault=1;return -3;}
 affine(z,a,w+800,w+56800,400,140);if(!finite_state(m,a,140))return -3;trace(t,0,a,140);
 affine(a,b,w+56940,w+91940,140,250);if(!finite_state(m,b,250))return -3;trace(t,1,b,250);relu(b,250);trace(t,2,b,250);
 for(int l=0;l<4;l++){
  size_t off=92190u+(size_t)l*65786u;
  affine(b,p,w+off,0,250,128);if(!finite_state(m,p,128))return -3;trace(t,3+4*l,p,128);
  memory(m,l,off,p,mem);
  if(!finite_state(m,mem,128))return -3;
  trace(t,4+4*l,mem,128);affine(mem,b,w+off+33536,w+off+65536,128,250);if(!finite_state(m,b,250))return -3;trace(t,5+4*l,b,250);relu(b,250);trace(t,6+4*l,b,250);
 }
 affine(b,a,w+355334,w+390334,250,140);if(!finite_state(m,a,140))return -3;trace(t,19,a,140);
 affine(a,y,w+390474,w+391314,140,A20_OUTPUTS);if(!finite_state(m,y,A20_OUTPUTS))return -3;trace(t,20,y,A20_OUTPUTS);
 for(unsigned i=0;i<A20_OUTPUTS;i++)if(!isfinite(y[i])){m->fault=1;return -3;}
 return 0;
}

size_t a20_model_bytes(void){return sizeof(a20_model);}

int a20_accumulation_bits(void){return 64;}
