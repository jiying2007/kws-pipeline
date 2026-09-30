#include "../fsmn.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
int main(void){
 float*w=calloc(DF_FLOATS,sizeof(float));float x[400]={0},y[2599];df_model a,b;
 assert(w);for(int i=400;i<800;i++)w[i]=1;
 assert(df_init(&a,w,DF_FLOATS)==0);assert(df_init(&b,w,DF_FLOATS)==0);
 for(int n=0;n<40;n++){x[n%400]=(float)n;assert(df_step(&a,x,y,0)==0);assert(df_step(&b,x,y,0)==0);}
 assert(df_init(&a,w,DF_FLOATS-1)==-1);assert(df_step(&a,x,y,0)==-1);
 assert(df_init(&a,w,DF_FLOATS)==0);x[0]=FLT_MAX;w[400]=2;assert(df_step(&a,x,y,0)==-3);assert(a.fault);
 x[0]=0;assert(df_step(&a,x,y,0)==-3);df_reset(&a);assert(df_step(&a,x,y,0)==0);
 /* Negative overflow in second affine must fault before ReLU can erase it. */
 df_reset(&a);w[400]=1;w[800]=1;x[0]=FLT_MAX;w[56940]=-2;assert(df_step(&a,x,y,0)==-3);assert(a.fault);
 free(w);return 0;
}
