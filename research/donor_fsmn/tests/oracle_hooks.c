/* Test-only hooks use the exact production operator bodies, no alternate math. */
#include "../fsmn.c"
void test_affine(const float*x,float*y,const float*w,const float*b,int ni,int no){affine(x,y,w,b,ni,no);}
void test_memory(df_model*m,int l,const float*p,float*y){memory(m,l,92190u+(size_t)l*65786u,p,y);}
void test_relu(float*x,int n){relu(x,n);}
