#include "a20_fsmn.h"
#include "sha256.h"
#include "../../export/a20_identity.h"
#include <stdint.h>
#include <string.h>
/* Caller materializes immutable bytes; identity and little-endian IEEE payload
 * are verified here before the model is usable. No file IO or allocation. */
int a20_init_verified(a20_model*m,const float*w,size_t n){
 uint32_t one=1;char digest[65];
 if(!m)return -1;
 m->weights=NULL;m->fault=1;
 if(!w||n!=A20_FLOATS||*(const uint8_t*)&one!=1)return -1;
 kws_sha256_memory_hex((const uint8_t*)w,n*sizeof(float),digest);
 if(strcmp(digest,A20_EXPECTED_SHA)!=0)return -4;
 return a20_init(m,w,n);
}
