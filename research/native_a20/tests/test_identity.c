/* Invented data only: hashes and fail-closed model identity. */
#include "../baseline/precision64/model/a20_fsmn.h"
#include "../baseline/precision64/model/sha256.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
 char digest[65];
 kws_sha256_memory_hex((const uint8_t*)"abc",3,digest);
 assert(strcmp(digest,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")==0);
 kws_sha256_memory_hex((const uint8_t*)"",0,digest);
 assert(strcmp(digest,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")==0);
 float *invented=calloc(A20_FLOATS,sizeof(float));a20_model state;assert(invented);
 assert(a20_init_verified(&state,invented,A20_FLOATS)==-4);
 assert(state.weights==NULL&&state.fault==1);
 assert(a20_init_verified(&state,invented,A20_FLOATS-1)==-1);
 assert(a20_init_verified(NULL,invented,A20_FLOATS)==-1);
 assert(a20_init_verified(&state,NULL,A20_FLOATS)==-1);
 assert(a20_accumulation_bits()==64);free(invented);
 puts("PASS SHA256 known vectors and invented payload identity rejection");return 0;
}
