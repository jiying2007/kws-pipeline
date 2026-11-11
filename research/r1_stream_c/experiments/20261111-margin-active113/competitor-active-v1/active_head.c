/* SPDX-License-Identifier: Apache-2.0
 * Frozen ACTIVE113 cached-head prediction only. No PCM/backbone/detector calls.
 * Original par4 OTHER arithmetic and original five precise rows are retained.
 */
#define _POSIX_C_SOURCE 200809L
#include "sha256.h"
#include "selected_rows.h"
#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { MODEL_BYTES=1960140, HEADER_BYTES=64, INPUT_STRIDE=1200, OUTPUT_STRIDE=40,
       QUANT=88+390474*4, SCALE=QUANT+2601*140+4, BIAS=SCALE+2601*8,
       PRECISE=BIAS+2601*4, MAX_FRAMES=512 };
static const uint32_t precise_ids[5]={0,1027,1401,1462,2093};
static const char model_sha[]="63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4";
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && sizeof(float)==4 &&
               FLT_MANT_DIG==24 && FLT_RADIX==2 && FLT_EVAL_METHOD==0,
               "IEEE binary64/binary32 with no excess precision required");

static uint32_t u32(const unsigned char *p){
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static uint64_t u64(const unsigned char *p){return (uint64_t)u32(p)|((uint64_t)u32(p+4)<<32);}
static double f64(const unsigned char *p){double v;memcpy(&v,p,8);return v;}
static float f32(const unsigned char *p){float v;memcpy(&v,p,4);return v;}
static void put32(unsigned char *p,uint32_t v){for(unsigned i=0;i<4;i++)p[i]=(unsigned char)(v>>(8*i));}
static void put64(unsigned char *p,uint64_t v){for(unsigned i=0;i<8;i++)p[i]=(unsigned char)(v>>(8*i));}
static int same_double(double value,const unsigned char *stored){return !memcmp(&value,stored,8);}
static int selected(uint32_t token){
    for(unsigned k=0;k<ACTIVE_OTHER_COUNT;k++)if(active_other_ids[k]==token)return 1;
    return 0;
}
static unsigned char *input(const char *path,size_t minimum,size_t maximum,size_t *bytes){
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NONBLOCK);
    struct stat st;
    if(fd<0)return NULL;
    if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0||
       (uintmax_t)st.st_size<minimum||(uintmax_t)st.st_size>maximum){close(fd);return NULL;}
    size_t n=(size_t)st.st_size;
    unsigned char *data=(unsigned char *)malloc(n);
    if(!data){close(fd);return NULL;}
    size_t got=0;
    while(got<n){ssize_t z=read(fd,data+got,n-got);if(z<0&&errno==EINTR)continue;
        if(z<=0){free(data);close(fd);return NULL;}got+=(size_t)z;}
    unsigned char extra;ssize_t z=read(fd,&extra,1);int closed=close(fd);
    if(z!=0||closed){free(data);return NULL;}
    *bytes=n;return data;
}
static int environment(void){
    const uint16_t one=1;volatile double tiny=DBL_MIN,half=0.5;
    volatile double sub=tiny*half,recovered=sub+sub;
    return *(const unsigned char *)&one==1&&fegetround()==FE_TONEAREST&&sub>0&&recovered==DBL_MIN;
}
int main(int argc,char **argv){
    if(argc!=6||(strcmp(argv[5],"train")&&strcmp(argv[5],"development"))||strlen(argv[3])!=64||!environment())return 2;
    int training=!strcmp(argv[5],"train");
    for(unsigned k=0;k<ACTIVE_OTHER_COUNT;k++){
        if(active_other_ids[k]>=2601||(k&&active_other_ids[k]<=active_other_ids[k-1]))return 2;
        for(unsigned j=0;j<5;j++)if(active_other_ids[k]==precise_ids[j])return 2;
    }
    size_t model_bytes=0,hidden_bytes=0;
    unsigned char *model=input(argv[1],MODEL_BYTES,MODEL_BYTES,&model_bytes);
    unsigned char *hidden=input(argv[2],HEADER_BYTES+INPUT_STRIDE,HEADER_BYTES+MAX_FRAMES*INPUT_STRIDE,&hidden_bytes);
    if(!model||!hidden){free(model);free(hidden);return 2;}
    char actual_model[65],actual_hidden[65];
    r1d_sha256_memory_hex(model,model_bytes,actual_model);
    r1d_sha256_memory_hex(hidden,hidden_bytes,actual_hidden);
    int bad=strcmp(actual_model,model_sha)||strcmp(actual_hidden,argv[3])||
        memcmp(hidden,"R1HD140\0",8)||u32(hidden+8)!=1||u32(hidden+12)!=64||
        u32(hidden+16)!=1200||u32(hidden+20)!=140||u32(hidden+24)!=5||u32(hidden+28)!=2601||
        u32(hidden+52)!=1||u64(hidden+56)!=0||(hidden_bytes-64)%1200;
    for(unsigned j=0;j<5;j++)if(u32(hidden+32+4*j)!=precise_ids[j])bad=1;
    if(bad){free(model);free(hidden);return 2;}
    size_t frames=(hidden_bytes-64)/1200;
    unsigned char header[64]={0};memcpy(header,"R1ACTV1\0",8);
    put32(header+8,1);put32(header+12,64);put32(header+16,40);put32(header+20,140);
    put32(header+24,5);put32(header+28,2601);
    for(unsigned j=0;j<5;j++)put32(header+32+4*j,precise_ids[j]);
    put32(header+52,ACTIVE_OTHER_COUNT);put64(header+56,frames);
    int fd=open(argv[4],O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd<0){free(model);free(hidden);return 2;}
    FILE *out=fdopen(fd,"wb");if(!out){close(fd);free(model);free(hidden);return 2;}
    if(fwrite(header,1,64,out)!=64)bad=1;
    size_t equal=0,covered=0;uint64_t last_delivery=0;
    for(size_t f=0;f<frames&&!bad;f++){
        const unsigned char *row=hidden+64+f*1200;
        uint64_t source=u64(row+8),delivery=u64(row+16);
        uint32_t original_other=u32(row+28);
        if(u64(row)!=f||source!=720+480*f||source>delivery||delivery<last_delivery||
           original_other>=2601||!isfinite(f64(row+1192))){bad=1;break;}
        last_delivery=delivery;
        double x[140];for(unsigned i=0;i<140;i++){x[i]=f64(row+32+8*i);if(!isfinite(x[i]))bad=1;}
        if(bad)break;
        for(unsigned j=0;j<5;j++){
            double sum=0.0;
            for(unsigned i=0;i<140;i++)sum+=x[i]*(double)f32(model+PRECISE+j*560+i*4);
            double precise=sum+(double)f32(model+BIAS+precise_ids[j]*4);
            if(!isfinite(precise)||!same_double(precise,row+1152+j*8)){bad=1;break;}
        }
        if(bad)break;
        double best=0.0;uint32_t best_id=0;
        for(unsigned k=0;k<ACTIVE_OTHER_COUNT;k++){
            uint32_t id=active_other_ids[k];const int8_t *q=(const int8_t *)(model+QUANT+id*140);
            double s0=0.0,s1=0.0,s2=0.0,s3=0.0;
            for(unsigned i=0;i<140;i+=4){s0+=x[i]*(double)q[i];s1+=x[i+1]*(double)q[i+1];
                s2+=x[i+2]*(double)q[i+2];s3+=x[i+3]*(double)q[i+3];}
            double sum=(s0+s1)+(s2+s3);sum*=f64(model+SCALE+id*8);
            double score=sum+(double)f32(model+BIAS+id*4);
            if(!isfinite(score)){bad=1;break;}
            if(k==0||score>best){best=score;best_id=id;}
        }
        if(bad||best>f64(row+1192)){bad=1;break;}
        int identical=same_double(best,row+1192),has=selected(original_other);
        if(has&&(!identical||best_id!=original_other)){bad=1;break;}
        if(training&&(!identical||!has||best_id!=original_other)){bad=1;break;}
        equal+=(size_t)identical;covered+=(size_t)has;
        unsigned char prediction[40]={0};put64(prediction,f);put64(prediction+8,source);
        put64(prediction+16,delivery);put32(prediction+24,best_id);memcpy(prediction+32,&best,8);
        if(fwrite(prediction,1,40,out)!=40)bad=1;
    }
    if(fflush(out)||ferror(out)||fsync(fd))bad=1;
    if(fclose(out))bad=1;
    free(model);free(hidden);
    if(bad){fputs("FAILED cached ACTIVE113 prediction; partial output retained\n",stderr);return 2;}
    if(printf("{\"status\":\"PREDICTED_SELECTED_ORIGINAL_HEAD\",\"frames\":%zu,\"precise_logits_bit_equal\":%zu,"
              "\"max_other_bit_equal_frames\":%zu,\"true_other_winner_covered_frames\":%zu,"
              "\"model_sha256\":\"%s\",\"hidden_sha256\":\"%s\",\"selection_sha256\":\"%s\","
              "\"pcm_calls\":0,\"backbone_calls\":0}\n",frames,frames*5,equal,covered,
              actual_model,actual_hidden,ACTIVE_SELECTION_SHA)<0||fflush(stdout)||ferror(stdout))return 2;
    return 0;
}
