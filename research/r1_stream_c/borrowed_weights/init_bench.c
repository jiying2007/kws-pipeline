/* SPDX-License-Identifier: Apache-2.0
 * R1 constructor-only resource comparator: one create/destroy, zero PCM feeds.
 * Run each mode as its own low-RSS process; this is not a board measurement.
 */
#define _POSIX_C_SOURCE 200809L
#include "fsmn_sdk.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static unsigned char *read_exact(const char *path, size_t bytes) {
    FILE *file=fopen(path,"rb");
    if(!file)return NULL;
    unsigned char *data=(unsigned char *)malloc(bytes);
    if(!data){(void)fclose(file);return NULL;}
    int bad=fread(data,1,bytes,file)!=bytes;
    if(fgetc(file)!=EOF || ferror(file))bad=1;
    if(fclose(file))bad=1;
    if(bad){free(data);return NULL;}
    return data;
}

/* Missing /proc metrics stay null. ru_maxrss is reported independently. */
static void proc_memory(long *hwm, long *rss) {
    FILE *file=fopen("/proc/self/status","r");
    char line[256];
    *hwm=-1;*rss=-1;
    if(!file)return;
    while(fgets(line,sizeof(line),file)) {
        long value=-1;
        char unit[3]={0};
        if(!strncmp(line,"VmHWM:",6) &&
           sscanf(line+6,"%ld %2s",&value,unit)==2 &&
           value>=0 && !strcmp(unit,"kB"))*hwm=value;
        if(!strncmp(line,"VmRSS:",6) &&
           sscanf(line+6,"%ld %2s",&value,unit)==2 &&
           value>=0 && !strcmp(unit,"kB"))*rss=value;
    }
    int bad=ferror(file);
    if(fclose(file))bad=1;
    if(bad){*hwm=-1;*rss=-1;}
}

static int valid_clock(const struct timespec *t) {
    return t->tv_sec>=0 && t->tv_nsec>=0 && t->tv_nsec<1000000000L;
}

int main(int argc,char **argv) {
    if(argc!=4 || (strcmp(argv[1],"copy") && strcmp(argv[1],"view"))) {
        fputs("usage: fsmn_init_bench (copy|view) MODEL FRONTEND\n",stderr);
        return 2;
    }
    int view=!strcmp(argv[1],"view");
    unsigned char *model=read_exact(argv[2],FSMN_MODEL_BYTES);
    unsigned char *tables=read_exact(argv[3],FSMN_TABLE_BYTES);
    if(!model || !tables){free(model);free(tables);return 2;}
    struct timespec begin,end;
    if(clock_gettime(CLOCK_MONOTONIC,&begin) || !valid_clock(&begin)) {
        free(model);free(tables);return 2;
    }
    fsmn_sdk *sdk=view?
        fsmn_sdk_create_view(model,FSMN_MODEL_BYTES,tables,FSMN_TABLE_BYTES,0,NULL,NULL,NULL):
        fsmn_sdk_create(model,FSMN_MODEL_BYTES,tables,FSMN_TABLE_BYTES,0,NULL,NULL,NULL);
    int bad=clock_gettime(CLOCK_MONOTONIC,&end) || !valid_clock(&end);
    free(tables);
    if(!view){free(model);model=NULL;}
    if(!sdk || bad) {
        if(sdk)(void)fsmn_sdk_destroy(sdk);
        free(model);return 2;
    }
    double init_ms=(double)(end.tv_sec-begin.tv_sec)*1000.0+
                   (double)(end.tv_nsec-begin.tv_nsec)/1e6;
    struct rusage usage;
    if(init_ms<0.0 || getrusage(RUSAGE_SELF,&usage) || usage.ru_maxrss<0)bad=1;
    long hwm=-1,rss=-1;
    proc_memory(&hwm,&rss);
    if(fsmn_sdk_frame_count(sdk) || fsmn_sdk_consumed_bytes(sdk) ||
       fsmn_sdk_event_count(sdk))bad=1;
    if(fsmn_sdk_destroy(sdk))bad=1;
    free(model);
    if(bad){fputs("FAILED: initialization/resource observation\n",stderr);return 2;}
    printf("{\"schema\":\"r1-constructor-only-resource-v1\",\"mode\":\"%s\",",argv[1]);
    printf("\"model_forwards\":0,\"pcm_bytes\":0,\"initializations\":1,");
    printf("\"model_init_wall_ms\":%.6f,\"process_peak_rss_kib\":%ld,",init_ms,usage.ru_maxrss);
    printf("\"proc_self_vm_hwm_kib\":");
    if(hwm<0)printf("null");else printf("%ld",hwm);
    printf(",\"proc_self_vm_rss_kib\":");
    if(rss<0)printf("null");else printf("%ld",rss);
    printf(",\"model_bytes\":%u,\"copied_model_bytes\":%u,\"physical_ssc305\":false}\n",
           FSMN_MODEL_BYTES,view?0u:FSMN_MODEL_BYTES);
    if(fflush(stdout) || ferror(stdout))return 2;
    return 0;
}
