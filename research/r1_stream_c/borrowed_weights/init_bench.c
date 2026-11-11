/* SPDX-License-Identifier: Apache-2.0
 * R1 constructor-only resource comparator: one create/destroy, zero PCM feeds.
 * Optional read-only, exact-sized mmap model for Linux squashfs/immutable files.
 * Run each mode as its own low-RSS process; this is NOT a board qualification.
 */
#define _POSIX_C_SOURCE 200809L
#include "fsmn_sdk.h"
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#error "read-only model mapping requires O_NOFOLLOW; no unsafe fallback"
#endif
#ifndef O_CLOEXEC
#error "read-only model mapping requires O_CLOEXEC; no unsafe fallback"
#endif

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

/* No anonymous model allocation or user-space copy. The SDK checks all
 * original model bytes/hash after mapping. The underlying regular file MUST
 * remain immutable and untruncated through SDK destroy + munmap.
 * Works naturally with a read-only squashfs model file on SSC305.
 */
static unsigned char *map_exact(const char *path, size_t bytes) {
    if(!path || !bytes)return NULL;
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if(fd<0)return NULL;
    struct stat st;
    int valid=fstat(fd,&st)==0 && S_ISREG(st.st_mode) &&
              st.st_size>=0 && (uintmax_t)st.st_size==(uintmax_t)bytes;
    if(!valid){(void)close(fd);return NULL;}
    void *mapped=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    int closed=close(fd);
    if(mapped==MAP_FAILED)return NULL;
    if(closed){(void)munmap(mapped,bytes);return NULL;}
    return (unsigned char *)mapped;
}

static int release_model(unsigned char *model,int mapped) {
    if(!model)return 0;
    if(mapped)return munmap(model,FSMN_MODEL_BYTES);
    free(model);
    return 0;
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

/* /proc/self/smaps_rollup is optional on older embedded Linux kernels.
 * PSS and Private_Dirty distinguish shared file-backed pages from anonymous
 * heap pressure; neither is a target-device measurement in host CI.
 */
static void proc_smaps(long *pss,long *dirty) {
    FILE *file=fopen("/proc/self/smaps_rollup","r");
    char line[256];
    *pss=-1;*dirty=-1;
    if(!file)return;
    while(fgets(line,sizeof(line),file)) {
        long value=-1;
        char unit[3]={0};
        if(!strncmp(line,"Pss:",4) &&
           sscanf(line+4,"%ld %2s",&value,unit)==2 &&
           value>=0 && !strcmp(unit,"kB"))*pss=value;
        if(!strncmp(line,"Private_Dirty:",14) &&
           sscanf(line+14,"%ld %2s",&value,unit)==2 &&
           value>=0 && !strcmp(unit,"kB"))*dirty=value;
    }
    int bad=ferror(file);
    if(fclose(file))bad=1;
    if(bad){*pss=-1;*dirty=-1;}
}

static int valid_clock(const struct timespec *t) {
    return t->tv_sec>=0 && t->tv_nsec>=0 && t->tv_nsec<1000000000L;
}

static double elapsed_ms(const struct timespec *begin,const struct timespec *end) {
    return (double)(end->tv_sec-begin->tv_sec)*1000.0+
           (double)(end->tv_nsec-begin->tv_nsec)/1e6;
}

int main(int argc,char **argv) {
    if(argc!=4 || (strcmp(argv[1],"copy") && strcmp(argv[1],"view") &&
                   strcmp(argv[1],"mmap"))) {
        fputs("usage: fsmn_init_bench (copy|view|mmap) MODEL FRONTEND\n",stderr);
        return 2;
    }
    int mode=!strcmp(argv[1],"copy")?0:(!strcmp(argv[1],"view")?1:2);
    int mapped=mode==2;
    struct timespec full_begin,full_cpu_begin,begin,end,full_cpu_end;
    if(clock_gettime(CLOCK_MONOTONIC,&full_begin) ||
       clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&full_cpu_begin) ||
       !valid_clock(&full_begin) || !valid_clock(&full_cpu_begin))return 2;
    unsigned char *model=mapped?
        map_exact(argv[2],FSMN_MODEL_BYTES):
        read_exact(argv[2],FSMN_MODEL_BYTES);
    unsigned char *tables=read_exact(argv[3],FSMN_TABLE_BYTES);
    if(!model || !tables) {
        (void)release_model(model,mapped);
        free(tables);
        return 2;
    }
    if(clock_gettime(CLOCK_MONOTONIC,&begin) || !valid_clock(&begin)) {
        (void)release_model(model,mapped);
        free(tables);
        return 2;
    }
    fsmn_sdk *sdk=mode?
        fsmn_sdk_create_view(model,FSMN_MODEL_BYTES,tables,FSMN_TABLE_BYTES,0,NULL,NULL,NULL):
        fsmn_sdk_create(model,FSMN_MODEL_BYTES,tables,FSMN_TABLE_BYTES,0,NULL,NULL,NULL);
    int bad=clock_gettime(CLOCK_MONOTONIC,&end) ||
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&full_cpu_end) ||
            !valid_clock(&end) || !valid_clock(&full_cpu_end);
    free(tables);
    if(mode==0) {
        if(release_model(model,0))bad=1;
        model=NULL;
    }
    if(!sdk || bad) {
        if(sdk)(void)fsmn_sdk_destroy(sdk);
        (void)release_model(model,mapped);
        return 2;
    }
    double init_ms=elapsed_ms(&begin,&end);
    double full_ms=elapsed_ms(&full_begin,&end);
    double full_cpu_ms=elapsed_ms(&full_cpu_begin,&full_cpu_end);
    struct rusage usage;
    if(init_ms<0.0 || full_ms<0.0 || full_cpu_ms<0.0 ||
       getrusage(RUSAGE_SELF,&usage) || usage.ru_maxrss<0)bad=1;
    long hwm=-1,rss=-1;
    proc_memory(&hwm,&rss);
    long pss=-1,private_dirty=-1;
    proc_smaps(&pss,&private_dirty);
    if(fsmn_sdk_frame_count(sdk) || fsmn_sdk_consumed_bytes(sdk) ||
       fsmn_sdk_event_count(sdk))bad=1;
    if(fsmn_sdk_destroy(sdk))bad=1;
    if(release_model(model,mapped))bad=1;
    if(bad){fputs("FAILED: initialization/resource observation\n",stderr);return 2;}
    printf("{\"schema\":\"r1-constructor-only-resource-v1\",\"mode\":\"%s\",",argv[1]);
    printf("\"model_forwards\":0,\"pcm_bytes\":0,\"initializations\":1,");
    printf("\"model_init_wall_ms\":%.6f,",init_ms);
    printf("\"model_load_and_init_wall_ms\":%.6f,\"model_load_and_init_cpu_ms\":%.6f,",
           full_ms,full_cpu_ms);
    printf("\"process_peak_rss_kib\":%ld,",usage.ru_maxrss);
    printf("\"proc_self_vm_hwm_kib\":");
    if(hwm<0)printf("null");else printf("%ld",hwm);
    printf(",\"proc_self_vm_rss_kib\":");
    if(rss<0)printf("null");else printf("%ld",rss);
    printf(",\"proc_self_pss_kib\":");
    if(pss<0)printf("null");else printf("%ld",pss);
    printf(",\"proc_self_private_dirty_kib\":");
    if(private_dirty<0)printf("null");else printf("%ld",private_dirty);
    printf(",\"process_minor_faults\":%ld,\"process_major_faults\":%ld,",
           usage.ru_minflt,usage.ru_majflt);
    printf("\"model_bytes\":%u,\"copied_model_bytes\":%u,",FSMN_MODEL_BYTES,
           mode==0?FSMN_MODEL_BYTES:0u);
    printf("\"model_storage\":\"%s\",\"physical_ssc305\":false}\n",
           mapped?"read_only_private_file_mapping":(mode==1?"borrowed_heap":"sdk_owned_heap"));
    if(fflush(stdout) || ferror(stdout))return 2;
    return 0;
}
