/* Host-only regression CLI; core mero43_stream.c has NO filesystem I/O.
 * Usage: mero43_host_cli model.pack original.pcm logits.f32 events.csv
 * Models/data MUST be independently SHA authenticated by the orchestration.
 * Not an SSC305 executable, not a production latency/quality benchmark. */
#include "mero43_stream.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

static uint8_t binary[EN_V5_M43_PACK_BYTES];
static en_v5_m43_model model;
static en_v5_m43_stream stream;

typedef struct {
    FILE *logits,*events;
    size_t frames,detected;
    int io_failed;
} outputs;

static void put32(uint8_t out[4],uint32_t number) {
    out[0]=(uint8_t)number;out[1]=(uint8_t)(number>>8);
    out[2]=(uint8_t)(number>>16);out[3]=(uint8_t)(number>>24);
}
static void receive(void *user,const float logits[4],uint64_t index,
                    uint64_t end_sample,const en_v5_mero_event *event) {
    outputs *ctx=(outputs*)user;
    if (ctx->io_failed) return;
    for (unsigned c=0u;c<4u;++c) {
        uint32_t bits;
        uint8_t raw[4];
        memcpy(&bits,&logits[c],4);
        put32(raw,bits);
        if (fwrite(raw,1,4,ctx->logits)!=4u) ctx->io_failed=1;
    }
    if (fprintf(ctx->events,"%" PRIu64 ",%" PRIu64 ",%u\n",
                index,end_sample,event->detected)<0)
        ctx->io_failed=1;
    ctx->frames++;
    ctx->detected+=event->detected;
}
int main(int argc,char **argv) {
    if (argc!=5) {
        fputs("Usage: mero43_host_cli model.pack original.pcm logits.f32 events.csv\n",
              stderr);
        return 2;
    }
    FILE *f=fopen(argv[1],"rb");
    if (!f) {perror("pack input");return 3;}
    size_t got=fread(binary,1,sizeof(binary),f);
    int extra=fgetc(f);
    int broken=ferror(f);
    fclose(f);
    if (got!=sizeof(binary) || extra!=EOF || broken ||
        en_v5_m43_model_load(&model,binary,sizeof(binary))!=EN_V5_M43_OK) {
        fputs("STOP: invalid or missing frozen 43K C model pack\n",stderr);
        return 4;
    }
    if (en_v5_m43_stream_init(&stream,&model,1000000u,0u)!=EN_V5_M43_OK) {
        fputs("STOP: C stream initialization failed\n",stderr);
        return 5;
    }
    FILE *pcm=fopen(argv[2],"rb");
    if (!pcm) {perror("PCM input");return 6;}
    outputs ctx={0};
    ctx.logits=fopen(argv[3],"wb");
    ctx.events=fopen(argv[4],"w");
    if (!ctx.logits || !ctx.events) {
        fputs("STOP: cannot open host output files\n",stderr);
        return 7;
    }
    uint8_t bytes[EN_V5_M43_HOP*2u];
    int16_t samples[EN_V5_M43_HOP];
    uint64_t total=0u;
    for (;;) {
        size_t n=fread(bytes,1,sizeof(bytes),pcm);
        if (!n) {
            if (ferror(pcm)) {fputs("STOP: PCM I/O error\n",stderr);return 8;}
            break;
        }
        if (n%2u) {fputs("STOP: incomplete PCM16 sample\n",stderr);return 9;}
        for (size_t i=0u;i<n/2u;++i) {
            uint16_t bits=(uint16_t)bytes[2u*i]|((uint16_t)bytes[2u*i+1u]<<8);
            memcpy(&samples[i],&bits,sizeof(bits));
        }
        if (en_v5_m43_feed(&stream,samples,n/2u,total,receive,&ctx)
                !=EN_V5_M43_OK || ctx.io_failed) {
            fputs("STOP: C frame processing or output failure\n",stderr);
            return 10;
        }
        total+=n/2u;
    }
    if (fclose(pcm) || fflush(ctx.logits) || fflush(ctx.events) ||
        fclose(ctx.logits) || fclose(ctx.events)) {
        fputs("STOP: flush output I/O error\n",stderr);
        return 11;
    }
    fprintf(stderr,"MERO43_C_PCM_STREAM_OK samples=%" PRIu64
          " frames=%zu events=%zu\n",total,ctx.frames,ctx.detected);
    return 0;
}
