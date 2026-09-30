#ifndef PCM_KWS_H
#define PCM_KWS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct PcmKws PcmKws;
typedef struct {const char *encoder,*decoder,*joiner,*tokens,*keywords;} PcmKwsFiles;
typedef struct {const char *keyword;int keyword_id;uint64_t available_samples;int eof_flush;} PcmKwsEvent;
/* Strings in event live only during callback. Single-threaded, non-reentrant;
 * callback must not throw, call this API, or destroy the instance. */
typedef void (*PcmKwsCallback)(void *user,const PcmKwsEvent *event);
enum {PCM_KWS_OK=0,PCM_KWS_ARGUMENT=1,PCM_KWS_STATE=2,PCM_KWS_RUNTIME=3};
/* Paths needed only during creation. Files load once; create a fresh stream.
 * Caller owns returned instance. Destroy(NULL) is allowed. */
int pcm_kws_create(const PcmKwsFiles*,PcmKwsCallback,void*,PcmKws **out,char *error,size_t capacity);
/* Mono16k signed PCM16. Internal320sample staging makes callback availability
 * independent of caller partition. No file I/O performed by adapter feed. */
int pcm_kws_feed(PcmKws*,const int16_t*,size_t frames,char*,size_t);
/* Flush partial block then EOF exactly once. Subsequent feed/finish rejected. */
int pcm_kws_finish(PcmKws*,char*,size_t);
/* Discard buffered partial block and stream state; reset sample counter to0. */
int pcm_kws_reset(PcmKws*,char*,size_t);
void pcm_kws_destroy(PcmKws*);
#ifdef __cplusplus
}
#endif
#endif
