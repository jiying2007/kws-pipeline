/* Opt-in observation only; candidate source/scoring remains unchanged.
 * Process singleton intentionally rejects concurrent handles and foreign threads.
 * A rejected stateful call invalidates capture. Callbacks may serialize only.
 * No reset/finish/pad/threshold API. No successful end on dropped or pending data. */
#include "trace_mode.h"
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
static pthread_mutex_t gate=PTHREAD_MUTEX_INITIALIZER;
static pthread_t thread;
static kws3_decoder *owner;
static int active, ended;
static _Atomic int valid;
static int enter(kws3_decoder *d) {
 if(pthread_mutex_trylock(&gate)!=0) { atomic_store(&valid,0); return 0; }
 if(!d || d!=owner || !pthread_equal(thread,pthread_self())) {
  atomic_store(&valid,0); pthread_mutex_unlock(&gate); return 0;
 }
 return 1;
}
static kws3_status leave(kws3_status rc) { pthread_mutex_unlock(&gate); return rc; }
static int healthy(void) { return atomic_load(&valid) && (!active || kws3_ledger_valid()); }
static kws3_status reject(void) { atomic_store(&valid,0); return leave(KWS3_INVALID); }
kws3_status kt_create(const char *mode,const char *id,kws3_decoder **out) {
 kws3_status rc;
 if(out) *out=NULL;
 if(pthread_mutex_trylock(&gate)!=0) { atomic_store(&valid,0); return KWS3_INVALID; }
 if(owner) return reject();
 if(!mode || strcmp(mode,"bounded-retention-trace-v1") || !out) return leave(KWS3_ARGUMENT);
 rc=kws3_create_with_threshold(id,0.7,out);
 if(rc==KWS3_OK) { owner=*out; thread=pthread_self(); active=ended=0; atomic_store(&valid,1); }
 return leave(rc);
}
kws3_status kt_begin(kws3_decoder *d) {
 kws3_summary s;
 if(!enter(d)) return KWS3_INVALID;
 if(!healthy() || active || ended || kws3_get_summary(d,&s)!=KWS3_OK ||
    !s.valid || s.rows_decoded || s.finished) return reject();
 kws3_ledger_begin(); active=1;
 if(!kws3_ledger_valid()) return reject();
 return leave(KWS3_OK);
}
kws3_status kt_step_f32(kws3_decoder *d,const float *row,size_t n,kws3_event *events,size_t cap,size_t *count) {
 kws3_status rc;
 if(count) *count=0;
 if(!enter(d)) return KWS3_INVALID;
 if(!healthy() || ended || (active && kws3_ledger_size())) return reject();
 rc=kws3_step_f32(d,row,n,events,cap,count);
 if(rc!=KWS3_OK || !healthy()) { atomic_store(&valid,0); if(count) *count=0; }
 return leave(rc==KWS3_OK && !healthy()?KWS3_INVALID:rc);
}
kws3_status kt_drain(kws3_decoder *d,kws3_ledger_sink sink,void *u) {
 int ok;
 if(!enter(d)) return KWS3_INVALID;
 if(!healthy() || !active || ended) return reject();
 ok=kws3_ledger_drain(sink,u);
 if(!ok || !healthy()) return reject();
 return leave(KWS3_OK);
}
kws3_status kt_status(kws3_decoder *d,kt_status_v1 *s) {
 kws3_summary summary;
 if(!enter(d)) return KWS3_INVALID;
 if(!s) return reject();
 memset(s,0,sizeof(*s)); s->version=1; s->active=active; s->ended=ended;
 s->valid=healthy();
 if(active) { s->dropped=kws3_ledger_dropped(); s->pending=kws3_ledger_size(); }
 if(kws3_get_summary(d,&summary)==KWS3_OK) {
  s->rows_committed=summary.rows_decoded; s->valid=s->valid && summary.valid;
 } else s->valid=0;
 return leave(KWS3_OK);
}
kws3_status kt_summary(kws3_decoder *d,kws3_summary *s) {
 kws3_status rc; if(!enter(d)) return KWS3_INVALID;
 rc=kws3_get_summary(d,s); return leave(rc);
}
kws3_status kt_retention(kws3_decoder *d,kws3_retention_summary *s) {
 kws3_status rc; if(!enter(d)) return KWS3_INVALID;
 rc=kws3_get_retention_summary(d,s); return leave(rc);
}
kws3_status kt_end(kws3_decoder *d) {
 kws3_summary s;
 if(!enter(d)) return KWS3_INVALID;
 if(!healthy() || !active || ended || kws3_ledger_size() || kws3_ledger_dropped() ||
    kws3_get_summary(d,&s)!=KWS3_OK || !s.valid) return reject();
 ended=1; return leave(KWS3_OK);
}
kws3_status kt_destroy(kws3_decoder *d) {
 kws3_status rc;
 if(!enter(d)) return KWS3_INVALID;
 rc=healthy() && (!active || ended)?KWS3_OK:KWS3_INVALID;
 kws3_destroy(d); owner=NULL; active=ended=0; atomic_store(&valid,0);
 return leave(rc);
}
size_t kt_status_bytes(void) {return sizeof(kt_status_v1);}
size_t kt_record_bytes(void) {return sizeof(kws3_ledger_record);}
