/* Research-only, separate from the aggregate kr_* ABI. One decoder/thread per library. */
#ifndef KWS_TRACE_MODE_H
#define KWS_TRACE_MODE_H
#include "kws3.h"
#include "kws3_retention.h"
#include "kws3_diagnostic_ledger.h"
typedef struct {
 uint32_t version, active, ended, valid;
 uint64_t dropped, pending, rows_committed;
} kt_status_v1;
kws3_status kt_create(const char *, const char *, kws3_decoder **);
kws3_status kt_begin(kws3_decoder *);
kws3_status kt_step_f32(kws3_decoder *,const float *,size_t,kws3_event *,size_t,size_t *);
kws3_status kt_drain(kws3_decoder *,kws3_ledger_sink,void *);
kws3_status kt_status(kws3_decoder *,kt_status_v1 *);
kws3_status kt_summary(kws3_decoder *,kws3_summary *);
kws3_status kt_retention(kws3_decoder *,kws3_retention_summary *);
kws3_status kt_end(kws3_decoder *);
kws3_status kt_destroy(kws3_decoder *);
size_t kt_status_bytes(void);
size_t kt_record_bytes(void);
#endif
