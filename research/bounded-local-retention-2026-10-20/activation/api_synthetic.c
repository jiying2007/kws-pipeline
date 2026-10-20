/* Invented one-hot blank rows only; never reads inputs or models. */
#include "kws3.h"
#include "kws3_retention.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
kws3_status kr_create(const char *,const char *,kws3_decoder **);
kws3_status kr_step_f32(kws3_decoder *,const float *,size_t,kws3_event *,size_t,size_t *);
kws3_status kr_summary(const kws3_decoder *,kws3_summary *);
kws3_status kr_retention(const kws3_decoder *,kws3_retention_summary *);
void kr_destroy(kws3_decoder *);
int main(void) {
    float row[KWS3_CLASSES]={0}, saved[KWS3_CLASSES];
    kws3_event events[3]; kws3_summary s; kws3_retention_summary r;
    kws3_decoder *d=NULL; size_t n=99; unsigned i;
    row[0]=1; memcpy(saved,row,sizeof(row));
    assert(kr_create("bounded-retention-research-v1","synthetic-blank-only",&d)==KWS3_OK);
    for(i=0;i<130;i++) {
        assert(kr_step_f32(d,row,KWS3_CLASSES,events,3,&n)==KWS3_OK);
        assert(n==0);assert(memcmp(row,saved,sizeof(row))==0);
    }
    assert(kr_summary(d,&s)==KWS3_OK); assert(s.rows_decoded==130 && s.events_emitted==0 && s.valid && !s.finished);
    assert(kr_retention(d,&r)==KWS3_OK);assert(r.rows_committed==130 && r.episodes_started==0);
    row[1]=NAN;
    assert(kr_step_f32(d,row,KWS3_CLASSES,events,3,&n)==KWS3_POSTERIOR); assert(n==0);
    assert(kr_summary(d,&s)==KWS3_OK);assert(s.rows_decoded==130 && !s.valid && s.invalid_reason==KWS3_POSTERIOR);
    assert(kr_retention(d,&r)==KWS3_OK);assert(r.rows_committed==130); /* Snapshot is not a success flag. */
    memcpy(row,saved,sizeof(row));assert(kr_step_f32(d,row,KWS3_CLASSES,events,3,&n)==KWS3_INVALID);
    kr_destroy(d);
    puts("{\"status\":\"PASS_PUBLIC_ABI_SYNTHETIC\",\"synthetic_valid_rows\":130,\"invalid_row_tests\":2,\"scientific_replays\":0,\"model_calls\":0,\"real_posterior_reads\":0,\"retention_branch_coverage\":false}");
    return 0;
}
