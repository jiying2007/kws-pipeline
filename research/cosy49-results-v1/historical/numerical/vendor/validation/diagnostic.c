/* Read-only diagnostic ABI. CMVN helper calls are budgeted separately. */
#include "../runtime/src/a20_stream_fft64.h"
int u8_cmvn(a20fft64s_state*s,const float*x,float*z){return a20_cmvn(&s->model,x,z);}
int64_t u8_total_frames(const a20fft64s_state*s){return s->decoder.total_frames;}
int64_t u8_last_active(const a20fft64s_state*s){return s->decoder.last_active_pos;}
double u8_hit_score(const a20fft64s_state*s){return s->decoder.hit_score;}
const a20d_decoder*u8_decoder(const a20fft64s_state*s){return &s->decoder;}
size_t u8_hyp_count(const a20fft64s_state*s){return s->decoder.hyp_count;}
size_t u8_front_bytes(void){return sizeof(donor_fft64_pcm_batch);}
size_t u8_result_bytes(void){return sizeof(a20d_result);}
size_t u8_batch_bytes(void){return sizeof(a20fft64s_batch);}
int u8_reset_valid(const a20fft64s_state*s){
 if(s->fault||s->busy||s->model.fault||s->pcm.total_ingress||s->pcm.total_calls||s->pcm.ingress_samples||s->pcm.splice.total_samples||s->pcm.splice.wave_samples||s->pcm.splice.feature_count||s->pcm.splice.offset||s->pcm.finished||s->decoder.total_frames||s->decoder.last_active_pos!=-1||s->decoder.hyp_count!=1||s->decoder.node_count||s->decoder.hit_score!=1.0)return 0;
 const unsigned char*p=(const unsigned char*)s->model.cache;for(size_t i=0;i<sizeof(s->model.cache);i++)if(p[i])return 0;return 1;
}
int u8_finished_valid(const a20fft64s_state*s){return !s->fault&&!s->busy&&!s->model.fault&&!s->pcm.faulted&&s->pcm.finished&&s->pcm.splice.finished&&!s->pcm.ingress_samples;}
