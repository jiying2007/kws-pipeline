#include "src/a20_stream_fft64.h"
#include "baseline/native/runtime/a20_stream.h"
#include "baseline/native/donor_fbank/frontend_tables.h"
#include "src/fft64_twiddles.h"
#include <stdio.h>
#define FIELD(name, value) printf("\"%s\":%zu,\n",name,(size_t)(value))
#define OFF(type, field) FIELD(#type "." #field,offsetof(type,field))
int main(void) {
 puts("{");
 FIELD("float_bytes",sizeof(float));FIELD("double_bytes",sizeof(double));
 FIELD("pointer_bytes",sizeof(void*));FIELD("frontend_original_bytes",sizeof(donor_fbank_state));
 FIELD("frontend_bytes",sizeof(donor_fft64_state));FIELD("frontend_alignment",_Alignof(donor_fft64_state));
 FIELD("frontend_added_double_workspace_bytes",sizeof(((donor_fft64_state*)0)->fft_re)+sizeof(((donor_fft64_state*)0)->fft_im));
 FIELD("frontend_existing_float_scratch_bytes",sizeof(((donor_fft64_state*)0)->re)+sizeof(((donor_fft64_state*)0)->im));
 FIELD("frontend_trace_bytes",sizeof(donor_fft64_trace));FIELD("frontend_local_output_row_bytes",80*sizeof(float));
 FIELD("pcm_original_bytes",sizeof(donor_pcm_state));FIELD("pcm_bytes",sizeof(donor_fft64_pcm_state));
 FIELD("pcm_alignment",_Alignof(donor_fft64_pcm_state));FIELD("splice_bytes",sizeof(donor_splice_state));
 FIELD("stream_original_bytes",sizeof(a20s_state));FIELD("stream_bytes",sizeof(a20fft64s_state));
 FIELD("stream_alignment",_Alignof(a20fft64s_state));FIELD("weights_bytes",A20_FLOATS*sizeof(float));
 FIELD("model_state_bytes",sizeof(a20_model));FIELD("model_cache_bytes",A20_CACHE_FLOATS*sizeof(float));
 FIELD("model_explicit_local_array_bytes",(400+250+250+128+128)*sizeof(float));
 FIELD("optional_model_trace_bytes",21*A20_TRACE_STRIDE*sizeof(float));
 FIELD("decoder_state_bytes",sizeof(a20d_decoder));FIELD("decoder_workspace_bytes",sizeof(a20d_workspace));
 FIELD("caller_arena_plus_weights_bytes",sizeof(a20fft64s_state)+A20_FLOATS*sizeof(float));
 FIELD("hamming_table_bytes",sizeof(donor_hamming));FIELD("mel_offsets_table_bytes",sizeof(donor_mel_offsets));
 FIELD("mel_bins_table_bytes",sizeof(donor_mel_bins));FIELD("mel_weights_table_bytes",sizeof(donor_mel_weights));
 FIELD("twiddle_tables_bytes",sizeof(donor_fft64_cos)+sizeof(donor_fft64_sin));
 FIELD("original_twiddle_tables_bytes",2*256*sizeof(float));
 FIELD("all_frontend_tables_bytes",sizeof(donor_hamming)+sizeof(donor_mel_offsets)+sizeof(donor_mel_bins)+sizeof(donor_mel_weights)+sizeof(donor_fft64_cos)+sizeof(donor_fft64_sin));
 OFF(donor_fft64_state,pcm);OFF(donor_fft64_state,re);OFF(donor_fft64_state,im);OFF(donor_fft64_state,fft_re);OFF(donor_fft64_state,fft_im);
 OFF(donor_fft64_pcm_state,frontend);OFF(donor_fft64_pcm_state,trace);OFF(donor_fft64_pcm_state,fbank);OFF(donor_fft64_pcm_state,rows);OFF(donor_fft64_pcm_state,centers);
 OFF(a20fft64s_state,pcm);OFF(a20fft64s_state,model);OFF(a20fft64s_state,decoder);OFF(a20fft64s_state,decoder_work);OFF(a20fft64s_state,logits);OFF(a20fft64s_state,initialized);OFF(a20fft64s_state,trace_buffer);
 FIELD("pcm_batch_bytes",sizeof(donor_fft64_pcm_batch));FIELD("stream_batch_bytes",sizeof(a20fft64s_batch));FIELD("decoder_result_bytes",sizeof(a20d_result));
 puts("\"a20_model_steps\":0,\"actual_audio_frames\":0,\"board_measured\":false}");
 return 0;
}
