/* Exact three-stream byte and integer geometry; no runtime execution. */
typedef struct {size_t call_index,available_samples,call_samples,waveform_samples,fbank_rows,splice_rows,selected_rows,is_final_short,wave_samples,feature_count,offset,decoder_total_frames; const char *phase; uint64_t centers[10];} callback_spec;
typedef struct {const char *id,*pcm,*wav_sha,*pcm_sha; size_t frames,callback_count,data_offset; const callback_spec *plan;} clip_spec;
static const callback_spec plan_0[4] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,14400,4800,5120,30,30,10,0,320,4,1,87,"feed",{57,60,63,66,69,72,75,78,81,84}},
{3,17246,2846,3166,18,18,6,1,286,4,1,105,"finish",{87,90,93,96,99,102}},
};
static const callback_spec plan_1[5] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,14400,4800,5120,30,30,10,0,320,4,1,87,"feed",{57,60,63,66,69,72,75,78,81,84}},
{3,19200,4800,5120,30,30,10,0,320,4,1,117,"feed",{87,90,93,96,99,102,105,108,111,114}},
{4,19847,647,967,4,4,1,1,327,4,0,120,"finish",{117}},
};
static const callback_spec plan_2[4] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,14400,4800,5120,30,30,10,0,320,4,1,87,"feed",{57,60,63,66,69,72,75,78,81,84}},
{3,16318,1918,2238,12,12,4,1,318,4,1,99,"finish",{87,90,93,96}},
};
static const clip_spec clips[3] = {
{"M3","data/M3.wav","8291e8ff90ad764d0ba3703d098786e500b6da8086863ea0b7a7b606aa3e8128","a2ae464c74824bb5cb1ae9b6146431d5f87f4900a46cc1222dd8afabeb187553",17246,4,44,plan_0},
{"M4","data/M4.wav","73b2aa90e2ca521f820a9208b1a184e50efbc27fe26e0f2b7fa842b0c2e2e5ec","d2a07149843653a4b14e06375b49c1e2500ef7c5c5918065e20757ea0ada0e6a",19847,5,44,plan_1},
{"M5","data/M5.wav","cbb15601711179d6486a1fa387f8a0ac322949675af099c796455a0e5f6d8275","d72baef8c5b3f8c5bbb8174947c7ce395c03f36dee8b4bf4cd5769771d0739b6",16318,4,44,plan_2},
};
#define CLIP_COUNT 3
#define TOTAL_SAMPLES 53411
