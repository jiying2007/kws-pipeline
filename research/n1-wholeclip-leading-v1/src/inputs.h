/* Exact one-stream byte and integer geometry; no runtime execution. */
typedef struct {size_t call_index,available_samples,call_samples,waveform_samples,fbank_rows,splice_rows,selected_rows,is_final_short,wave_samples,feature_count,offset,decoder_total_frames; const char *phase; uint64_t centers[10];} callback_spec;
typedef struct {const char *id,*pcm,*wav_sha,*pcm_sha; size_t frames,callback_count,data_offset; const callback_spec *plan;} clip_spec;
static const callback_spec plan_0[11] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,14400,4800,5120,30,30,10,0,320,4,1,87,"feed",{57,60,63,66,69,72,75,78,81,84}},
{3,19200,4800,5120,30,30,10,0,320,4,1,117,"feed",{87,90,93,96,99,102,105,108,111,114}},
{4,24000,4800,5120,30,30,10,0,320,4,1,147,"feed",{117,120,123,126,129,132,135,138,141,144}},
{5,28800,4800,5120,30,30,10,0,320,4,1,177,"feed",{147,150,153,156,159,162,165,168,171,174}},
{6,33600,4800,5120,30,30,10,0,320,4,1,207,"feed",{177,180,183,186,189,192,195,198,201,204}},
{7,38400,4800,5120,30,30,10,0,320,4,1,237,"feed",{207,210,213,216,219,222,225,228,231,234}},
{8,43200,4800,5120,30,30,10,0,320,4,1,267,"feed",{237,240,243,246,249,252,255,258,261,264}},
{9,48000,4800,5120,30,30,10,0,320,4,1,297,"feed",{267,270,273,276,279,282,285,288,291,294}},
{10,52578,4578,4898,29,29,10,1,258,4,2,327,"finish",{297,300,303,306,309,312,315,318,321,324}},
};
static const clip_spec clips[1] = {
{"N1","data/N1.wav","2752dab0ca534795cf29da1726f8abd17fb76a5bc2723f1e11c91a9ddb9c46ae","bd3236facf96bef048b8d4b45d9a24729866bcf4048e7366d7f406d0a2d9f99e",52578,11,44,plan_0},
};
#define CLIP_COUNT 1
#define TOTAL_SAMPLES 52578
