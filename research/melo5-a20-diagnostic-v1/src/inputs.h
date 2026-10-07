/* Exact five-source byte and integer geometry; no runtime execution. */
typedef struct {size_t call_index,available_samples,call_samples,waveform_samples,fbank_rows,splice_rows,selected_rows,is_final_short,wave_samples,feature_count,offset,decoder_total_frames; const char *phase; uint64_t centers[10];} callback_spec;
typedef struct {const char *id,*pcm,*wav_sha,*pcm_sha; size_t frames,callback_count,data_offset; const callback_spec *plan;} clip_spec;
static const callback_spec plan_0[3] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,12075,2475,2795,15,15,5,1,395,4,1,72,"finish",{57,60,63,66,69}},
};
static const callback_spec plan_1[3] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,13004,3404,3724,21,21,7,1,364,4,1,78,"finish",{57,60,63,66,69,72,75}},
};
static const callback_spec plan_2[3] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,12446,2846,3166,18,18,6,1,286,4,1,75,"finish",{57,60,63,66,69,72}},
};
static const callback_spec plan_3[4] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,14400,4800,5120,30,30,10,0,320,4,1,87,"feed",{57,60,63,66,69,72,75,78,81,84}},
{3,15047,647,967,4,4,1,1,327,4,0,90,"finish",{87}},
};
static const callback_spec plan_4[3] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,11518,1918,2238,12,12,4,1,318,4,1,69,"finish",{57,60,63,66}},
};
static const clip_spec clips[5] = {
{"M1","data/M1.wav","c5cf2886eed0929f611cc6cad3135c56cf55620a2be2cc26b030845207d8e46a","b0085e8e55e81fb3865a44f76bb55a1ceb8ca132fe70513690a797cb490aeb74",12075,3,44,plan_0},
{"M2","data/M2.wav","306e17a73d4970534c7a2e85d203924cb8b3b2baf23d7d8ebafd8da89fb3f9fb","df483c3bc72c8364bc30987bac35a32b4f41cfd56a6acf1a0e2d5fcd234f735c",13004,3,44,plan_1},
{"M3","data/M3.wav","82d3ce86355a9475a2057099c75e41d87fbf096b621a9bb20eed80c7976f51c7","4a69d875a04bfd4477f96c7685f1b19832fc540d24ccdeb0c5d40586f2dc2bb3",12446,3,44,plan_2},
{"M4","data/M4.wav","b70bc3effb0fca66e3efdb4f762dc2b2830311cb956ffe216aad569342071f97","774e55e34e0744327f56e43c6da3a91abdbea91febb1891db05acb454110f6de",15047,4,44,plan_3},
{"M5","data/M5.wav","5dc8b44de88d747de8ef890f89937a9d91d2ca9e432f624646a959304d86f027","3c301bda58704126572d4b6a7a1f48493e727927e171cfe9a3ddfa3fe7479e96",11518,3,44,plan_4},
};
#define CLIP_COUNT 5
#define TOTAL_SAMPLES 64090
