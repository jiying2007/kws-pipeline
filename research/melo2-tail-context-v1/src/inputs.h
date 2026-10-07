/* Exact two-stream byte and integer geometry; no runtime execution. */
typedef struct {size_t call_index,available_samples,call_samples,waveform_samples,fbank_rows,splice_rows,selected_rows,is_final_short,wave_samples,feature_count,offset,decoder_total_frames; const char *phase; uint64_t centers[10];} callback_spec;
typedef struct {const char *id,*pcm,*wav_sha,*pcm_sha; size_t frames,callback_count,data_offset; const callback_spec *plan;} clip_spec;
static const callback_spec plan_0[4] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,14400,4800,5120,30,30,10,0,320,4,1,87,"feed",{57,60,63,66,69,72,75,78,81,84}},
{3,16875,2475,2795,15,15,5,1,395,4,1,102,"finish",{87,90,93,96,99}},
};
static const callback_spec plan_1[4] = {
{0,4800,4800,4800,28,26,9,0,320,4,1,27,"feed",{0,3,6,9,12,15,18,21,24}},
{1,9600,4800,5120,30,30,10,0,320,4,1,57,"feed",{27,30,33,36,39,42,45,48,51,54}},
{2,14400,4800,5120,30,30,10,0,320,4,1,87,"feed",{57,60,63,66,69,72,75,78,81,84}},
{3,17804,3404,3724,21,21,7,1,364,4,1,108,"finish",{87,90,93,96,99,102,105}},
};
static const clip_spec clips[2] = {
{"M1","data/M1.wav","f4fd514d2e492e1126453b933dd107a62ead80590229f9dce380be2b1f9f64b5","ed139c2338c9cfbca9a9c53ab94f353fe1ccb81d7ea6067cc67e1af90ea9a012",16875,4,44,plan_0},
{"M2","data/M2.wav","cf6dd4f436eb1003947d53fbb71c2200f1604bdd856ceef3bdca339ba8698ce9","3e93b417cef88af1663294c03cafc84000038dd6ad786bc79cc07f6f458eac13",17804,4,44,plan_1},
};
#define CLIP_COUNT 2
#define TOTAL_SAMPLES 34679
