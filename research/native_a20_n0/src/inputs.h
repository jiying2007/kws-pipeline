/* Fixed three complete N0 streams; metadata only. */
typedef struct {const char *id,*pcm,*wav_sha,*pcm_sha; size_t frames;} clip_spec;
static const clip_spec clips[3] = {
{"n0-low-triangular","inputs/pcm/n0-low-triangular.pcm","3d41adac31062b1a3176120ed1df8662da1d2395463a60d2f20373d49d5e47cc","18e16a761a2a9b2cf638854c5a8af939d10e6d036273eebb8aa9661795850a07",4800000},
{"n0-colored-ma32","inputs/pcm/n0-colored-ma32.pcm","35fff1350de000becd105ab57800e1f4f78c9a8bcd926167bdd54a3878b6f025","d3810ff259d1da943b520b520b8a27ff227f6563ba687713ef53333c61f1e22e",4800000},
{"n0-sparse-transients","inputs/pcm/n0-sparse-transients.pcm","b91fcff6324f1239267de24ff50e75414155c04af0e1b9f9cffe1e898ce3f9b2","f8d5a2e5d7c55ee58c5a6fb868342518ad3c759c986afd667591506a0af986c6",4800000},
};
