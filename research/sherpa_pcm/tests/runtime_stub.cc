// Model-free lifecycle fake only. It makes no speech/quality/performance claim.
#include "sherpa-onnx/c-api/c-api.h"
#include <cstring>
struct SherpaOnnxKeywordSpotter {};
struct SherpaOnnxOnlineStream {mutable int pending=0,total=0;mutable bool hit=false,bad=false;};
extern "C" {
const SherpaOnnxKeywordSpotter* SherpaOnnxCreateKeywordSpotter(const SherpaOnnxKeywordSpotterConfig*c){
 if(!std::strcmp(c->model_config.transducer.encoder,"fail"))return nullptr;
 if(c->feat_config.sample_rate!=16000||c->feat_config.feature_dim!=80||c->model_config.num_threads!=1||c->max_active_paths!=4||c->num_trailing_blanks!=1||c->keywords_score!=1.0f||c->keywords_threshold!=.25f)return nullptr;
 return new SherpaOnnxKeywordSpotter;
}
void SherpaOnnxDestroyKeywordSpotter(const SherpaOnnxKeywordSpotter*p){delete p;}
const SherpaOnnxOnlineStream* SherpaOnnxCreateKeywordStream(const SherpaOnnxKeywordSpotter*){return new SherpaOnnxOnlineStream;}
void SherpaOnnxDestroyOnlineStream(const SherpaOnnxOnlineStream*p){delete p;}
void SherpaOnnxOnlineStreamAcceptWaveform(const SherpaOnnxOnlineStream*s,int32_t,const float*p,int32_t n){s->pending+=n;if(n&&p[0]<0)s->bad=true;}
int32_t SherpaOnnxIsKeywordStreamReady(const SherpaOnnxKeywordSpotter*,const SherpaOnnxOnlineStream*s){return s->pending>0;}
void SherpaOnnxDecodeKeywordStream(const SherpaOnnxKeywordSpotter*,const SherpaOnnxOnlineStream*s){s->total+=s->pending;s->pending=0;s->hit=s->total>=640;}
const SherpaOnnxKeywordResult* SherpaOnnxGetKeywordResult(const SherpaOnnxKeywordSpotter*,const SherpaOnnxOnlineStream*s){if(s->bad)return nullptr;auto*r=new SherpaOnnxKeywordResult{};r->keyword=s->hit?"你好小窝":"";return r;}
void SherpaOnnxDestroyKeywordResult(const SherpaOnnxKeywordResult*r){delete r;}
void SherpaOnnxResetKeywordStream(const SherpaOnnxKeywordSpotter*,const SherpaOnnxOnlineStream*s){s->hit=false;s->total=0;}
void SherpaOnnxOnlineStreamInputFinished(const SherpaOnnxOnlineStream*){}
}
