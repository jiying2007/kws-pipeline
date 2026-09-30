#include "pcm_kws.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
struct PcmKws {
 const SherpaOnnxKeywordSpotter *model=nullptr;
 const SherpaOnnxOnlineStream *stream=nullptr;
 PcmKwsCallback callback=nullptr;void *user=nullptr;
 float pending[320]{};size_t used=0;uint64_t received=0,fed=0;
 bool finished=false,faulted=false;
 ~PcmKws(){if(stream)SherpaOnnxDestroyOnlineStream(stream);if(model)SherpaOnnxDestroyKeywordSpotter(model);}
};
static int error(int code,char*b,size_t n,const char*text){if(b&&n)std::snprintf(b,n,"%s",text);return code;}
static void drain(PcmKws*x,bool eof){
 while(SherpaOnnxIsKeywordStreamReady(x->model,x->stream)){
  SherpaOnnxDecodeKeywordStream(x->model,x->stream);
  auto*r=SherpaOnnxGetKeywordResult(x->model,x->stream);
  if(!r)throw std::runtime_error("null keyword result");
  std::unique_ptr<const SherpaOnnxKeywordResult,decltype(&SherpaOnnxDestroyKeywordResult)> holder(r,SherpaOnnxDestroyKeywordResult);
  if(r->keyword&&r->keyword[0]){
   int id=!std::strcmp(r->keyword,"你好小窝")?1:(!std::strcmp(r->keyword,"小窝小窝")?2:0);
   PcmKwsEvent e{r->keyword,id,x->fed,int(eof)};
   if(x->callback)x->callback(x->user,&e);
   SherpaOnnxResetKeywordStream(x->model,x->stream);
  }
 }
}
static void push(PcmKws*x){if(!x->used)return;SherpaOnnxOnlineStreamAcceptWaveform(x->stream,16000,x->pending,int(x->used));x->fed+=x->used;x->used=0;drain(x,false);}
extern "C" int pcm_kws_create(const PcmKwsFiles*f,PcmKwsCallback cb,void*u,PcmKws**out,char*b,size_t n){
 if(out)*out=nullptr;
 if(!f||!out||!f->encoder||!f->decoder||!f->joiner||!f->tokens||!f->keywords||!f->encoder[0]||!f->decoder[0]||!f->joiner[0]||!f->tokens[0]||!f->keywords[0])return error(PCM_KWS_ARGUMENT,b,n,"missing model/token/keyword paths or output");
 try{
  auto x=std::make_unique<PcmKws>();SherpaOnnxKeywordSpotterConfig c{};
  c.feat_config={16000,80};c.model_config.transducer={f->encoder,f->decoder,f->joiner};c.model_config.tokens=f->tokens;c.model_config.provider="cpu";c.model_config.num_threads=1;
  c.max_active_paths=4;c.num_trailing_blanks=1;c.keywords_score=1.0f;c.keywords_threshold=.25f;c.keywords_file=f->keywords;
  x->model=SherpaOnnxCreateKeywordSpotter(&c);if(!x->model)throw std::runtime_error("keyword model creation failed");
  x->stream=SherpaOnnxCreateKeywordStream(x->model);if(!x->stream)throw std::runtime_error("stream creation failed");
  x->callback=cb;x->user=u;*out=x.release();return error(PCM_KWS_OK,b,n,"");
 }catch(const std::exception&e){return error(PCM_KWS_RUNTIME,b,n,e.what());}catch(...){return error(PCM_KWS_RUNTIME,b,n,"unknown runtime failure");}
}
extern "C" int pcm_kws_feed(PcmKws*x,const int16_t*p,size_t frames,char*b,size_t n){
 if(!x||(!p&&frames))return error(PCM_KWS_ARGUMENT,b,n,"null state or PCM");
 if(x->finished||x->faulted)return error(PCM_KWS_STATE,b,n,"finished or faulted; reset required");
 if(frames>std::numeric_limits<uint64_t>::max()-x->received)return error(PCM_KWS_ARGUMENT,b,n,"sample counter overflow");
 try{size_t pos=0;while(pos<frames){size_t take=std::min(size_t(320)-x->used,frames-pos);for(size_t j=0;j<take;++j)x->pending[x->used+j]=p[pos+j]/32768.0f;x->used+=take;pos+=take;x->received+=take;if(x->used==320)push(x);}return error(PCM_KWS_OK,b,n,"");}
 catch(const std::exception&e){x->faulted=true;return error(PCM_KWS_RUNTIME,b,n,e.what());}catch(...){x->faulted=true;return error(PCM_KWS_RUNTIME,b,n,"unknown runtime failure");}
}
extern "C" int pcm_kws_finish(PcmKws*x,char*b,size_t n){
 if(!x)return error(PCM_KWS_ARGUMENT,b,n,"null state");
 if(x->finished||x->faulted)return error(PCM_KWS_STATE,b,n,"finished or faulted; reset required");
 try{push(x);SherpaOnnxOnlineStreamInputFinished(x->stream);drain(x,true);x->finished=true;return error(PCM_KWS_OK,b,n,"");}catch(const std::exception&e){x->faulted=true;return error(PCM_KWS_RUNTIME,b,n,e.what());}catch(...){x->faulted=true;return error(PCM_KWS_RUNTIME,b,n,"unknown runtime failure");}
}
extern "C" int pcm_kws_reset(PcmKws*x,char*b,size_t n){
 if(!x)return error(PCM_KWS_ARGUMENT,b,n,"null state");
 try{auto*s=SherpaOnnxCreateKeywordStream(x->model);if(!s)throw std::runtime_error("stream creation failed");SherpaOnnxDestroyOnlineStream(x->stream);x->stream=s;x->used=0;x->fed=x->received=0;x->finished=x->faulted=false;return error(PCM_KWS_OK,b,n,"");}catch(const std::exception&e){x->faulted=true;return error(PCM_KWS_RUNTIME,b,n,e.what());}catch(...){x->faulted=true;return error(PCM_KWS_RUNTIME,b,n,"unknown runtime failure");}
}
extern "C" void pcm_kws_destroy(PcmKws*x){delete x;}
