# Generated mechanically from TU-CONTRACTS.json; no configure or compiler ran.
# All usage requirements are PRIVATE. No upstream target graph is imported.

add_library(kws-private-kaldi-decoder STATIC
  "${KWS_SOURCE_ROOT}/vendor/kaldi-decoder/kaldi-decoder/csrc/faster-decoder.cc"
)
set_target_properties(kws-private-kaldi-decoder PROPERTIES
  POSITION_INDEPENDENT_CODE FALSE
  INTERPROCEDURAL_OPTIMIZATION FALSE
  CXX_SCAN_FOR_MODULES FALSE
)
target_compile_definitions(kws-private-kaldi-decoder PRIVATE
  "SHERPA_ONNX_ENABLE_DIRECTML=0"
  "SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=0"
  "SHERPA_ONNX_ENABLE_TTS=0"
  "NDEBUG"
)
target_compile_options(kws-private-kaldi-decoder PRIVATE
  "-O3"
  "-std=c++17"
  "-fPIC"
  "-fvisibility=hidden"
  "-fvisibility-inlines-hidden"
)
target_include_directories(kws-private-kaldi-decoder PRIVATE
  "${KWS_SOURCE_ROOT}/vendor/kaldi-decoder"
  "${KWS_SOURCE_ROOT}/source"
  "${KWS_SOURCE_ROOT}/vendor/kaldifst"
  "${KWS_SOURCE_ROOT}/vendor/openfst/src/include"
  "${KWS_SOURCE_ROOT}/vendor/eigen"
)

add_library(kws-private-fbank STATIC
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/feature-fbank.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/feature-functions.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/feature-mfcc.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/feature-raw-audio-samples.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/feature-window.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/kaldi-math.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/mel-computations.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/online-feature.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/rfft.cc"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kaldi-native-fbank/csrc/whisper-feature.cc"
)
set_target_properties(kws-private-fbank PROPERTIES
  POSITION_INDEPENDENT_CODE FALSE
  INTERPROCEDURAL_OPTIMIZATION FALSE
  CXX_SCAN_FOR_MODULES FALSE
)
target_compile_definitions(kws-private-fbank PRIVATE
  "SHERPA_ONNX_ENABLE_DIRECTML=0"
  "SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=0"
  "SHERPA_ONNX_ENABLE_TTS=0"
  "kiss_fft_scalar=float"
  "NDEBUG"
)
target_compile_options(kws-private-fbank PRIVATE
  "-O3"
  "-std=c++17"
  "-fPIC"
  "-fvisibility=hidden"
  "-fvisibility-inlines-hidden"
)
target_include_directories(kws-private-fbank PRIVATE
  "${KWS_SOURCE_ROOT}/source"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank/kissfft"
  "${KWS_SOURCE_ROOT}/vendor/kissfft"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank"
)

add_library(kws-private-kissfft STATIC
  "${KWS_SOURCE_ROOT}/vendor/kissfft/kiss_fft.c"
  "${KWS_SOURCE_ROOT}/vendor/kissfft/kiss_fftr.c"
)
set_target_properties(kws-private-kissfft PROPERTIES
  POSITION_INDEPENDENT_CODE FALSE
  INTERPROCEDURAL_OPTIMIZATION FALSE
  CXX_SCAN_FOR_MODULES FALSE
)
target_compile_definitions(kws-private-kissfft PRIVATE
  "SHERPA_ONNX_ENABLE_DIRECTML=0"
  "SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=0"
  "SHERPA_ONNX_ENABLE_TTS=0"
  "kiss_fft_scalar=float"
  "NDEBUG"
)
target_compile_options(kws-private-kissfft PRIVATE
  "-O3"
  "-fPIC"
  "-fvisibility=hidden"
  "-ffast-math"
  "-fomit-frame-pointer"
  "-W"
  "-Wall"
  "-Wcast-align"
  "-Wcast-qual"
  "-Wshadow"
  "-Wwrite-strings"
  "-Wstrict-prototypes"
  "-Wmissing-prototypes"
  "-Wnested-externs"
  "-Wbad-function-cast"
)
target_include_directories(kws-private-kissfft PRIVATE
  "${KWS_SOURCE_ROOT}/source"
  "${KWS_SOURCE_ROOT}/vendor/kissfft"
)

add_library(sherpa-onnx-kws-core STATIC
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/base64-decode.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/bbpe.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/cat.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/context-graph.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/features.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/file-utils.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/hypothesis.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/keyword-spotter-impl.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/keyword-spotter.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-conformer-transducer-model.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-ebranchformer-transducer-model.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-lstm-transducer-model.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-model-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-nemo-ctc-model-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-paraformer-model-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-stream.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-t-one-ctc-model-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-transducer-decoder.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-transducer-model-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-transducer-model.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-wenet-ctc-model-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-zipformer-transducer-model.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-zipformer2-ctc-model-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/online-zipformer2-transducer-model.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/onnx-utils.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/parse-options.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/provider-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/provider.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/qnn-config.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/resample.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/session.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/symbol-table.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/text-utils.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/transducer-keyword-decoder.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/unbind.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/utils.cc"
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/csrc/version.cc"
)
set_target_properties(sherpa-onnx-kws-core PROPERTIES
  POSITION_INDEPENDENT_CODE FALSE
  INTERPROCEDURAL_OPTIMIZATION FALSE
  CXX_SCAN_FOR_MODULES FALSE
)
target_compile_definitions(sherpa-onnx-kws-core PRIVATE
  "SHERPA_ONNX_ENABLE_DIRECTML=0"
  "SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=0"
  "SHERPA_ONNX_ENABLE_TTS=0"
  "kiss_fft_scalar=float"
  "NDEBUG"
)
target_compile_options(sherpa-onnx-kws-core PRIVATE
  "-O3"
  "-std=c++17"
  "-fPIC"
  "-fvisibility=hidden"
  "-fvisibility-inlines-hidden"
)
target_include_directories(sherpa-onnx-kws-core PRIVATE
  "${KWS_SOURCE_ROOT}/vendor/kaldi-decoder"
  "${KWS_SOURCE_ROOT}/vendor/nlohmann-json/include"
  "${KWS_SOURCE_ROOT}/source"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank"
  "${KWS_SOURCE_ROOT}/vendor/kissfft"
  "${KWS_SOURCE_ROOT}/vendor/kaldifst"
  "${KWS_SOURCE_ROOT}/vendor/openfst/src/include"
  "${KWS_SOURCE_ROOT}/vendor/eigen"
  "${KWS_SOURCE_ROOT}/vendor/simple-sentencepiece"
)
target_include_directories(sherpa-onnx-kws-core SYSTEM PRIVATE
  "${KWS_SOURCE_ROOT}/vendor/onnxruntime-headers/include"
)

add_library(kws-private-sentencepiece STATIC
  "${KWS_SOURCE_ROOT}/vendor/simple-sentencepiece/ssentencepiece/csrc/ssentencepiece.cc"
)
set_target_properties(kws-private-sentencepiece PROPERTIES
  POSITION_INDEPENDENT_CODE FALSE
  INTERPROCEDURAL_OPTIMIZATION FALSE
  CXX_SCAN_FOR_MODULES FALSE
)
target_compile_definitions(kws-private-sentencepiece PRIVATE
  "SHERPA_ONNX_ENABLE_DIRECTML=0"
  "SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=0"
  "SHERPA_ONNX_ENABLE_TTS=0"
  "NDEBUG"
)
target_compile_options(kws-private-sentencepiece PRIVATE
  "-O3"
  "-std=c++17"
  "-fPIC"
  "-fvisibility=hidden"
  "-fvisibility-inlines-hidden"
)
target_include_directories(kws-private-sentencepiece PRIVATE
  "${KWS_SOURCE_ROOT}/vendor/kaldi-decoder"
  "${KWS_SOURCE_ROOT}/vendor/onnxruntime-headers/include"
  "${KWS_SOURCE_ROOT}/source"
  "${KWS_SOURCE_ROOT}/vendor/simple-sentencepiece"
)

add_library(sherpa-onnx-kws-c-api SHARED
  "${KWS_SOURCE_ROOT}/source/sherpa-onnx/c-api/kws-c-api.cc"
)
set_target_properties(sherpa-onnx-kws-c-api PROPERTIES
  POSITION_INDEPENDENT_CODE FALSE
  INTERPROCEDURAL_OPTIMIZATION FALSE
  CXX_SCAN_FOR_MODULES FALSE
  DEFINE_SYMBOL sherpa_onnx_c_api_EXPORTS
)
target_compile_definitions(sherpa-onnx-kws-c-api PRIVATE
  "SHERPA_ONNX_BUILD_MAIN_LIB=1"
  "SHERPA_ONNX_BUILD_SHARED_LIBS=1"
  "SHERPA_ONNX_ENABLE_DIRECTML=0"
  "SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=0"
  "SHERPA_ONNX_ENABLE_TTS=0"
  "kiss_fft_scalar=float"
  "NDEBUG"
)
target_compile_options(sherpa-onnx-kws-c-api PRIVATE
  "-O3"
  "-std=c++17"
  "-fPIC"
  "-fvisibility=hidden"
  "-fvisibility-inlines-hidden"
)
target_include_directories(sherpa-onnx-kws-c-api PRIVATE
  "${KWS_SOURCE_ROOT}/vendor/kaldi-decoder"
  "${KWS_SOURCE_ROOT}/vendor/nlohmann-json/include"
  "${KWS_SOURCE_ROOT}/source"
  "${KWS_SOURCE_ROOT}/vendor/kaldi-native-fbank"
  "${KWS_SOURCE_ROOT}/vendor/kissfft"
  "${KWS_SOURCE_ROOT}/vendor/kaldifst"
  "${KWS_SOURCE_ROOT}/vendor/openfst/src/include"
  "${KWS_SOURCE_ROOT}/vendor/eigen"
  "${KWS_SOURCE_ROOT}/vendor/simple-sentencepiece"
)
target_include_directories(sherpa-onnx-kws-c-api SYSTEM PRIVATE
  "${KWS_SOURCE_ROOT}/vendor/onnxruntime-headers/include"
)
