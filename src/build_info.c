#include "kws_pipeline/kws.h"
#include "kws_build_config.h"

const kws_build_info_t *kws_build_info(void) {
  static const kws_build_info_t info = {
      (uint32_t)sizeof(kws_build_info_t),
      KWS_BUILD_INFO_API_VERSION,
      KWS_BUILD_VERSION,
      KWS_BUILD_SOURCE_REVISION,
      KWS_BUILD_COMPILER_ID,
      KWS_BUILD_COMPILER_VERSION,
      KWS_BUILD_TARGET_TRIPLE,
      KWS_BUILD_TYPE,
      KWS_BUILD_CONFIG_DIGEST,
      {0u},
  };
  return &info;
}
