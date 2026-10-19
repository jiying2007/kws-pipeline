/* SPDX-License-Identifier: Apache-2.0. Compile/link only; no model execution. */
#include "b_stream_adapter.h"
#include "bsa_capture.h"
#ifdef KWS_B_PROFILE
#include "b_profile.h"
#endif
int main() {
#ifdef KWS_B_PROFILE
  if (b_profile_size() != sizeof(b_profile_snapshot)) return 2;
#endif
  const bsa_descriptor *model = bsa_model();
  return model && bsa_abi_size(1) == sizeof(bsa_row) && bsa_destroy(nullptr) == BSA_OK ? 0 : 1;
}
