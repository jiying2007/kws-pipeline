/* Opaque workspace ABI; no DSP changes and no model/CMVN/decoder symbols. */
#include "feature_collector.h"
__attribute__((visibility("default"))) size_t a20_recipe_workspace_bytes(void) { return sizeof(a20_recipe_features); }
__attribute__((visibility("default"))) size_t a20_recipe_rows_count(const a20_recipe_features *p) { return p->row_count; }
__attribute__((visibility("default"))) size_t a20_recipe_calls_count(const a20_recipe_features *p) { return p->call_count; }
__attribute__((visibility("default"))) const float *a20_recipe_rows_pointer(const a20_recipe_features *p) { return &p->rows[0][0]; }
__attribute__((visibility("default"))) const a20_recipe_call *a20_recipe_calls_pointer(const a20_recipe_features *p) { return p->calls; }

__attribute__((visibility("default"))) int a20_recipe_execute(a20_recipe_features *p, const int16_t *x, size_t n) { return a20_recipe_collect(p, x, n); }
