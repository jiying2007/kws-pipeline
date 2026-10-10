#ifndef KWS_BENCH_STATISTICS_H
#define KWS_BENCH_STATISTICS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define KWS_BENCH_PERCENTILE_ESTIMATOR "nearest-rank-ceil-v1"

/* Nearest-rank empirical percentile for finite values sorted ascending:
 * rank = ceil(p * count), one-based. Endpoints select the first/last value;
 * an empty input returns zero. No interpolation or lower-rank truncation.
 * The benchmark calls this with p in {0.50, 0.95, 0.99}.
 */
static inline double kws_bench_percentile_nearest_rank(const double *values,
                                                      size_t count, double p) {
  size_t rank;
  if (count == 0u) {
    return 0.0;
  }
  if (p <= 0.0) {
    return values[0];
  }
  if (p >= 1.0) {
    return values[count - 1u];
  }
  rank = (size_t)ceil(p * (double)count);
  if (rank == 0u) {
    rank = 1u;
  }
  if (rank > count) {
    rank = count;
  }
  return values[rank - 1u];
}

/* Runtime statistics are lifetime counters: an algorithm/frontend reset does
 * not clear them. Admit only an exact, positive observed interval and reject
 * counter regression before subtracting unsigned values. */
static inline int kws_bench_counter_delta(uint64_t before, uint64_t after,
                                          uint64_t expected,
                                          uint64_t *out_delta) {
  if (out_delta == NULL) {
    return 0;
  }
  *out_delta = 0u;
  if (expected == 0u || after < before || after - before != expected) {
    return 0;
  }
  *out_delta = after - before;
  return 1;
}

#endif
