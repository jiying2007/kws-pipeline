#ifndef KWS_BENCH_STATISTICS_H
#define KWS_BENCH_STATISTICS_H

#include <math.h>
#include <stddef.h>

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

#endif
