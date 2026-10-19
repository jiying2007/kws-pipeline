static __attribute__((noinline)) void affine_reference(const float *x, float *y, const float *w, const float *b, int ni, int no) {
    for (int o = 0; o < no; ++o) {
        float sum[4] = {0, 0, 0, 0};
        for (int i = 0; i < ni; ++i) sum[i % 4] += x[i] * w[o * ni + i];
        y[o] = ((sum[0] + sum[1]) + (sum[2] + sum[3])) + (b ? b[o] : 0.0f);
    }
}
