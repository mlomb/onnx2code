template<
    int mr,
    int nr,
    int kc,

    int CStrideRow
>
inline void ref_microkernel(
    const float* __restrict__ A_kernel,  // (mr x kc) column major
    const float* __restrict__ B_kernel,  // (kc x nr) row major
    float* __restrict__ C
) {
    float AB[mr * nr];
    memset(AB, 0, mr * nr * sizeof(float));

    for (int k = 0; k < kc; k++) {
        for (int n = 0; n < nr; n++) {
            for (int m = 0; m < mr; m++) {
                AB[n * mr + m] +=
                    A_kernel[k * mr + m] *
                    B_kernel[k * nr + n];
            }
        }
    }

    for (int j = 0; j < nr; j++) {
        for (int i = 0; i < mr; i++) {
            C[i * CStrideRow + j] += AB[mr * j + i];
        }
    }
}
