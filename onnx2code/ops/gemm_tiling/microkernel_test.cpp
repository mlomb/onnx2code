template <
    int mv,
    int nu,

    int CStrideRow,
    int CStrideCol>
inline void unit_update(
    const float* __restrict__ a,  // mv
    const float* __restrict__ b,  // nu
    float* __restrict__ C         // mv x nu
) {
    for (int i = 0; i < mv; i++) {
        for (int j = 0; j < nu; j++) {
            C[i * CStrideRow + j * CStrideCol] += a[i] * b[j];
        }
    }
}

template <
    int mr,
    int nr,
    int mv,
    int nu>
inline void test_microkernel(
    int kc,
    const float* __restrict__ A_kernel,  // (mr x kc) column major
    const float* __restrict__ B_kernel,  // (kc x nr) row major
    float* __restrict__ AB               // (mr x nr)
) {
    static_assert(mr % mv == 0, "must be conforming");
    static_assert(nr % nu == 0, "must be conforming");

    for (int k = 0; k < kc; k++) {
        // single outer product
        // en una columna de A y una fila de B (del zigzag)

        // loop tiling
        for (int j = 0; j < nr; j += nu) {
            for (int i = 0; i < mr; i += mv) {
                // unit update (small outer product)
                unit_update<mv, nu, nr, 1>(
                    A_kernel + i,
                    B_kernel + j,
                    AB + i * nr + j);
            }
        }

        // advance one column of A and one row of B
        A_kernel += mr;
        B_kernel += nr;
    }
}
