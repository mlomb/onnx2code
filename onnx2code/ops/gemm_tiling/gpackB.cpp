template <int kc, int nr, int StrideCol, int StrideRow>
inline void gpackB_panel(
    const float* __restrict__ B,
    float* __restrict__ B_panel  // kc x nr
) {
    for (int r = 0; r < kc; r++) {
        // copy row of nr
        for (int c = 0; c < nr; c++) {
            B_panel[c] = B[c * StrideCol];
        }

        // advance row
        B_panel += nr;
        B += StrideRow;
    }
}

template <int kc, int nc, int nr, int StrideCol, int StrideRow>
inline void gpackB(
    const float* __restrict__ B,
    float* __restrict__ B_panel  // kc x nc
) {
    const int NP = nc / nr;
    // const int NPl = nc % nr;

    // if (NPl > 0)
    //     memset(B_panel, 0, kc * nc * sizeof(float));

    for (int p = 0; p < NP; p++) {
        gpackB_panel<kc, nr, StrideCol, StrideRow>(B, B_panel);

        // advance panel
        B_panel += kc * nr;
        B += nr * StrideCol;
    }
}
