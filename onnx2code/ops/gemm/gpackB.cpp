

template <int KC, int NR, int StrideCol, int StrideRow>
inline void gpackB_block(
    float* __restrict__ B,
    float* __restrict__ B_panel  // kc x nr
) {
    for (int r = 0; r < KC; r++) {
        // copy row of nr
        for (int c = 0; c < NR; c++) {
            B_panel[c] = B[c * StrideCol];
        }

        // advance row
        B_panel += NR;
        B += StrideRow;
    }
}

template <int KC, int NC, int NR, int StrideCol, int StrideRow>
inline void gpackB(
    float* __restrict__ B,
    float* __restrict__ B_panel  // kc x nc
) {
    const int NP = NC / NR;
    const int NPl = NC % NR;

    for (int p = 0; p < NP; p++) {
        gpackB_block<KC, NR, StrideCol, StrideRow>(B, B_panel);

        // advance block
        B_panel += KC * NR;
        B += NR * StrideCol;
    }
    if (NPl > 0) {
        // TODO: handle leftover (padding with zeros)
        assert(false);
    }
}
