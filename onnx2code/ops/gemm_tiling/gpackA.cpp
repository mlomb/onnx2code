// Referencias:
//  - [Automating the Last-Mile for High Performance Dense Linear Algebra] Figure 1 ~Ai (verde)
//  - https://github.com/michael-lehn/ulmBLAS/blob/191efa54ddb595760353a1ca557a886fa74a864a/ulmblas/level3/pack/gepack.tcc

// for (1..mp) bloques de mr filas (dentro de las mc filas del panel)
//   for (1..kc) columnas del bloque de A (kc)
//       for (1..mr) filas de mr (dentro de las mc filas)

// La matriz A se lee en forma ROW MAJOR (onnx)
// El panel de A se puede pensar como un tensor de 3 dimensiones: (mp, kc, mr)

// Target (los numeros son los indices originales de A que queremos en el packeado)
// -----------------  -|      -|
// | 0 | 2 | 4 | 6 |   |       |
// | 1 | 3 | 5 | 7 |   |  mr   |
// -----------------  -|       |  mc
// | 8 | 10| 12| 14|   |  mr   |
// | 9 | 11| 13| 15|   |       |
// -----------------  -|      -|
// |---------------|
//         kc
// mp = 2

template <int kc, int mr, int StrideCol, int StrideRow>
inline void gpackA_panel(
    const float* __restrict__ A,
    float* __restrict__ A_panel  // mr x kc
) {
    for (int c = 0; c < kc; c++) {
        // copy column of mr
        for (int r = 0; r < mr; r++) {
            A_panel[r] = A[r * StrideRow];
        }

        // advance column
        A_panel += mr;
        A += StrideCol;
    }
}

template <int kc, int mc, int mr, int StrideCol, int StrideRow>
inline void gpackA(
    const float* __restrict__ A,
    float* __restrict__ A_panel  // mc x kc
) {
    for (int p = 0; p < mc; p += mr) {
        gpackA_panel<kc, mr, StrideCol, StrideRow>(A, A_panel);

        // advance panel
        A_panel += mr * kc;
        A += mr * StrideRow;
    }
}

// Edge case

template <int mr, int StrideCol, int StrideRow>
inline void gpackA_panel_edge(
    int _kc,
    int _mr,
    const float* __restrict__ A,
    float* __restrict__ A_panel  // mr x kc
) {
    for (int c = 0; c < _kc; c++) {
        // copy column of mr
        for (int r = 0; r < _mr; r++) {
            A_panel[r] = A[r * StrideRow];
        }

        // advance column
        A_panel += mr;
        A += StrideCol;
    }
}
template <int kc, int mc, int mr, int StrideCol, int StrideRow>
inline void gpackA_edge(
    int _kc,
    int _mc,
    const float* __restrict__ A,
    float* __restrict__ A_panel  // mc x kc
) {
    const int MP = _mc / mr;
    const int MPl = _mc % mr;

    memset(A_panel, 0, mc * kc * sizeof(float));

    for (int p = 0; p < MP; p++) {
        gpackA_panel_edge<mr, StrideCol, StrideRow>(_kc, mr, A, A_panel);

        // advance panel
        A_panel += mr * kc;
        A += mr * StrideRow;
    }

    if (MPl != 0) {
        gpackA_panel_edge<mr, StrideCol, StrideRow>(_kc, MPl, A, A_panel);
    }
}
