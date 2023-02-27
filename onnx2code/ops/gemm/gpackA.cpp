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

template <int MR, int KC, int StrideCol, int StrideRow>
inline void gpackA_panel(
    float* __restrict__ A,
    float* __restrict__ A_panel  // mr x kc
) {
    for (int c = 0; c < KC; c++) {
        // copy column of mr
        for (int r = 0; r < MR; r++) {
            A_panel[r] = A[r * StrideRow];
        }

        // advance column
        A_panel += MR;
        A += StrideCol;
    }
}

template <int MC, int KC, int MR, int StrideCol, int StrideRow>
inline void gpackA(
    float* __restrict__ A,
    float* __restrict__ A_panel  // mc x kc
) {
    const int MP = MC / MR;
    const int MPl = MC % MR;

    for (int p = 0; p < MP; p++) {
        gpackA_panel<MR, KC, StrideCol, StrideRow>(A, A_panel);

        // advance panel
        A_panel += MR * KC;
        A += MR * StrideRow;
    }
    if (MPl > 0) {
        // TODO: handle leftover (padding with zeros)
        assert(false);
    }
}
