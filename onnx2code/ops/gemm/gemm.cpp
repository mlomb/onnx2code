template<
    // matrix sizes
    int M,
    int N,
    int K,

    int nc = N, // Columnas de panel de B
    int kc = 2, // Filas de panel de B
    int mc = 32, // Filas de bloque de A

    int mr = 2, // Filas de microkernel
    int nr = 2 // Columnas de microkernel
>
void gemm(
    const float* __restrict__ A, // MxN
    const float* __restrict__ B, // 
    float* __restrict__ OUT
) {
    memset(OUT, 0, M * N * sizeof(float));

    float A_panel[mc * kc];
    float B_panel[nc * kc];
    float AB[mr * nr];

    for (int jc = 0; jc < N; jc += nc) {
        for (int pc = 0; pc < K; pc += kc) {
            gpackB<kc, nc, nr, 1, N>((float*)B + pc * N + jc , B_panel);

            for (int ic = 0; ic < M; ic += mc) {
                gpackA<mc, kc, mr, 1, K>((float*)A + ic * K + pc, A_panel);

                int _nc = min(N - jc, nc); // evitar que se pase "matrices grandes?"
                int _mc = min(M - ic, mc); // evitar que se pase el panel

                for (int jr = 0; jr < _nc; jr += nr) {
                    for (int ir = 0; ir < _mc; ir += mr) {
                        int _kc = min(K - pc, kc); // evitar que se pase el panel
                        int _nr = min(_nc - jr, nr); // evitar que se pase el bloque
                        int _mr = min(_mc - ir, mr); // evitar que se pase el bloque

                        // (_mr x _kc) * (_kc x _nr)

                        memset(AB, 0, mr * nr * sizeof(float));

                        for (int k = 0; k < _kc; k++) {
                            for (int n = 0; n < _nr; n++) {
                                for (int m = 0; m < _mr; m++) {
                                    AB[m + n * mr] +=
                                        A_panel[k * mr + m] *
                                        B_panel[k * nr + n];
                                }
                            }
                        }

                        float* Ckernel = (float*)OUT + (ic + ir) * N + (jc + jr);

                        for (int j = 0; j < _nr; j++) {
                            for (int i = 0; i < _mr; i++) {
                                Ckernel[i * N + j] +=
                                    AB[i + j * mr];
                            }
                        }

                        // --
                    }
                }
            }
        }
    }
}
