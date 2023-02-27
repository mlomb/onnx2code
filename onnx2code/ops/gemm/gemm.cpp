template <
    // matrix sizes
    int M,
    int N,
    int K,

    int nc,  // Columnas de panel de B
    int kc,  // Filas de panel de B
    int mc,  // Filas de bloque de A

    int mr,  // Filas de microkernel
    int nr   // Columnas de microkernel
    >
void gemm(
    const float* __restrict__ A,  // MxN
    const float* __restrict__ B,  //
    float* __restrict__ OUT) {
    memset(OUT, 0, M * N * sizeof(float));

    float A_panel[mc * kc];
    float B_panel[nc * kc];

    for (int jc = 0; jc < N; jc += nc) {
        for (int pc = 0; pc < K; pc += kc) {
            gpackB<kc, nc, nr, 1, N>((float*)B + pc * N + jc, B_panel);

            for (int ic = 0; ic < M; ic += mc) {
                gpackA<mc, kc, mr, 1, K>((float*)A + ic * K + pc, A_panel);

                int _nc = min(N - jc, nc);  // evitar que se pase "matrices grandes?"
                int _mc = min(M - ic, mc);  // evitar que se pase el panel

                for (int jr = 0; jr < _nc; jr += nr) {  // jr es el offset del panel de ancho nr (violeta)
                    for (int ir = 0; ir < _mc; ir += mr) {  // ir es el offset del panel de ancho mr (verde)
                        int _kc = min(K - pc, kc);    // evitar que se pase el panel
                        int _nr = min(_nc - jr, nr);  // evitar que se pase el bloque
                        int _mr = min(_mc - ir, mr);  // evitar que se pase el bloque

                        // (_mr x _kc) * (_kc x _nr)

                        const float* A_kernel = A_panel + ir * kc; // (_mr x _kc) column major
                        const float* B_kernel = B_panel + jr * kc; // (_kc x _nr) row major

                        float* C_writeback = (float*)OUT + (ic + ir) * N + (jc + jr);

                        // TODO: pasar _mr y _nr para evitar escribir fuera en C
                        //       quizas un branch entre optimized y ref?
                        assert(_mr == mr);
                        assert(_nr == nr);

                        // ref_microkernel<mr, nr, kc, N>(A_kernel, B_kernel, C_writeback);

                        test_microkernel<mr, nr, kc, N>(A_kernel, B_kernel, C_writeback);

                        // --
                    }
                }
            }
        }
    }
}
