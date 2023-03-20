template <
    // matrix sizes
    int M,
    int N,
    int K,

    int nc,  // Columnas de panel de B
    int kc,  // Filas de panel de B
    int mc,  // Filas de bloque de A

    int mr,  // Filas de microkernel
    int nr,  // Columnas de microkernel

    int mv,  // Filas de unit update
    int nu   // Columnas de unit update
    >
void gemm(
    const float* __restrict__ A,  // MxN
    const float* __restrict__ B,  // NxK
    float* __restrict__ OUT      // MxK
) {
    memset(OUT, 0, M * N * sizeof(float));

    // Si no se cumplen, las multiplicaciones de microkernel acceden
    // a memoria inválida en los edge cases
    // Podría solucionarse agregando un sliver más a A_panel y B_panel
    static_assert(mc % mr == 0, "must be conforming");
    static_assert(nc % nr == 0, "must be conforming");

    float A_panel[mc * kc];
    float B_panel[nc * kc];

    float AB_microkernel[mr * nr];

    for (int jc = 0; jc < N; jc += nc) {
        for (int pc = 0; pc < K; pc += kc) {
            gpackB<kc, nc, nr, 1, N>((float*)B + pc * N + jc, B_panel);

            int _kc = min(K - pc, kc);  // evitar que se pase el panel

            for (int ic = 0; ic < M; ic += mc) {
                gpackA<mc, kc, mr, 1, K>((float*)A + ic * K + pc, A_panel);

                int _nc = min(N - jc, nc);  // evitar que se pase "matrices grandes?"
                int _mc = min(M - ic, mc);  // evitar que se pase el panel

                for (int jr = 0; jr < _nc; jr += nr) {      // jr es el offset del panel de ancho nr (violeta)
                    for (int ir = 0; ir < _mc; ir += mr) {  // ir es el offset del panel de ancho mr (verde)
                        // (_mr x _kc) * (_kc x _nr)

                        const float* A_kernel = A_panel + ir * kc;  // (_mr x _kc) column major
                        const float* B_kernel = B_panel + jr * kc;  // (_kc x _nr) row major

                        // ref_microkernel<mr, nr, kc, N>(A_kernel, B_kernel, AB_microkernel);

                        memset(AB_microkernel, 0, mr * nr * sizeof(float));
                        test_microkernel<mr, nr, mv, nu>(_kc, A_kernel, B_kernel, AB_microkernel);

                        // TODO: pasar _mr y _nr para evitar escribir fuera en C
                        //       quizas un branch entre optimized y ref?
                        int _nr = min(_nc - jr, nr);  // evitar que se pase el bloque
                        int _mr = min(_mc - ir, mr);  // evitar que se pase el bloque

                        // assert(_mr == mr);
                        // assert(_nr == nr);

                        float* C_writeback = (float*)OUT + (ic + ir) * N + (jc + jr);

                        if (_mr == mr && _nr == nr) {
                            // Versión optimizada
                            for (int i = 0; i < mr; i++) {
                                for (int j = 0; j < nr; j++) {
                                    C_writeback[i * N + j] += AB_microkernel[i * nr + j];
                                }
                            }
                        } else {
                            // Edge case
                            for (int i = 0; i < _mr; i++) {
                                for (int j = 0; j < _nr; j++) {
                                    C_writeback[i * N + j] += AB_microkernel[i * nr + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
