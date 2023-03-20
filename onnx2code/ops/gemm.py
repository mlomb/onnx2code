import subprocess
from typing import Iterable

from onnx2code.util import get_attribute

from .operation import OpCall, Operation, OpImpl
from .gemm_tiling.GEMM import external_paths_GEMM, call_GEMM


class GEMM(Operation):
    """
    GEneral Matrix Multiplication operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    """

    node_types = {"Gemm", "MatMul"}

    def parse(self) -> None:
        assert (
            len(self.inputs) == 2 or len(self.inputs) == 3
        ), "expected two or three inputs"
        assert len(self.outputs) == 1, "expected one output"

        self.hasC = len(self.inputs) == 3
        self.transA = get_attribute(self.node, "transA", 0) > 0.5
        self.transB = get_attribute(self.node, "transB", 0) > 0.5
        self.alpha = get_attribute(self.node, "alpha", 1.0)
        self.beta = get_attribute(self.node, "beta", 1.0)

        # normalize
        self.alpha = None if self.alpha == 1.0 else self.alpha
        self.beta = None if self.beta == 1.0 else self.beta

        if self.transA:
            raise NotImplementedError("transA not supported")
        if self.alpha is not None:
            raise NotImplementedError("alpha not supported")
        if self.beta is not None:
            raise NotImplementedError("beta not supported")

        A = self.inputs[0]
        B = self.inputs[1]
        Y = self.outputs[0]

        self.N = A.shape[0]
        self.M = B.shape[1] if self.transB else B.shape[0]
        self.K = B.shape[0] if self.transB else B.shape[1]

        assert Y.shape[0] == self.N
        assert Y.shape[1] == self.K

    def call(self) -> OpCall:
        return OpCall(
            sig_name="GEMM",
            sig_params=[
                self.hasC,
                self.N,
                self.M,
                self.K,
                self.transB,
            ],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@GEMM.variant(["c", "gemm-naive"], priority=2)
class GEMMC(GEMM):
    def impl(self) -> OpImpl:
        N, M, K = self.N, self.M, self.K

        index_B = f"i * {K} + col" if not self.transB else f"col * {M} + i"

        source = f"""
        for(int row = 0; row < {N}; row++) {{
            for(int col = 0; col < {K}; col++) {{
                float sum = 0;
                for(int i = 0; i < {M}; i++) {{
                    sum += A[row * {M} + i] * B[{index_B}];
                }}
                OUT[row * {K} + col] = sum{f' + C[row * {K} + col]' if self.hasC else ''};
            }}
        }}
        """

        return OpImpl(lang="c", source=source)


# Make sure this executable is in your PATH
LIBXSMM_PATH = "libxsmm_gemm_generator"


@GEMM.variant(["asm", "libxsmm"], priority=0)
class GEMMAsm(GEMM):
    def impl(self) -> OpImpl:
        N, M, K = self.N, self.M, self.K

        aux_fn_name = f"libxsmm_GEMM_{N}_{M}_{K}"

        # Reference: https://scalable.uni-jena.de/opt/hpc/chapters/assignment_small_gemms.html
        generator_args = [
            LIBXSMM_PATH,
            # matrix type
            "dense",
            # output file name
            "/dev/stdout",
            # function name
            aux_fn_name,
            # matrix size
            str(K),
            str(N),
            str(M),
            # lda, ldb, ldc
            str(K),
            str(M),
            str(K),
            # alpha beta
            # C := alpha*A*B + beta*C
            "1",
            "0",
            # 0: unaligned A, C
            "0",
            "0",
            # arch
            "hsw",  # haswell, targets AVX2
            # prefetch
            "nopf",  # no prefetch
            # precision
            "SP",  # single precision (f32)
        ]

        try:
            libxsmm_generator_process = subprocess.run(
                generator_args,
                capture_output=True,
                encoding="utf-8",
            )
        except PermissionError:
            raise RuntimeError(f"libxsmm not found at '{LIBXSMM_PATH}'")

        if (
            libxsmm_generator_process.returncode != 0
            or libxsmm_generator_process.stderr != ""
        ):
            raise RuntimeError(f"libxsmm: {libxsmm_generator_process.stderr}")

        lines: Iterable[str] = libxsmm_generator_process.stdout.splitlines()

        aux_fn = "\n".join(
            filter(
                # Filter out the flops line
                lambda line: not (
                    line.startswith("libxsmm_num_total_flops") or line == ""
                ),
                lines,
            )
        )

        if aux_fn == "":
            raise RuntimeError("libxsmm: no output")

        # tensors MUST be reversed since libxsmm uses BLAS' column-major order
        # and we use onnx's row-major order
        source = f"""
        {aux_fn_name}(B, A, OUT);
        """ + (
            f"""
            for(int i = 0; i < {N * K}; i++) {{
                OUT[i] += C[i];
            }}
            """
            if self.hasC
            else ""
        )

        return OpImpl(lang="c", source=source, cpp_aux_functions=(aux_fn,))


@GEMM.variant(["c", "loop-tiling"], priority=1)
class GEMMLoopTiling(GEMM):
    def impl(self) -> OpImpl:
        M, K, N = self.N, self.M, self.K

        if self.hasC:
            raise NotImplementedError("hasC not supported")

        # unit_update_asm = ASMAuxFunction(
        #     signature="void unit_update(const float*, const float*, float*)",
        #     source="""
        #         vbroadcastss ymm0, [rsi]
        #         vmovups ymm1, [rdi]
        #         vfmadd213ps ymm0, ymm1, [rdx]
        #         vmovups [rdx], ymm0
        #         vzeroupper
        #         ret
        #     """,
        # )

        return OpImpl(
            lang="c",
            source=call_GEMM(M, K, N, "A, B, OUT"),
            external_paths=external_paths_GEMM,
            # asm_aux_functions=(unit_update_asm,),
        )
