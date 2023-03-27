from dataclasses import dataclass
import math
from pathlib import Path


@dataclass
class LoopTilingParams:
    nc: int  # Columnas de panel de B
    kc: int  # Filas de panel de B
    mc: int  # Filas de bloque de A
    mr: int  # Filas de microkernel
    nr: int  # Columnas de microkernel
    mv: int  # Filas de unit-update
    nu: int  # Columnas de unit-update


tiling_params = LoopTilingParams(
    nc=4096,
    kc=256,
    mc=256,
    mr=4,
    nr=8,
    mv=4,
    nu=4,
)


def set_tiling_params(params: LoopTilingParams) -> None:
    global tiling_params
    tiling_params = params


external_paths_GEMM = (
    Path(__file__).parent / "gpackA.cpp",
    Path(__file__).parent / "gpackB.cpp",
    Path(__file__).parent / "microkernel_ref.cpp",
    Path(__file__).parent / "microkernel_test.cpp",
    Path(__file__).parent / "gemm.cpp",
)


def call_GEMM(M: int, K: int, N: int, params: str) -> str:
    nc = min(2 ** math.ceil(math.log2(N)), tiling_params.nc)
    kc = tiling_params.kc
    mc = tiling_params.mc
    mr = tiling_params.mr
    nr = tiling_params.nr

    mv = tiling_params.mv
    nu = tiling_params.nu

    return f"gemm<{M},{K},{N},{nc},{kc},{mc},{mr},{nr},{mv},{nu}>({params});"
