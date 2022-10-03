from pathlib import Path
from subprocess import call

from .output import Output

CHECKER_PATH = Path(__file__).parent / "model_check.c"


def model_check(output: Output, n_inputs: int = 2) -> bool:
    """
    Checks if the generated output matches the reference (ONNX Runtime)

    <n_inputs> random inputs will be generated
    """

    tmp_folder = Path("tmp/")
    tmp_folder.mkdir(exist_ok=True)
    cpp_file = tmp_folder / "model.cpp"
    hpp_file = tmp_folder / "model.hpp"
    asm_file = tmp_folder / "model.asm"

    asm_object = tmp_folder / "model-asm.o"
    final_object = tmp_folder / "model-check"

    with open(cpp_file, "w") as f:
        f.write(output.source_cpp)
    with open(hpp_file, "w") as f:
        f.write(output.source_hpp)
    with open(asm_file, "w") as f:
        f.write(output.source_asm)

    run_commands(
        [
            ["nasm", "-f", "elf64", str(asm_file), "-o", str(asm_object), "-g"],
            [
                "gcc",
                "-m64",  # 64 bit env
                str(asm_object),
                str(hpp_file),
                str(cpp_file),
                str(CHECKER_PATH),
                "-o",
                str(final_object),
                "-O0",
                "-g",
                "-fsanitize=address",
            ],
        ]
    )

    exit_code = call([str(final_object), str(n_inputs)])
    
    return exit_code == 0


def run_commands(cmds: list[list[str]]) -> None:
    for cmd in cmds:
        exit_code = call(cmd)
        if exit_code != 0:
            raise Exception(f"Command failed: {cmd}")
