import argparse
import sys
from pathlib import Path

import onnx
from rich import print

from .checker import check_model_result
from .generator import Generator


def main() -> None:
    parser = argparse.ArgumentParser(prog="onnx2code")
    parser.add_argument("input_model", help="input .onnx file")
    parser.add_argument("output_folder", help="output folder to write files")
    parser.add_argument(
        "--variations",
        "--vars",
        type=str,
        help="variation priority",
        default="asm, c",
        action="store",
    )
    parser.add_argument(
        "--checks",
        type=int,
        help="compile and test the model with the provided amount of inputs",
        default=0,
        action="store",
    )

    args = parser.parse_args()

    try:
        model_proto = onnx.load(args.input_model)
    except Exception as e:
        print("Error loading ONNX model: ", e)
        sys.exit(1)

    variations = [v.strip() for v in args.variations.split(",")]

    try:
        result = Generator(model_proto, variations).generate()
    except Exception as e:
        print("Error generating code: ", e)
        sys.exit(2)

    print("Input shapes:", result.input_shapes)
    print("Output shapes:", result.output_shapes)
    print("Weights size (floats):", result.weights.size)

    path = Path(args.output_folder)
    print("Writing files to", path.resolve())

    path.mkdir(parents=True, exist_ok=True)
    c_file = path / "model.c"
    h_file = path / "model.h"
    asm_file = path / "model.asm"
    weights_file = path / "weights.bin"
    result.weights.tofile(weights_file)

    for file, content in [
        (c_file, result.source_c),
        (h_file, result.source_h),
        (asm_file, result.source_asm),
    ]:
        with open(file, "w") as f:
            f.write(content)

    if args.checks > 0:
        print("Checking model with", args.checks, "random inputs")

        try:
            check_model_result(model_proto, result, args.checks)
        except Exception as e:
            print("Error checking model: ", e)
            sys.exit(3)

    print("Done")


if __name__ == "__main__":
    main()
