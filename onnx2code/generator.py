from itertools import chain
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from textwrap import dedent, indent

import numpy as np
import onnx
import onnxsim.onnx_simplifier as onnx_simplifier

from .memory import TensorUsageRecord, find_best_layout
from .ops.operation import OpCall, Operation, OpImpl
from .result import ModelResult
from .tensor import TensorData, parse_tensors
from .util import get_fixed_input_shapes

REGISTER_ORDER = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
INFERENCE_SIGNATURE = "void __attribute__ ((noinline)) inference(const float* weights, const float* inputs, float* outputs)"


class Generator:
    """
    Code generator

    Proto ref: https://github.com/onnx/onnx/blob/main/docs/IR.md
    """

    def __init__(self, _model_proto: onnx.ModelProto, variations: list[str] = []):
        try:
            model_proto, check = onnx_simplifier.simplify(
                model=_model_proto,
                overwrite_input_shapes=get_fixed_input_shapes(_model_proto),
            )
            assert check, "ONNX model could not be simplified"
        except Exception as e:
            model_proto = _model_proto
            warnings.warn("Model could not be simplified, using as is (" + str(e) + ")")

        # save model for later inspection
        if os.getenv("ONNX2CODE_DEBUG", "0") == "1":
            tmp = Path(__file__).parent.parent / "tmp"
            tmp.mkdir(exist_ok=True)
            onnx.save_model(model_proto, (tmp / "model.onnx").__str__())

        self.model_proto = model_proto
        self.tensors = {tensor.name: tensor for tensor in parse_tensors(model_proto)}
        self.variations = variations + ["c", "asm"]

        self.impls: dict[OpImpl, OpCall] = {}
        self.calls: list[OpCall] = []

    def weld_tensors(self, name_from: str, name_to: str) -> None:
        """
        Weld tensors together
        This means they should point to the same variable in runtime

        :param name_from: Name of the origin tensor
        :param name_to: Name of the destination tensor
        :raises KeyError: If the tensor names are not found
        """
        self.tensors[name_to].variable = self.tensors[name_from].variable

        if self.tensors[name_to].tag != "output":
            self.tensors[name_to].tag = "welded"

    def generate(self) -> ModelResult:
        """
        Generate C and ASM code to run the model
        """
        for node in self.model_proto.graph.node:
            if node.op_type in [
                # Reshape/Squeeze/Unsqueeze operator ⚠️ SPECIAL CASE ⚠️
                #
                # https://github.com/onnx/onnx/blob/main/docs/Operators.md#reshape
                # https://github.com/onnx/onnx/blob/main/docs/Operators.md#squeeze
                # https://github.com/onnx/onnx/blob/main/docs/Operators.md#unsqueeze
                "Reshape",
                "Squeeze",
                "Unsqueeze",
                # have no effect during inference
                "Dropout",
                "BatchNormalization",  # are we sure about this one?
                # other kind of reshape
                "Flatten",
            ]:
                # Since it just reshapes the tensor, we don't need to do anything in runtime
                # But we must must be weld the input and output tensors (variables/data)
                self.weld_tensors(node.input[0], node.output[0])

                continue

            variants = Operation.get(node.op_type, self.variations)

            impl: (OpImpl | None) = None
            call: (OpCall | None) = None
            ex: (Exception | None) = None

            # we try all the variants we have available, in the order specified
            # if one throws NotImplemented, we try the next one
            for var in variants:
                try:
                    op = var(
                        node,
                        [self.tensors[name] for name in node.input],
                        [self.tensors[name] for name in node.output],
                    )
                    impl = op.impl()
                    call = op.call()
                    break
                except NotImplementedError as _ex:
                    # keep first
                    if ex is None:
                        ex = _ex

            if impl is None or call is None:
                assert ex is not None
                raise ex

            if call is not None and impl is not None:
                if impl in self.impls:
                    new_name = call.fn_name()
                    prev_name = self.impls[impl].fn_name()
                    assert (
                        new_name == prev_name
                    ), "function name should coincide if the implementation is the same"

                self.impls[impl] = call
                self.calls.append(call)

        self._compute_memory_layout()

        inputs = [tensor for tensor in self.tensors.values() if tensor.tag == "input"]
        outputs = [tensor for tensor in self.tensors.values() if tensor.tag == "output"]

        return ModelResult(
            input_shapes={tensor.name: tensor.shape for tensor in inputs},
            output_shapes={tensor.name: tensor.shape for tensor in outputs},
            source_c=self._gen_c_source(),
            source_h=f"extern {INFERENCE_SIGNATURE};",
            source_asm=self._gen_asm_source(),
            weights=self._gen_weights(),
        )

    def _compute_memory_layout(self) -> None:
        """
        Finds a good memory layout for intermediate tensors
        """
        MAX = 999999999
        MIN = -1

        inter_tensors: dict[str, TensorUsageRecord] = {}

        # add all intermediate tensors
        for t in self.tensors.values():
            if t.tag == "intermediate":
                inter_tensors[t.variable] = TensorUsageRecord(MAX, MIN, t.size)

        # build usage records knowing the order of calls and data dependencies
        for index, call in enumerate(self.calls):
            # for inputs, make sure we reserve the tensor up to index
            for tensor in call.inputs:
                if tensor.tag == "intermediate":
                    rec = inter_tensors[tensor.variable]
                    rec.last_op = max(rec.last_op, index)

            # for outputs, make sure we reserve the tensor from at least index
            for tensor in call.outputs:
                if tensor.tag == "intermediate":
                    rec = inter_tensors[tensor.variable]
                    rec.first_op = min(rec.first_op, index)

        # tensors that connect with the output don't have last_op set
        # set to first_op + 1
        for var, rec in inter_tensors.items():
            assert rec.first_op != MAX, "tensor is never used"

            if rec.last_op == -1:
                rec.last_op = rec.first_op + 1

        self.inter_size, offsets = find_best_layout(list(inter_tensors.values()))
        self.inter_offsets = {}

        # map tensor names to variables
        for var, offset in zip(inter_tensors.keys(), offsets):
            self.inter_offsets[var] = offset

    def _gen_weights(self) -> TensorData:
        return np.concatenate(
            [
                tensor.data.reshape(-1)
                for tensor in self.tensors.values()
                if tensor.tag == "weight"
                and tensor.data is not None
                and tensor.data.dtype == np.float32
            ]
            # concatenate needs at least one array
            + [np.array([], dtype=np.float32)],
        )

    def _gen_c_source(self) -> str:
        source = "\n".join(
            [
                "#include <stdio.h>",
                "#include <assert.h>",
                "#include <math.h>",
                "#include <string.h>",
                "#define min(a,b) ((a)<(b)?(a):(b))",
                "#define max(a,b) ((a)>(b)?(a):(b))",
                "float im2col[50000000]; // TODO: do this correctly...",
                "",
            ]
        )

        # asm auxiliary function declarations

        source += "// Auxiliary functions (ASM):\n\n"

        asm_aux_declarations = [
            f"{asm_aux_function.signature};"
            for impl in self.impls.keys()
            for asm_aux_function in impl.asm_aux_functions
        ]

        source += 'extern "C" {\n' + "\n\n".join(asm_aux_declarations) + "\n}\n\n"

        # loading external files
        source += "// External files:\n\n"

        efp = [path for impl in self.impls.keys() for path in impl.external_paths]
        external_file_paths = sorted(set(efp), key=efp.index)

        for path in external_file_paths:
            source += f"// {path}\n\n"
            with open(path, "r") as f:
                source += f.read() + "\n"

        source += "\n" * 2

        # c++ auxiliary functions

        source += "// Auxiliary functions (C++):\n\n"

        cpp_aux_functions = list(
            dict.fromkeys(
                chain.from_iterable(
                    impl.cpp_aux_functions for impl in self.impls.keys()
                )
            )
        )

        source += "\n".join(cpp_aux_functions) + "\n" * 2

        # define ASM functions in C

        source += "// ASM functions:\n\n"

        for impl, call in self.impls.items():
            if impl.lang == "asm":
                source += f"extern {call.signature()};"

        source += "\n" * 2

        # implementations

        source += "// Implementations:\n\n"

        for impl, call in self.impls.items():
            if impl.lang == "c":
                source += call.signature() + " {\n"
                source += indent(impl.full_source().strip(), prefix=" " * 4)
                source += "\n}\n"

        # define intermediate tensor
        # it is a shared buffer
        source += "\n" * 2
        source += f"float intermediates[{self.inter_size}];"
        source += "\n" * 2

        inference_source = ""
        io_offsets: defaultdict[str, int] = defaultdict(int)
        # build tensor variables
        for tensor in self.tensors.values():
            if tensor.tag != "welded":
                if (
                    tensor.tag == "weight"
                    and tensor.data is not None
                    and tensor.data.dtype != np.float32
                ):
                    # weight with no data or invalid, skip
                    continue

                if tensor.tag == "intermediate":
                    # IF an intermediate tensor is welded with the output
                    # we want to preserve the output tensor instead of the intermediate one
                    # so we skip the definition of the intermediate in favor of the output
                    skip = False
                    for other in self.tensors.values():
                        if other.tag == "output" and other.variable == tensor.variable:
                            # already defined as output
                            skip = True
                            break
                    if skip:
                        continue

                if tensor.tag == "intermediate":
                    offset = self.inter_offsets[tensor.variable]
                    assert offset is not None, "invliad offset"
                else:  # input, output or weight
                    offset = io_offsets[tensor.tag]
                    io_offsets[tensor.tag] += tensor.size

                decl = "const " if tensor.tag in ["input", "weight"] else ""
                decl += f"float* {tensor.variable} = "
                decl += f"{tensor.tag}s + {offset};"

            else:
                # welded
                continue

            decl = f"\n{decl : <34} // ({tensor.shape_str()}) {tensor.name}"
            inference_source += decl

        # make op calls
        inference_source += "\n"
        for call in self.calls:
            inference_source += f"\n{call.invocation()};"

        source += INFERENCE_SIGNATURE + " {"
        source += indent(inference_source, prefix=" " * 4)
        source += "\n}"

        return source

    def _gen_asm_source(self) -> str:
        source = ""

        # asm auxiliary functions

        for impl in self.impls.keys():
            for asm_aux_function in impl.asm_aux_functions:
                # extract name from signature
                regex = re.compile(r"(\w+)\s*\(")
                match = regex.search(asm_aux_function.signature)
                assert match is not None, "invalid signature"
                name = match.group(1)

                function_source = indent(
                    dedent(asm_aux_function.source), prefix=" " * 4
                )

                source += f"global {name}\n{name}:{function_source}\n\n"

        for impl, call in self.impls.items():
            if impl.lang == "asm":
                comments = [call.signature()] + [
                    f"{p}: {REGISTER_ORDER[i]}"
                    for i, p in enumerate(
                        call.input_names[: len(call.inputs)]
                        + call.output_names[: len(call.outputs)]
                    )
                ]
                source += "\n\n".join(
                    [
                        *[f";; {c}" for c in comments],
                        f"global {call.fn_name()}",
                        f"{call.fn_name()}:",
                        indent(impl.full_source(), prefix=" " * 4),
                    ]
                )

        return source.strip() + "\n"
