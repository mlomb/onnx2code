from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from textwrap import dedent
from typing import Callable, Literal

import onnx

from ..tensor import TensorInfo

# used as tensor names
LETTERS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
)


@dataclass
class OpCall:
    sig_name: str
    sig_params: list[int | str | list[int] | list[str]]
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    input_names: tuple[str, ...] = LETTERS
    output_names: tuple[str, ...] = ("OUT",)

    def fn_name(self) -> str:
        str_sig_params = []
        for sig_param in self.sig_params:
            if isinstance(sig_param, list):
                str_sig_params.append("x".join(map(str, sig_param)))
            else:
                str_sig_params.append(str(sig_param))

        return f"{self.sig_name}{'_' if len(str_sig_params) > 0 else ''}" + "_".join(
            str_sig_params
        )

    def signature(self) -> str:
        params = []
        for i in range(len(self.inputs)):
            params.append(f"const float* restrict {self.input_names[i]}")
        for i in range(len(self.outputs)):
            params.append(f"float* restrict {self.output_names[i]}")

        return f"void {self.fn_name()}({', '.join(params)})"

    def invocation(self) -> str:
        return (
            self.fn_name()
            + f"({', '.join(t.variable for t in self.inputs + self.outputs)})"
        )


@dataclass(frozen=True)
class OpImpl:
    lang: Literal["c", "asm"]
    source: str | tuple[str, ...]
    aux_functions: frozenset[str] = frozenset()

    def full_source(self) -> str:
        code = self.source if isinstance(self.source, str) else "\n".join(self.source)
        return dedent(code).strip().strip("\n")


class Operation(ABC):
    node_types: set[str]
    _registry: defaultdict[str, dict[str, type["Operation"]]] = defaultdict(dict)

    def __init__(
        self,
        node: onnx.NodeProto,
        inputs: list[TensorInfo],
        outputs: list[TensorInfo],
    ):
        self.node = node
        self.inputs = inputs
        self.outputs = outputs
        self.parse()

    @abstractmethod
    def parse(self) -> None:
        pass

    @abstractmethod
    def call(self) -> OpCall | None:
        return None

    @abstractmethod
    def impl(self) -> OpImpl | None:
        pass

    @classmethod
    def variant(
        cls, name: str | list[str]
    ) -> Callable[[type["Operation"]], type["Operation"]]:
        names = [name] if isinstance(name, str) else name

        def decorator(newcls: type[Operation]) -> type[Operation]:
            for node_type in newcls.node_types:
                for name in names:
                    cls._registry[node_type][name] = newcls

            return newcls

        return decorator

    @staticmethod
    def get(node_type: str, variant: list[str]) -> list[type["Operation"]]:
        if node_type not in Operation._registry:
            raise NotImplementedError(f"Operation {node_type} not implemented")

        variants = set()

        for variant_name in variant:
            if variant_name in Operation._registry[node_type]:
                variants.add(Operation._registry[node_type][variant_name])

        if len(variants) == 0:
            raise ValueError(f"No valid variant found for {node_type}")
        else:
            return list(variants)
