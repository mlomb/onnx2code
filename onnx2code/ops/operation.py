from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Literal

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
            params.append(f"const float* __restrict__ {self.input_names[i]}")
        for i in range(len(self.outputs)):
            params.append(f"float* __restrict__ {self.output_names[i]}")

        return f"void {self.fn_name()}({', '.join(params)})"

    def invocation(self) -> str:
        return (
            self.fn_name()
            + f"({', '.join(t.variable for t in self.inputs + self.outputs)})"
        )


@dataclass(frozen=True)
class ASMAuxFunction:
    signature: str
    source: str


@dataclass(frozen=True)
class OpImpl:
    lang: Literal["c", "asm"]
    source: str | tuple[str, ...]
    cpp_aux_functions: tuple[str, ...] = ()
    asm_aux_functions: tuple[ASMAuxFunction, ...] = ()
    external_paths: tuple[Path, ...] = ()

    def full_source(self) -> str:
        code = self.source if isinstance(self.source, str) else "\n".join(self.source)
        return dedent(code).strip().strip("\n")


@dataclass(frozen=True)
class RegistryEntry:
    variant_tags: list[str]
    priority: int
    klass: type["Operation"]

    def __lt__(self, other: Any) -> bool:
        return self.priority < other.priority  # type: ignore


class Operation(ABC):
    node_types: set[str]
    _registry: defaultdict[str, list[RegistryEntry]] = defaultdict(list)

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
        cls, var: str | list[str], priority: int = 0
    ) -> Callable[[type["Operation"]], type["Operation"]]:
        vars = [var] if isinstance(var, str) else var

        def decorator(newcls: type[Operation]) -> type[Operation]:
            for node_type in newcls.node_types:
                cls._registry[node_type].append(
                    RegistryEntry(variant_tags=vars, priority=priority, klass=newcls)
                )
                # always keep sorted
                cls._registry[node_type].sort()

            return newcls

        return decorator

    @staticmethod
    def get(node_type: str, variant_order: list[str]) -> list[type["Operation"]]:
        if node_type not in Operation._registry:
            raise NotImplementedError(f"Operation {node_type} not implemented")

        variants = []

        for variant_tag in variant_order:
            for entry in Operation._registry[node_type]:
                if variant_tag in entry.variant_tags:
                    variants.append(entry.klass)

        if len(variants) == 0:
            raise ValueError(f"No valid variant found for {node_type}")
        else:
            return list(dict.fromkeys(variants))
