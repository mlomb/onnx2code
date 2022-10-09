from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal
from collections import defaultdict
import onnx


from ..tensor import TensorInfo


@dataclass
class OpCall:
    name: str
    params: list[str]
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]


@dataclass
class OpImpl:
    lang: Literal["c", "asm"]
    source: str | list[str]


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
    def variant(cls, name: str) -> Callable[[type["Operation"]], type["Operation"]]:
        def decorator(newcls: type[Operation]) -> type[Operation]:
            for node_type in newcls.node_types:
                cls._registry[node_type][name] = newcls

            return newcls

        return decorator

    @staticmethod
    def get(node_type: str, variant: list[str]) -> type["Operation"]:
        if node_type not in Operation._registry:
            raise NotImplementedError(f"Operation {node_type} not implemented")

        for variant_name in variant:
            if variant_name in Operation._registry[node_type]:
                return Operation._registry[node_type][variant_name]

        raise ValueError(f"No valid variant found for {node_type}")
