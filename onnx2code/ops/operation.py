from abc import ABC, abstractmethod
from typing import Callable

import onnx

from ..generator import Generator


class Operation(ABC):
    node_types: set[str]
    _registry: dict[str, dict[str, type["Operation"]]] = {}

    def __init__(self, gen: Generator, node: onnx.NodeProto):
        self.node = node
        self.inputs = [gen.tensors[name] for name in node.input]
        self.outputs = [gen.tensors[name] for name in node.output]
        self.asserts()

    @abstractmethod
    def asserts(self) -> None:
        pass

    @abstractmethod
    def emit(self, gen: Generator) -> None:
        pass

    @classmethod
    def variant(cls, name: str) -> Callable[[type["Operation"]], type["Operation"]]:
        def decorator(newcls: type[Operation]) -> type[Operation]:
            if cls.__name__ not in cls._registry:
                cls._registry[cls.__name__] = {}
            cls._registry[cls.__name__][name] = newcls

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
