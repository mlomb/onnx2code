from abc import ABC, abstractmethod
from typing import Callable

import onnx

from ..generator import Generator


class Operation(ABC):
    node_types: set[str]
    variant_name: str | None = None
    variants: dict[str, type["Operation"]] = {}
    _registry: dict[str, type["Operation"]] = {}

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
            newcls.node_types = cls.node_types
            newcls.variant_name = name
            cls.variants[name] = newcls

            return newcls

        return decorator

    def __init_subclass__(cls) -> None:
        if cls.variant_name is None:
            for node_type in cls.node_types:
                Operation._registry[node_type] = cls

    @staticmethod
    def get(node_type: str, variant: list[str]) -> type["Operation"]:
        if node_type not in Operation._registry:
            raise NotImplementedError(f"Operation {node_type} not implemented")

        basecls = Operation._registry[node_type]

        for variant_name in variant:
            return basecls.variants[variant_name]

        if len(basecls.variants) > 0:
            raise ValueError(f"No valid variant found for {node_type}")

        return basecls
