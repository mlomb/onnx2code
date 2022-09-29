import collections.abc
from abc import ABC, abstractmethod
from typing import Callable

import onnx

from ..generator import Generator


class Operation(ABC):
    node_types: list[str]
    variant_name: str | None = None

    def __init__(self, node: onnx.NodeProto):
        self.node = node
        self.asserts()

    @abstractmethod
    def asserts(self) -> None:
        pass

    @abstractmethod
    def generate(self, gen: Generator) -> None:
        pass

    @classmethod
    def variant(cls, name: str) -> Callable[[type["Operation"]], type["Operation"]]:
        def decorator(newcls: type[Operation]) -> type[Operation]:
            newcls.node_types = cls.node_types
            newcls.variant_name = name
            return newcls

        return decorator

    @classmethod
    def get_subclasses(
        cls: type["Operation"],
    ) -> collections.abc.Generator[type["Operation"], None, None]:
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass
