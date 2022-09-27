from abc import ABC, abstractmethod


class Operation(ABC):
    node_types: list[str]
    variant_name: str | None = None

    def __init__(self, node):
        self.node = node
        self.asserts()

    @abstractmethod
    def asserts(self):
        pass

    @classmethod
    def variant(cls, name: str):
        def decorator(newcls: type[Operation]):
            newcls.node_types = cls.node_types
            newcls.variant_name = name
            return newcls

        return decorator

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass
