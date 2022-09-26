from abc import ABC, abstractmethod

class Operation(ABC):

    def __init__(self, node):
        self.node = node
        self.asserts()

    @abstractmethod
    def asserts(self):
        pass

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


def register_op(cls, node_type: str, variant: str):
    print (f"Registering {cls} for {node_type} and {variant}")
    return cls
