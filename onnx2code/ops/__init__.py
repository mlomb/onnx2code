from . import identity
from . import operation

for cls in operation.Operation.get_subclasses():
    if hasattr(cls, "variant"):
        print(cls, cls.nodes_type, cls.variant)

