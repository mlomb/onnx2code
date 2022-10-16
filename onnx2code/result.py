from dataclasses import dataclass

from .tensor import TensorData
from .util import ShapesMap


@dataclass
class ModelResult:
    input_shapes: ShapesMap
    output_shapes: ShapesMap
    source_c: str
    source_h: str
    source_asm: str
    weights: TensorData
