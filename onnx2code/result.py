from dataclasses import dataclass

from .util import ShapesMap
from .tensor import TensorData


@dataclass
class ModelResult:
    input_shapes: ShapesMap
    ouput_shapes: ShapesMap
    source_c: str
    source_h: str
    source_asm: str
    weights: TensorData
