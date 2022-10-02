import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

ShapesMap = dict[str, list[int]]


@dataclass
class Output:
    input_shapes: ShapesMap
    ouput_shapes: ShapesMap
    source_cpp: str
    source_hpp: str
    source_asm: str
    weights: npt.NDArray[np.float32]
