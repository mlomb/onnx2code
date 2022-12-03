from onnx2code.util import compute_strides, get_attribute

from .operation import OpCall, Operation, OpImpl


class MaxPool(Operation):
    """
    MaxPool operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
    """

    node_types = {"MaxPool"}

    def parse(self) -> None:
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"

        self.X = self.inputs[0]
        self.Y = self.outputs[0]

        self.pads = get_attribute(self.node, "pads", [0] * len(self.X.shape) * 2)
        self.strides = get_attribute(self.node, "strides", [1] * len(self.X.shape))

        kernel_shape = get_attribute(self.node, "kernel_shape", [1] * len(self.X.shape))

        self.KH = kernel_shape[0]
        self.KW = kernel_shape[1]

    def call(self) -> OpCall:
        return OpCall(
            name=f"MaxPool_{0}",
            params=["X", "Y"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@MaxPool.variant("c")
class MaxPoolC(MaxPool):
    def impl(self) -> OpImpl:
        KH, KW = self.KH, self.KW

        H = self.X.shape[2]
        W = self.X.shape[3]

        pads_start = [self.pads[0], self.pads[1]]
        # pads_end = [self.pads[2], self.pads[3]]

        input_strides = compute_strides(self.X.shape)
        output_strides = compute_strides(self.Y.shape)

        source = f"""
        // start position of kernel
        for(int c = 0; c < {self.Y.shape[1]}; c++) {{
            for(int h = 0; h < {self.Y.shape[2]}; h++) {{
                for(int w = 0; w < {self.Y.shape[3]}; w++) {{
                    float mmax = -INFINITY;
                    // position in kernel
                    for(int hh = 0; hh < {KH}; hh++) {{
                        for(int ww = 0; ww < {KW}; ww++) {{
                            const int ih = {-pads_start[0]} + (h * {self.strides[0]}) + hh;
                            const int iw = {-pads_start[1]} + (w * {self.strides[1]}) + ww;
                            if(ih >= 0 && ih < {H} && iw >= 0 && iw < {W}) {{
                                const float val = X[
                                    c * {input_strides[1]} +
                                    ih * {input_strides[2]} +
                                    iw * {input_strides[3]}
                                ];
                                mmax = mmax > val ? mmax : val;
                            }}
                        }}
                    }}
                    Y[
                        c * {output_strides[1]} +
                        h * {output_strides[2]} +
                        w * {output_strides[3]}
                    ] = mmax;
                }}
            }}
        }}
        """

        return OpImpl(lang="c", source=source)
