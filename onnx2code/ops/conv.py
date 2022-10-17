from onnx2code.util import compute_strides, get_attribute

from .operation import OpCall, Operation, OpImpl


class Conv(Operation):
    """
    Conv operator

    Only 2D convolutions are supported

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#conv
    """

    node_types = {"Conv"}

    def parse(self) -> None:
        assert len(self.inputs) == 2, "expected two inputs"
        assert len(self.outputs) == 1, "expected one output"

        self.X = self.inputs[0]
        self.W = self.inputs[1]
        self.Y = self.outputs[0]

        self.pads = get_attribute(self.node, "pads", [0] * len(self.X.shape) * 2)
        self.strides = get_attribute(self.node, "strides", [1] * 2)

    def call(self) -> OpCall:
        return OpCall(
            name=f"Conv_{self.X.shape_str()}_{self.W.shape_str()}",
            params=["X", "W", "Y"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Conv.variant("c")
class ConvC(Conv):
    def impl(self) -> OpImpl:
        # onnx is NCHW
        # N = self.X.shape[0]
        # C = self.X.shape[1]
        H = self.X.shape[2]
        W = self.X.shape[3]
        F = self.W.shape[0]  # filters
        KC = self.W.shape[1]
        KH = self.W.shape[2]
        KW = self.W.shape[3]

        pads_start = [self.pads[0], self.pads[1]]
        pads_end = [self.pads[2], self.pads[3]]

        input_strides = compute_strides(self.X.shape)
        output_strides = compute_strides(self.Y.shape)
        kernel_strides = compute_strides((KC, KH, KW))

        source = ""

        source += f"""
        for(int f = 0; f < {F}; f++) {{
            // start position of kernel
            for(int h = {-pads_start[0]}; h <= {H - KH + pads_end[0]}; h += {self.strides[0]}) {{
                for(int w = {-pads_start[1]}; w <= {W - KW + pads_end[1]}; w += {self.strides[1]}) {{
                    float accum = 0.0f;

                    // position in kernel
                    for(int hh = 0; hh < {KH}; hh++) {{
                        for(int ww = 0; ww < {KW}; ww++) {{
                            const int ih = h + hh;
                            const int iw = w + ww;
                            if(ih >= 0 && ih < {H} && iw >= 0 && iw < {W}) {{
                                accum += X[
                                    ih * {input_strides[2]} +
                                    iw * {input_strides[3]}
                                ] * W[
                                    f * {kernel_strides[0]} +
                                    hh * {kernel_strides[1]} +
                                    ww * {kernel_strides[2]}
                                ];
                            }}
                        }}
                    }}

                    Y[
                        f * {output_strides[1]} +
                        (h + {pads_start[0]}) * {output_strides[2]} +
                        (w + {pads_start[1]}) * {output_strides[3]}
                    ] = accum;
                }}
            }}
        }}
        """

        return OpImpl(lang="c", source=source)
