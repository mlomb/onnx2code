from onnx2code.util import compute_strides, resolve_padding, resolve_stride

from .operation import OpCall, Operation, OpImpl


class Conv(Operation):
    """
    Conv operator

    Only 2D convolutions are supported

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#conv
    """

    node_types = {"Conv"}

    def parse(self) -> None:
        assert (
            len(self.inputs) == 2 or len(self.inputs) == 3
        ), "expected two or three inputs"
        assert len(self.outputs) == 1, "expected one output"

        self.X = self.inputs[0]
        self.W = self.inputs[1]
        self.B = self.inputs[2] if len(self.inputs) == 3 else None
        self.Y = self.outputs[0]

        self.strides = resolve_stride(self.node)
        self.pads = resolve_padding(self.node, self.X.shape, self.W.shape)

    def call(self) -> OpCall:
        return OpCall(
            name="Conv",
            sig_params=[self.X.shape, self.W.shape, self.strides, self.pads],
            params=["X", "W", "B", "Y"] if self.B is not None else ["X", "W", "Y"],
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
        # pads_end = [self.pads[2], self.pads[3]]

        input_strides = compute_strides(self.X.shape)
        output_strides = compute_strides(self.Y.shape)
        kernel_strides = compute_strides(self.W.shape)

        source = ""

        source += f"""
        for(int f = 0; f < {F}; f++) {{
            // start position of kernel
            for(int h = 0; h < {self.Y.shape[2]}; h++) {{
                for(int w = 0; w < {self.Y.shape[3]}; w++) {{
                    float accum = {"0.0f" if self.B is None else "B[f]" };

                    // position in kernel
                    for(int cc = 0; cc < {KC}; cc++) {{
                        for(int hh = 0; hh < {KH}; hh++) {{
                            for(int ww = 0; ww < {KW}; ww++) {{
                                const int ih = {-pads_start[0]} + (h * {self.strides[0]}) + hh;
                                const int iw = {-pads_start[1]} + (w * {self.strides[1]}) + ww;
                                if(ih >= 0 && ih < {H} && iw >= 0 && iw < {W}) {{
                                    accum += X[
                                        cc * {input_strides[1]} +
                                        ih * {input_strides[2]} +
                                        iw * {input_strides[3]}
                                    ] * W[
                                        f * {kernel_strides[0]} +
                                        cc * {kernel_strides[1]} +
                                        hh * {kernel_strides[2]} +
                                        ww * {kernel_strides[3]}
                                    ];
                                }}
                            }}
                        }}
                    }}

                    Y[
                        f * {output_strides[1]} +
                        h * {output_strides[2]} +
                        w * {output_strides[3]}
                    ] = accum;
                }}
            }}
        }}
        """

        return OpImpl(lang="c", source=source)
