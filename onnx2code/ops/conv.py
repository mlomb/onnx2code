from math import ceil
import numpy as np
from onnx2code.ops.gemm_tiling.GEMM import call_GEMM, external_paths_GEMM
from onnx2code.util import (
    compute_strides,
    get_attribute,
    resolve_padding_attribute,
    resolve_stride_attribute,
)

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

        group = get_attribute(self.node, "group", 1)
        if group != 1:
            raise NotImplementedError("depthwise is not supported (group != 1)")

        self.X = self.inputs[0]
        self.W = self.inputs[1]
        self.B = self.inputs[2] if len(self.inputs) == 3 else None
        self.Y = self.outputs[0]

        self.dilations = get_attribute(self.node, "dilations", [1] * 2)
        self.strides = resolve_stride_attribute(self.node)
        self.pads = resolve_padding_attribute(self.node, self.X.shape, self.W.shape)

    def call(self) -> OpCall:
        return OpCall(
            sig_name="Conv",
            sig_params=[self.X.shape, self.W.shape, self.strides, self.pads],
            inputs=self.inputs,
            outputs=self.outputs,
            input_names=("X", "W", "B"),
        )


@Conv.variant("c", priority=1)
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

                    OUT[
                        f * {output_strides[1]} +
                        h * {output_strides[2]} +
                        w * {output_strides[3]}
                    ] = accum;
                }}
            }}
        }}
        """

        return OpImpl(lang="c", source=source)


@Conv.variant(["im2col", "loop-tiling"], priority=0)
class ConvIm2col(Conv):
    def impl(self) -> OpImpl:
        input_shape = self.X.shape
        weight_shape = self.W.shape
        has_bias = self.B is not None
        pads, dilations, strides = self.pads, self.dilations, self.strides

        assert len(pads) == 4 or np.allclose(
            pads, 0
        ), "expected padding only in two dimensions"

        # onnx is NCHW
        N = input_shape[0]
        C = input_shape[1]
        H = input_shape[2]
        W = input_shape[3]
        F = weight_shape[0]  # filters
        KC = weight_shape[1]
        KH = weight_shape[2]
        KW = weight_shape[3]

        input_strides = compute_strides(input_shape)
        kernel_strides = compute_strides([KC, KH, KW])
        pads_start = [pads[0], pads[1]]
        pads_end = [pads[2], pads[3]]
        patch_stride = KC * KH * KW
        num_patches = ceil(
            (H - KH + 1 + pads_start[0] + pads_end[0] - (dilations[0] - 1) * (KH - 1))
            / strides[0]
        ) * ceil(
            (W - KW + 1 + pads_start[1] + pads_end[1] - (dilations[1] - 1) * (KW - 1))
            / strides[0]
        )
        im2col_shape = [patch_stride, num_patches]

        bias_code = (
            ""
            if has_bias
            else f"""
        // bias
        for (int f = 0; f < {F}; f++) {{
            for (int i = 0; i < {num_patches}; i++) {{
                output[f * {num_patches} + i] += biases[f];
            }}
        }}
        """
        )

        _N = F # weight_shape[0]
        _M = patch_stride # weight_shape[1]
        _K = im2col_shape[1]

        source = f"""
        // padding, dilations, strides
        // im2col
        float im2col[{np.prod(im2col_shape)}];
        int patch = 0;
        for(int c = 0; c < {C - KC + 1}; c++) {{
            for(int h = {-pads_start[0]}; h < {H - KH + 1 + pads_end[0] - (dilations[0] - 1) * (KH - 1)}; h += {strides[0]}) {{
                for(int w = {-pads_start[1]}; w < {W - KW + 1 + pads_end[1] - (dilations[1] - 1) * (KW - 1)}; w += {strides[1]}) {{
                    // copy patch
                    for(int cc = 0; cc < {KC}; cc++) {{
                        for(int hh = 0; hh < {KH}; hh++) {{
                            for(int ww = 0; ww < {KW}; ww++) {{
                                const int ih = h + hh * {dilations[0]};
                                const int iw = w + ww * {dilations[1]};
                                float value;
                                if(ih < 0 || ih >= {H} || iw < 0 || iw >= {W}) {{
                                    value = 0.0f;
                                }} else {{
                                    value = X[
                                        (c + cc) * {input_strides[1]} +
                                        ih * {input_strides[2]} +
                                        iw * {input_strides[3]}
                                    ];
                                }}
                                im2col[
                                    (cc * {kernel_strides[0]} +
                                    hh * {kernel_strides[1]} +
                                    ww * {kernel_strides[2]}) * {num_patches} +
                                    patch
                                ] = value;
                            }}
                        }}
                    }}
                    patch++;
                }}
            }}
        }}
        // gemm ({self.Y.shape})
        for(int row = 0; row < {_N}; row++) {{
            for(int col = 0; col < {_K}; col++) {{
                float sum = 0;
                for(int i = 0; i < {_M}; i++) {{
                    sum += W[row * {_M} + i] * im2col[i * {_K} + col];
                }}
                OUT[row * {_K} + col] = sum;
            }}
        }}
        // {call_GEMM(1,2,3,"W, im2col, OUT")}
        // bias
        {bias_code}
        """

        return OpImpl(lang="c", source=source, external_paths=external_paths_GEMM)
