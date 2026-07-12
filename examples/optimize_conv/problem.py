# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# KernelBench level2 #1: Conv2D + ReLU + BiasAdd (a fused conv workload).
import torch
import torch.nn as nn


class Model(nn.Module):
    """Convolution, ReLU, then add a bias term."""

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x + self.bias
        return x


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]


def get_workload_spec():
    """Conv tensor-math FLOPs plus separately reported fused epilogue work."""
    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    output_elements = (
        batch_size * out_channels * output_height * output_width
    )
    input_elements = batch_size * in_channels * height * width
    weight_elements = (
        out_channels * in_channels * kernel_size * kernel_size
    )
    conv_flops = (
        2 * output_elements * in_channels * kernel_size * kernel_size
    )
    return {
        "operation": "conv2d_relu_bias_add",
        "flops": conv_flops,
        # Conv bias + the post-ReLU broadcast bias. ReLU is a comparison.
        "epilogue_flops": 2 * output_elements,
        "minimum_io_elements": (
            input_elements
            + weight_elements
            + out_channels
            + output_elements
            + out_channels
        ),
        "flop_convention": "2_per_fma",
        "flop_scope": "primary_tensor_math",
        "io_scope": "semantic_minimum",
        "details": {
            "batch": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input_height": height,
            "input_width": width,
            "output_height": output_height,
            "output_width": output_width,
            "kernel_height": kernel_size,
            "kernel_width": kernel_size,
            "groups": 1,
        },
    }
