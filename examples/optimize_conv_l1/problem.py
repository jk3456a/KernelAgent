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

# KernelBench level1 #50: standard 2D convolution (square input, square kernel).
# AlexNet first conv: Conv2d(3 -> 96, k=11, stride=4, padding=2) on 256x3x224x224.
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


batch_size = 256
num_classes = 1000


def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]


def get_workload_spec():
    """AlexNet first-conv tensor work and semantic minimum tensor traffic."""
    in_channels = 3
    out_channels = 96
    input_height = input_width = 224
    kernel_size = 11
    stride = 4
    padding = 2
    groups = 1
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    output_elements = (
        batch_size * out_channels * output_height * output_width
    )
    input_elements = (
        batch_size * in_channels * input_height * input_width
    )
    weight_elements = (
        out_channels * in_channels * kernel_size * kernel_size // groups
    )
    conv_flops = (
        2
        * output_elements
        * in_channels
        * kernel_size
        * kernel_size
        // groups
    )
    return {
        "operation": "conv2d",
        "flops": conv_flops,
        "epilogue_flops": output_elements,  # Conv bias add.
        "minimum_io_elements": (
            input_elements + weight_elements + out_channels + output_elements
        ),
        "flop_convention": "2_per_fma",
        "flop_scope": "primary_tensor_math",
        "io_scope": "semantic_minimum",
        "details": {
            "batch": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width,
            "kernel_height": kernel_size,
            "kernel_width": kernel_size,
            "stride": stride,
            "padding": padding,
            "groups": groups,
        },
    }
