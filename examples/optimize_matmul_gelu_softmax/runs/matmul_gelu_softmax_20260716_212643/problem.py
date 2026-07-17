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

# KernelBench level2 #99: Matmul + GELU + Softmax (BF16 variant).
import torch
import torch.nn as nn


class Model(nn.Module):
    """Linear layer followed by GELU and a row softmax."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.bfloat16)]


def get_init_inputs():
    return [in_features, out_features]


def get_workload_spec():
    """Semantic work used for throughput/MFU reporting (one FMA = two FLOPs)."""
    return {
        "operation": "matmul_gelu_softmax",
        "flops": 2 * batch_size * in_features * out_features,
        # Per output element: bias add (1), GELU (1), and the stable softmax's
        # max/subtract/exp/sum/divide (5).
        "epilogue_flops": 7 * batch_size * out_features,
        "minimum_io_elements": (
            batch_size * in_features
            + in_features * out_features
            + out_features
            + batch_size * out_features
        ),
        "flop_convention": "2_per_fma",
        "flop_scope": "primary_tensor_math",
        "io_scope": "semantic_minimum",
        "details": {
            "batch": batch_size,
            "in_features": in_features,
            "out_features": out_features,
            "softmax_dim": 1,
        },
    }
