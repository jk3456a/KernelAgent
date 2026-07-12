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

import torch
import torch.nn as nn


class Model(nn.Module):
    """Single square matrix multiplication (C = A @ B). KernelBench level1 #1."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


N = 4096


def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


def get_workload_spec():
    """Semantic work used for throughput/MFU reporting (one FMA = two FLOPs)."""
    return {
        "operation": "gemm",
        "flops": 2 * N * N * N,
        "epilogue_flops": 0,
        "minimum_io_elements": 3 * N * N,  # read A/B once, write C once
        "flop_convention": "2_per_fma",
        "flop_scope": "primary_tensor_math",
        "io_scope": "semantic_minimum",
        "details": {"M": N, "N": N, "K": N},
    }
