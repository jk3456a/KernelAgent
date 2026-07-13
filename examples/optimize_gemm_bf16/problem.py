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
    """Square BF16 matrix multiplication: C = A @ B."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


M = 4096
N = 4096
K = 4096


def get_inputs():
    """Return BF16 inputs with GEMM shapes (M, K) and (K, N)."""
    A = torch.rand(M, K, dtype=torch.bfloat16)
    B = torch.rand(K, N, dtype=torch.bfloat16)
    return [A, B]


def get_init_inputs():
    return []


def get_workload_spec():
    """Semantic work used for throughput/MFU reporting (one FMA = two FLOPs)."""
    return {
        "operation": "gemm",
        "flops": 2 * M * N * K,
        "epilogue_flops": 0,
        "minimum_io_elements": M * K + K * N + M * N,
        "flop_convention": "2_per_fma",
        "flop_scope": "primary_tensor_math",
        "io_scope": "semantic_minimum",
        "details": {"M": M, "N": N, "K": K},
    }
