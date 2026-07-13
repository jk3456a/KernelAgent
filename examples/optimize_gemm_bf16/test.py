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

"""Strict correctness test for the 4096 x 4096 x 4096 BF16 GEMM."""

import sys

import torch


M = 4096
N = 4096
K = 4096


def validate_output(ref_output, kernel_output):
    """Require an accurate BF16 tensor with the exact GEMM output shape."""
    if not isinstance(kernel_output, torch.Tensor):
        print("FAIL: kernel_function must return a torch.Tensor")
        return False

    expected_shape = (M, N)
    if tuple(kernel_output.shape) != expected_shape:
        print(
            f"FAIL: output shape is {tuple(kernel_output.shape)}, "
            f"expected {expected_shape}"
        )
        return False

    if kernel_output.dtype != torch.bfloat16:
        print(
            "FAIL: output dtype is "
            f"{kernel_output.dtype}, expected {torch.bfloat16}"
        )
        return False

    if torch.allclose(ref_output, kernel_output, rtol=1e-2, atol=1e-2):
        print("PASS")
        return True

    max_diff = (
        ref_output.to(torch.float32) - kernel_output.to(torch.float32)
    ).abs().max().item()
    print(f"FAIL: max difference = {max_diff}")
    return False


def test_kernel():
    from kernel import kernel_function
    from timing import bind_kernel_function

    device = "cuda"
    dtype = torch.bfloat16
    inputs = [
        torch.rand((M, K), device=device, dtype=dtype),
        torch.rand((K, N), device=device, dtype=dtype),
    ]

    with torch.no_grad():
        ref_output = torch.matmul(*inputs)

    invoke = bind_kernel_function(kernel_function, inputs, model=None)
    kernel_output = invoke()
    return validate_output(ref_output, kernel_output)


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
