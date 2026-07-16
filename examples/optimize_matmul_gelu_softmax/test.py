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

"""Strict correctness test for the fused BF16 Matmul + GELU + Softmax.

The kernel/problem/timing imports stay inside ``test_kernel()`` so this module
can be imported on a CPU-only machine to reuse ``validate_output``.
"""

import sys

import torch


batch_size = 1024
in_features = 8192
out_features = 8192


def validate_output(
    ref_output, kernel_output, expected_shape=(batch_size, out_features)
):
    """Require an accurate BF16 softmax tensor with the exact fused shape."""
    if not isinstance(kernel_output, torch.Tensor):
        print("FAIL: kernel_function must return a torch.Tensor")
        return False

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

    # Softmax rows must sum to 1 by definition; the allclose tolerances alone
    # cannot distinguish tiny softmax values from zeros.
    row_sums = kernel_output.to(torch.float32).sum(dim=1)
    max_row_sum_error = (row_sums - 1.0).abs().max().item()
    if max_row_sum_error > 2e-2:
        print(f"FAIL: softmax row sums deviate from 1 by {max_row_sum_error}")
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
    from problem import Model, get_init_inputs, get_inputs
    from timing import bind_kernel_function

    device = "cuda"
    dtype = torch.bfloat16

    model = Model(*get_init_inputs()).to(device).to(dtype)
    inputs = [x.to(device).to(dtype) for x in get_inputs()]

    with torch.no_grad():
        ref_output = model(*inputs)

    invoke = bind_kernel_function(kernel_function, inputs, model)
    kernel_output = invoke()
    return validate_output(ref_output, kernel_output)


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
