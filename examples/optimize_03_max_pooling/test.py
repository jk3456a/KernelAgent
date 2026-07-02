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

"""Task-agnostic correctness test for a Triton kernel.

Argument binding is delegated to the shared SSOT binder in ``timing.py``
(``bind_kernel_function``), the SAME one used by benchmarking, so a kernel that
verifies here binds identically under benchmarking. Tensor parameters (inputs +
model weights/biases) are bound positionally — kernel tensor names may be
chosen freely — while scalar hyperparameters bind by name.
"""

import sys

import torch
from kernel import kernel_function
from problem import Model, get_init_inputs, get_inputs
from timing import bind_kernel_function


def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    # Setup reference model
    model = Model(*get_init_inputs()).to(device).to(dtype)
    inputs = [
        (
            x.to(device).to(dtype)
            if isinstance(x, torch.Tensor) and x.is_floating_point()
            else (x.to(device) if isinstance(x, torch.Tensor) else x)
        )
        for x in get_inputs()
    ]

    # Get reference output
    with torch.no_grad():
        ref_output = model(*inputs)

    # Bind kernel arguments via the shared planner (positional tensors +
    # named config scalars) and invoke.
    invoke = bind_kernel_function(kernel_function, inputs, model)
    kernel_output = invoke()

    # Compare
    # Handle in-place kernels that return None
    if kernel_output is None:
        # Assume in-place modification of first input
        kernel_output = inputs[0]
    # Handle shape mismatch: kernel may return per-sample loss vs reference scalar mean
    if ref_output.dim() == 0 and kernel_output.dim() >= 1:
        kernel_output = kernel_output.mean()
    elif kernel_output.dim() == 0 and ref_output.dim() >= 1:
        ref_output = ref_output.mean()
    # Align dtypes for comparison
    if ref_output.dtype != kernel_output.dtype:
        # If kernel outputs higher precision, recompute reference at that precision
        # using the SAME inputs to ensure fair comparison
        if kernel_output.dtype == torch.float32 and ref_output.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            model_f32 = Model(*get_init_inputs()).to(device).to(torch.float32)
            inputs_f32 = [
                x.to(torch.float32) if x.is_floating_point() else x for x in inputs
            ]
            with torch.no_grad():
                ref_output = model_f32(*inputs_f32)
        else:
            kernel_output = kernel_output.to(ref_output.dtype)
    if torch.allclose(ref_output, kernel_output, rtol=1e-2, atol=1e-2):
        print("PASS")
        return True
    else:
        max_diff = (ref_output - kernel_output).abs().max().item()
        print(f"FAIL: max difference = {max_diff}")
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
