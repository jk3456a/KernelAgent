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

"""Tests for the single-source kernel argument binding planner.

The planner is the SSOT used by BOTH correctness verification (test.py) and
benchmarking (kernel_subprocess.py). It must bind agent1's *freely named*
tensor parameters (e.g. ``conv_weight`` / ``extra_bias``) positionally, since
name-guessing against a fixed table misses those names and silently mis-binds.
"""

import sys
from pathlib import Path

# The binding planner lives next to the benchmark scripts (pushed to every
# remote workdir via timing.py). Import it the same way the workdir scripts do.
_BENCH_DIR = (
    Path(__file__).resolve().parent.parent
    / "triton_kernel_agent"
    / "opt_worker_component"
    / "benchmarking"
)
sys.path.insert(0, str(_BENCH_DIR))

from kernel_binding import plan_kernel_binding  # noqa: E402


def test_conv_free_names_bind_positionally():
    """Conv kernel with free tensor names binds all params (the regression).

    kernel_function(x, conv_weight, conv_bias, extra_bias)
    inputs   = [x]                       (1 tensor input)
    tensors  = [conv.weight, conv.bias, model.bias]  (3 model tensors)
    config   = {stride, padding, ...}    (scalars — irrelevant here)

    Every kernel param must be bound; nothing UNBOUND, nothing left over.
    """
    plan = plan_kernel_binding(
        kernel_params=["x", "conv_weight", "conv_bias", "extra_bias"],
        has_var_positional=False,
        num_inputs=1,
        num_model_tensors=3,
        config_names=["stride", "padding", "dilation", "groups", "output_padding"],
    )
    # 4 positional slots filled from the tensor queue (1 input + 3 model tensors)
    assert plan.positional_sources == [
        ("input", 0),
        ("tensor", 0),
        ("tensor", 1),
        ("tensor", 2),
    ]
    assert plan.config_kwargs == []
    assert plan.needs_model is True


def test_gemm_no_weights_binds_inputs_only():
    """GEMM matmul(A, B): two inputs, no model tensors, no config."""
    plan = plan_kernel_binding(
        kernel_params=["A", "B"],
        has_var_positional=False,
        num_inputs=2,
        num_model_tensors=0,
        config_names=[],
    )
    assert plan.positional_sources == [("input", 0), ("input", 1)]
    assert plan.needs_model is False


def test_maxpool_config_scalars_bind_by_name():
    """MaxPool kernel: one input tensor + named scalar hyperparameters.

    kernel_function(x, kernel_size, stride, padding, dilation) — no weights,
    so kernel_size/stride/... are config scalars matched by name.
    """
    plan = plan_kernel_binding(
        kernel_params=["x", "kernel_size", "stride", "padding", "dilation"],
        has_var_positional=False,
        num_inputs=1,
        num_model_tensors=0,
        config_names=["kernel_size", "stride", "padding", "dilation"],
    )
    assert plan.positional_sources == [("input", 0)]
    assert set(plan.config_kwargs) == {
        "kernel_size",
        "stride",
        "padding",
        "dilation",
    }


def test_varargs_kernel_passes_all_tensors_positionally():
    """*args kernel (e.g. rmsnorm ``kernel_function(x, *args, **kwargs)``).

    All tensors (inputs + model tensors) go positionally; config as kwargs.
    """
    plan = plan_kernel_binding(
        kernel_params=["x"],
        has_var_positional=True,
        num_inputs=1,
        num_model_tensors=1,
        config_names=["eps"],
    )
    assert plan.is_varargs is True
    # inputs + all model tensors, positionally
    assert plan.positional_sources == [("input", 0), ("tensor", 0)]
    assert set(plan.config_kwargs) == {"eps"}


def test_more_tensors_than_slots_is_flagged():
    """If model tensors exceed the kernel's free positional slots, flag it.

    A fixed-signature kernel that cannot receive all model tensors is a real
    mismatch, not something to silently truncate.
    """
    plan = plan_kernel_binding(
        kernel_params=["x", "weight"],  # only 2 slots
        has_var_positional=False,
        num_inputs=1,
        num_model_tensors=3,  # 1 input + 3 tensors = 4 > 2 slots
        config_names=[],
    )
    assert plan.unassigned_tensors > 0
