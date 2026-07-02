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

"""Single source of truth for binding kernel_function arguments.

Both correctness verification (``test.py``) and benchmarking
(``kernel_subprocess.py``) must feed the *same* arguments to a candidate
``kernel_function`` in the *same* order, or a kernel that verifies will crash
under benchmarking (and vice versa). Historically the two paths kept forked,
drifting copies of a name-matching heuristic; conv kernels named their tensors
freely (``conv_weight`` / ``extra_bias``) and neither fixed name table matched,
so binding silently failed.

The planner here is deliberately torch-free (pure signature/shape reasoning) so
it can be unit-tested without a GPU. The thin torch glue that materialises the
actual tensors lives in the caller.

Binding rules (positional-first, name-only for scalars):

* **Tensor arguments** — the ordered queue ``[*inputs, *model_tensors]`` (model
  weights/biases in forward/module order) fills the kernel's non-config
  positional parameters left to right. Tensor *names* are never matched, so
  agent1's free naming binds correctly.
* **Config scalars** — parameters whose name is a known layer hyperparameter
  (``stride`` / ``kernel_size`` / ``eps`` / …) are filled by name as kwargs,
  not consumed from the tensor queue.
* **``*args`` kernels** — all tensors are passed positionally; config as kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CONFIG_PARAM_NAMES", "BindingPlan", "plan_kernel_binding"]


# Names that denote scalar layer hyperparameters (NOT tensors). A kernel
# parameter with one of these names is filled from the extracted model config
# by name; everything else is treated as a positional tensor slot.
CONFIG_PARAM_NAMES = frozenset(
    {
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "output_padding",
        "groups",
        "eps",
        "num_groups",
        "normalized_shape",
        "dim",
        "negative_slope",
        "min_val",
        "max_val",
        "beta",
        "threshold",
        "alpha",
        "lambd",
        "upper",
        "lower",
        "p",
    }
)


@dataclass
class BindingPlan:
    """How to call a candidate ``kernel_function``.

    Attributes:
        positional_sources: Ordered list of ``(kind, index)`` for each
            positional (non-config) kernel parameter, where ``kind`` is
            ``"input"`` or ``"tensor"`` and ``index`` selects from that queue.
        config_kwargs: Names of kernel parameters to fill from model config by
            name (scalars).
        needs_model: Whether the kernel consumes any model-derived value
            (tensors or config). ``False`` means "call with inputs only".
        is_varargs: The kernel uses ``*args``; pass all tensors positionally.
        unassigned_tensors: Count of model tensors that had no positional slot
            (a real signature mismatch, surfaced rather than silently dropped).
    """

    positional_sources: list[tuple[str, int]] = field(default_factory=list)
    config_kwargs: list[str] = field(default_factory=list)
    needs_model: bool = False
    is_varargs: bool = False
    unassigned_tensors: int = 0


def plan_kernel_binding(
    kernel_params: list[str],
    has_var_positional: bool,
    num_inputs: int,
    num_model_tensors: int,
    config_names: list[str],
) -> BindingPlan:
    """Plan how to bind kernel arguments from inputs + extracted model params.

    Args:
        kernel_params: Named (non-vararg) parameters of ``kernel_function`` in
            declaration order.
        has_var_positional: Whether the signature contains ``*args``.
        num_inputs: Number of tensors from ``get_inputs()``.
        num_model_tensors: Number of model weight/bias tensors extracted (in
            forward/module order).
        config_names: Config scalar names available from the extracted model
            (e.g. ``stride``, ``eps``). Only those that also appear as kernel
            parameters (or, for varargs kernels, all of them) are passed.

    Returns:
        A :class:`BindingPlan`.
    """
    config_available = set(config_names)

    # A kernel "needs the model" if it declares more tensor slots than there
    # are plain inputs (extra slots want weights), names a config scalar, or is
    # a varargs kernel with model tensors/config to forward.
    if has_var_positional:
        # All tensors go positionally; forward every available config scalar.
        positional = [("input", i) for i in range(num_inputs)]
        positional += [("tensor", j) for j in range(num_model_tensors)]
        needs_model = num_model_tensors > 0 or bool(config_available)
        return BindingPlan(
            positional_sources=positional,
            config_kwargs=sorted(config_available),
            needs_model=needs_model,
            is_varargs=True,
            unassigned_tensors=0,
        )

    # Fixed signature: split kernel params into config (by name) vs tensor slots.
    config_kwargs = [p for p in kernel_params if p in config_available]
    tensor_slots = [p for p in kernel_params if p not in config_available]

    needs_model = (
        bool(config_kwargs)
        or num_model_tensors > 0
        or len(tensor_slots) > num_inputs
    )

    if not needs_model:
        # Pure functional kernel (e.g. matmul): inputs only, positionally.
        return BindingPlan(
            positional_sources=[("input", i) for i in range(len(tensor_slots))],
            config_kwargs=[],
            needs_model=False,
            is_varargs=False,
            unassigned_tensors=0,
        )

    # Fill tensor slots from the queue [*inputs, *model_tensors] in order.
    positional: list[tuple[str, int]] = []
    input_idx = 0
    tensor_idx = 0
    for _ in tensor_slots:
        if input_idx < num_inputs:
            positional.append(("input", input_idx))
            input_idx += 1
        elif tensor_idx < num_model_tensors:
            positional.append(("tensor", tensor_idx))
            tensor_idx += 1
        # else: slot left unfilled (fewer values than slots) — the caller will
        # raise a clear TypeError when invoking, which is the correct signal.

    unassigned = max(0, num_model_tensors - tensor_idx)

    return BindingPlan(
        positional_sources=positional,
        config_kwargs=config_kwargs,
        needs_model=True,
        is_varargs=False,
        unassigned_tensors=unassigned,
    )
