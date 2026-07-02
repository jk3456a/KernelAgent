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

"""Core timing and model loading utilities for kernel benchmarking.

This module consolidates:
- Timing functions (CUDA events, do_bench, host timing)
- Model/kernel loading utilities
- Statistics computation

Inspired by KernelBench's timing.py
"""

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from kernel_binding import CONFIG_PARAM_NAMES, plan_kernel_binding


# =============================================================================
# Model and Kernel Loading Utilities
# =============================================================================


class CompilationError(RuntimeError):
    """Raised when a kernel or problem file fails to compile/import."""

    pass


def import_module(path: Path, module_name: Optional[str] = None):
    """Dynamically import a Python file.

    Args:
        path: Path to the Python file
        module_name: Optional name for the module (auto-generated if None)

    Returns:
        The imported module

    Raises:
        FileNotFoundError: If path doesn't exist
        CompilationError: If import fails
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if module_name is None:
        module_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CompilationError(f"Failed to create spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CompilationError(f"Failed to import {path}: {exc}") from exc

    return module


def load_problem_interface(
    problem_file: Path,
) -> Tuple[type, Callable, Optional[Callable]]:
    """Load the standard problem interface from a problem file.

    Args:
        problem_file: Path to problem file

    Returns:
        Tuple of (Model class, get_inputs function, get_init_inputs function)

    Raises:
        CompilationError: If problem file doesn't define required interface
    """
    module = import_module(problem_file, "problem")

    Model = getattr(module, "Model", None)
    get_inputs = getattr(module, "get_inputs", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)

    if Model is None:
        raise CompilationError("Problem file must define 'Model' class")
    if get_inputs is None:
        raise CompilationError("Problem file must define 'get_inputs()' function")

    return Model, get_inputs, get_init_inputs


def prepare_inputs(
    get_inputs: Callable,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, ...]:
    """Prepare inputs by converting to target device and dtype.

    Args:
        get_inputs: Function that returns inputs
        device: Target device
        dtype: Target dtype for floating-point tensors

    Returns:
        Tuple of prepared inputs
    """
    inputs = get_inputs()
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)

    # Convert inputs to target device and dtype
    # IMPORTANT: Only convert floating-point tensors; preserve integer/bool tensors
    converted_inputs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            inp = inp.to(device=device)
            # Preserve integer/bool tensors (e.g., targets for classification)
            if inp.is_floating_point():
                inp = inp.to(dtype=dtype)
        converted_inputs.append(inp)

    return tuple(converted_inputs)


def prepare_pytorch_model(
    problem_file: Path,
    device: torch.device | str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    """Prepare PyTorch model and inputs for benchmarking.

    This handles the full workflow:
    1. Load problem interface (Model, get_inputs, get_init_inputs)
    2. Initialize model with init inputs
    3. Move model to device
    4. Handle dtype conversion based on whether model has parameters

    Args:
        problem_file: Path to problem file
        device: Target device
        dtype: Target dtype (auto-detected if None)

    Returns:
        Tuple of (model, inputs) ready for benchmarking
    """
    Model, get_inputs, get_init_inputs = load_problem_interface(problem_file)

    # Get initialization inputs (e.g., features, eps for RMSNorm)
    init_inputs = get_init_inputs() if get_init_inputs is not None else []
    if not isinstance(init_inputs, (tuple, list)):
        init_inputs = [init_inputs]

    model = Model(*init_inputs) if init_inputs else Model()
    model = model.cuda()
    has_parameters = any(p.numel() > 0 for p in model.parameters())

    inputs = get_inputs()
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)

    # Default to bfloat16 unless explicitly specified or model is a loss function
    target_dtype = dtype or torch.bfloat16
    is_loss_function = isinstance(model, torch.nn.modules.loss._Loss)

    if has_parameters or not is_loss_function:
        # Models with parameters (Conv, Linear, etc.) OR compute operations (matmul, etc.)
        # → use bfloat16 (or user-specified dtype)
        if has_parameters:
            model = model.to(target_dtype)
        inputs = [
            (
                inp.cuda().to(target_dtype)
                if isinstance(inp, torch.Tensor) and inp.is_floating_point()
                else inp.cuda()
                if isinstance(inp, torch.Tensor)
                else inp
            )
            for inp in inputs
        ]
    else:
        # Loss functions (no parameters) → use float32 for compatibility
        # PyTorch cross_entropy doesn't support bf16 on CUDA
        processed_inputs = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                if i == 0 and inp.is_floating_point():
                    # First input (predictions) - convert to float32 for compatibility
                    processed_inputs.append(inp.cuda().to(torch.float32))
                else:
                    # Other inputs (like targets) - just move to CUDA, preserve dtype
                    processed_inputs.append(inp.cuda())
            else:
                processed_inputs.append(inp)
        inputs = processed_inputs

    return model, tuple(inputs)


def load_kernel_function(kernel_file: Path) -> Callable:
    """Load kernel_function from a kernel file.

    Args:
        kernel_file: Path to kernel file

    Returns:
        The kernel_function callable

    Raises:
        CompilationError: If kernel file doesn't define kernel_function
    """
    module = import_module(kernel_file, "kernel")

    kernel_function = getattr(module, "kernel_function", None)
    if kernel_function is None:
        raise CompilationError(
            f"Kernel file {kernel_file.name} must define 'kernel_function'"
        )

    return kernel_function


# =============================================================================
# Timing Utilities
# =============================================================================


def clear_l2_cache(device: torch.device | str = "cuda") -> None:
    """Clear L2 cache by thrashing with a large tensor.

    This ensures we measure cold cache performance, which is more representative
    of real-world scenarios where data isn't already cached.

    Reference: KernelBench timing.py
    L2 cache sizes: A100=40MB, H100=50MB, H200=90MB, RTX4090=72MB, L40S=48MB
    We overwrite >256MB to fully thrash L2 cache.

    Args:
        device: CUDA device to use
    """
    # 32 * 1024 * 1024 * 8B = 256MB - enough to thrash most GPU L2 caches
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)  # Write to tensor to ensure cache thrashing
    del dummy


def time_with_cuda_events(
    kernel_fn: Callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    clear_cache: bool = True,
    discard_first: int = 0,
    verbose: bool = False,
    device: Optional[torch.device | str] = None,
) -> list[float]:
    """Time a CUDA kernel using CUDA events for accurate device-side timing.

    This measures actual GPU execution time without host-side overhead.
    Each trial clears L2 cache to measure cold-cache performance.

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_warmup: Number of warmup iterations
        num_trials: Number of timing trials
        clear_cache: Whether to clear L2 cache between trials
        discard_first: Number of initial trials to discard
        verbose: Print per-trial timing info
        device: CUDA device to use (None = current device)

    Returns:
        List of elapsed times in milliseconds (length = num_trials)
    """
    if device is None:
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        # Warmup
        for _ in range(num_warmup):
            kernel_fn(*args)
            torch.cuda.synchronize(device=device)

        torch.cuda.empty_cache()

        if verbose:
            print(
                f"[Timing] Device: {torch.cuda.get_device_name(device)}, "
                f"warmup={num_warmup}, trials={num_trials}"
            )

        elapsed_times: list[float] = []

        # Timing trials
        for trial in range(num_trials + discard_first):
            torch.cuda.synchronize(device=device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            if clear_cache:
                clear_l2_cache(device=device)

            start_event.record()
            kernel_fn(*args)
            end_event.record()

            torch.cuda.synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)

            if trial >= discard_first:
                if verbose:
                    print(
                        f"  Trial {trial - discard_first + 1}: {elapsed_time_ms:.3f} ms"
                    )
                elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def time_with_inductor_benchmarker(
    kernel_fn: Callable,
    args: list[Any],
    num_warmup: int = 25,
    verbose: bool = False,
) -> float:
    """Time using PyTorch Inductor's benchmarker (simplest approach).

    This is a thin wrapper around torch._inductor.runtime.benchmarking.benchmarker,
    which handles CUDA synchronization and timing internally.

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_warmup: Number of warmup iterations
        verbose: Print timing info

    Returns:
        Elapsed time in milliseconds (single value, not a list)

    Note:
        This uses a private PyTorch API (_inductor) which may change without notice.
    """
    from torch._inductor.runtime.benchmarking import benchmarker

    # Warmup
    for _ in range(num_warmup):
        kernel_fn(*args)

    ms = benchmarker.benchmark_gpu(lambda: kernel_fn(*args))

    if verbose:
        print(f"[Timing] Inductor benchmarker: {ms:.4f} ms")

    return ms


def time_with_triton_do_bench(
    kernel_fn: Callable,
    args: list[Any],
    warmup: int = 25,
    rep: int = 100,
    verbose: bool = False,
    device: Optional[torch.device | str] = None,
) -> list[float]:
    """Time using Triton's do_bench with adaptive trial count.

    Triton's do_bench automatically determines the number of trials based on
    warmup/rep time budgets. This is convenient but gives less control.

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        warmup: Warmup time budget in milliseconds
        rep: Repetition time budget in milliseconds
        verbose: Print timing info
        device: CUDA device to use

    Returns:
        List of all trial times in milliseconds
    """
    if device is None:
        device = torch.cuda.current_device()

    import triton.testing as triton_testing

    with torch.cuda.device(device):
        if verbose:
            print(
                f"[Timing] Using triton.do_bench on {torch.cuda.get_device_name(device)}"
            )

        def wrapped_fn():
            return kernel_fn(*args)

        times = triton_testing.do_bench(
            fn=wrapped_fn,
            warmup=warmup,
            rep=rep,
            grad_to_none=None,
            quantiles=None,
            return_mode="all",
        )

    return times


def compute_timing_stats(
    elapsed_times: list[float],
    device: Optional[torch.device | str] = None,
) -> dict[str, Any]:
    """Compute essential timing statistics.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device (for recording hardware info)

    Returns:
        Dictionary with timing statistics:
            - mean: Mean time in ms
            - std: Standard deviation in ms
            - min: Minimum time in ms
            - max: Maximum time in ms
            - num_trials: Number of trials
            - all_times: All trial times
            - hardware: GPU name (if device provided)
    """
    times_array = np.array(elapsed_times)

    stats = {
        "mean": float(np.mean(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "num_trials": len(elapsed_times),
        "all_times": [float(t) for t in elapsed_times],
    }

    if device is not None:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)

    return stats


# =============================================================================
# Kernel argument binding (single source of truth)
# =============================================================================


_CONV_LINEAR_TYPES = (
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Linear,
)
_NORM_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
)


def extract_model_tensors_and_config(
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Extract model weight/bias tensors (ordered) and config scalars.

    The tensor list is ordered by module traversal (== forward order for the
    fused single-path models used here): each conv/linear contributes its
    ``weight`` then ``bias``; each norm its ``weight`` then ``bias``; finally a
    top-level ``model.bias`` parameter (fusion bias). This ordered queue is what
    the binding planner maps onto the kernel's positional tensor slots — so the
    kernel's tensor parameters may be named freely.

    Returns:
        Tuple of (ordered tensor list, config scalar dict keyed by hyperparameter
        name such as ``stride`` / ``eps`` / ``kernel_size``).
    """
    tensors: list[torch.Tensor] = []
    config: dict[str, Any] = {}

    for _, module in model.named_modules():
        if isinstance(module, _CONV_LINEAR_TYPES):
            if getattr(module, "weight", None) is not None:
                tensors.append(module.weight)
            if getattr(module, "bias", None) is not None:
                tensors.append(module.bias)
            for attr in ("stride", "padding", "dilation", "output_padding"):
                val = getattr(module, attr, None)
                if val is not None:
                    config.setdefault(attr, val)
            if hasattr(module, "groups"):
                config.setdefault("groups", module.groups)
        elif isinstance(module, _NORM_TYPES):
            if getattr(module, "weight", None) is not None:
                tensors.append(module.weight)
            if getattr(module, "bias", None) is not None:
                tensors.append(module.bias)
            if hasattr(module, "eps"):
                config.setdefault("eps", module.eps)
            if hasattr(module, "num_groups"):
                config.setdefault("num_groups", module.num_groups)
            if hasattr(module, "normalized_shape"):
                config.setdefault("normalized_shape", module.normalized_shape)

    # Top-level fusion bias (e.g. Conv+ReLU+BiasAdd stores it on the Model).
    top_bias = getattr(model, "bias", None)
    if isinstance(top_bias, (torch.Tensor, torch.nn.Parameter)):
        tensors.append(top_bias)

    # Simple scalar attributes stored directly on the Model (dim, eps, ...).
    for attr_name in CONFIG_PARAM_NAMES:
        if hasattr(model, attr_name):
            val = getattr(model, attr_name)
            if not isinstance(val, (torch.Tensor, torch.nn.Module)):
                config.setdefault(attr_name, val)

    return tensors, config


def _collapse_uniform(value: Any) -> Any:
    """Collapse a uniform tuple/list (e.g. ``(3, 3)``) to its scalar element.

    Conv layers expose stride/kernel_size as tuples, but kernels commonly take
    a scalar int; when all entries are equal, pass the scalar.
    """
    if isinstance(value, (tuple, list)) and len(value) >= 1 and all(
        e == value[0] for e in value
    ):
        return value[0]
    return value


def bind_kernel_function(
    kernel_function: Callable,
    inputs: list,
    model: Optional[torch.nn.Module],
) -> Callable:
    """Return a zero-arg-adapted callable that invokes ``kernel_function``.

    Uses the shared :func:`plan_kernel_binding` planner so verification and
    benchmarking bind identically. Tensor parameters (inputs + model weights)
    are bound positionally — kernel tensor names are never matched — while
    scalar hyperparameters are bound by name.

    Args:
        kernel_function: The candidate kernel.
        inputs: Tensors from ``get_inputs()`` (already on device/dtype).
        model: The reference model to extract weights/config from, or ``None``
            for purely functional kernels.

    Returns:
        A callable taking no arguments that runs the kernel with bound args.
    """
    sig = inspect.signature(kernel_function)
    kernel_params = [
        name
        for name, p in sig.parameters.items()
        if p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )

    model_tensors: list = []
    config: dict[str, Any] = {}
    if model is not None:
        model_tensors, config = extract_model_tensors_and_config(model)

    plan = plan_kernel_binding(
        kernel_params=kernel_params,
        has_var_positional=has_var_positional,
        num_inputs=len(inputs),
        num_model_tensors=len(model_tensors),
        config_names=list(config.keys()),
    )

    if not plan.needs_model:
        return lambda: kernel_function(*inputs)

    def _invoke():
        args = []
        for kind, idx in plan.positional_sources:
            args.append(inputs[idx] if kind == "input" else model_tensors[idx])
        kwargs = {name: _collapse_uniform(config[name]) for name in plan.config_kwargs}
        return kernel_function(*args, **kwargs)

    return _invoke
