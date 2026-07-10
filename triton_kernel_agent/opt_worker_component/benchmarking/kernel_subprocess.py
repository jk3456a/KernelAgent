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

"""
A task-agnostic, profiling-only benchmark script for Triton kernels.
This script ONLY benchmarks candidate kernels without correctness checks.
Assumes correctness has been verified upstream.

Design:
- Skips correctness verification (assumes already verified)
- Only runs candidate kernels
- Fast profiling for iterative optimization loops
- Uses shared utilities from timing.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend_probe import inspect_pytorch_backend
from timing import (
    bind_kernel_function,
    import_module,
    load_kernel_function,
    load_problem_interface,
    prepare_inputs,
)
from typing import Any, Callable

import torch
import triton.testing as tt


def _run_once(invoke: Callable[[], torch.Tensor], name: str) -> torch.Tensor:
    """Run the bound kernel once to verify execution and get output shape."""
    try:
        with torch.inference_mode():
            return invoke()
    except Exception as exc:
        raise RuntimeError(f"{name} failed to execute: {exc}") from exc


def _benchmark(
    invoke: Callable[[], torch.Tensor],
    name: str,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark a bound kernel callable using triton.testing.do_bench."""
    try:
        ms = tt.do_bench(
            invoke,
            warmup=warmup,
            rep=rep,
            return_mode="mean",
        )
        print(f"{name}: {ms:.4f} ms (mean over {rep} runs)")
        return ms
    except Exception as exc:
        print(f"❌ {name}: Benchmark failed: {exc}")
        return float("inf")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Task-agnostic Triton kernel benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--problem",
        type=Path,
        required=True,
        help="Path to problem file (must define Model and get_inputs)",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        required=True,
        help="Path to kernel file (must define kernel_function)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include PyTorch reference model in benchmark",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--size", type=int, default=4096, help="Problem size N")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--json", type=Path, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    args.problem = args.problem.resolve()
    args.kernel = args.kernel.resolve()
    return args


def _load_problem(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[type, tuple, list, torch.nn.Module | None]:
    """Load problem interface, prepare inputs, and optionally create baseline model.

    Returns:
        Tuple of (Model class, inputs, init_inputs, baseline_model or None)
    """
    Model, get_inputs, get_init_inputs = load_problem_interface(args.problem)

    # Check for optional benchmark config override
    try:
        problem_mod = import_module(args.problem, "problem")
        get_benchmark_config = getattr(problem_mod, "get_benchmark_config", None)
        if get_benchmark_config is not None:
            config = get_benchmark_config()
            args.warmup = config.get("warmup", args.warmup)
            args.repeat = config.get("repeat", args.repeat)
            if not args.quiet:
                print(
                    f"Using problem-specific config: "
                    f"warmup={args.warmup}, repeat={args.repeat}"
                )
    except Exception:
        pass

    inputs = prepare_inputs(get_inputs, device=device, dtype=dtype)

    init_inputs = get_init_inputs() if get_init_inputs is not None else []
    if not isinstance(init_inputs, (tuple, list)):
        init_inputs = [init_inputs]

    # Create baseline model if requested
    baseline_model = None
    if args.baseline:
        baseline_model = (
            Model(*init_inputs).to(device=device, dtype=dtype)
            if init_inputs
            else Model().to(device=device, dtype=dtype)
        )
        baseline_model.eval()
        out = _run_once(lambda: baseline_model(*inputs), "Reference")
        if not args.quiet:
            print(f"Reference output shape: {out.shape}, dtype: {out.dtype}")
            print()

    return Model, inputs, init_inputs, baseline_model


def _prepare_kernel(
    kernel_file: Path,
    Model: type,
    baseline_model: torch.nn.Module | None,
    init_inputs: list,
    inputs: list,
    device: torch.device,
    dtype: torch.dtype,
    quiet: bool = False,
) -> Callable[[], torch.Tensor]:
    """Load the kernel and bind its arguments via the shared SSOT planner.

    Verification (``test.py``) and benchmarking use the *same*
    :func:`bind_kernel_function`, so a kernel that verifies binds identically
    here — tensor parameters are matched positionally (names are irrelevant),
    scalar hyperparameters by name.

    Returns:
        A zero-argument callable that invokes the kernel with bound arguments.
    """
    kernel_function = load_kernel_function(kernel_file)

    extract_model = baseline_model
    if extract_model is None and Model is not None:
        extract_model = (
            Model(*init_inputs).to(device=device, dtype=dtype)
            if init_inputs
            else Model().to(device=device, dtype=dtype)
        )

    return bind_kernel_function(kernel_function, inputs, extract_model)


def _save_results(results: dict[str, Any], path: Path) -> None:
    """Save benchmark results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {path}")


def main():
    args = _parse_args()

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # Auto-detect dtype from kernel source (matches NCU wrapper's dtype inference)
    try:
        kernel_source = args.kernel.read_text()
        if "bfloat16" in kernel_source.lower():
            dtype = torch.bfloat16
        elif "float16" in kernel_source.lower() or "half" in kernel_source.lower():
            dtype = torch.float16
        elif "float32" in kernel_source.lower():
            dtype = torch.float32
    except Exception:
        pass

    if not args.quiet:
        print("=" * 80)
        print("TRITON KERNEL PROFILING")
        print("=" * 80)
        print(f"Problem: {args.problem.name}")
        print(f"Size: {args.size}")
        print(f"Device: {device}, Dtype: {dtype}")
        print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
        print()

    # Load problem and prepare inputs
    try:
        Model, inputs, init_inputs, baseline_model = _load_problem(args, device, dtype)
    except Exception as exc:
        print(f"❌ Failed to load problem: {exc}")
        sys.exit(1)

    results: dict[str, Any] = {
        "problem": str(args.problem),
        "size": args.size,
        "device": str(device),
        "dtype": str(dtype),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "kernels": {},
    }

    # Benchmark baseline (if requested)
    baseline_time = None
    if baseline_model is not None:
        if not args.quiet:
            print("1. PyTorch Reference")
        backend = inspect_pytorch_backend(baseline_model, inputs)
        baseline_time = _benchmark(
            lambda: baseline_model(*inputs), "PyTorch", args.warmup, args.repeat
        )
        results["kernels"]["pytorch_reference"] = {
            "time_ms": baseline_time,
            "speedup": 1.0,
            "backend": backend,
        }
        if not args.quiet:
            print()

    # Load and prepare kernel
    kernel_name = args.kernel.stem
    if not args.quiet:
        idx = 2 if args.baseline else 1
        print(f"{idx}. Candidate: {kernel_name}")

    try:
        kernel_invoke = _prepare_kernel(
            args.kernel,
            Model,
            baseline_model,
            init_inputs,
            inputs,
            device,
            dtype,
            args.quiet,
        )
    except Exception as exc:
        print(f"❌ Failed to load kernel: {exc}")
        results["kernels"][kernel_name] = {"time_ms": float("inf"), "error": str(exc)}
        if args.json:
            _save_results(results, args.json)
        sys.exit(1)

    # Verify kernel executes
    try:
        out = _run_once(kernel_invoke, kernel_name)
        if not args.quiet:
            print(f"✓ {kernel_name} executes successfully")
            print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    except Exception as exc:
        print(f"❌ {kernel_name} failed: {exc}")
        results["kernels"][kernel_name] = {"time_ms": float("inf"), "error": str(exc)}
        if args.json:
            _save_results(results, args.json)
        sys.exit(1)

    # Benchmark kernel
    kernel_time = _benchmark(
        kernel_invoke, kernel_name, args.warmup, args.repeat
    )
    results["kernels"][kernel_name] = {"time_ms": kernel_time, "path": str(args.kernel)}

    # Calculate speedup
    if baseline_time is not None and kernel_time != float("inf"):
        speedup = baseline_time / kernel_time
        results["kernels"][kernel_name]["speedup"] = speedup
        if not args.quiet:
            print(f"Speedup vs PyTorch: {speedup:.2f}x")

    if args.json:
        _save_results(results, args.json)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(130)
    except Exception as exc:
        print(f"❌ Unexpected error: {exc}")
        sys.exit(1)
