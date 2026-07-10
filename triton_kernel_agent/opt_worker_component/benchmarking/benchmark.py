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

"""Unified benchmarking for Triton kernels and PyTorch baselines.

This module consolidates kernel and PyTorch benchmarking with improved timing
utilities, L2 cache clearing, and comprehensive statistics.
"""

import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from utils import remote_config, remote_exec

# ``torch`` and the in-process timing helpers are imported lazily inside the
# methods that need them. When a remote target is configured (kind="ssh") the
# GPU lives on the remote host and this process never touches torch, so a
# torch-free control machine can still drive the optimization loop.
if TYPE_CHECKING:
    import torch


class BenchmarkLockManager:
    """Manages GPU benchmarking locks to prevent resource contention."""

    def __init__(self, lock: Any, worker_id: int, logger: logging.Logger):
        """Initialize the lock manager.

        Args:
            lock: Shared multiprocessing lock for serializing GPU access
            worker_id: Worker ID for logging
            logger: Logger instance
        """
        self.lock = lock
        self.worker_id = worker_id
        self.logger = logger

    def __enter__(self):
        """Acquire the benchmarking lock."""
        self.logger.info(f"⏳ Waiting for benchmark lock (worker {self.worker_id})...")
        self.lock.acquire()
        self.logger.info(f"🔓 Acquired benchmark lock (worker {self.worker_id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the benchmarking lock."""
        try:
            self.lock.release()
            self.logger.info(f"🔒 Released benchmark lock (worker {self.worker_id})")
        except Exception as e:
            self.logger.warning(f"Failed to release benchmark lock: {e}")
        return False


class Benchmark:
    """Unified benchmark for Triton kernels and PyTorch baselines.

    Supports two modes:
    1. Subprocess mode: Runs benchmarks in isolated processes (for compatibility)
    2. Direct mode: Uses in-process timing utilities (faster, more flexible)
    """

    def __init__(
        self,
        logger: logging.Logger,
        artifacts_dir: Path,
        benchmark_lock: Any,
        worker_id: int = 0,
        warmup: int = 25,
        repeat: int = 100,
        timing_method: str = "cuda_event",
    ):
        """Initialize the benchmark.

        Args:
            logger: Logger instance
            artifacts_dir: Directory for benchmark artifacts
            benchmark_lock: Shared lock to serialize GPU benchmarking
            worker_id: Worker ID
            warmup: Number of warmup iterations (or warmup time in ms for do_bench)
            repeat: Number of repeat iterations (or rep time in ms for do_bench)
            timing_method: Timing method ("cuda_event", "do_bench", "host_time")
        """
        self.logger = logger
        self.artifacts_dir = artifacts_dir
        self.lock_manager = BenchmarkLockManager(benchmark_lock, worker_id, logger)
        self.warmup = warmup
        self.repeat = repeat
        self.timing_method = timing_method

    def benchmark_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        baseline_file: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Benchmark Triton kernel performance using subprocess isolation.

        Uses subprocess for crash protection of potentially buggy kernels.

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            baseline_file: Path to baseline kernel (optional)

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - speedup: Speedup vs baseline
        """
        try:
            with self.lock_manager:
                results_json = self.artifacts_dir / "benchmark_results.json"
                benchmark_script = Path(__file__).parent / "kernel_subprocess.py"

                remote_cfg = remote_config.load_remote_config()
                if remote_config.is_remote_enabled(remote_cfg):
                    return self._benchmark_kernel_remote(
                        remote_cfg, kernel_file, problem_file, baseline_file
                    )

                # Use KERNEL_PROFILER_PYTHON (the PAR bootstrap) when set, like
                # ncu_profiler.py; bare sys.executable is un-bootstrapped in a PAR.
                bench_python = (
                    os.environ.get("KERNEL_PROFILER_PYTHON") or sys.executable
                )

                cmd = [
                    bench_python,
                    str(benchmark_script),
                    "--problem",
                    str(problem_file),
                    "--kernel",
                    str(kernel_file),
                    "--warmup",
                    str(self.warmup),
                    "--repeat",
                    str(self.repeat),
                    "--json",
                    str(results_json),
                    "--quiet",
                ]

                if baseline_file:
                    cmd.extend(["--baseline"])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode != 0:
                    error_msg = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or "Unknown error"
                    )
                    self.logger.error(f"Kernel benchmark failed: {error_msg}")
                    return {"time_ms": float("inf"), "speedup": 0.0}

                with open(results_json, "r") as f:
                    results = json.load(f)

                kernel_name = kernel_file.stem
                kernel_results = results.get("kernels", {}).get(kernel_name, {})

                return {
                    "time_ms": kernel_results.get("time_ms", float("inf")),
                    "speedup": kernel_results.get("speedup", 1.0),
                }

        except Exception as e:
            self.logger.error(f"Kernel benchmark failed: {e}")
            return {"time_ms": float("inf"), "speedup": 0.0}

    def _benchmark_kernel_remote(
        self,
        remote_cfg: dict[str, str],
        kernel_file: Path,
        problem_file: Path,
        baseline_file: Optional[Path],
    ) -> dict[str, Any]:
        """Benchmark a kernel on the remote GPU host.

        Pushes ``kernel_subprocess.py`` + the kernel + the problem into a remote
        workdir, runs the benchmark there, and pulls ``benchmark_results.json``
        back so the local parsing path is unchanged.

        The caller (``benchmark_kernel``) already holds ``self.lock_manager``; the
        lock serializes *local* GPU access and must NOT be re-acquired here (it is
        non-reentrant — doing so deadlocks). Remote GPU contention is handled on
        the remote host, so running unlocked here is correct.
        """
        workdir = self.artifacts_dir / f"remote_bench_{kernel_file.stem}"
        workdir.mkdir(parents=True, exist_ok=True)
        bench_script = Path(__file__).parent / "kernel_subprocess.py"
        timing_mod = Path(__file__).parent / "timing.py"
        backend_probe_mod = Path(__file__).parent / "backend_probe.py"
        binding_mod = Path(__file__).parent / "kernel_binding.py"
        for src in (
            bench_script,
            timing_mod,
            backend_probe_mod,
            binding_mod,
            kernel_file,
            problem_file,
        ):
            (workdir / src.name).write_bytes(src.read_bytes())

        results_name = "benchmark_results.json"
        command = (
            f"python3 -u {bench_script.name} "
            f"--problem {problem_file.name} --kernel {kernel_file.name} "
            f"--warmup {self.warmup} --repeat {self.repeat} "
            f"--json {results_name} --quiet"
        )
        if baseline_file:
            command += " --baseline"

        rc, out, err = remote_exec.run_command_with_artifacts(
            remote_cfg, workdir, command, artifacts=[results_name], timeout_s=300
        )
        results_json = workdir / results_name
        if rc != 0 or not results_json.exists():
            self.logger.error(
                f"Remote kernel benchmark failed (rc={rc}): {(err or out).strip()[:500]}"
            )
            return {"time_ms": float("inf"), "speedup": 0.0}

        with open(results_json, "r") as f:
            results = json.load(f)
        kernel_results = results.get("kernels", {}).get(kernel_file.stem, {})
        return {
            "time_ms": kernel_results.get("time_ms", float("inf")),
            "speedup": kernel_results.get("speedup", 1.0),
        }

    def benchmark_pytorch(
        self,
        problem_file: Path,
        dtype: Optional["torch.dtype"] = None,
        kernel_file: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Benchmark PyTorch baseline using direct in-process timing.

        Always uses direct mode (PyTorch is stable, doesn't need subprocess isolation).

        Args:
            problem_file: Path to problem file (must define Model class and get_inputs())
            dtype: Data type to use (default: auto-detect based on model parameters)
            kernel_file: Optional kernel file used only by the remote path to drive
                ``kernel_subprocess.py --baseline`` (the GPU lives on the remote host).

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - stats: Full timing statistics (mean, std, min, max, all_times, etc.)
        """
        remote_cfg = remote_config.load_remote_config()
        if remote_config.is_remote_enabled(remote_cfg):
            return self._benchmark_pytorch_remote(remote_cfg, problem_file, kernel_file)

        from triton_kernel_agent.opt_worker_component.benchmarking.backend_probe import (
            inspect_pytorch_backend,
        )
        from triton_kernel_agent.opt_worker_component.benchmarking.timing import (
            compute_timing_stats,
            prepare_pytorch_model,
            time_with_cuda_events,
            time_with_triton_do_bench,
        )

        try:
            with self.lock_manager:
                model, inputs = prepare_pytorch_model(
                    problem_file=problem_file,
                    device="cuda",
                    dtype=dtype,
                )
                backend = inspect_pytorch_backend(model, inputs)

                if self.timing_method == "do_bench":
                    times = time_with_triton_do_bench(
                        lambda: model(*inputs),
                        [],
                        warmup=self.warmup,
                        rep=self.repeat,
                        verbose=False,
                    )
                else:  # cuda_event
                    times = time_with_cuda_events(
                        lambda: model(*inputs),
                        [],
                        num_warmup=self.warmup,
                        num_trials=self.repeat,
                        clear_cache=True,
                        verbose=False,
                    )

                stats = compute_timing_stats(times)

                return {
                    "time_ms": stats["mean"],
                    "stats": stats,
                    "backend": backend,
                }

        except Exception as e:
            self.logger.error(f"PyTorch baseline benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"time_ms": float("inf")}

    def _benchmark_pytorch_remote(
        self,
        remote_cfg: dict[str, str],
        problem_file: Path,
        kernel_file: Optional[Path],
    ) -> dict[str, Any]:
        """PyTorch baseline timing on the remote GPU host.

        ``kernel_subprocess.py --baseline`` times the PyTorch reference alongside
        the kernel and reports it under ``kernels.pytorch_reference``. We need a
        kernel to drive it; the orchestrator's round-0 kernel is the natural
        choice and is passed through ``kernel_file``.
        """
        if kernel_file is None or not kernel_file.exists():
            self.logger.error(
                "Remote PyTorch baseline requires a kernel_file to drive "
                "kernel_subprocess.py --baseline; none was provided."
            )
            return {"time_ms": float("inf")}

        with self.lock_manager:
            workdir = self.artifacts_dir / "remote_bench_pytorch"
            workdir.mkdir(parents=True, exist_ok=True)
            bench_script = Path(__file__).parent / "kernel_subprocess.py"
            timing_mod = Path(__file__).parent / "timing.py"
            backend_probe_mod = Path(__file__).parent / "backend_probe.py"
            binding_mod = Path(__file__).parent / "kernel_binding.py"
            for src in (
                bench_script,
                timing_mod,
                backend_probe_mod,
                binding_mod,
                kernel_file,
                problem_file,
            ):
                (workdir / src.name).write_bytes(src.read_bytes())

            results_name = "pytorch_baseline.json"
            command = (
                f"python3 -u {bench_script.name} "
                f"--problem {problem_file.name} --kernel {kernel_file.name} "
                f"--warmup {self.warmup} --repeat {self.repeat} "
                f"--json {results_name} --baseline --quiet"
            )
            rc, out, err = remote_exec.run_command_with_artifacts(
                remote_cfg, workdir, command, artifacts=[results_name], timeout_s=300
            )
            results_json = workdir / results_name
            if rc != 0 or not results_json.exists():
                self.logger.error(
                    f"Remote PyTorch baseline failed (rc={rc}): {(err or out).strip()[:500]}"
                )
                return {"time_ms": float("inf")}
            with open(results_json, "r") as f:
                results = json.load(f)
            ref = results.get("kernels", {}).get("pytorch_reference", {})
            return {
                "time_ms": ref.get("time_ms", float("inf")),
                "backend": ref.get("backend"),
            }

    def benchmark_pytorch_compile(
        self,
        problem_file: Path,
        dtype: Optional["torch.dtype"] = None,
    ) -> dict[str, Any]:
        """Benchmark torch.compile'd PyTorch baseline using direct in-process timing.

        Mirrors benchmark_pytorch() but wraps the model with torch.compile()
        and uses extended warmup (3 forward calls) before timing to allow
        compilation and warm caches.

        Args:
            problem_file: Path to problem file (must define Model class and get_inputs())
            dtype: Data type to use (default: auto-detect based on model parameters)

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - stats: Full timing statistics (mean, std, min, max, all_times, etc.)
        """
        remote_cfg = remote_config.load_remote_config()
        if remote_config.is_remote_enabled(remote_cfg):
            # torch.compile baseline is an informational reference line, not part
            # of the optimization loop, and kernel_subprocess.py has no compile
            # path. Skip it gracefully on remote rather than ship a new script.
            self.logger.info(
                "Skipping torch.compile baseline under remote execution "
                "(informational reference only)."
            )
            return {"time_ms": float("inf")}

        from triton_kernel_agent.opt_worker_component.benchmarking.timing import (
            compute_timing_stats,
            prepare_pytorch_model,
            time_with_cuda_events,
            time_with_triton_do_bench,
        )

        import torch

        try:
            with self.lock_manager:
                model, inputs = prepare_pytorch_model(
                    problem_file=problem_file,
                    device="cuda",
                    dtype=dtype,
                )

                model = torch.compile(model)

                # Extended warmup: 3 forward calls to trigger compilation
                for _ in range(3):
                    model(*inputs)
                torch.cuda.synchronize()

                if self.timing_method == "do_bench":
                    times = time_with_triton_do_bench(
                        lambda: model(*inputs),
                        [],
                        warmup=self.warmup,
                        rep=self.repeat,
                        verbose=False,
                    )
                else:  # cuda_event
                    times = time_with_cuda_events(
                        lambda: model(*inputs),
                        [],
                        num_warmup=self.warmup,
                        num_trials=self.repeat,
                        clear_cache=True,
                        verbose=False,
                    )

                stats = compute_timing_stats(times)

                return {
                    "time_ms": stats["mean"],
                    "stats": stats,
                }

        except Exception as e:
            self.logger.error(f"PyTorch compile benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"time_ms": float("inf")}
