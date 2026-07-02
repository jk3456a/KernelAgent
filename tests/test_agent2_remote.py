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
"""Tests for agent2 (optimization) remote execution wiring.

These pin the remote NCU-profiling path: the Jinja-rendered wrapper hard-codes
local absolute parent paths in ``sys.path.insert(...)``; the remote path must
rewrite them to ``"."`` and push wrapper + kernel + problem into one workdir,
run ncu there, and pull the CSV back. Transport is mocked (no real SSH).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from kernel_perf_agent.kernel_opt.profiler import ncu_profiler


class TestRemoteNCUProfiling:
    def _make_inputs(self, tmp_path):
        kdir = tmp_path / "art"
        kdir.mkdir()
        kernel = kdir / "kernel_round_0.py"
        kernel.write_text("# kernel\n", encoding="utf-8")
        problem = kdir / "problem.py"
        problem.write_text("# problem\n", encoding="utf-8")
        # Wrapper as the factory would render it: absolute local parent paths.
        wrapper = kdir / "ncu_wrapper.py"
        wrapper.write_text(
            f"import sys\n"
            f"sys.path.insert(0, {repr(str(kernel.parent))})\n"
            f"sys.path.insert(0, {repr(str(problem.parent))})\n"
            f"from kernel_round_0 import kernel_function\n",
            encoding="utf-8",
        )
        return kdir, kernel, problem, wrapper

    def test_rewrites_paths_and_pulls_csv(self, tmp_path):
        kdir, kernel, problem, wrapper = self._make_inputs(tmp_path)

        captured = {}

        def fake_run_cmd(cfg, workdir, command, *, artifacts, timeout_s):
            captured["workdir"] = workdir
            captured["command"] = command
            captured["artifacts"] = artifacts
            # Simulate ncu writing a valid CSV into the workdir.
            (workdir / artifacts[0]).write_text("x" * 200, encoding="utf-8")
            return 0, "ok", ""

        cfg = {"kind": "ssh", "hostname": "h100box", "workspace": "/data/ka"}
        with patch("utils.remote_config.load_remote_config", return_value=cfg), \
             patch("utils.remote_exec.run_command_with_artifacts", side_effect=fake_run_cmd):
            csv_path = ncu_profiler.profile_triton_kernel(
                benchmark_script=wrapper,
                workdir=kdir,
                out_csv="ncu_round_1.csv",
                kernel_file=kernel,
                problem_file=problem,
            )

        # CSV returned and non-trivial
        assert csv_path.exists()
        assert csv_path.stat().st_size >= 100
        # ncu command runs the wrapper in-workdir
        assert "ncu --csv" in captured["command"]
        assert "python3 ncu_wrapper.py" in captured["command"]
        assert captured["artifacts"] == ["ncu_round_1.csv"]
        # the pushed wrapper had its absolute parents rewritten to "."
        pushed_wrapper = (captured["workdir"] / "ncu_wrapper.py").read_text()
        assert repr(str(kernel.parent)) not in pushed_wrapper
        assert "sys.path.insert(0, '.')" in pushed_wrapper

    def test_missing_kernel_problem_raises(self, tmp_path):
        kdir, kernel, problem, wrapper = self._make_inputs(tmp_path)
        cfg = {"kind": "ssh", "hostname": "h100box", "workspace": ""}
        with patch("utils.remote_config.load_remote_config", return_value=cfg):
            try:
                ncu_profiler.profile_triton_kernel(
                    benchmark_script=wrapper, workdir=kdir, kernel_file=None,
                    problem_file=None,
                )
            except RuntimeError as exc:
                assert "kernel_file" in str(exc)
            else:
                raise AssertionError("expected RuntimeError when kernel/problem missing")

    def test_nonzero_rc_raises(self, tmp_path):
        kdir, kernel, problem, wrapper = self._make_inputs(tmp_path)

        def fake_run_cmd(cfg, workdir, command, *, artifacts, timeout_s):
            return 1, "", "ncu boom"

        cfg = {"kind": "ssh", "hostname": "box", "workspace": ""}
        with patch("utils.remote_config.load_remote_config", return_value=cfg), \
             patch("utils.remote_exec.run_command_with_artifacts", side_effect=fake_run_cmd):
            try:
                ncu_profiler.profile_triton_kernel(
                    benchmark_script=wrapper, workdir=kdir,
                    kernel_file=kernel, problem_file=problem,
                )
            except RuntimeError as exc:
                assert "NCU" in str(exc)
            else:
                raise AssertionError("expected RuntimeError on nonzero ncu rc")


class TestBenchmarkRemoteNoDeadlock:
    """Regression: benchmark_kernel acquires the GPU lock, then delegates to the
    remote helper. The helper must NOT re-acquire the same (non-reentrant) lock,
    or the optimization loop deadlocks. We drive it with a real threading.Lock
    so a double-acquire would hang (caught by the test's own timeout)."""

    def test_remote_kernel_benchmark_does_not_redacquire_lock(self, tmp_path):
        import logging
        import threading
        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            Benchmark,
        )

        kernel = tmp_path / "kernel_round_0.py"
        kernel.write_text("# kernel\n", encoding="utf-8")
        problem = tmp_path / "problem.py"
        problem.write_text("# problem\n", encoding="utf-8")

        bench = Benchmark(
            logger=logging.getLogger("t"),
            artifacts_dir=tmp_path,
            benchmark_lock=threading.Lock(),  # non-reentrant: re-acquire would hang
            worker_id=-1,
            warmup=1,
            repeat=1,
        )

        cfg = {"kind": "ssh", "hostname": "box", "workspace": ""}

        def fake_run_cmd(c, workdir, command, *, artifacts, timeout_s):
            (workdir / artifacts[0]).write_text(
                '{"kernels": {"kernel_round_0": {"time_ms": 2.0, "speedup": 1.0}}}',
                encoding="utf-8",
            )
            return 0, "ok", ""

        with patch("utils.remote_config.load_remote_config", return_value=cfg), \
             patch("utils.remote_exec.run_command_with_artifacts", side_effect=fake_run_cmd):
            result = bench.benchmark_kernel(kernel, problem)

        assert result["time_ms"] == 2.0
