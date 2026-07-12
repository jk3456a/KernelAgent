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
"""Tests for the optimization dashboard data layer (discover_runs / load_run)."""

from __future__ import annotations

import pytest

from scripts.optimization_dashboard import discover_runs, load_run
from triton_kernel_agent.opt_worker_component.searching.trajectory import TrajectoryWriter


def _make_run(run_dir, rounds):
    run_dir.mkdir(parents=True, exist_ok=True)
    w = TrajectoryWriter(run_dir / "trajectory.jsonl")
    w.record_baseline(
        time_ms=16.0,
        pytorch_ms=1.4,
        sol_pct=2.0,
        bottleneck="memory",
        kernel_performance={"achieved_tflops": 10.0, "mfu_pct": 20.0},
        pytorch_performance={"achieved_tflops": 12.0, "mfu_pct": 24.0},
    )
    for n, t in rounds:
        w.record_round(
            round_num=n, time_ms=t, baseline_ms=16.0,
            improvement_pct=0.0, compute_sol_pct=0.0, memory_sol_pct=0.0,
            combined_sol_pct=0.0, bottleneck="memory", config_changes={},
            is_improvement=t < 16.0, is_best=False, verified=t == t,
            performance={
                "achieved_tflops": 160.0 / t,
                "mfu_pct": 32.0,
                "roofline_utilization_pct": 40.0,
            },
        )
        # Per-round detail lives in the worker artifact dir, as the real run does.
        art = run_dir / "workers" / "w0" / f"r{n}" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "kernel_round_1.py").write_text(f"# kernel r{n}\n", encoding="utf-8")
        (art / f"round{n:03d}_opt_prompt.txt").write_text(f"prompt r{n}\n", encoding="utf-8")
        (art / f"round{n:03d}_opt_reply.txt").write_text(f"reply r{n}\n", encoding="utf-8")
        (art / f"round{n:03d}_strategy.json").write_text(
            '[{"category":"underutilized","summary":"low tensor",'
            '"root_causes":[{"fixes":[{"fix":"try TMA"}]}]}]',
            encoding="utf-8",
        )


def test_discover_summarizes_best(tmp_path):
    _make_run(tmp_path / "runA" / "artifacts", [(1, 12.0), (2, 8.0), (3, 9.0)])
    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    run = runs[0]
    assert run["rounds"] == 3
    assert run["baseline_ms"] == 16.0
    assert run["best_ms"] == 8.0
    assert run["best_round"] == 2
    assert abs(run["best_speedup"] - 16.0 / 8.0) < 1e-6


def test_discover_multiple_runs_sorted_by_mtime(tmp_path):
    _make_run(tmp_path / "r1" / "art", [(1, 10.0)])
    _make_run(tmp_path / "r2" / "art", [(1, 9.0)])
    runs = discover_runs(tmp_path)
    assert len(runs) == 2
    # ids are derived from the path relative to root
    ids = {r["id"] for r in runs}
    assert any("r1" in i for i in ids) and any("r2" in i for i in ids)


def test_load_run_attaches_artifacts(tmp_path):
    _make_run(tmp_path / "runA" / "artifacts", [(1, 12.0)])
    runs = discover_runs(tmp_path)
    data = load_run(tmp_path, runs[0]["id"])
    round_rows = [r for r in data["rows"] if r["kind"] == "round"]
    r = round_rows[0]
    assert r["kernel_code"].startswith("# kernel r1")
    assert "prompt r1" in r["opt_prompt"]
    assert "reply r1" in r["opt_reply"]
    # the diagnosis / prescription is parsed from strategy.json
    assert r["strategy"][0]["category"] == "underutilized"
    assert r["strategy"][0]["root_causes"][0]["fixes"][0]["fix"] == "try TMA"
    assert r["used_tma"] is False  # kernel stub has no TMA calls
    assert r["performance"]["mfu_pct"] == 32.0
    baseline = next(row for row in data["rows"] if row["kind"] == "baseline")
    assert baseline["kernel_performance"]["mfu_pct"] == 20.0
    assert baseline["pytorch_performance"]["mfu_pct"] == 24.0


def test_load_run_rejects_escape(tmp_path):
    _make_run(tmp_path / "runA" / "artifacts", [(1, 12.0)])
    with pytest.raises(ValueError):
        load_run(tmp_path, "..~..~etc")


def test_no_runs_when_empty(tmp_path):
    assert discover_runs(tmp_path) == []
