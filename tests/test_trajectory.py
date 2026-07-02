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
"""Tests for the optimization trajectory writer.

The trajectory is the single, append-only source the dashboard reads to show
what each optimization round did and how it performed. These tests pin the
record schema and the append/read contract.
"""

from __future__ import annotations

import json

from triton_kernel_agent.opt_worker_component.searching.trajectory import (
    TrajectoryWriter,
    read_trajectory,
)


def test_writes_jsonl_records_in_order(tmp_path):
    writer = TrajectoryWriter(tmp_path / "trajectory.jsonl")
    writer.record_baseline(time_ms=15.6, pytorch_ms=1.43, sol_pct=2.0, bottleneck="memory")
    writer.record_round(
        round_num=1,
        time_ms=8.0,
        baseline_ms=15.6,
        improvement_pct=48.7,
        compute_sol_pct=10.0,
        memory_sol_pct=40.0,
        combined_sol_pct=40.0,
        bottleneck="memory",
        config_changes={"BLOCK_SIZE": "256→512"},
        is_improvement=True,
        is_best=True,
        verified=True,
    )

    rows = read_trajectory(tmp_path / "trajectory.jsonl")
    assert len(rows) == 2
    assert rows[0]["kind"] == "baseline"
    assert rows[0]["time_ms"] == 15.6
    assert rows[0]["pytorch_ms"] == 1.43
    assert rows[1]["kind"] == "round"
    assert rows[1]["round"] == 1
    # speedup vs baseline is derived for convenience
    assert abs(rows[1]["speedup_vs_baseline"] - (15.6 / 8.0)) < 1e-6
    assert rows[1]["is_best"] is True
    assert rows[1]["config_changes"] == {"BLOCK_SIZE": "256→512"}


def test_each_record_has_timestamp_and_is_valid_json(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    writer = TrajectoryWriter(path)
    writer.record_round(
        round_num=1, time_ms=10.0, baseline_ms=10.0, improvement_pct=0.0,
        compute_sol_pct=0.0, memory_sol_pct=0.0, combined_sol_pct=0.0,
        bottleneck="unknown", config_changes={}, is_improvement=False,
        is_best=False, verified=False,
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])  # must be valid JSON
    assert "ts" in rec and isinstance(rec["ts"], (int, float))


def test_failed_round_records_inf_time(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    writer = TrajectoryWriter(path)
    writer.record_round(
        round_num=2, time_ms=float("inf"), baseline_ms=10.0, improvement_pct=0.0,
        compute_sol_pct=0.0, memory_sol_pct=0.0, combined_sol_pct=0.0,
        bottleneck="memory", config_changes={}, is_improvement=False,
        is_best=False, verified=False,
    )
    rows = read_trajectory(path)
    # inf is not JSON-native; the writer must serialize it as null and the
    # reader surface it back as None so the dashboard can skip the point.
    assert rows[0]["time_ms"] is None
    assert rows[0]["speedup_vs_baseline"] is None


def test_read_missing_file_returns_empty(tmp_path):
    assert read_trajectory(tmp_path / "nope.jsonl") == []
