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
"""Tests for resume-from-prior-run state loading.

Resuming a run must: pick the prior run's best (finite-time) kernel to seed the
strategy, re-expose all prior programs to the database, and continue the round
numbering past the prior run's last round — not restart at round 1.
"""

from __future__ import annotations

import json

from triton_kernel_agent.opt_manager import OptimizationManager


def _write_db(path, programs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"programs": programs}), encoding="utf-8")


def _prog(pid, time_ms, gen, code="# k"):
    return {
        "program_id": pid,
        "kernel_code": code,
        "metrics": {"time_ms": time_ms},
        "problem_id": "p",
        "generation": gen,
    }


def _manager_stub():
    # Build without running __init__ (which needs providers/config); we only
    # exercise the pure _load_resume_state method.
    import logging

    m = OptimizationManager.__new__(OptimizationManager)
    m.logger = logging.getLogger("t")
    return m


def test_resume_picks_best_and_continues_round(tmp_path):
    db = tmp_path / "greedy" / "program_db.json"
    _write_db(db, [
        _prog("initial", float("inf"), 0),
        _prog("r1_w0", 2.0, 1, code="# fast"),
        _prog("r2_w0", 2.5, 2, code="# slower"),
    ])
    info = _manager_stub()._load_resume_state(tmp_path / "greedy")
    assert info is not None
    assert info["best_ms"] == 2.0
    assert info["kernel_code"] == "# fast"
    assert info["next_round"] == 3  # past r2
    assert len(info["programs"]) == 3


def test_resume_accepts_run_root(tmp_path):
    # Passing the run root (parent of the strategy dir) should still find the db.
    db = tmp_path / "greedy_glm" / "program_db.json"
    _write_db(db, [_prog("r1_w0", 1.5, 1)])
    info = _manager_stub()._load_resume_state(tmp_path)
    assert info is not None
    assert info["next_round"] == 2


def test_resume_no_finite_programs_returns_none(tmp_path):
    db = tmp_path / "greedy" / "program_db.json"
    _write_db(db, [_prog("initial", float("inf"), 0)])
    assert _manager_stub()._load_resume_state(tmp_path / "greedy") is None


def test_resume_missing_db_returns_none(tmp_path):
    assert _manager_stub()._load_resume_state(tmp_path / "nope") is None
