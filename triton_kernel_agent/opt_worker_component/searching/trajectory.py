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
"""Optimization trajectory: an append-only JSONL log of what each round did.

This is the single source the optimization dashboard reads. The orchestrator
appends one record per baseline + round; the dashboard renders the performance
curve and per-round detail from it. Keeping it a flat JSONL (not the program DB)
means it is cheap to tail, survives crashes, and needs no schema migration.

``inf`` times (a failed/un-timed round) are stored as JSON ``null`` and read
back as ``None`` so consumers can skip those points rather than choke on a
non-JSON ``Infinity`` token.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["TrajectoryWriter", "read_trajectory"]


def _clean_ms(value: float | None) -> float | None:
    """Map inf/NaN/None to None; keep finite floats as-is."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


class TrajectoryWriter:
    """Append round records to ``<artifact_dir>/trajectory.jsonl``."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def record_baseline(
        self,
        *,
        time_ms: float,
        pytorch_ms: float | None,
        sol_pct: float,
        bottleneck: str,
        kernel_performance: dict[str, Any] | None = None,
        pytorch_performance: dict[str, Any] | None = None,
    ) -> None:
        self._append(
            {
                "kind": "baseline",
                "round": 0,
                "time_ms": _clean_ms(time_ms),
                "pytorch_ms": _clean_ms(pytorch_ms),
                "combined_sol_pct": sol_pct,
                "bottleneck": bottleneck,
                "kernel_performance": kernel_performance,
                "pytorch_performance": pytorch_performance,
            }
        )

    def record_resume(
        self, *, resumed_from: str, from_round: int, best_ms: float
    ) -> None:
        """Mark that this run continues a prior run (for dashboard display)."""
        self._append(
            {
                "kind": "resume",
                "round": from_round,
                "resumed_from": resumed_from,
                "best_ms": _clean_ms(best_ms),
            }
        )

    def record_round(
        self,
        *,
        round_num: int,
        time_ms: float,
        baseline_ms: float,
        improvement_pct: float,
        compute_sol_pct: float,
        memory_sol_pct: float,
        combined_sol_pct: float,
        bottleneck: str,
        config_changes: dict[str, str],
        is_improvement: bool,
        is_best: bool,
        verified: bool,
        kernel_file: str | None = None,
        performance: dict[str, Any] | None = None,
    ) -> None:
        clean_time = _clean_ms(time_ms)
        clean_base = _clean_ms(baseline_ms)
        speedup = (
            clean_base / clean_time
            if clean_time and clean_base and clean_time > 0
            else None
        )
        self._append(
            {
                "kind": "round",
                "round": round_num,
                "time_ms": clean_time,
                "baseline_ms": clean_base,
                "speedup_vs_baseline": speedup,
                "improvement_pct": improvement_pct,
                "compute_sol_pct": compute_sol_pct,
                "memory_sol_pct": memory_sol_pct,
                "combined_sol_pct": combined_sol_pct,
                "bottleneck": bottleneck,
                "config_changes": config_changes,
                "is_improvement": bool(is_improvement),
                "is_best": bool(is_best),
                "verified": bool(verified),
                "kernel_file": kernel_file,
                "performance": performance,
            }
        )


def read_trajectory(path: Path | str) -> list[dict[str, Any]]:
    """Read all records from a trajectory file; missing file → empty list.

    Corrupt trailing lines (e.g. a crash mid-write) are skipped rather than
    raising, so the dashboard can render a partial in-progress run.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
