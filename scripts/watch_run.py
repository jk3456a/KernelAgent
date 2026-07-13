#!/usr/bin/env python3
"""Watch progress events emitted by a KernelAgent run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _age(ts: float | None) -> str:
    if not ts:
        return "unknown"
    seconds = max(0, int(time.time() - ts))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s ago"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m ago"


def _fmt_elapsed(start_ts: float | None) -> str:
    if not start_ts:
        return "unknown"
    seconds = max(0, int(time.time() - start_ts))
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _load_sources(status_dir: Path) -> list[dict[str, Any]]:
    sources = []
    for path in sorted((status_dir / "sources").glob("*.json")):
        data = _read_json(path)
        if data:
            data["_path"] = str(path)
            sources.append(data)
    if sources:
        return sources

    # Backward-compatible reader for runs created by the first monitor version.
    latest = _read_json(status_dir / "status.json")
    if latest:
        latest.setdefault("source", "legacy")
        sources.append(latest)
    for path in sorted((status_dir / "workers").glob("w*/status.json")):
        data = _read_json(path)
        if data:
            data.setdefault("source", "legacy.worker")
            data.setdefault("worker_id", path.parent.name[1:])
            if data.get("stage") in {"worker_done", "worker_process_done"}:
                data["status"] = "completed"
            sources.append(data)
    return sources


def _find_start_ts(
    status_dir: Path, sources: list[dict[str, Any]]
) -> float | None:
    timestamps = []
    event_paths = list((status_dir / "events").glob("*.jsonl"))
    legacy_events = status_dir / "events.jsonl"
    if legacy_events.exists():
        event_paths.append(legacy_events)
    for path in event_paths:
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            timestamp = json.loads(first).get("ts")
            if timestamp:
                timestamps.append(float(timestamp))
        except (OSError, IndexError, json.JSONDecodeError, TypeError, ValueError):
            continue
    if timestamps:
        return min(timestamps)
    source_timestamps = [float(row["ts"]) for row in sources if row.get("ts")]
    return min(source_timestamps) if source_timestamps else None


def _run_state(sources: list[dict[str, Any]], stale_after_s: int) -> str:
    now = time.time()
    fresh_running = [
        row
        for row in sources
        if row.get("status") == "running"
        and now - float(row.get("ts", 0) or 0) <= stale_after_s
    ]
    if fresh_running:
        return "running"

    # Manager/agent terminal events are authoritative over stale child workers.
    for source in ("agent2.manager", "agent1"):
        rows = [row for row in sources if row.get("source") == source]
        if rows:
            latest = max(rows, key=lambda row: float(row.get("ts", 0) or 0))
            if latest.get("status") in {"completed", "failed"}:
                return str(latest["status"])
    legacy = [row for row in sources if row.get("source") == "legacy"]
    if legacy and legacy[0].get("status") in {"completed", "failed"}:
        return str(legacy[0]["status"])

    if any(row.get("status") == "running" for row in sources):
        return "possibly_stuck"
    if any(row.get("status") == "failed" for row in sources):
        return "failed"
    if any(row.get("status") == "completed" for row in sources):
        return "completed"
    return "unknown"


def _source_row(data: dict[str, Any], stale_after_s: int) -> str:
    age_s = time.time() - float(data.get("ts", 0) or 0)
    stale = " STALE" if data.get("status") == "running" and age_s > stale_after_s else ""
    source = str(data.get("source", "unknown"))
    worker_id = data.get("worker_id")
    label = f"w{worker_id}" if worker_id is not None else source
    manager_round = data.get("manager_round", data.get("round"))
    round_part = f" r{manager_round}" if manager_round is not None else ""
    pid = data.get("pid")
    pid_part = f" pid={pid}" if pid else ""
    occurrence = data.get("occurrence")
    attempt_part = f" attempt={occurrence}" if occurrence and occurrence > 1 else ""
    return (
        f"  {label}{round_part:<5} {data.get('status', 'unknown'):<9} "
        f"{data.get('stage', 'unknown'):<30} last={_age(data.get('ts'))}{stale}"
        f"{pid_part}{attempt_part}  {data.get('message', '')}"
    )


def render(run_dir: Path, stale_after_s: int) -> str:
    status_dir = run_dir / "run_status"
    sources = _load_sources(status_dir)
    if not sources:
        return f"No run status found under {status_dir}"

    start_ts = _find_start_ts(status_dir, sources)
    latest = max(sources, key=lambda row: float(row.get("ts", 0) or 0))
    state = _run_state(sources, stale_after_s)
    workers = [row for row in sources if row.get("worker_id") is not None]
    processes = [row for row in sources if row.get("worker_id") is None]

    lines = [
        "KernelAgent Run Monitor",
        f"run: {run_dir.name}",
        f"status: {state}",
        f"elapsed: {_fmt_elapsed(start_ts)}",
        f"last heartbeat: {_age(latest.get('ts'))}",
        "",
        "processes:",
    ]
    lines.extend(_source_row(row, stale_after_s) for row in processes)

    if workers:
        lines.extend(
            [
                "",
                "workers:",
                *(_source_row(row, stale_after_s) for row in workers),
            ]
        )

    source_dir = status_dir / "sources"
    status_path = source_dir if source_dir.exists() else status_dir / "status.json"
    lines.extend(["", "files:", f"  status: {status_path}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a KernelAgent run")
    parser.add_argument("run_dir", type=Path, help="Run directory printed as RUN_DIR")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--stale-after", type=int, default=120)
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    while True:
        if not args.once:
            os.system("clear")
        print(render(args.run_dir, args.stale_after), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
