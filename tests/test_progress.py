"""Tests for the pluggable progress reporter and run watcher."""

from __future__ import annotations

import json
import threading
import time

import pytest

from scripts.watch_run import render
from utils.progress import (
    JsonFileProgressSink,
    ProgressReporter,
    configure_progress,
    progress_stage,
    set_progress_sink,
    shutdown_progress,
)


class MemorySink:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.lock = threading.Lock()

    def write(self, event: dict) -> None:
        with self.lock:
            self.events.append(dict(event))


@pytest.fixture(autouse=True)
def reset_global_progress():
    shutdown_progress(clear_environment=True)
    yield
    shutdown_progress(clear_environment=True)


def test_json_backend_writes_process_local_status(tmp_path):
    reporter = configure_progress(tmp_path, heartbeat_interval_s=10)
    reporter.emit(
        "test.stage",
        source="test",
        message="working",
        round=2,
    )

    latest_files = list((tmp_path / "run_status" / "sources").glob("*.json"))
    event_files = list((tmp_path / "run_status" / "events").glob("*.jsonl"))
    assert len(latest_files) == 1
    assert len(event_files) == 1

    latest = json.loads(latest_files[0].read_text(encoding="utf-8"))
    assert latest["source"] == "test"
    assert latest["stage"] == "test.stage"
    assert latest["round"] == 2


def test_heartbeat_refreshes_running_stage():
    sink = MemorySink()
    reporter = set_progress_sink(sink, heartbeat_interval_s=0.01)
    reporter.emit("test.blocking", source="test")

    deadline = time.time() + 0.5
    while time.time() < deadline:
        with sink.lock:
            if any(event.get("heartbeat") for event in sink.events):
                break
        time.sleep(0.01)

    with sink.lock:
        heartbeat = next(event for event in sink.events if event.get("heartbeat"))
        initial = sink.events[0]
    assert heartbeat["stage"] == initial["stage"]
    assert heartbeat["stage_started_ts"] == initial["stage_started_ts"]
    assert heartbeat["ts"] > initial["ts"]


def test_progress_stage_marks_unsuccessful_result_failed():
    sink = MemorySink()
    set_progress_sink(sink, heartbeat_interval_s=10)

    @progress_stage("test.operation", source="test", result_ok=bool)
    def operation():
        return False

    assert operation() is False
    assert sink.events[-1]["status"] == "failed"
    assert sink.events[-1]["stage"] == "test.operation"


def test_watcher_merges_manager_and_worker_sources(tmp_path):
    manager = ProgressReporter(
        JsonFileProgressSink(tmp_path, "manager"), heartbeat_interval_s=10
    )
    worker = ProgressReporter(
        JsonFileProgressSink(tmp_path, "worker"), heartbeat_interval_s=10
    )
    worker.bind(worker_id=0, manager_round=3)
    manager.emit(
        "agent2.workers",
        source="agent2.manager",
        message="workers running",
        round=3,
    )
    worker.emit(
        "agent2.worker_generate",
        source="agent2.worker",
        message="generating",
    )

    output = render(tmp_path, stale_after_s=120)
    manager.close()
    worker.close()

    assert "status: running" in output
    assert "agent2.manager" in output
    assert "w0" in output
    assert "agent2.worker_generate" in output
