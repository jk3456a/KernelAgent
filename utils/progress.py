"""Pluggable progress reporting with process-local automatic heartbeats.

Business code emits semantic progress events through :func:`get_progress`.
The default backend is a no-op, so progress reporting stays optional.  The
built-in JSON backend stores one latest/event file per process; this avoids
multiple managers and workers overwriting one shared status file.
"""

from __future__ import annotations

import atexit
import functools
import json
import multiprocessing
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Protocol, TypeVar

ENV_BACKEND = "KERNEL_PROGRESS_BACKEND"
ENV_ROOT = "KERNEL_PROGRESS_ROOT"
ENV_HEARTBEAT_INTERVAL = "KERNEL_PROGRESS_HEARTBEAT_S"

_DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
_F = TypeVar("_F", bound=Callable[..., Any])
_BackendFactory = Callable[[Path, str], "ProgressSink"]


class ProgressSink(Protocol):
    """Storage/transport plugin for progress events."""

    def write(self, event: dict[str, Any]) -> None:
        """Persist or publish one progress event."""


class NullProgressSink:
    """Default sink used when progress reporting is disabled."""

    def write(self, event: dict[str, Any]) -> None:
        del event


class JsonFileProgressSink:
    """Write process-isolated latest status and append-only JSONL events."""

    def __init__(self, root: Path | str, source_id: str) -> None:
        self.status_dir = Path(root) / "run_status"
        self.latest_path = self.status_dir / "sources" / f"{source_id}.json"
        self.events_path = self.status_dir / "events" / f"{source_id}.jsonl"
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, sort_keys=True, default=str)
        tmp_path = self.latest_path.with_name(
            f"{self.latest_path.name}.{threading.get_ident()}.tmp"
        )
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.latest_path)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


class ProgressReporter:
    """Emit events and heartbeat the most recent running stage."""

    def __init__(
        self,
        sink: ProgressSink | None = None,
        *,
        heartbeat_interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._sink = sink or NullProgressSink()
        self._enabled = not isinstance(self._sink, NullProgressSink)
        self._heartbeat_interval_s = max(0.01, heartbeat_interval_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: dict[str, Any] | None = None
        self._occurrences: dict[tuple[str, str], int] = {}
        self._stage_started: dict[tuple[str, str], float] = {}
        self._context: dict[str, Any] = {}
        self._pid = os.getpid()
        self._process_name = multiprocessing.current_process().name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind(self, **fields: Any) -> None:
        """Attach fields to all later events from this process."""
        with self._lock:
            self._context.update(fields)

    def emit(
        self,
        stage: str,
        *,
        source: str,
        status: str = "running",
        message: str | None = None,
        **fields: Any,
    ) -> None:
        """Publish a semantic stage transition.

        A ``running`` event becomes the process's heartbeat target until another
        event is emitted. Terminal events stop heartbeating that stage.
        """
        if not self._enabled:
            return

        now = time.time()
        key = (source, stage)
        with self._lock:
            if status == "running":
                self._occurrences[key] = self._occurrences.get(key, 0) + 1
                self._stage_started[key] = now
            occurrence = self._occurrences.get(key, 0)
            stage_started_ts = self._stage_started.get(key, now)
            event = {
                **self._context,
                **fields,
                "source": source,
                "stage": stage,
                "status": status,
                "message": message,
                "pid": self._pid,
                "process_name": self._process_name,
                "ts": now,
                "stage_started_ts": stage_started_ts,
                "occurrence": occurrence,
                "heartbeat": False,
            }
            event = {key: value for key, value in event.items() if value is not None}
            self._safe_write(event)
            self._current = dict(event) if status == "running" else None
            if status != "running":
                self._stage_started.pop(key, None)
            if status == "running":
                self._ensure_heartbeat_thread()

    def stage(
        self,
        stage: str,
        *,
        source: str,
        message: str | None = None,
        **fields: Any,
    ) -> ContextManager[None]:
        """Context manager form for a single blocking stage."""

        @contextmanager
        def _stage() -> Iterator[None]:
            self.emit(
                stage,
                source=source,
                status="running",
                message=message,
                **fields,
            )
            try:
                yield
            except BaseException as exc:
                self.emit(
                    stage,
                    source=source,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                    **fields,
                )
                raise
            else:
                self.emit(
                    stage,
                    source=source,
                    status="completed",
                    message=message,
                    **fields,
                )

        return _stage()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1)

    def _ensure_heartbeat_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="kernel-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_interval_s):
            with self._lock:
                if self._current is None:
                    continue
                heartbeat = dict(self._current)
                heartbeat["ts"] = time.time()
                heartbeat["heartbeat"] = True
                self._safe_write(heartbeat)

    def _safe_write(self, event: dict[str, Any]) -> None:
        try:
            self._sink.write(event)
        except Exception:
            # Observability must never break kernel generation/optimization.
            pass


_backend_factories: dict[str, _BackendFactory] = {}
_reporter: ProgressReporter | None = None
_reporter_pid: int | None = None
_reporter_lock = threading.Lock()


def register_progress_backend(name: str, factory: _BackendFactory) -> None:
    """Register a backend factory for application-specific transports."""
    _backend_factories[name] = factory


def configure_progress(
    root: Path | str,
    *,
    backend: str = "json",
    heartbeat_interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S,
) -> ProgressReporter:
    """Enable a backend in this process and inherited child processes."""
    if backend not in _backend_factories:
        raise ValueError(f"Unknown progress backend: {backend}")
    os.environ[ENV_BACKEND] = backend
    os.environ[ENV_ROOT] = str(Path(root))
    os.environ[ENV_HEARTBEAT_INTERVAL] = str(heartbeat_interval_s)
    return _replace_reporter(
        _create_reporter(backend, Path(root), heartbeat_interval_s)
    )


def get_progress() -> ProgressReporter:
    """Return the process-local reporter, lazily configured from the environment."""
    global _reporter, _reporter_pid
    pid = os.getpid()
    if _reporter is not None and _reporter_pid == pid:
        return _reporter

    with _reporter_lock:
        if _reporter is not None and _reporter_pid == pid:
            return _reporter
        backend = os.environ.get(ENV_BACKEND)
        root = os.environ.get(ENV_ROOT)
        if backend in _backend_factories and root:
            try:
                interval = float(
                    os.environ.get(
                        ENV_HEARTBEAT_INTERVAL,
                        str(_DEFAULT_HEARTBEAT_INTERVAL_S),
                    )
                )
            except ValueError:
                interval = _DEFAULT_HEARTBEAT_INTERVAL_S
            reporter = _create_reporter(backend, Path(root), interval)
        else:
            reporter = ProgressReporter()
        _reporter = reporter
        _reporter_pid = pid
        return reporter


def set_progress_sink(
    sink: ProgressSink,
    *,
    heartbeat_interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S,
) -> ProgressReporter:
    """Inject a sink directly, primarily for embedding and tests."""
    return _replace_reporter(
        ProgressReporter(sink, heartbeat_interval_s=heartbeat_interval_s)
    )


def shutdown_progress(*, clear_environment: bool = False) -> None:
    """Stop the process-local heartbeat thread."""
    global _reporter, _reporter_pid
    with _reporter_lock:
        reporter = _reporter
        _reporter = None
        _reporter_pid = None
    if reporter is not None:
        reporter.close()
    if clear_environment:
        for key in (ENV_BACKEND, ENV_ROOT, ENV_HEARTBEAT_INTERVAL):
            os.environ.pop(key, None)


def progress_stage(
    stage: str,
    *,
    source: str,
    message: str | None = None,
    result_ok: Callable[[Any], bool] | None = None,
) -> Callable[[_F], _F]:
    """Decorate a blocking operation with start/success/failure events."""

    def decorator(function: _F) -> _F:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            reporter = get_progress()
            reporter.emit(stage, source=source, message=message)
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                reporter.emit(
                    stage,
                    source=source,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                )
                raise

            ok = result_ok(result) if result_ok is not None else True
            reporter.emit(
                stage,
                source=source,
                status="completed" if ok else "failed",
                message=message if ok else "operation returned an unsuccessful result",
            )
            return result

        return wrapped  # type: ignore[return-value]

    return decorator


def _create_reporter(
    backend: str,
    root: Path,
    heartbeat_interval_s: float,
) -> ProgressReporter:
    source_id = _process_source_id()
    sink = _backend_factories[backend](root, source_id)
    return ProgressReporter(sink, heartbeat_interval_s=heartbeat_interval_s)


def _replace_reporter(reporter: ProgressReporter) -> ProgressReporter:
    global _reporter, _reporter_pid
    with _reporter_lock:
        previous = _reporter
        _reporter = reporter
        _reporter_pid = os.getpid()
    if previous is not None:
        previous.close()
    return reporter


def _process_source_id() -> str:
    process_name = multiprocessing.current_process().name
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", process_name)
    return f"{os.getpid()}-{safe_name}"


register_progress_backend(
    "json",
    lambda root, source_id: JsonFileProgressSink(root, source_id),
)
atexit.register(shutdown_progress)
