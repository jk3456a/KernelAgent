#!/usr/bin/env python3
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
Devspace heartbeat daemon: keep Cybertron GPU devspaces alive.

Discovers running GPU devspaces via cctl, validates connectivity with a
single heartbeat round, then forks a background loop that periodically
pings the Cybertron API to prevent idle-timeout reclamation.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SERVER = "https://cybertron.modelbest.co"
DEFAULT_TOKEN_FILE = Path.home() / ".cybertronctl" / "profiles" / "modelbest" / "token"
DEFAULT_INTERVAL = 300
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
PID_FILE = Path.home() / ".devspace-heartbeat.pid"
LOG_FILE = Path.home() / ".devspace-heartbeat.log"


# ------------------------------------------------------------------
# Token
# ------------------------------------------------------------------

def read_token(token_file: Path) -> str:
    try:
        return token_file.read_text().strip()
    except FileNotFoundError:
        print(f"ERROR: token file not found at {token_file}", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------

def discover_gpu_devspaces() -> list[dict]:
    """Return list of dicts with keys: id, gpu, model, project."""
    try:
        result = subprocess.run(
            ["cctl", "devspace", "list", "--own", "--status", "Running",
             "--limit", "50", "-o", "json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return []

    devspaces: list[dict] = []
    for task in data.get("tasks", []):
        gpu_count = task.get("resourcesPerNode", {}).get("gpuCount", 0)
        if gpu_count > 0:
            devspaces.append({
                "id": task["name"].split("/")[-1],
                "gpu": gpu_count,
                "model": task.get("gpuModel", "?"),
                "project": task.get("project", "?"),
            })
    return devspaces


# ------------------------------------------------------------------
# Heartbeat
# ------------------------------------------------------------------

def send_heartbeat(server: str, token: str, devspace_id: str) -> str:
    """Send one heartbeat request. Returns 'OK' or 'FAIL:<detail>'."""
    url = f"{server}/api/job/devspace/heartbeat?id={devspace_id}"
    req = urllib.request.Request(
        url, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=b"",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            if resp.status == 200 and '"success":' in body and "true" in body.lower():
                return "OK"
            return f"FAIL:{resp.status}"
    except urllib.error.HTTPError as e:
        return f"FAIL:{e.code}"
    except Exception as e:
        return f"ERR:{e}"


# ------------------------------------------------------------------
# Log management
# ------------------------------------------------------------------

def _redirect_stdio_to_log(log_file: Path) -> None:
    """Redirect fd 1 (stdout) and fd 2 (stderr) to *log_file* (append)."""
    fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


def _last_ok_ago() -> str | None:
    """Parse the log for the last '-> OK' line and return a human-readable age."""
    if not LOG_FILE.exists():
        return None
    last_ok_ts: str | None = None
    for line in LOG_FILE.read_text().splitlines():
        if "-> OK" in line and line.startswith("["):
            last_ok_ts = line[1:line.index("]")]
    if last_ok_ts is None:
        return None
    try:
        ts = datetime.strptime(last_ok_ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc,
        )
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s ago"
        return f"{secs // 3600}h {(secs % 3600) // 60}m ago"
    except (ValueError, IndexError):
        return None


def _maybe_rotate_log(log_file: Path, max_bytes: int = LOG_MAX_BYTES) -> None:
    """Rotate *log_file* → *.log.1* when it exceeds *max_bytes*."""
    if not log_file.exists():
        return
    if log_file.stat().st_size <= max_bytes:
        return
    backup = log_file.with_suffix(".log.1")
    log_file.replace(backup)


# ------------------------------------------------------------------
# Daemon loop
# ------------------------------------------------------------------

def _daemon_loop(server: str, token_file: Path, interval: int) -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    while True:
        _maybe_rotate_log(LOG_FILE)
        token = read_token(token_file)
        devspaces = discover_gpu_devspaces()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for ds in devspaces:
            result = send_heartbeat(server, token, ds["id"])
            print(
                f"[{ts}] heartbeat devspace={ds['id']} "
                f"({ds['gpu']}x{ds['model']}, {ds['project']}) -> {result}",
                flush=True,
            )
        time.sleep(interval)


# ------------------------------------------------------------------
# PID management
# ------------------------------------------------------------------

def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

def cmd_start(server: str, token_file: Path, interval: int) -> None:
    if pid := _read_pid():
        print(f"Already running (PID {pid}). Use 'stop' first.", file=sys.stderr)
        sys.exit(1)

    token = read_token(token_file)
    devspaces = discover_gpu_devspaces()
    if not devspaces:
        print("ERROR: no running GPU devspaces found", file=sys.stderr)
        sys.exit(1)

    success_lines: list[str] = []
    for ds in devspaces:
        result = send_heartbeat(server, token, ds["id"])
        if result == "OK":
            success_lines.append(f"  {ds['id']} / {ds['gpu']}x{ds['model']} / {ds['project']}")
        else:
            print(f"WARN: heartbeat failed for {ds['id']} ({result})", file=sys.stderr)

    if not success_lines:
        print("ERROR: all heartbeats failed, aborting", file=sys.stderr)
        sys.exit(1)

    print(f"=== Heartbeat validated ({len(success_lines)} devspace(s) OK) ===")
    print()
    print("First successful devspaces (id / gpu / project):")
    for line in success_lines[:3]:
        print(line)
    print()

    child_pid = os.fork()
    if child_pid > 0:
        PID_FILE.write_text(str(child_pid))
        print(f"Daemon started (PID {child_pid}), interval={interval}s")
        return

    # Child: detach, redirect output to log, run the daemon loop
    os.setsid()
    _redirect_stdio_to_log(LOG_FILE)
    try:
        _wrap_caffeinate(server, token_file, interval)
    except Exception:
        _daemon_loop(server, token_file, interval)


def _wrap_caffeinate(server: str, token_file: Path, interval: int) -> None:
    """Exec under caffeinate on macOS to prevent system sleep."""
    if sys.platform != "darwin":
        _daemon_loop(server, token_file, interval)
        return
    import shutil
    if not shutil.which("caffeinate"):
        _daemon_loop(server, token_file, interval)
        return
    os.execlp(
        "caffeinate", "caffeinate", "-i",
        sys.executable, __file__,
        "--server", server,
        "--token-file", str(token_file),
        "--interval", str(interval),
        "_loop",
    )


def cmd_stop() -> None:
    pid = _read_pid()
    if pid is None:
        print("Not running.")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(10):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    PID_FILE.unlink(missing_ok=True)
    print(f"Stopped (PID {pid})")


def cmd_status(token_file: Path, *, n: int = 5, follow: bool = False) -> None:
    pid = _read_pid()
    if pid is None:
        print("Not running.")
        return

    try:
        result = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        elapsed = result.stdout.strip() or "unknown"
    except Exception:
        elapsed = "unknown"

    ago = _last_ok_ago()
    if ago:
        print(f"Running (PID {pid}, elapsed {elapsed}) — Last OK: {ago}")
    else:
        print(f"Running (PID {pid}, elapsed {elapsed}) — No successful heartbeat in log")
    print()
    print("Current GPU devspaces:")
    devspaces = discover_gpu_devspaces()
    if not devspaces:
        print("  (none)")
    else:
        for ds in devspaces:
            print(f"  {ds['id']} / {ds['gpu']}x{ds['model']} / {ds['project']}")

    print()
    if not LOG_FILE.exists():
        print("(no log file yet)")
        return

    lines = LOG_FILE.read_text().splitlines()
    tail = lines[-n:] if n < len(lines) else lines
    print(f"Recent log ({len(tail)} lines):")
    for line in tail:
        print(f"  {line}")

    if follow:
        print()
        print("Following log (Ctrl+C to stop) ...")
        _follow_log()


def _follow_log() -> None:
    """Tail the log file continuously until interrupted."""
    with open(LOG_FILE) as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                print(f"  {line}", end="", flush=True)
            else:
                time.sleep(0.5)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devspace-heartbeat",
        description="Keep Cybertron GPU devspaces alive with periodic heartbeat requests.",
    )
    parser.add_argument(
        "--server", default=os.environ.get("HEARTBEAT_SERVER", DEFAULT_SERVER),
        help=f"Cybertron API base URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--token-file",
        default=os.environ.get("HEARTBEAT_TOKEN_FILE", str(DEFAULT_TOKEN_FILE)),
        help=f"Path to cctl token file (default: {DEFAULT_TOKEN_FILE})",
    )
    parser.add_argument(
        "--interval", type=int,
        default=int(os.environ.get("HEARTBEAT_INTERVAL", str(DEFAULT_INTERVAL))),
        help=f"Seconds between heartbeat rounds (default: {DEFAULT_INTERVAL})",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="Validate heartbeat and start background daemon")
    sub.add_parser("stop", help="Stop the heartbeat daemon")
    status_parser = sub.add_parser("status", help="Show daemon status and current GPU devspaces")
    status_parser.add_argument(
        "-n", type=int, default=5,
        help="Number of recent log lines to show (default: 5)",
    )
    status_parser.add_argument(
        "-f", "--follow", action="store_true", default=False,
        help="Follow log output (like tail -f)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    raw = argv if argv is not None else sys.argv[1:]
    if raw and raw[-1] == "_loop":
        parser = build_parser()
        args = parser.parse_args(raw[:-1])
        _redirect_stdio_to_log(LOG_FILE)
        _daemon_loop(args.server, Path(args.token_file).expanduser(), args.interval)
        return

    parser = build_parser()
    args = parser.parse_args(raw)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    token_file = Path(args.token_file).expanduser()

    if args.command == "start":
        cmd_start(args.server, token_file, args.interval)
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status(token_file, n=args.n, follow=args.follow)


if __name__ == "__main__":
    main()
