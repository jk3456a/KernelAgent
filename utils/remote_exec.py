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
"""SSH remote execution of candidate kernels (ported from Train_engine).

This is a faithful, minimal port of Train_engine's two-piece remote stack:

* ``remote_sync.py``  — ``rsync`` the workspace to the remote host.
* ``remote_run.sh``   — ``ssh <host> 'cd <workdir> && exec setsid <cmd>'`` with
  persistent Triton/Inductor caches and ``ServerAlive`` keepalives.

KernelAgent's only GPU touchpoint is ``Fuser.runner.run_candidate``, which runs
a *self-contained* ``candidate_main.py`` (candidate kernel + embedded test that
prints ``PASS`` / ``ALL_TESTS_PASSED``). So the remote port is narrow: per
candidate we

1. ``ssh mkdir -p`` a per-run remote workdir,
2. ``rsync`` the run directory's files (candidate + optional sitecustomize),
3. build the same ``ssh`` argv Train_engine uses and hand it to the existing
   ``runner._run_candidate`` Popen loop, so timeout / cancel / output-capture /
   PASS-classification are all reused unchanged.

Like Train_engine, this depends only on the system ``ssh`` / ``rsync`` binaries
plus a ``~/.ssh/config`` host alias — no paramiko, no extra Python deps.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

__all__ = ["build_remote_argv", "run_workdir_tests", "run_command_with_artifacts"]

# Remote root (under the configured workspace) that holds per-run workdirs.
_REMOTE_ROOT = ".kernelagent_remote"

# Mirror runner._allowlist_env()'s determinism knobs so remote and local runs
# behave identically.
_DETERMINISM_ENV = (
    "PYTHONHASHSEED=0",
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
)

# Persistent compile caches: Triton JIT + TorchInductor rebuild their object
# files on every cold ssh hop unless these point at a stable on-disk location.
# Train_engine documented a ~3 min/invocation compile tax without this; the
# user-home volume survives ``cctl devspace copy`` so $HOME/.cache is the right
# anchor.
_TRITON_CACHE = "$HOME/.cache/triton-persistent"
_INDUCTOR_CACHE = "$HOME/.cache/inductor-persistent"


def _remote_workdir(workspace: str, local_run_dir: Path) -> str:
    """Per-run remote workdir, namespaced by the local attempt directory name.

    ``workspace`` empty → default to the remote ``$HOME`` (resolved by the
    remote shell at exec time, exactly like Train_engine's ssh.toml semantics).
    """
    root = (workspace.strip() or "$HOME").rstrip("/")
    return f"{root}/{_REMOTE_ROOT}/{local_run_dir.name}"


def _build_remote_script(
    remote_workdir: str,
    exec_filename: str,
    *,
    isolated: bool,
    deny_network: bool,
) -> str:
    """The inner command run on the remote host (before ssh quoting).

    Mirrors ``remote_run.sh``: ``mkdir`` the persistent caches, export them,
    ``cd`` into the workdir, then ``exec setsid python3 ...`` so the remote
    process group is reaped as a unit when ssh tears the connection down (the
    remote analogue of the local ``start_new_session=True``).
    """
    # -I (isolated) drops cwd from sys.path, which would defeat the
    # sitecustomize network block; match runner.run_candidate's precedence.
    py = "python3 -u"
    if isolated and not deny_network:
        py += " -I"
    py += f" {shlex.quote(exec_filename)}"

    env_prefix = " ".join(_DETERMINISM_ENV)
    # Cache paths contain $HOME and MUST stay unquoted so the remote shell
    # expands them (Train_engine uses "\$HOME/..." for the same reason). They
    # contain no spaces, so leaving them bare is safe. The workdir likewise may
    # be "$HOME/..." when no workspace is configured.
    return (
        f'mkdir -p {_TRITON_CACHE} {_INDUCTOR_CACHE} && '
        f'export TRITON_CACHE_DIR={_TRITON_CACHE} '
        f'TORCHINDUCTOR_CACHE_DIR={_INDUCTOR_CACHE} && '
        f'cd {remote_workdir} && '
        f'exec setsid --wait env {env_prefix} {py}'
    )


def _build_ssh_argv(hostname: str, remote_cmd: str) -> list[str]:
    """``ssh`` argv with the same keepalive options Train_engine uses.

    ServerAliveInterval/CountMax mean a dropped connection surfaces as a
    nonzero exit within ~120s rather than hanging the local poll loop forever.
    """
    return [
        "ssh",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
        hostname,
        remote_cmd,
    ]


def _build_mkdir_argv(hostname: str, remote_workdir: str) -> list[str]:
    return ["ssh", hostname, f"mkdir -p {remote_workdir}"]


def _build_rsync_argv(hostname: str, sources: list[Path], remote_workdir: str) -> list[str]:
    cmd = ["rsync", "-az"]
    cmd.extend(str(p) for p in sources)
    cmd.append(f"{hostname}:{remote_workdir}/")
    return cmd


def _remaining_timeout(
    deadline: float | None,
    command: list[str],
) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(command, 0)
    return remaining


def _push_candidate(
    hostname: str,
    sources: list[Path],
    remote_workdir: str,
    *,
    deadline: float | None = None,
) -> None:
    """Create the remote workdir and rsync the run files into it.

    Fail fast (RuntimeError) on mkdir/rsync failure — a half-pushed workdir
    must not be executed.
    """
    mkdir_argv = _build_mkdir_argv(hostname, remote_workdir)
    mkdir = subprocess.run(
        mkdir_argv,
        capture_output=True, text=True,
        timeout=_remaining_timeout(deadline, mkdir_argv),
    )
    if mkdir.returncode != 0:
        raise RuntimeError(
            f"remote mkdir failed on {hostname}:{remote_workdir} "
            f"(exit {mkdir.returncode}): {mkdir.stderr.strip()}"
        )
    rsync_argv = _build_rsync_argv(hostname, sources, remote_workdir)
    rsync = subprocess.run(
        rsync_argv,
        capture_output=True, text=True,
        timeout=_remaining_timeout(deadline, rsync_argv),
    )
    if rsync.returncode != 0:
        raise RuntimeError(
            f"remote rsync failed to {hostname}:{remote_workdir} "
            f"(exit {rsync.returncode}): {rsync.stderr.strip()}"
        )


def build_remote_argv(
    cfg: dict[str, str],
    local_run_dir: Path,
    exec_filename: str,
    sources: list[Path],
    *,
    isolated: bool,
    deny_network: bool,
) -> list[str]:
    """Push *sources* to the remote host and return the ssh argv to execute.

    The returned argv is meant to be passed straight to ``runner._run_candidate``
    (a Popen loop), so all timeout / cancel / capture / classification logic is
    shared with the local path.
    """
    hostname = cfg["hostname"]
    remote_workdir = _remote_workdir(cfg.get("workspace", ""), local_run_dir)
    _push_candidate(hostname, sources, remote_workdir)
    remote_cmd = _build_remote_script(
        remote_workdir, exec_filename, isolated=isolated, deny_network=deny_network
    )
    return _build_ssh_argv(hostname, remote_cmd)


def _build_workdir_test_script(remote_workdir: str, test_filenames: list[str]) -> str:
    """Remote command that runs several test files sequentially with && semantics.

    The TritonKernelAgent worker writes ``kernel.py`` plus one or more
    ``test_*.py`` into a single workdir; each test imports the sibling kernel.
    So we ``cd`` into the pushed workdir once and chain the tests — the first
    nonzero exit aborts the chain, exactly like the local ``_run_test`` loop.
    """
    env_prefix = " ".join(_DETERMINISM_ENV)
    chain = " && ".join(f"python3 -u {shlex.quote(t)}" for t in test_filenames)
    return (
        f'mkdir -p {_TRITON_CACHE} {_INDUCTOR_CACHE} && '
        f'export TRITON_CACHE_DIR={_TRITON_CACHE} '
        f'TORCHINDUCTOR_CACHE_DIR={_INDUCTOR_CACHE} && '
        f'cd {remote_workdir} && '
        f'exec setsid --wait env {env_prefix} bash -c {shlex.quote(chain)}'
    )


def run_workdir_tests(
    cfg: dict[str, str],
    workdir: Path,
    test_filenames: list[str],
    *,
    timeout_s: float,
) -> tuple[bool, str, str]:
    """Rsync *workdir* to the remote host and run its test files there.

    Returns the same ``(success, stdout, stderr)`` contract the local worker's
    ``_run_test`` uses, so the worker can swap execution targets transparently.
    Fails fast (RuntimeError) on push errors; a nonzero remote exit is a normal
    test failure (``success=False``), not an exception.
    """
    hostname = cfg["hostname"]
    remote_workdir = _remote_workdir(cfg.get("workspace", ""), workdir)

    # Push the entire workdir (kernel + tests) so imports resolve remotely.
    sources = sorted(p for p in workdir.iterdir() if p.is_file())
    _push_candidate(hostname, sources, remote_workdir)

    remote_cmd = _build_workdir_test_script(remote_workdir, test_filenames)
    argv = _build_ssh_argv(hostname, remote_cmd)
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout_s
    )
    return completed.returncode == 0, completed.stdout, completed.stderr


def _build_command_script(remote_workdir: str, command: str) -> str:
    """Remote command that ``cd``s into the pushed workdir and runs *command*.

    Used by the optimization stack (agent2): benchmark and NCU profiling each
    run a single command in the workdir and write artifact files (JSON / CSV)
    that are pulled back afterwards. ``command`` is a fully-formed shell command
    (e.g. ``python3 -u bench.py --json out.json`` or an ``ncu ...`` invocation).
    """
    env_prefix = " ".join(_DETERMINISM_ENV)
    return (
        f'mkdir -p {_TRITON_CACHE} {_INDUCTOR_CACHE} && '
        f'export TRITON_CACHE_DIR={_TRITON_CACHE} '
        f'TORCHINDUCTOR_CACHE_DIR={_INDUCTOR_CACHE} && '
        f'cd {remote_workdir} && '
        f'exec setsid --wait env {env_prefix} bash -c {shlex.quote(command)}'
    )


def _build_pull_argv(
    hostname: str, remote_workdir: str, artifact: str, local_dest: Path
) -> list[str]:
    return [
        "rsync",
        "-az",
        f"{hostname}:{remote_workdir}/{artifact}",
        str(local_dest),
    ]


def remote_command_for(cfg: dict[str, str], workdir: Path, command: str) -> str:
    """Rewrite a local ``command`` string so its python/ncu paths are remote-safe.

    Callers build their command against local absolute paths; on the remote the
    files live in the pushed workdir (same basenames). This helper is a no-op
    today (commands are built with bare basenames), kept as the single seam for
    future path rewriting if needed.
    """
    return command


def run_command_with_artifacts(
    cfg: dict[str, str],
    workdir: Path,
    command: str,
    *,
    artifacts: list[str],
    timeout_s: float,
) -> tuple[int, str, str]:
    """Push *workdir*, run *command* on the remote host, pull *artifacts* back.

    The agent2 (optimization) execution model: a single command runs in the
    workdir and writes result files. We rsync the whole workdir up, run the
    command over ssh, then rsync each named artifact back into the local
    workdir so the caller reads them exactly as in the local path.

    Returns ``(returncode, stdout, stderr)``. Push failure raises (fail fast);
    a nonzero command exit is returned, not raised — the caller decides. The
    artifact pull is best-effort (a failed run may not produce them).
    """
    hostname = cfg["hostname"]
    remote_workdir = _remote_workdir(cfg.get("workspace", ""), workdir)
    deadline = time.monotonic() + timeout_s

    sources = sorted(p for p in workdir.iterdir() if p.is_file())
    _push_candidate(hostname, sources, remote_workdir, deadline=deadline)

    remote_cmd = _build_command_script(remote_workdir, command)
    argv = _build_ssh_argv(hostname, remote_cmd)
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=_remaining_timeout(deadline, argv),
    )

    # Pull artifacts back regardless of rc — a profiler may write a partial CSV
    # that the caller still inspects; missing files just fail the pull quietly.
    for artifact in artifacts:
        pull = _build_pull_argv(hostname, remote_workdir, artifact, workdir / artifact)
        subprocess.run(
            pull,
            capture_output=True,
            text=True,
            timeout=_remaining_timeout(deadline, pull),
        )

    return completed.returncode, completed.stdout, completed.stderr


