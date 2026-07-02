# Remote Kernel Verification over SSH

KernelAgent can run candidate-kernel verification on a **remote GPU host** while
the LLM generation runs locally. This lets you drive the full pipeline from a
machine with no GPU and no PyTorch/Triton install — only the LLM API keys are
needed locally; the remote host owns the GPU and the `torch` + `triton` install.

The mechanism is a faithful port of Train_engine's SSH stack: `rsync` the
self-contained candidate file to a per-run remote workdir, then run it over
`ssh <host> 'cd <workdir> && exec setsid python3 candidate_main.py'` with
persistent Triton/Inductor caches and `ServerAlive` keepalives. It depends only
on the system `ssh` / `rsync` binaries plus an `~/.ssh/config` host alias — no
extra Python packages.

## How it routes

KernelAgent's only GPU touchpoint is `Fuser.runner.run_candidate`. It now
resolves a remote config and branches:

- `kind = "local"` (default) → original local `subprocess` path, unchanged.
- `kind = "ssh"` → push the candidate to the remote host and execute it over
  SSH. The same Popen loop drives both, so timeout, cancellation, output
  capture, and `PASS` / `ALL_TESTS_PASSED` classification are identical.

Every other component (`pipeline`, `compose_end_to_end`, `auto_agent`,
`dispatch_kernel_agent`) calls `run_candidate` and is transparently remoted.

### Generation (agent1) vs optimization (agent2)

Both agents are remoted:

- **agent1 (generation)** — `triton_kernel_agent/worker.py::_run_test` and
  `Fuser/runner.py::run_candidate` push the candidate workdir to the remote host
  and run the correctness tests there.
- **agent2 (optimization / speedup)** — three GPU steps run remotely:
  - **benchmark** (`benchmark.py`): pushes `kernel_subprocess.py` + kernel +
    problem, runs the timing there, pulls `benchmark_results.json` back.
  - **PyTorch baseline**: driven by `kernel_subprocess.py --baseline` on the
    remote host. (The `torch.compile` reference line is skipped under remote —
    it is informational only.)
  - **NCU profiling** (`ncu_profiler.py`): rewrites the wrapper's hard-coded
    local `sys.path` entries to the pushed workdir, runs `ncu` remotely, pulls
    the metrics CSV back for local roofline/bottleneck analysis.

The control machine **never imports torch** — the optimization modules import it
lazily, only on the local path. So a torch-free laptop can drive the full
generate→verify→profile→optimize loop as long as the LLM is reachable locally
and the GPU host is reachable over SSH.

## Configure

Pick **either** a TOML file **or** environment variables (env overrides TOML
per field).

### TOML

```bash
cp remote.toml.example remote.toml   # then edit hostname
```

```toml
[remote]
kind = "ssh"
hostname = "h100box"     # an alias in ~/.ssh/config
workspace = "/data/ka"   # optional; empty = remote $HOME
```

By default `./remote.toml` (in the run CWD) is read. Override the path with
`KERNEL_REMOTE_CONFIG=/abs/path/to/remote.toml`.

### Environment variables

```bash
export KERNEL_REMOTE_KIND=ssh
export KERNEL_REMOTE_HOST=h100box
export KERNEL_REMOTE_WORKSPACE=/data/ka   # optional
```

These also work in KernelAgent's `.env` file.

## Pre-conditions on the remote host

1. `~/.ssh/config` has the `hostname` alias configured (proxy / cert / port /
   identity) so plain `ssh <hostname> <cmd>` works without extra flags.
2. `rsync` 3.x on the remote non-interactive `$PATH`.
3. `torch` + `triton` importable on the remote default `python3`.

Candidates run under `<workspace>/.kernelagent_remote/<attempt_id>/`. Triton and
Inductor compile caches persist at `$HOME/.cache/{triton,inductor}-persistent`
on the remote so repeated runs avoid the cold-compile tax.

## Keeping a managed devspace alive

If the remote GPU host is an auto-reclaimed Cybertron devspace, run the
heartbeat daemon locally so the devspace is not idle-timed out:

```bash
devspace-heartbeat start     # validates connectivity, forks a background daemon
devspace-heartbeat status
devspace-heartbeat stop
```

It discovers running GPU devspaces via `cctl devspace list` and POSTs a periodic
heartbeat (default every 300 s). Requires `cctl` installed and logged in. Not
needed for a static on-prem SSH box.
