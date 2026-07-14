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

### PyTorch baseline cuBLAS/cuDNN evidence

The eager PyTorch baseline records which CUDA library appears to execute its
workload. `torch.profiler` is the primary probe. For GEMM/matmul/linear
workloads it evaluates cuBLAS; for convolution workloads it evaluates cuDNN.
An unrelated library's medium-confidence `not_detected` result does not trigger
extra profiling.

If the relevant library remains `unknown` or medium-confidence, agent2 launches
one best-effort NCU sidecar. The sidecar uses
`cudaProfilerStart`/`cudaProfilerStop` around exactly one eager forward and
`ncu --profile-from-start=off`, then merges the demangled kernel names with the
primary evidence. NCU failure, timeout, missing permissions, or an empty CSV
does not fail the baseline benchmark; the original profiler result is retained
with an `ncu.status` of `failed` or `inconclusive`. Override the sidecar timeout
with `KERNELAGENT_BACKEND_NCU_TIMEOUT_S` (default: 300 seconds).

The merged schema is written to `artifacts/pytorch_backend.json`, logged with
the baseline result, and returned from `OptimizationManager.run_optimization()`
as `pytorch_baseline_backend`. The top-level library fields remain
`status`, `detected`, `confidence`, and `evidence`; schema v2 additionally
records evidence sources, conflicts, and the NCU probe status.

This is kernel-launch evidence, not an API-call audit. A kernel name containing
cuBLAS/cuDNN is strong evidence, but a missing or renamed symbol cannot prove
that `cublasLtMatmul` or `cudnnConvolutionForward` was never called. Use CUPTI
callbacks or an Nsight Systems API trace when host API-level proof is required.

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
4. `ncu` on the non-interactive `$PATH` for candidate profiling and optional
   baseline backend refinement.

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

## Request and manage a devspace

When the remote host is a Cybertron devspace, the whole lifecycle — provision,
inspect, keep alive, and reclaim — is driven by the `cctl` CLI. Install it and
log in first; every command below assumes an authenticated profile.

```bash
cctl login                 # browser sign-in, or: cctl login <token>
cctl profile current       # confirm which credential profile is active
```

> Agent tip: `cctl` prints JSON whenever stdout is not a TTY, so pipe to `jq`
> or add `-o json` explicitly in scripts. Use `-F name,status` to pull just the
> fields you need, and `-q` to print only IDs.

### 1. Find a pool and an image

```bash
cctl pool list --own                            # pools you can schedule on
cctl image list --usage training --limit 20     # base images; prefer loopharness
```

Prefer a `loopharness` image — it ships `rsync` and `ncu` (Nsight Compute) out
of the box, which the candidate verification and NCU profiling steps both need.

### 2. Create the devspace

The account-specific parameters (project / cluster / pool / image / billing) live
in the gitignored `.env.local` — fill them in once, then source it:

```bash
set -a && source .env.local && set +a

cctl devspace create \
  --project "$CCTL_PROJECT" --cluster "$CCTL_CLUSTER" \
  --resource-pool "$CCTL_RESOURCE_POOL" \
  --image "$CCTL_IMAGE" \
  --gpu 1 --gpu-model H100 --cpu 16 --memory 128 \
  --priority PREEMPTABLE \
  --billing-account-id "$CCTL_BILLING_ACCOUNT_ID"
```

Notes:

- `--priority PREEMPTABLE` is required unless you hold *staff* permission on the
  resource pool. The default `NORMAL` (and `HIGH`) is rejected for non-staff
  members with `task.priority: NORMAL/HIGH requires resource pool staff permission`.
- A devspace has no `--entry`; it is an interactive environment, not a batch job.
- Duration is heartbeat-managed (forced to 0), so the box stays up only while it
  receives heartbeats — see [Keeping a managed devspace alive](#keeping-a-managed-devspace-alive).
- Add `--dry-run` to preview the request payload without submitting.
- For repeatable specs use `-f task.json` instead of long flag lists.

### 3. Inspect and connect

```bash
cctl devspace list --own --status Running        # find your devspace + its id
cctl devspace get <id>                            # full detail (node name, ip, ports)
cctl devspace get <id> -F name,status
cctl devspace logs <id>                            # startup / runtime logs
cctl devspace metrics <id>                         # GPU / power / temp / util
```

Use the node name from `get` to build the `~/.ssh/config` alias (proxy/cert/port)
so that plain `ssh <hostname>` works — that alias is what `remote.toml`'s
`hostname` points at (see [Configure](#configure) and the SSH setup in
`HANDOFF.md §2.3`). If you serve a dashboard or notebook from the box, expose its
port:

```bash
cctl devspace expose <id> --port 8086 --open       # enable + open in browser
cctl devspace expose <id> --disable                # turn it back off
```

### 4. Snapshot, restore, and stop

```bash
cctl devspace stop <id> --snapshot-before-stop best-effort   # save state, then stop
cctl devspace stop <id> -y                                     # stop immediately
cctl devspace restore <id> -y                                 # new devspace from a snapshot
```

Snapshotting before a stop lets you rehydrate the same environment later with
`restore`. After creating or restoring, update `remote.toml` (and your
`~/.ssh/config` alias) to the new node name, and restart the heartbeat.

> If `cctl` cannot see the devspace (token scope) the API-based heartbeat won't
> work either; fall back to the direct-SSH heartbeat documented in `HANDOFF.md §3`.
