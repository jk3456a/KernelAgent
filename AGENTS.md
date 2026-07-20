# Agent Notes

This fork is designed to run from a Mac or other non-GPU development machine.
LLM calls run locally through llm-center, while correctness checks, benchmarks,
and NCU profiling run on a remote H100 devspace over SSH.

## Requese a devspace GPU

See [docs/remote-execution.md](docs/remote-execution.md) for the full remote execution and devspace GPU setup guide.

## Environment

- Install locally with `pip install -e .` inside the repository virtualenv.
- Store secrets only in ignored files such as `.env.local`; load it before runs with
  `set -a && source .env.local && set +a`.
- `remote.toml` must point at the active Teleport devspace, and `ssh <hostname>`
  must work without extra flags.
- Remote prerequisites are `python3`, `torch`, `triton`, `rsync`, `ncu`, and
  `numpy` on the non-interactive PATH.
- Keep a heartbeat running for long jobs; macOS does not provide GNU `timeout`,
  so prefer an SSH loop using `ssh -o ConnectTimeout=35 -o BatchMode=yes`.

## Common Runs

- Fused Conv2d pipeline: `python3 -u run_conv_pipeline.py`.
- Level1 Conv2d pipeline: `python3 -u run_conv_l1_pipeline.py`.
- GEMM pipeline: `python3 -u run_gemm_pipeline.py`.
- Isolated BF16 GEMM pipeline: `python3 -u run_gemm_bf16_pipeline.py`.
- Use long timeouts for model and worker subprocesses:
  `KERNEL_WORKER_TIMEOUT_S=3600 LLM_TIMEOUT_S=3600`.
- Write logs with `tee /tmp/<name>.log` so long-running progress can be inspected.

## Project Conventions

- Do not commit local secrets, `remote.toml`, `.env.local`, generated runs, or
  optimized kernels unless explicitly requested.
- Generated optimization artifacts live under `examples/**/runs/` or
  `examples/**/opt_manager_logs/`.
- Agent1 writes `input.py`; agent2 writes `optimized_kernel_greedy_glm.py`.
- GLM-5.2 currently uses llm-center with a 128K output-token cap, even if the
  context window is larger.
- For remote execution issues, first check Teleport inventory with `tsh ls`, then
  verify `ssh`, `rsync`, `ncu`, `torch`, `triton`, and `numpy`.

## Worktree Setup

- A new git worktree does NOT inherit gitignored files. Some of them are required
  for runs, so copy them from the primary checkout into the new worktree before
  running anything:
  - `.env.local` — local secrets loaded via `set -a && source .env.local && set +a`.
  - `remote.toml` — pointer to the active Teleport devspace GPU.
- Do NOT copy regenerable ignored paths: `__pycache__/`, `*.egg-info/`, `.venv/`,
  `examples/**/runs/`, `examples/**/opt_manager_logs/`, and generated kernels
  (`input.py`, `*_kernel.py`). They are rebuilt or produced by the pipeline runs.
- Copy example, then adapt paths as needed:
  `cp .env.local remote.toml <worktree-dir>/`.
