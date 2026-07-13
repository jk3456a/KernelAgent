# GEMM GLM-5.2 BF16 Run Config

This document records the isolated GEMM pipeline configuration used for BF16
kernel generation and optimization.

## Pipeline

- Entry point: `run_gemm_bf16_pipeline.py`
- Problem directory: `examples/optimize_gemm_bf16/`
- Agent1 model: `glm-5.2`
- Agent1 workers: `4`
- Agent1 max attempts: `10`
- Agent2 strategy: `greedy_glm`
- Agent2 max rounds: `10`
- Run directory pattern:
  `examples/optimize_gemm_bf16/runs/gemm_bf16_<timestamp>/`
- Correctness contract: Agent1 and Agent2 both use the copied strict `test.py`

## Problem Contract

- Operation: square matrix multiplication, `C = A @ B`
- Formal benchmark shape: `A: (4096, 4096)`, `B: (4096, 4096)`
- Input dtype: `torch.bfloat16` on CUDA
- Output dtype: exactly `torch.bfloat16`; FP32 output tensors are rejected
- Kernel requirement: support BF16 inputs and use FP32 accumulation internally
- Exposed function: `kernel_function(A, B)` returning a `(4096, 4096)` tensor

## Runtime Environment

- Remote kind: SSH devspace via `remote.toml`
- Current remote host at setup time: `devspace-dengxianglong-kernel-agent-552125`
- Required remote tools: `python3`, `torch`, `triton`, `numpy`, `rsync`, `ncu`
- Recommended env:
  - `KERNEL_WORKER_TIMEOUT_S=3600`
  - `LLM_TIMEOUT_S=3600`
  - `LLM_CENTER_API_KEY` loaded from `.env.local`

## Command

```bash
set -a && source .env.local && set +a
KERNEL_WORKER_TIMEOUT_S=3600 LLM_TIMEOUT_S=3600 \
  ./.venv/bin/python -u run_gemm_bf16_pipeline.py 2>&1 \
  | tee /tmp/gemm_run_glm52_bf16_552125.log
```
