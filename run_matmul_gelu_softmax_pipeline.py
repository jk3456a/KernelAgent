"""Generate and optimize a fused BF16 Matmul+GELU+Softmax kernel (KernelBench L2 #99)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from triton_kernel_agent import TritonKernelAgent
from utils.progress import configure_progress


REPO = Path(__file__).resolve().parent
DST = REPO / "examples" / "optimize_matmul_gelu_softmax"
RUNS_DIR = DST / "runs"
RUN_PREFIX = "matmul_gelu_softmax"

MODEL_NAME = "glm-5.2"
AGENT1_WORKERS = 4
AGENT1_MAX_ROUNDS = 10
MAX_ATTEMPTS = 10
AGENT2_MAX_ROUNDS = 10
DEFAULT_AGENT2_STRATEGY = "greedy_glm_rag"
AGENT2_STRATEGY_ENV = "MATMUL_GELU_SOFTMAX_AGENT2_STRATEGY"

PROBLEM = (
    "Implement a fused Triton kernel for y = softmax(gelu(x @ W.T + b), dim=1), "
    "matching KernelBench level2 #99 (Matmul_GELU_Softmax). x is a (1024, 8192) "
    "torch.bfloat16 CUDA tensor, W is a (8192, 8192) torch.bfloat16 weight in "
    "nn.Linear layout, and b is a (8192,) torch.bfloat16 bias. "
    "kernel_function(x, W, b) must return y with shape (1024, 8192) and dtype "
    "torch.bfloat16. Use a tiled blocked matmul with FP32 accumulation, apply "
    "exact (erf-based) GELU to the FP32 accumulator, then a numerically stable "
    "row softmax over dim=1 (subtract the row max) in FP32, casting to BF16 "
    "once when storing the output. Prefer fusing the whole computation into as "
    "few Triton kernels as possible."
)


def create_run_dir() -> Path:
    """Create an isolated timestamped directory for one fused pipeline run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{RUN_PREFIX}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{RUN_PREFIX}_{timestamp}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    for filename in ("problem.py", "test.py"):
        shutil.copy2(DST / filename, run_dir / filename)

    print(f"RUN_DIR {run_dir}", flush=True)
    configure_progress(run_dir)
    return run_dir


def run_agent1(run_dir: Path) -> bool:
    """Generate a kernel that passes the same strict test used by agent2."""
    test_code = (run_dir / "test.py").read_text(encoding="utf-8")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(
            f"\n========== agent1 attempt {attempt}/{MAX_ATTEMPTS} ==========",
            flush=True,
        )
        agent = TritonKernelAgent(
            num_workers=AGENT1_WORKERS,
            max_rounds=AGENT1_MAX_ROUNDS,
            log_dir=str(run_dir / "agent1_logs"),
            model_name=MODEL_NAME,
        )
        result = agent.generate_kernel(
            problem_description=PROBLEM,
            test_code=test_code,
            generate_default_test=False,
        )
        print(f"[attempt {attempt}] success={result.get('success')}", flush=True)

        if result.get("success") and result.get("kernel_code"):
            input_file = run_dir / "input.py"
            input_file.write_text(result["kernel_code"], encoding="utf-8")
            print(f"WROTE {input_file} after {attempt} attempt(s)", flush=True)
            return True

        print(f"[attempt {attempt}] failed: {result.get('message')}", flush=True)

    print("AGENT1_EXHAUSTED: no verified kernel after max attempts", flush=True)
    return False


def run_agent2(run_dir: Path) -> int:
    """Optimize the verified kernel with the configured agent2 strategy."""
    strategy = os.environ.get(AGENT2_STRATEGY_ENV, DEFAULT_AGENT2_STRATEGY)
    print(
        f"\n========== agent2: {strategy}, "
        f"{AGENT2_MAX_ROUNDS}-round optimization ==========",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(REPO / "examples" / "run_opt_manager.py"),
        "--kernel-dir",
        str(run_dir),
        "--strategy",
        strategy,
        "--max-rounds",
        str(AGENT2_MAX_ROUNDS),
    ]
    env = os.environ.copy()
    env["KERNEL_OPT_LOG_DIR"] = str(run_dir / "opt_manager_logs")
    result = subprocess.run(cmd, cwd=str(REPO), env=env)
    return result.returncode


def main() -> int:
    run_dir = create_run_dir()
    if not run_agent1(run_dir):
        return 1
    return run_agent2(run_dir)


if __name__ == "__main__":
    sys.exit(main())
