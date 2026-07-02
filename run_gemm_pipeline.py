# GEMM pipeline: retry agent1 until it produces a verified square-GEMM kernel,
# then run agent2 (10-round optimization) on it. agent1 verification and agent2
# benchmark/NCU all run on the remote GPU (KERNEL_REMOTE_* env). Nothing in the
# harness is modified — the retry loop lives entirely here.
import subprocess
import sys
from pathlib import Path

from triton_kernel_agent import TritonKernelAgent

REPO = Path(__file__).resolve().parent
DST = REPO / "examples" / "optimize_gemm"
MAX_ATTEMPTS = 10

PROBLEM = (
    "Implement a Triton kernel for square matrix multiplication C = A @ B, "
    "where A and B are both (2048, 2048) float32 tensors on the GPU. "
    "Expose a function `kernel_function(A, B)` returning the (2048, 2048) "
    "result tensor. Use a tiled blocked-matmul Triton kernel with FP32 "
    "accumulation."
)


def run_agent1() -> bool:
    """Retry agent1 until a kernel verifies; write it to optimize_gemm/input.py."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n========== agent1 attempt {attempt}/{MAX_ATTEMPTS} ==========", flush=True)
        agent = TritonKernelAgent(num_workers=4, max_rounds=10, model_name="glm-5.2")
        result = agent.generate_kernel(problem_description=PROBLEM)
        print(f"[attempt {attempt}] success={result.get('success')}", flush=True)
        if result.get("success") and result.get("kernel_code"):
            (DST / "input.py").write_text(result["kernel_code"], encoding="utf-8")
            print(f"WROTE {DST/'input.py'} after {attempt} attempt(s)", flush=True)
            return True
        print(f"[attempt {attempt}] failed: {result.get('message')}", flush=True)
    print("AGENT1_EXHAUSTED: no verified kernel after max attempts", flush=True)
    return False


def run_agent2() -> int:
    print("\n========== agent2: 10-round optimization ==========", flush=True)
    cmd = [
        sys.executable,
        str(REPO / "examples" / "run_opt_manager.py"),
        "--kernel-dir", str(DST),
        "--strategy", "greedy_glm",
        "--max-rounds", "10",
    ]
    return subprocess.run(cmd, cwd=str(REPO)).returncode


if __name__ == "__main__":
    if not run_agent1():
        sys.exit(1)
    sys.exit(run_agent2())
