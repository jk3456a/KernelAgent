# Level1 conv pipeline: retry agent1 until it produces a verified standard
# Conv2d kernel (KernelBench level1 #50), then run agent2 (10-round
# optimization). All verification / benchmark / NCU runs on the remote GPU.
# Mirrors run_conv_pipeline.py but targets a plain (unfused) level1 conv.
import subprocess
import sys
import time
from pathlib import Path

from triton_kernel_agent import TritonKernelAgent

REPO = Path(__file__).resolve().parent
DST = REPO / "examples" / "optimize_conv_l1"
MAX_ATTEMPTS = 10

PROBLEM = (
    "Implement a Triton kernel for a standard 2D convolution (KernelBench "
    "level1 #50, the AlexNet first conv). The reference module is "
    "nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, "
    "padding=2, bias=True) applied to a (256, 3, 224, 224) input, producing a "
    "(256, 96, 55, 55) output. Expose kernel_function(x, weight, bias, stride, "
    "padding) where weight has shape (96, 3, 11, 11), bias has shape (96,), "
    "stride=4 and padding=2. Compute the convolution directly (no torch conv "
    "fallback) and return the result. Inputs are bfloat16."
)


def run_agent1() -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n===== agent1 attempt {attempt}/{MAX_ATTEMPTS} =====", flush=True)
        agent = TritonKernelAgent(num_workers=4, max_rounds=10, model_name="glm-5.2")
        result = agent.generate_kernel(problem_description=PROBLEM)
        print(f"[attempt {attempt}] success={result.get('success')}", flush=True)
        if result.get("success") and result.get("kernel_code"):
            (DST / "input.py").write_text(result["kernel_code"], encoding="utf-8")
            print(f"WROTE {DST/'input.py'} after {attempt} attempt(s)", flush=True)
            return True
        print(f"[attempt {attempt}] failed: {result.get('message')}", flush=True)
    print("AGENT1_EXHAUSTED", flush=True)
    return False


def run_agent2() -> int:
    print("\n===== agent2: 10-round optimization =====", flush=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = DST / "runs" / f"v1_{ts}"
    import os

    env = dict(os.environ)
    env["KERNEL_OPT_LOG_DIR"] = str(log_dir)
    print(f"log dir: {log_dir}", flush=True)
    cmd = [
        sys.executable,
        str(REPO / "examples" / "run_opt_manager.py"),
        "--kernel-dir", str(DST),
        "--strategy", "greedy_glm",
        "--max-rounds", "10",
    ]
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    if not run_agent1():
        sys.exit(1)
    sys.exit(run_agent2())
