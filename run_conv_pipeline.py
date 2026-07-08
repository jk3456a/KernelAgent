# Conv pipeline: retry agent1 until it produces a verified Conv2D+ReLU+BiasAdd
# kernel, then run agent2 (10-round optimization). All verification / benchmark
# / NCU runs on the remote GPU. Nothing in the harness is modified here.
import subprocess
import sys
import time
from pathlib import Path

from triton_kernel_agent import TritonKernelAgent

REPO = Path(__file__).resolve().parent
DST = REPO / "examples" / "optimize_conv"
MAX_ATTEMPTS = 10

PROBLEM = (
    "Implement a Triton kernel for a fused 2D convolution followed by ReLU and a "
    "per-output-channel bias add. The module is nn.Conv2d(in_channels=64, "
    "out_channels=128, kernel_size=3) (no padding, stride 1) applied to a "
    "(128, 64, 128, 128) float32 input, then torch.relu, then add a bias of "
    "shape (128, 1, 1) broadcast over the output. Expose kernel_function(x, "
    "conv_weight, conv_bias, extra_bias) returning the (128, 128, 126, 126) "
    "result. Fuse ReLU and bias-add into the conv epilogue."
)


def run_agent1() -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n===== agent1 attempt {attempt}/{MAX_ATTEMPTS} =====", flush=True)
        # 2 seeds, not 4: GLM-5.2 emits one long reasoning trace per requested
        # completion for a complex fused-conv kernel, and n=4 in a single request
        # took ~67 min (past the LLM hard-timeout wall). Two seeds keep the seed
        # phase under the timeout while still giving the verifier a choice.
        agent = TritonKernelAgent(num_workers=2, max_rounds=10, model_name="glm-5.2")
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
