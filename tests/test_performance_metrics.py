"""GPU-free tests for workload FLOPs, MFU, and semantic roofline metrics."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent
_BENCH_DIR = (
    _ROOT / "triton_kernel_agent" / "opt_worker_component" / "benchmarking"
)
sys.path.insert(0, str(_BENCH_DIR))

from performance_metrics import (  # noqa: E402
    add_hardware_efficiency,
    compute_latency_performance,
    infer_kernel_math_mode,
    infer_pytorch_math_mode,
    resolve_workload_spec,
)

from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (  # noqa: E402
    get_gpu_specs,
)


def _load_workload_spec(relative_path: str) -> dict:
    """Execute only constants plus get_workload_spec from a problem file."""
    path = _ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "get_workload_spec"
        )
    ]
    namespace: dict = {}
    exec(compile(ast.Module(selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_workload_spec"]()


def test_gemm_workload_formula():
    spec = _load_workload_spec("examples/optimize_gemm/problem.py")

    assert spec["flops"] == 2 * 4096**3
    assert spec["minimum_io_elements"] == 3 * 4096**2
    assert spec["details"] == {"M": 4096, "N": 4096, "K": 4096}


def test_isolated_bf16_gemm_workload_formula():
    spec = _load_workload_spec("examples/optimize_gemm_bf16/problem.py")

    assert spec["flops"] == 2 * 4096**3
    assert spec["minimum_io_elements"] == 3 * 4096**2
    assert spec["details"] == {"M": 4096, "N": 4096, "K": 4096}


def test_fused_conv_workload_formula():
    spec = _load_workload_spec("examples/optimize_conv/problem.py")
    output_elements = 128 * 128 * 126 * 126

    assert spec["flops"] == 2 * output_elements * 64 * 3 * 3
    assert spec["epilogue_flops"] == 2 * output_elements
    assert spec["details"]["output_height"] == 126


def test_alexnet_conv_workload_formula():
    spec = _load_workload_spec("examples/optimize_conv_l1/problem.py")
    output_elements = 256 * 96 * 55 * 55

    assert spec["flops"] == 2 * output_elements * 3 * 11 * 11
    assert spec["epilogue_flops"] == output_elements
    assert spec["details"]["stride"] == 4
    assert spec["details"]["padding"] == 2


def test_bf16_mfu_and_semantic_roofline_use_dense_h100_peak():
    raw = _load_workload_spec("examples/optimize_gemm/problem.py")
    workload = resolve_workload_spec(raw, "torch.bfloat16")
    performance = compute_latency_performance(workload, time_ms=1.0)
    specs = get_gpu_specs("NVIDIA H100 NVL 94GB")

    result = add_hardware_efficiency(performance, workload, specs)

    assert result["achieved_tflops"] == pytest.approx(137.438953472)
    assert result["dense_peak_tflops"] == 835.5
    assert result["mfu_pct"] == pytest.approx(137.438953472 / 835.5 * 100)
    assert result["math_mode"] == "bf16_tensor_core"
    assert result["roofline_attainable_tflops"] == 835.5
    assert result["limiting_resource"] == "compute"


def test_fp32_requires_explicit_ieee_or_tf32_math_mode():
    raw = _load_workload_spec("examples/optimize_gemm/problem.py")
    workload = resolve_workload_spec(raw, "float32")
    performance = compute_latency_performance(workload, time_ms=1.0)
    specs = get_gpu_specs("NVIDIA H100 NVL 94GB")

    unknown = add_hardware_efficiency(performance, workload, specs)
    ieee = add_hardware_efficiency(
        performance, workload, specs, math_mode="ieee"
    )
    tf32 = add_hardware_efficiency(
        performance, workload, specs, math_mode="tf32"
    )

    assert unknown["mfu_pct"] is None
    assert any("math mode is unknown" in w for w in unknown["warnings"])
    assert ieee["dense_peak_tflops"] == 60.0
    assert tf32["dense_peak_tflops"] == 417.5


def test_math_mode_inference_uses_kernel_source_and_backend_flags():
    workload = {"dtype": "float32", "operation": "gemm"}
    conv_workload = {"dtype": "float32", "operation": "conv2d"}

    assert (
        infer_kernel_math_mode(
            "acc = tl.dot(a, b, input_precision='ieee')", "float32"
        )
        == "ieee"
    )
    assert infer_kernel_math_mode("acc = tl.dot(a, b)", "float32") == "tf32"
    assert (
        infer_pytorch_math_mode(
            workload,
            {"pytorch_config": {"matmul_allow_tf32": False}},
        )
        == "ieee"
    )
    assert (
        infer_pytorch_math_mode(
            conv_workload,
            {"pytorch_config": {"cudnn_allow_tf32": True}},
        )
        == "tf32"
    )


def test_missing_hook_and_unmodeled_gpu_degrade_without_guessing():
    workload = resolve_workload_spec(None, "bfloat16")
    performance = compute_latency_performance(workload, 1.0)
    assert workload["status"] == "unavailable"
    assert performance["achieved_tflops"] is None

    valid = resolve_workload_spec(
        {"operation": "gemm", "flops": 1000, "minimum_io_elements": 100},
        "bfloat16",
    )
    throughput = compute_latency_performance(valid, 1.0)
    rtx_specs = get_gpu_specs("NVIDIA RTX 4090")
    result = add_hardware_efficiency(throughput, valid, rtx_specs)
    assert result["achieved_tflops"] is not None
    assert result["mfu_pct"] is None
    assert any("dense peak unavailable" in w for w in result["warnings"])
