"""GPU-free tests for PyTorch baseline backend event classification."""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path


_BENCH_DIR = (
    Path(__file__).resolve().parent.parent
    / "triton_kernel_agent"
    / "opt_worker_component"
    / "benchmarking"
)
sys.path.insert(0, str(_BENCH_DIR))

from backend_probe import classify_backend_events  # noqa: E402


def test_explicit_library_kernel_names_are_high_confidence():
    result = classify_backend_events(
        ["aten::mm", "aten::cudnn_convolution"],
        [
            "cublasLt::matmul_kernel",
            "void cudnn::cnn::implicit_convolve_sgemm()",
        ],
    )

    assert result["libraries"]["cublas"] == {
        "status": "detected",
        "detected": True,
        "confidence": "high",
        "evidence": ["cublasLt::matmul_kernel"],
    }
    assert result["libraries"]["cudnn"]["status"] == "detected"
    assert result["libraries"]["cudnn"]["confidence"] == "high"


def test_vendor_gemm_name_needs_matmul_dispatch():
    kernel = "ampere_bf16_s16816gemm_bf16_128x128"

    with_matmul = classify_backend_events(["aten::matmul", "aten::mm"], [kernel])
    without_matmul = classify_backend_events(["aten::relu"], [kernel])

    assert with_matmul["libraries"]["cublas"]["status"] == "detected"
    assert with_matmul["libraries"]["cublas"]["confidence"] == "medium"
    assert without_matmul["libraries"]["cublas"]["status"] == "not_detected"


def test_aten_ops_or_backend_config_are_not_execution_evidence():
    result = classify_backend_events(
        ["aten::mm", "aten::convolution", "aten::cudnn_convolution"],
        [],
    )

    assert result["libraries"]["cublas"]["status"] == "unknown"
    assert result["libraries"]["cublas"]["detected"] is None
    assert result["libraries"]["cudnn"]["status"] == "unknown"
    assert result["libraries"]["cudnn"]["detected"] is None


def test_cutlass_and_triton_names_do_not_count_as_cublas():
    result = classify_backend_events(
        ["aten::matmul"],
        [
            "void cutlass::Kernel<cutlass_tensorop_gemm>()",
            "triton_red_fused_kernel",
        ],
    )

    assert result["libraries"]["cublas"]["status"] == "unknown"
    assert result["libraries"]["cublas"]["evidence"] == []


def test_empty_trace_is_unknown_for_both_libraries():
    result = classify_backend_events([], [])

    assert result["cuda_kernels"] == []
    assert result["libraries"]["cublas"]["status"] == "unknown"
    assert result["libraries"]["cudnn"]["status"] == "unknown"


def test_nvidia_benchmarker_saves_backend_evidence(tmp_path, monkeypatch, caplog):
    from triton_kernel_agent.platform.nvidia import NvidiaBenchmarker

    backend = classify_backend_events(
        ["aten::mm"],
        ["cublasLt::matmul_kernel"],
    )
    backend.update({"warnings": [], "schema_version": 1})

    class FakeBenchmark:
        @staticmethod
        def benchmark_pytorch(problem_file, kernel_file=None):
            return {"time_ms": 1.25, "backend": backend}

    problem = tmp_path / "problem.py"
    problem.write_text("# problem\n", encoding="utf-8")
    benchmarker = NvidiaBenchmarker(
        log_dir=tmp_path,
        logger=logging.getLogger("backend-probe-test"),
        benchmark_lock=threading.Lock(),
    )
    monkeypatch.setattr(benchmarker, "_get_benchmarker", FakeBenchmark)

    with caplog.at_level(logging.INFO):
        elapsed = benchmarker.benchmark_reference(problem)

    artifact = json.loads(
        (tmp_path / "artifacts" / "pytorch_backend.json").read_text()
    )
    assert elapsed == 1.25
    assert artifact["problem"] == str(problem)
    assert artifact["libraries"]["cublas"]["status"] == "detected"
    assert "cuBLAS=detected[high]" in caplog.text
