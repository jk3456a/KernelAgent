"""GPU-free tests for PyTorch baseline backend event classification."""

from __future__ import annotations

import ast
import json
import logging
import subprocess
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

from backend_probe import (  # noqa: E402
    classify_backend_events,
    get_backend_probe_targets,
    get_ncu_fallback_targets,
    merge_ncu_backend_evidence,
    parse_ncu_kernel_names,
)


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


def test_ncu_fallback_targets_only_the_workload_relevant_library():
    gemm = classify_backend_events(
        ["aten::mm"],
        ["cublasLt::matmul_kernel"],
    )
    assert gemm["libraries"]["cudnn"]["confidence"] == "medium"
    assert get_ncu_fallback_targets(gemm, {"operation": "gemm"}) == []

    conv = classify_backend_events(
        ["aten::conv2d"],
        ["some_generic_cuda_kernel"],
    )
    assert get_ncu_fallback_targets(
        conv, {"operation": "conv2d_relu_bias_add"}
    ) == ["cudnn"]


def test_backend_targets_fall_back_to_aten_events():
    assert get_backend_probe_targets(None, ["aten::bmm"]) == ["cublas"]
    assert get_backend_probe_targets({}, ["aten::convolution"]) == ["cudnn"]
    assert get_backend_probe_targets(
        {"operation": "custom"}, ["aten::convolution"]
    ) == ["cudnn"]


def test_parse_ncu_kernel_names_handles_preamble_units_and_quoted_commas(tmp_path):
    csv_path = tmp_path / "backend.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==WARNING== section files unavailable",
                '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"',
                '"","","","",""',
                '"1","void cublasLt::kernel<float, 128>()",'
                '"gpu__time_duration.sum","nsecond","42"',
                '"2","void cudnn::cnn::kernel(int, float)",'
                '"gpu__time_duration.sum","nsecond","21"',
                '"3","void cublasLt::kernel<float, 128>()",'
                '"gpu__time_duration.sum","nsecond","42"',
            ]
        ),
        encoding="utf-8",
    )

    assert parse_ncu_kernel_names(csv_path) == [
        "void cublasLt::kernel<float, 128>()",
        "void cudnn::cnn::kernel(int, float)",
    ]


def test_merge_ncu_positive_evidence_overrides_weak_negative_with_conflict():
    primary = classify_backend_events(["aten::relu"], ["relu_kernel"])
    primary.update(
        {
            "schema_version": 1,
            "method": "torch.profiler",
            "pytorch_config": {},
            "warnings": [],
        }
    )

    merged = merge_ncu_backend_evidence(
        primary,
        ncu_status="succeeded",
        target_libraries=["cublas"],
        ncu_kernel_names=["cublasLt::matmul_kernel"],
    )

    assert merged["schema_version"] == 2
    assert merged["method"] == "torch.profiler"
    assert merged["libraries"]["cublas"]["status"] == "detected"
    assert merged["libraries"]["cublas"]["confidence"] == "medium"
    assert merged["libraries"]["cublas"]["conflict"] is True
    assert merged["libraries"]["cublas"]["sources"] == [
        "torch.profiler",
        "ncu",
    ]
    assert merged["ncu"]["status"] == "succeeded"


def test_failed_ncu_probe_preserves_primary_result_and_adds_warning():
    primary = classify_backend_events(
        ["aten::matmul"],
        ["ampere_bf16_s16816gemm_bf16_128x128"],
    )
    primary.update(
        {
            "schema_version": 1,
            "method": "torch.profiler",
            "pytorch_config": {},
            "warnings": [],
        }
    )

    merged = merge_ncu_backend_evidence(
        primary,
        ncu_status="failed",
        target_libraries=["cublas"],
        warning="NCU backend refinement failed: permission denied",
    )

    assert merged["libraries"]["cublas"]["status"] == "detected"
    assert merged["libraries"]["cublas"]["confidence"] == "medium"
    assert merged["ncu"]["status"] == "failed"
    assert merged["methods"] == ["torch.profiler", "ncu"]
    assert "permission denied" in merged["warnings"][0]


def test_local_ncu_sidecar_uses_single_forward_mode(tmp_path, monkeypatch):
    from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
        Benchmark,
    )

    problem = tmp_path / "problem.py"
    problem.write_text("# problem\n", encoding="utf-8")
    backend = classify_backend_events(["aten::matmul"], [])
    backend.update(
        {
            "schema_version": 1,
            "method": "torch.profiler",
            "pytorch_config": {
                "cudnn_enabled": False,
                "matmul_allow_tf32": True,
            },
            "warnings": [],
        }
    )
    captured = {}

    def fake_run(args, *, cwd, capture_output, text, timeout):
        captured["args"] = args
        captured["cwd"] = cwd
        output_arg = next(arg for arg in args if arg.startswith("--log-file="))
        output_name = output_arg.split("=", 1)[1]
        (cwd / output_name).write_text(
            '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"1","cublasLt::matmul_kernel","gpu__time_duration.sum",'
            '"nsecond","42"\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(
        "triton_kernel_agent.opt_worker_component.benchmarking."
        "benchmark.shutil.which",
        lambda _: "/usr/local/bin/ncu",
    )
    monkeypatch.setattr(
        "triton_kernel_agent.opt_worker_component.benchmarking."
        "benchmark.subprocess.run",
        fake_run,
    )
    bench = Benchmark(
        logger=logging.getLogger("local-ncu"),
        artifacts_dir=tmp_path,
        benchmark_lock=threading.Lock(),
    )

    result = bench._maybe_refine_backend_with_ncu(
        backend=backend,
        workload={"operation": "gemm"},
        problem_file=problem,
        dtype_name="bfloat16",
        device_name="cuda:1",
        remote_cfg=None,
    )

    assert captured["args"][0] == "/usr/local/bin/ncu"
    assert "--profile-from-start=off" in captured["args"]
    assert "--ncu-baseline-once" in captured["args"]
    assert "--launch-count" not in captured["args"]
    device_index = captured["args"].index("--device")
    assert captured["args"][device_index + 1] == "cuda:1"
    config_index = captured["args"].index("--pytorch-config-json")
    config = json.loads(captured["args"][config_index + 1])
    assert config == {"cudnn_enabled": False, "matmul_allow_tf32": True}
    assert result["ncu"]["status"] == "succeeded"
    assert result["libraries"]["cublas"]["status"] == "detected"


def test_ncu_staging_failure_preserves_primary_evidence(tmp_path):
    from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
        Benchmark,
    )

    backend = classify_backend_events(["aten::matmul"], [])
    backend.update(
        {
            "schema_version": 1,
            "method": "torch.profiler",
            "pytorch_config": {},
            "warnings": [],
        }
    )
    bench = Benchmark(
        logger=logging.getLogger("ncu-staging-failure"),
        artifacts_dir=tmp_path,
        benchmark_lock=threading.Lock(),
    )

    result = bench._maybe_refine_backend_with_ncu(
        backend=backend,
        workload={"operation": "gemm"},
        problem_file=tmp_path / "missing_problem.py",
        dtype_name="bfloat16",
        device_name="cuda",
        remote_cfg=None,
    )

    assert result["ncu"]["status"] == "failed"
    assert result["libraries"]["cublas"]["status"] == "unknown"
    assert result["warnings"]


def test_nvidia_benchmarker_saves_backend_evidence(tmp_path, monkeypatch, caplog):
    from triton_kernel_agent.platform.nvidia import NvidiaBenchmarker

    backend = classify_backend_events(
        ["aten::mm"],
        ["cublasLt::matmul_kernel"],
    )
    backend.update({"warnings": [], "schema_version": 1})
    backend = merge_ncu_backend_evidence(
        backend,
        ncu_status="not_requested",
        target_libraries=["cublas"],
    )

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
    assert artifact["ncu"]["status"] == "not_requested"
    assert "cuBLAS=detected[high]" in caplog.text
    assert "NCU refinement: not_requested" in caplog.text


def test_optimization_manager_return_exposes_baseline_backend():
    manager_path = (
        Path(__file__).resolve().parent.parent
        / "triton_kernel_agent"
        / "opt_manager.py"
    )
    tree = ast.parse(manager_path.read_text(encoding="utf-8"))
    run_optimization = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_optimization"
    )
    result_dict = next(
        node.value
        for node in ast.walk(run_optimization)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "pytorch_baseline_backend"
            for key in node.value.keys
        )
    )
    backend_index = next(
        index
        for index, key in enumerate(result_dict.keys)
        if isinstance(key, ast.Constant)
        and key.value == "pytorch_baseline_backend"
    )
    backend_value = result_dict.values[backend_index]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "backend"
        for node in ast.walk(backend_value)
    )
