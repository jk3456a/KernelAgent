"""Inspect which CUDA libraries execute a PyTorch eager baseline.

The profiler records execution evidence, while :func:`classify_backend_events`
is deliberately torch-free so its heuristics can be unit-tested on a control
machine without a CUDA PyTorch installation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_MATMUL_ATEN_OPS = (
    "aten::addmm",
    "aten::baddbmm",
    "aten::bmm",
    "aten::linear",
    "aten::matmul",
    "aten::mm",
)
_CONV_ATEN_OPS = (
    "aten::_convolution",
    "aten::conv1d",
    "aten::conv2d",
    "aten::conv3d",
    "aten::convolution",
)
_CUDNN_ATEN_MARKERS = ("cudnn_convolution", "cudnn::")
_NON_CUBLAS_KERNEL_MARKERS = ("cutlass", "inductor", "nvfuser", "triton")
_VENDOR_GEMM_RE = re.compile(
    r"(?:ampere|blackwell|hopper|turing|volta|sm\d+)[^\s]*gemm"
    r"|(?:^|[^a-z0-9])(?:[sdhcz])?gemm(?:[^a-z0-9]|$)"
    r"|xmma_gemm",
    re.IGNORECASE,
)
_MAX_RECORDED_NAMES = 32
_MAX_EVIDENCE_NAMES = 8


def _unique_names(names: Iterable[str], limit: int | None = None) -> list[str]:
    """Return non-empty names in first-seen order, capped for small artifacts."""
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        clean_name = str(name).strip()
        if not clean_name or clean_name in seen:
            continue
        seen.add(clean_name)
        result.append(clean_name)
        if limit is not None and len(result) >= limit:
            break
    return result


def _library_result(
    status: str,
    confidence: str,
    evidence: Iterable[str] = (),
) -> dict[str, Any]:
    detected = {"detected": True, "not_detected": False}.get(status)
    return {
        "status": status,
        "detected": detected,
        "confidence": confidence,
        "evidence": _unique_names(evidence, _MAX_EVIDENCE_NAMES),
    }


def classify_backend_events(
    aten_ops: Iterable[str],
    cuda_kernels: Iterable[str],
) -> dict[str, Any]:
    """Classify cuBLAS/cuDNN use from one eager-forward profiler trace.

    CUDA kernel names are the primary evidence. Dispatcher events are used only
    to disambiguate a vendor GEMM name or to report medium-confidence cuDNN
    dispatch when Kineto did not preserve a cuDNN kernel name.
    """
    aten_names = _unique_names(aten_ops)
    kernel_names = _unique_names(cuda_kernels)
    aten_lower = [name.lower() for name in aten_names]
    kernel_lower = [(name, name.lower()) for name in kernel_names]

    has_matmul = any(
        any(marker in name for marker in _MATMUL_ATEN_OPS) for name in aten_lower
    )
    has_conv = any(
        any(marker in name for marker in _CONV_ATEN_OPS) for name in aten_lower
    )

    explicit_cublas = [
        name for name, lower in kernel_lower if "cublas" in lower
    ]
    heuristic_cublas = [
        name
        for name, lower in kernel_lower
        if has_matmul
        and not any(marker in lower for marker in _NON_CUBLAS_KERNEL_MARKERS)
        and "cudnn" not in lower
        and _VENDOR_GEMM_RE.search(lower)
    ]
    cudnn_kernels = [
        name for name, lower in kernel_lower if "cudnn" in lower
    ]
    cudnn_dispatch = [
        name
        for name in aten_names
        if any(marker in name.lower() for marker in _CUDNN_ATEN_MARKERS)
    ]

    if explicit_cublas:
        cublas = _library_result("detected", "high", explicit_cublas)
    elif heuristic_cublas:
        cublas = _library_result("detected", "medium", heuristic_cublas)
    elif kernel_names and not has_matmul:
        cublas = _library_result("not_detected", "medium")
    else:
        cublas = _library_result("unknown", "none")

    if cudnn_kernels:
        cudnn = _library_result("detected", "high", cudnn_kernels)
    elif cudnn_dispatch and kernel_names:
        cudnn = _library_result("detected", "medium", cudnn_dispatch)
    elif kernel_names and not has_conv:
        cudnn = _library_result("not_detected", "medium")
    else:
        cudnn = _library_result("unknown", "none")

    return {
        "aten_ops": aten_names[:_MAX_RECORDED_NAMES],
        "cuda_kernels": kernel_names[:_MAX_RECORDED_NAMES],
        "libraries": {
            "cublas": cublas,
            "cudnn": cudnn,
        },
    }


def _unknown_probe_result(warning: str) -> dict[str, Any]:
    """Return a stable result when profiling cannot produce CUDA evidence."""
    result = classify_backend_events([], [])
    result.update(
        {
            "schema_version": 1,
            "method": "torch.profiler",
            "torch_version": None,
            "cuda_device": None,
            "pytorch_config": {},
            "warnings": [warning],
        }
    )
    return result


def inspect_pytorch_backend(model: Any, inputs: Iterable[Any]) -> dict[str, Any]:
    """Profile one eager forward and report cuBLAS/cuDNN execution evidence.

    This diagnostic must never make a valid benchmark fail. Any profiler or
    CUPTI failure is represented as an ``unknown`` result with a warning.
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:
        return _unknown_probe_result(
            f"torch profiler unavailable: {type(exc).__name__}: {exc}"
        )

    if not torch.cuda.is_available():
        result = _unknown_probe_result("CUDA is unavailable; backend not inspected")
        result["torch_version"] = str(torch.__version__)
        return result

    try:
        args = tuple(inputs)
        with torch.inference_mode():
            model(*args)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                model(*args)
                torch.cuda.synchronize()

        aten_ops: list[str] = []
        cuda_kernels: list[str] = []
        for event in prof.events():
            name = getattr(event, "name", None) or getattr(event, "key", None)
            if not name:
                continue
            device_type = getattr(event, "device_type", "")
            device_label = str(device_type).lower()
            device_name = str(getattr(device_type, "name", "")).lower()
            if device_name == "cuda" or "cuda" in device_label:
                cuda_kernels.append(str(name))
            elif str(name).startswith("aten::"):
                aten_ops.append(str(name))

        result = classify_backend_events(aten_ops, cuda_kernels)
        warnings: list[str] = []
        if not result["cuda_kernels"]:
            warnings.append(
                "torch.profiler captured no CUDA kernel names; backend is unknown"
            )
        cuda_backends = getattr(torch.backends, "cuda", None)
        matmul_backend = getattr(cuda_backends, "matmul", None)
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        result.update(
            {
                "schema_version": 1,
                "method": "torch.profiler",
                "torch_version": str(torch.__version__),
                "cuda_device": torch.cuda.get_device_name(),
                "pytorch_config": {
                    "cudnn_enabled": getattr(cudnn_backend, "enabled", None),
                    "cudnn_benchmark": getattr(cudnn_backend, "benchmark", None),
                    "cudnn_allow_tf32": getattr(
                        cudnn_backend, "allow_tf32", None
                    ),
                    "matmul_allow_tf32": getattr(
                        matmul_backend, "allow_tf32", None
                    ),
                    "usage_evidence": False,
                },
                "warnings": warnings,
            }
        )
        return result
    except Exception as exc:
        result = _unknown_probe_result(
            f"backend inspection failed: {type(exc).__name__}: {exc}"
        )
        result["torch_version"] = str(torch.__version__)
        try:
            result["cuda_device"] = torch.cuda.get_device_name()
        except Exception:
            pass
        return result
