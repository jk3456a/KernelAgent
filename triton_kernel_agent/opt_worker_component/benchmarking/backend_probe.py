"""Inspect which CUDA libraries execute a PyTorch eager baseline.

The profiler records execution evidence, while :func:`classify_backend_events`
is deliberately torch-free so its heuristics can be unit-tested on a control
machine without a CUDA PyTorch installation.
"""

from __future__ import annotations

import copy
import csv
import re
from collections.abc import Iterable
from pathlib import Path
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
_CUBLAS_OPERATION_MARKERS = ("gemm", "matmul", "linear", "bmm", "addmm")
_CUDNN_OPERATION_MARKERS = ("conv", "convolution")
_CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_NCU_STATUSES = {"not_requested", "succeeded", "inconclusive", "failed"}


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
    *,
    target_libraries: Iterable[str] = (),
) -> dict[str, Any]:
    """Classify cuBLAS/cuDNN use from one eager-forward profiler trace.

    CUDA kernel names are the primary evidence. Dispatcher events are used only
    to disambiguate a vendor GEMM name or to report medium-confidence cuDNN
    dispatch when Kineto did not preserve a cuDNN kernel name.
    """
    aten_names = _unique_names(aten_ops)
    kernel_names = _unique_names(cuda_kernels)
    targets = set(_unique_names(target_libraries))
    aten_lower = [name.lower() for name in aten_names]
    kernel_lower = [(name, name.lower()) for name in kernel_names]

    has_matmul = "cublas" in targets or any(
        any(marker in name for marker in _MATMUL_ATEN_OPS) for name in aten_lower
    )
    has_conv = "cudnn" in targets or any(
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


def get_backend_probe_targets(
    workload: dict[str, Any] | None,
    aten_ops: Iterable[str],
) -> list[str]:
    """Return workload-relevant CUDA libraries in stable order.

    Workload metadata is preferred because profiler failures can leave no ATen
    events. ATen events are a fallback for problems without an operation label.
    """
    operation = ""
    if isinstance(workload, dict):
        operation = str(workload.get("operation") or "").strip().lower()

    targets: list[str] = []
    if operation:
        if any(marker in operation for marker in _CUBLAS_OPERATION_MARKERS):
            targets.append("cublas")
        if any(marker in operation for marker in _CUDNN_OPERATION_MARKERS):
            targets.append("cudnn")
        if targets:
            return targets

    aten_lower = [str(name).lower() for name in aten_ops]
    if any(
        any(marker in name for marker in _MATMUL_ATEN_OPS) for name in aten_lower
    ):
        targets.append("cublas")
    if any(
        any(marker in name for marker in _CONV_ATEN_OPS) for name in aten_lower
    ):
        targets.append("cudnn")
    return targets


def get_ncu_fallback_targets(
    backend: dict[str, Any] | None,
    workload: dict[str, Any] | None,
) -> list[str]:
    """Return relevant libraries whose primary evidence needs NCU refinement."""
    if not isinstance(backend, dict):
        return []

    targets = get_backend_probe_targets(workload, backend.get("aten_ops", []))
    libraries = backend.get("libraries", {})
    fallback_targets: list[str] = []
    for library in targets:
        info = libraries.get(library, {}) if isinstance(libraries, dict) else {}
        status = info.get("status", "unknown")
        confidence = info.get("confidence", "none")
        if status == "unknown" or confidence in {"none", "low", "medium"}:
            fallback_targets.append(library)
    return fallback_targets


def parse_ncu_kernel_names(csv_path: str | Path) -> list[str]:
    """Extract every unique kernel name from an NCU raw CSV artifact."""
    path = Path(csv_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_index: int | None = None
    for index, line in enumerate(lines):
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        normalized = {field.strip().strip('"') for field in fields}
        if "ID" in normalized and "Kernel Name" in normalized:
            header_index = index
            break

    if header_index is None:
        raise ValueError("NCU CSV does not contain an ID/Kernel Name header")

    reader = csv.DictReader(lines[header_index:])
    names: list[str] = []
    for row in reader:
        name = str(row.get("Kernel Name") or "").strip()
        if not name or name == "Kernel Name":
            continue
        names.append(name)
    return _unique_names(names)


def _confidence_max(first: str, second: str) -> str:
    return max(
        (first, second),
        key=lambda value: _CONFIDENCE_RANK.get(value, 0),
    )


def _merge_library_evidence(
    primary: dict[str, Any],
    ncu: dict[str, Any],
) -> dict[str, Any]:
    """Merge two name-based observations without inventing negative evidence."""
    primary_status = primary.get("status", "unknown")
    ncu_status = ncu.get("status", "unknown")
    primary_confidence = primary.get("confidence", "none")
    ncu_confidence = ncu.get("confidence", "none")
    evidence = _unique_names(
        [
            *primary.get("evidence", []),
            *ncu.get("evidence", []),
        ],
        _MAX_EVIDENCE_NAMES,
    )
    conflict = {primary_status, ncu_status} == {"detected", "not_detected"}

    if ncu_status == "unknown":
        merged = copy.deepcopy(primary)
    elif primary_status == "unknown":
        merged = copy.deepcopy(ncu)
    elif primary_status == ncu_status:
        merged = copy.deepcopy(primary)
        merged["confidence"] = _confidence_max(
            primary_confidence, ncu_confidence
        )
        merged["detected"] = {
            "detected": True,
            "not_detected": False,
        }.get(primary_status)
    elif conflict:
        detected_source = primary if primary_status == "detected" else ncu
        detected_confidence = detected_source.get("confidence", "none")
        if detected_confidence == "high" and primary_status == "detected":
            merged = copy.deepcopy(primary)
        elif ncu_status == "detected":
            merged = _library_result("detected", "medium", evidence)
        else:
            merged = _library_result("unknown", "none", evidence)
    else:
        merged = copy.deepcopy(primary)

    merged["evidence"] = evidence
    merged["sources"] = ["torch.profiler", "ncu"]
    merged["conflict"] = conflict
    return merged


def merge_ncu_backend_evidence(
    primary: dict[str, Any],
    *,
    ncu_status: str,
    target_libraries: Iterable[str],
    ncu_kernel_names: Iterable[str] = (),
    warning: str | None = None,
) -> dict[str, Any]:
    """Attach optional NCU evidence while preserving the v1 top-level contract."""
    if ncu_status not in _NCU_STATUSES:
        raise ValueError(f"Unsupported NCU backend status: {ncu_status}")

    result = copy.deepcopy(primary)
    targets = _unique_names(target_libraries)
    kernel_names = _unique_names(ncu_kernel_names)
    ncu_classification = classify_backend_events(
        result.get("aten_ops", []),
        kernel_names,
        target_libraries=targets,
    )
    ncu_libraries = {
        library: ncu_classification["libraries"].get(
            library, _library_result("unknown", "none")
        )
        for library in targets
    }

    libraries = result.setdefault("libraries", {})
    for library, info in libraries.items():
        normalized = copy.deepcopy(info)
        normalized.setdefault("sources", ["torch.profiler"])
        normalized.setdefault("conflict", False)
        libraries[library] = normalized

    if ncu_status == "succeeded":
        for library in targets:
            primary_info = libraries.get(
                library, _library_result("unknown", "none")
            )
            libraries[library] = _merge_library_evidence(
                primary_info,
                ncu_libraries[library],
            )

    warnings = list(result.get("warnings", []))
    if warning:
        warnings.append(warning)

    methods = ["torch.profiler"]
    if ncu_status != "not_requested":
        methods.append("ncu")
    result.update(
        {
            "schema_version": 2,
            "methods": methods,
            "ncu": {
                "status": ncu_status,
                "target_libraries": targets,
                "cuda_kernels": kernel_names[:_MAX_RECORDED_NAMES],
                "libraries": ncu_libraries,
                "warnings": [warning] if warning else [],
            },
            "warnings": _unique_names(warnings),
        }
    )
    return result


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
