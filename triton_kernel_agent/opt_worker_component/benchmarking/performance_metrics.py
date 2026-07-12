"""Formula-based performance metrics for benchmarked semantic workloads.

The helpers in this module are intentionally torch-free. GPU subprocesses use
them to derive workload-level throughput from latency, while the control
process later adds dtype-aware hardware peaks and MFU.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "fp32": "float32",
    "float": "float32",
    "float32": "float32",
    "torch.float32": "float32",
}
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


def normalize_dtype_name(dtype: Any) -> str | None:
    """Return a stable dtype name accepted by the MFU peak selector."""
    if dtype is None:
        return None
    return _DTYPE_ALIASES.get(str(dtype).strip().lower())


def resolve_workload_spec(
    spec: Mapping[str, Any] | None,
    dtype: Any,
) -> dict[str, Any]:
    """Normalize an optional problem workload specification.

    ``minimum_io_elements`` is a semantic lower bound: it counts each logical
    input/weight/output once and deliberately excludes implementation-specific
    temporary tensors.
    """
    dtype_name = normalize_dtype_name(dtype)
    if not isinstance(spec, Mapping):
        return {
            "status": "unavailable",
            "dtype": dtype_name,
            "warnings": ["problem.py does not define get_workload_spec()"],
        }

    warnings: list[str] = []
    try:
        flops = int(spec["flops"])
        minimum_io_elements = int(spec["minimum_io_elements"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {
            "status": "unavailable",
            "dtype": dtype_name,
            "warnings": [f"invalid workload spec: {type(exc).__name__}: {exc}"],
        }

    if flops <= 0 or minimum_io_elements <= 0:
        return {
            "status": "unavailable",
            "dtype": dtype_name,
            "warnings": ["workload flops and minimum_io_elements must be positive"],
        }

    bytes_per_element = _DTYPE_BYTES.get(dtype_name)
    if bytes_per_element is None:
        minimum_io_bytes = None
        arithmetic_intensity = None
        warnings.append(f"unsupported benchmark dtype: {dtype}")
    else:
        minimum_io_bytes = minimum_io_elements * bytes_per_element
        arithmetic_intensity = flops / minimum_io_bytes

    workload = {
        "status": "available",
        "operation": str(spec.get("operation", "unknown")),
        "dtype": dtype_name,
        "flops": flops,
        "epilogue_flops": int(spec.get("epilogue_flops", 0)),
        "minimum_io_elements": minimum_io_elements,
        "minimum_io_bytes": minimum_io_bytes,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "flop_convention": str(spec.get("flop_convention", "2_per_fma")),
        "flop_scope": str(spec.get("flop_scope", "primary_tensor_math")),
        "io_scope": str(spec.get("io_scope", "semantic_minimum")),
        "warnings": warnings,
    }
    details = spec.get("details")
    if isinstance(details, Mapping):
        workload["details"] = dict(details)
    return workload


def compute_latency_performance(
    workload: Mapping[str, Any] | None,
    time_ms: float | int | None,
) -> dict[str, Any]:
    """Compute achieved throughput from a resolved workload and latency."""
    warnings = list((workload or {}).get("warnings", []))
    result: dict[str, Any] = {
        "status": "unavailable",
        "time_ms": None,
        "achieved_tflops": None,
        "mfu_pct": None,
        "dense_peak_tflops": None,
        "roofline_attainable_tflops": None,
        "roofline_utilization_pct": None,
        "limiting_resource": None,
        "math_mode": None,
        "peak_source": None,
        "gpu_name": None,
        "warnings": warnings,
    }
    if not isinstance(workload, Mapping) or workload.get("status") != "available":
        if not warnings:
            result["warnings"].append("workload metrics unavailable")
        return result

    try:
        latency = float(time_ms)
    except (TypeError, ValueError, OverflowError):
        result["warnings"].append("latency is unavailable")
        return result
    if not math.isfinite(latency) or latency <= 0:
        result["warnings"].append("latency must be finite and positive")
        return result

    flops = int(workload["flops"])
    achieved_tflops = flops / (latency * 1e9)
    result.update(
        {
            "status": "throughput_only",
            "time_ms": latency,
            "achieved_tflops": achieved_tflops,
        }
    )
    return result


def infer_kernel_math_mode(kernel_source: str, dtype: Any) -> str | None:
    """Best-effort FP32 Triton dot precision inference from candidate source."""
    if normalize_dtype_name(dtype) != "float32":
        return None
    if re.search(
        r"input_precision\s*=\s*['\"]ieee['\"]",
        kernel_source,
        re.IGNORECASE,
    ):
        return "ieee"
    if re.search(
        r"input_precision\s*=\s*['\"]tf32",
        kernel_source,
        re.IGNORECASE,
    ):
        return "tf32"
    if "tl.dot" in kernel_source:
        return "tf32"
    return "ieee"


def infer_pytorch_math_mode(
    workload: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> str | None:
    """Resolve FP32 eager math mode from recorded PyTorch policy flags."""
    if workload.get("dtype") != "float32":
        return None
    config = backend.get("pytorch_config", {})
    operation = str(workload.get("operation", ""))
    flag_name = (
        "cudnn_allow_tf32" if operation.startswith("conv") else "matmul_allow_tf32"
    )
    allow_tf32 = config.get(flag_name)
    if isinstance(allow_tf32, bool):
        return "tf32" if allow_tf32 else "ieee"
    return None


def _select_dense_peak(
    dtype_name: str | None,
    math_mode: str | None,
    gpu_specs: Mapping[str, Any],
) -> tuple[float | None, str | None, str | None]:
    if gpu_specs.get("mfu_supported") is False:
        return None, None, None
    if dtype_name == "bfloat16":
        key, mode = "peak_bf16_tflops", "bf16_tensor_core"
    elif dtype_name == "float16":
        key, mode = "peak_fp16_tflops", "fp16_tensor_core"
    elif dtype_name == "float32" and math_mode == "tf32":
        key, mode = "peak_tf32_tflops", "tf32_tensor_core"
    elif dtype_name == "float32" and math_mode == "ieee":
        key, mode = "peak_fp32_tflops", "fp32_ieee"
    else:
        return None, None, None

    try:
        peak = float(gpu_specs[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, key, mode
    if not math.isfinite(peak) or peak <= 0:
        return None, key, mode
    return peak, key, mode


def add_hardware_efficiency(
    performance: Mapping[str, Any],
    workload: Mapping[str, Any] | None,
    gpu_specs: Mapping[str, Any] | None,
    *,
    math_mode: str | None = None,
) -> dict[str, Any]:
    """Add dense-peak MFU and semantic roofline utilization."""
    result = dict(performance)
    result["warnings"] = list(performance.get("warnings", []))
    if performance.get("achieved_tflops") is None:
        return result
    if not isinstance(gpu_specs, Mapping):
        result["warnings"].append("GPU specifications unavailable; MFU not computed")
        return result

    dtype_name = (workload or {}).get("dtype")
    peak, peak_key, resolved_mode = _select_dense_peak(
        dtype_name, math_mode, gpu_specs
    )
    if peak is None:
        if dtype_name == "float32" and math_mode not in ("ieee", "tf32"):
            result["warnings"].append(
                "FP32 math mode is unknown; choose ieee or tf32 before computing MFU"
            )
        else:
            result["warnings"].append(
                f"dense peak unavailable for dtype={dtype_name}, math_mode={math_mode}"
            )
        return result

    achieved = float(performance["achieved_tflops"])
    result.update(
        {
            "status": "available",
            "mfu_pct": achieved / peak * 100.0,
            "dense_peak_tflops": peak,
            "math_mode": resolved_mode,
            "peak_source": f"gpu_specs.{peak_key}",
            "gpu_name": gpu_specs.get("name"),
        }
    )

    intensity = (workload or {}).get("arithmetic_intensity_flops_per_byte")
    bandwidth = gpu_specs.get("peak_memory_bw_gbps")
    if intensity is not None and bandwidth is not None:
        memory_roof_tflops = float(intensity) * float(bandwidth) / 1000.0
        attainable = min(peak, memory_roof_tflops)
        result.update(
            {
                "roofline_attainable_tflops": attainable,
                "roofline_utilization_pct": achieved / attainable * 100.0,
                "limiting_resource": (
                    "compute" if peak <= memory_roof_tflops else "memory"
                ),
            }
        )

    if result["mfu_pct"] > 100.0:
        result["warnings"].append(
            "MFU exceeds 100%; verify FLOP count, latency, GPU peak, and math mode"
        )
    roofline_util = result.get("roofline_utilization_pct")
    if roofline_util is not None and roofline_util > 100.0:
        result["warnings"].append(
            "roofline utilization exceeds 100%; semantic minimum IO is not a "
            "measured traffic model"
        )
    return result


def format_performance_summary(performance: Mapping[str, Any] | None) -> str:
    """Render a compact log line without raising on partial metrics."""
    if not isinstance(performance, Mapping):
        return "performance unavailable"

    def _fmt(key: str, suffix: str = "") -> str:
        value = performance.get(key)
        return f"{float(value):.2f}{suffix}" if value is not None else "unknown"

    return (
        f"achieved={_fmt('achieved_tflops')} TFLOPS, "
        f"MFU={_fmt('mfu_pct', '%')}, "
        f"attainable={_fmt('roofline_attainable_tflops')} TFLOPS, "
        f"attainable_util={_fmt('roofline_utilization_pct', '%')}"
    )
