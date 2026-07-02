# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the roofline efficiency fix for tensor-core kernels.

Bug being fixed: a naive GEMM saturates the shared-memory subsystem, so
``gpu__compute_memory_throughput`` reads ~76% while the tensor cores run at ~7%.
The old ``efficiency = max(compute_sol, memory_sol)`` reported ~79% and made the
optimizer stop early. For tensor-core kernels the efficiency must reflect tensor
pipe utilization, exposing the real headroom.
"""

from __future__ import annotations

from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer


# The exact NCU readout from the GEMM that triggered this fix.
_NAIVE_GEMM = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": 13.55,
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 76.43,
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": 5.13,
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 7.0,
}


def test_tensor_core_kernel_efficiency_uses_tensor_sol():
    """A 7% tensor-core kernel must NOT be reported as ~79% efficient."""
    analyzer = RooflineAnalyzer()
    result = analyzer.analyze(_NAIVE_GEMM)
    assert result.uses_tensor_cores is True
    # efficiency reflects tensor utilization, not the SMEM-subsystem 76%.
    assert result.efficiency_pct < 20, result.efficiency_pct
    assert abs(result.efficiency_pct - 7.0) < 1e-6
    assert result.at_roofline is False


def test_tensor_core_kernel_not_classified_memory_bound():
    """GEMM is compute(tensor)-bound; DRAM is 5% so it must not say 'memory'."""
    analyzer = RooflineAnalyzer()
    result = analyzer.analyze(_NAIVE_GEMM)
    assert result.bottleneck != "memory"


def test_high_tensor_util_is_at_roofline():
    """A genuinely good GEMM (high tensor SOL) should read as near-roofline."""
    analyzer = RooflineAnalyzer()
    good = dict(_NAIVE_GEMM)
    good["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"] = 80.0
    result = analyzer.analyze(good)
    assert result.efficiency_pct >= 80.0
    assert result.uses_tensor_cores is True


def test_non_tensor_kernel_unchanged():
    """Without tensor-core activity, fall back to the original max() behavior."""
    analyzer = RooflineAnalyzer()
    elementwise = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 0.0,
    }
    result = analyzer.analyze(elementwise)
    assert result.uses_tensor_cores is False
    assert result.efficiency_pct == 85.0  # max(compute, memory), unchanged
    assert result.bottleneck == "memory"
