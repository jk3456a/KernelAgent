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

"""Static check that kernel_subprocess's internal helper calls stay in sync.

kernel_subprocess.py imports torch/triton at module top level, so it cannot be
imported on this (GPU-less) control machine. But signature drift between a
helper's definition and its call sites -- e.g. changing _run_once(fn, inputs,
init, name) to _run_once(invoke, name) and forgetting the --baseline call site
-- is a real bug that silently broke the PyTorch baseline. Parse the AST and
assert every internal call passes the exact positional arity the definition
requires.
"""

import ast
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parent.parent
    / "triton_kernel_agent"
    / "opt_worker_component"
    / "benchmarking"
    / "kernel_subprocess.py"
)


def _positional_arity(func: ast.FunctionDef) -> int:
    """Number of positional params without defaults (required positional)."""
    args = func.args
    n = len(args.args)
    n_defaults = len(args.defaults)
    return n - n_defaults


def test_internal_helper_calls_match_signatures():
    tree = ast.parse(_SRC.read_text())
    defs = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    checked = 0
    for helper in ("_run_once", "_benchmark"):
        assert helper in defs, f"{helper} not found"
        func = defs[helper]
        required = _positional_arity(func)
        total = len(func.args.args)
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == helper
            ):
                # No *args unpacking expected in these call sites.
                assert not any(
                    isinstance(a, ast.Starred) for a in call.args
                ), f"{helper} called with *args unpacking"
                n = len(call.args)
                assert required <= n <= total, (
                    f"{helper} called with {n} positional args at line "
                    f"{call.lineno}; signature needs {required}..{total}"
                )
                checked += 1
    assert checked >= 3, f"expected >=3 helper call sites, found {checked}"


def test_ncu_baseline_mode_runs_one_forward_before_normal_benchmark_path():
    tree = ast.parse(_SRC.read_text())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    helper = functions["_run_ncu_baseline_once"]
    model_forwards = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "model"
    ]
    profiler_calls = [
        node.func.attr
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"cudaProfilerStart", "cudaProfilerStop"}
    ]
    set_device_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_device"
    ]

    assert len(model_forwards) == 1
    assert profiler_calls.count("cudaProfilerStart") == 1
    assert profiler_calls.count("cudaProfilerStop") == 1
    assert len(set_device_calls) == 1

    main = functions["main"]
    ncu_dispatch = next(
        node
        for node in main.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "ncu_baseline_once"
    )
    assert any(isinstance(node, ast.Return) for node in ncu_dispatch.body)
    normal_load_line = min(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_problem"
    )
    assert ncu_dispatch.lineno < normal_load_line
