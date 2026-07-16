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

"""GPU- and LLM-free tests for the fused Matmul+GELU+Softmax pipeline."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import run_matmul_gelu_softmax_pipeline as pipeline


_ROOT = Path(__file__).resolve().parent.parent
_TASK_DIR = _ROOT / "examples" / "optimize_matmul_gelu_softmax"
_PROBLEM_FILE = _TASK_DIR / "problem.py"
_TEST_FILE = _TASK_DIR / "test.py"
_BENCHMARKING_DIR = (
    _ROOT / "triton_kernel_agent" / "opt_worker_component" / "benchmarking"
)

_BATCH = 1024
_IN_FEATURES = 8192
_OUT_FEATURES = 8192


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_problem_generates_bf16_input():
    get_inputs = _function_node(_PROBLEM_FILE, "get_inputs")
    rand_calls = [
        node
        for node in ast.walk(get_inputs)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.rand"
    ]

    assert [[ast.unparse(arg) for arg in call.args] for call in rand_calls] == [
        ["batch_size", "in_features"],
    ]
    assert [
        ast.unparse(next(kw.value for kw in call.keywords if kw.arg == "dtype"))
        for call in rand_calls
    ] == ["torch.bfloat16"]


def test_problem_workload_spec_values():
    problem = _load_module(_PROBLEM_FILE, "matmul_gelu_softmax_problem")
    spec = problem.get_workload_spec()

    assert spec["operation"] == "matmul_gelu_softmax"
    assert spec["flops"] == 2 * _BATCH * _IN_FEATURES * _OUT_FEATURES
    assert spec["epilogue_flops"] == 7 * _BATCH * _OUT_FEATURES
    assert spec["minimum_io_elements"] == (
        _BATCH * _IN_FEATURES
        + _IN_FEATURES * _OUT_FEATURES
        + _OUT_FEATURES
        + _BATCH * _OUT_FEATURES
    )
    assert spec["details"]["softmax_dim"] == 1


def test_strict_test_rejects_wrong_output_shape_dtype_and_row_sums():
    validate_output = _function_node(_TEST_FILE, "validate_output")
    rendered = ast.unparse(validate_output)

    assert "isinstance(kernel_output, torch.Tensor)" in rendered
    assert "tuple(kernel_output.shape) != expected_shape" in rendered
    assert "kernel_output.dtype != torch.bfloat16" in rendered
    assert "sum(dim=1)" in rendered

    allclose = next(
        node
        for node in ast.walk(validate_output)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.allclose"
    )
    tolerances = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in allclose.keywords
    }
    assert tolerances == {"rtol": 1e-2, "atol": 1e-2}


def test_strict_test_defers_kernel_imports_into_test_function():
    tree = ast.parse(_TEST_FILE.read_text(encoding="utf-8"))
    top_level_imports = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert top_level_imports <= {"sys", "torch"}


def test_create_run_dir_copies_isolated_contract(tmp_path, monkeypatch):
    source_dir = tmp_path / "optimize_matmul_gelu_softmax"
    source_dir.mkdir()
    (source_dir / "problem.py").write_text("problem contract", encoding="utf-8")
    (source_dir / "test.py").write_text("strict test", encoding="utf-8")
    configured = []

    monkeypatch.setattr(pipeline, "DST", source_dir)
    monkeypatch.setattr(pipeline, "RUNS_DIR", source_dir / "runs")
    monkeypatch.setattr(pipeline, "configure_progress", configured.append)

    run_dir = pipeline.create_run_dir()

    assert run_dir.parent == source_dir / "runs"
    assert run_dir.name.startswith("matmul_gelu_softmax_")
    assert (run_dir / "problem.py").read_text(encoding="utf-8") == "problem contract"
    assert (run_dir / "test.py").read_text(encoding="utf-8") == "strict test"
    assert configured == [run_dir]


def test_agent1_uses_strict_static_test_and_writes_kernel(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    strict_test = "STRICT FUSED BF16 TEST"
    (run_dir / "test.py").write_text(strict_test, encoding="utf-8")
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate_kernel(self, **kwargs):
            captured["generate"] = kwargs
            return {"success": True, "kernel_code": "generated kernel"}

    monkeypatch.setattr(pipeline, "TritonKernelAgent", FakeAgent)

    assert pipeline.run_agent1(run_dir)
    assert (run_dir / "input.py").read_text(encoding="utf-8") == "generated kernel"
    assert captured["init"] == {
        "num_workers": 4,
        "max_rounds": 10,
        "log_dir": str(run_dir / "agent1_logs"),
        "model_name": "glm-5.2",
    }
    assert captured["generate"] == {
        "problem_description": pipeline.PROBLEM,
        "test_code": strict_test,
        "generate_default_test": False,
    }


def test_agent2_targets_run_dir_and_isolated_logs(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    captured = {}

    def fake_run(cmd, *, cwd, env):
        captured.update(cmd=cmd, cwd=cwd, env=env)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.delenv(pipeline.AGENT2_STRATEGY_ENV, raising=False)

    assert pipeline.run_agent2(run_dir) == 7
    assert captured["cmd"] == [
        pipeline.sys.executable,
        str(pipeline.REPO / "examples" / "run_opt_manager.py"),
        "--kernel-dir",
        str(run_dir),
        "--strategy",
        "greedy_glm_rag",
        "--max-rounds",
        "10",
    ]
    assert captured["cwd"] == str(pipeline.REPO)
    assert captured["env"]["KERNEL_OPT_LOG_DIR"] == str(
        run_dir / "opt_manager_logs"
    )


def test_agent2_strategy_can_be_overridden(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    captured = {}

    def fake_run(cmd, *, cwd, env):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setenv(pipeline.AGENT2_STRATEGY_ENV, "greedy_glm")

    assert pipeline.run_agent2(run_dir) == 0
    strategy_index = captured["cmd"].index("--strategy") + 1
    assert captured["cmd"][strategy_index] == "greedy_glm"


def test_module_defaults_glm_thinking_disabled(monkeypatch):
    monkeypatch.delenv("LLM_CENTER_GLM_THINKING", raising=False)
    importlib.reload(pipeline)
    assert os.environ.get("LLM_CENTER_GLM_THINKING") == "disabled"


def test_module_respects_existing_glm_thinking_setting(monkeypatch):
    monkeypatch.setenv("LLM_CENTER_GLM_THINKING", "enabled")
    importlib.reload(pipeline)
    assert os.environ.get("LLM_CENTER_GLM_THINKING") == "enabled"


def test_agent1_llm_error_consumes_attempt_and_retries(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test.py").write_text("strict test", encoding="utf-8")
    calls = {"generate": 0}

    class FlakyAgent:
        def __init__(self, **kwargs):
            pass

        def generate_kernel(self, **kwargs):
            calls["generate"] += 1
            if calls["generate"] == 1:
                raise RuntimeError(
                    "passthrough stream idle timeout after 120s waiting for next chunk"
                )
            return {"success": True, "kernel_code": "recovered kernel"}

    monkeypatch.setattr(pipeline, "TritonKernelAgent", FlakyAgent)

    assert pipeline.run_agent1(run_dir)
    assert calls["generate"] == 2
    assert (run_dir / "input.py").read_text(encoding="utf-8") == "recovered kernel"


def test_binding_and_validation_end_to_end_cpu(monkeypatch):
    monkeypatch.syspath_prepend(str(_BENCHMARKING_DIR))
    for module_name in ("timing", "kernel_binding"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    timing = importlib.import_module("timing")

    test_module = _load_module(_TEST_FILE, "matmul_gelu_softmax_test_module")
    validate_output = test_module.validate_output

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            x = self.linear(x)
            x = torch.nn.functional.gelu(x)
            return torch.nn.functional.softmax(x, dim=1)

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    x = torch.rand(4, 8, dtype=torch.bfloat16)

    seen = []

    def kernel_function(x_arg, weight, bias):
        seen.extend([x_arg, weight, bias])
        acc = x_arg.float() @ weight.float().t() + bias.float()
        acc = torch.nn.functional.gelu(acc)
        return torch.nn.functional.softmax(acc, dim=1).to(torch.bfloat16)

    invoke = timing.bind_kernel_function(kernel_function, [x], model)
    kernel_output = invoke()

    # The binder must fill tensor slots as [*inputs, *model_tensors] with the
    # nn.Linear weight extracted before its bias.
    assert len(seen) == 3
    assert seen[0].shape == (4, 8) and torch.equal(seen[0], x)
    assert seen[1].shape == (8, 8) and torch.equal(seen[1], model.linear.weight)
    assert seen[2].shape == (8,) and torch.equal(seen[2], model.linear.bias)

    with torch.no_grad():
        ref_output = model(x)

    assert validate_output(ref_output, kernel_output, expected_shape=(4, 8))
    assert not validate_output(ref_output, kernel_output.float(), expected_shape=(4, 8))
    assert not validate_output(ref_output, kernel_output[:2], expected_shape=(4, 8))
    assert not validate_output(
        ref_output, torch.zeros_like(kernel_output), expected_shape=(4, 8)
    )
