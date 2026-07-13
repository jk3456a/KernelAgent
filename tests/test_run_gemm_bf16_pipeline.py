"""GPU- and LLM-free tests for the isolated BF16 GEMM pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import run_gemm_bf16_pipeline as pipeline


_ROOT = Path(__file__).resolve().parent.parent
_PROBLEM_FILE = _ROOT / "examples" / "optimize_gemm_bf16" / "problem.py"
_TEST_FILE = _ROOT / "examples" / "optimize_gemm_bf16" / "test.py"


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_problem_generates_bf16_gemm_inputs():
    get_inputs = _function_node(_PROBLEM_FILE, "get_inputs")
    rand_calls = [
        node
        for node in ast.walk(get_inputs)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.rand"
    ]

    assert [[ast.unparse(arg) for arg in call.args] for call in rand_calls] == [
        ["M", "K"],
        ["K", "N"],
    ]
    assert [
        ast.unparse(next(kw.value for kw in call.keywords if kw.arg == "dtype"))
        for call in rand_calls
    ] == ["torch.bfloat16", "torch.bfloat16"]


def test_strict_test_rejects_wrong_output_shape_and_dtype():
    validate_output = _function_node(_TEST_FILE, "validate_output")
    rendered = ast.unparse(validate_output)

    assert "isinstance(kernel_output, torch.Tensor)" in rendered
    assert "expected_shape = (M, N)" in rendered
    assert "tuple(kernel_output.shape) != expected_shape" in rendered
    assert "kernel_output.dtype != torch.bfloat16" in rendered

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


def test_create_run_dir_copies_isolated_contract(tmp_path, monkeypatch):
    source_dir = tmp_path / "optimize_gemm_bf16"
    source_dir.mkdir()
    (source_dir / "problem.py").write_text("problem contract", encoding="utf-8")
    (source_dir / "test.py").write_text("strict test", encoding="utf-8")
    configured = []

    monkeypatch.setattr(pipeline, "DST", source_dir)
    monkeypatch.setattr(pipeline, "RUNS_DIR", source_dir / "runs")
    monkeypatch.setattr(pipeline, "configure_progress", configured.append)

    run_dir = pipeline.create_run_dir()

    assert run_dir.parent == source_dir / "runs"
    assert run_dir.name.startswith("gemm_bf16_")
    assert (run_dir / "problem.py").read_text(encoding="utf-8") == "problem contract"
    assert (run_dir / "test.py").read_text(encoding="utf-8") == "strict test"
    assert configured == [run_dir]


def test_agent1_uses_strict_static_test_and_writes_kernel(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    strict_test = "STRICT BF16 TEST"
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
