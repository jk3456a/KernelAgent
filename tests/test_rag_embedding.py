"""GPU- and network-free tests for pluggable RAG embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_kernel_agent.opt_manager import OptimizationManager
from triton_kernel_agent.opt_worker import OptimizationWorker
from triton_kernel_agent.opt_worker_component.prescribing.RAG_based_prescriber import (
    RAGPrescriber,
)
from triton_kernel_agent.opt_worker_component.prescribing.embedding_backend import (
    EmbeddingBackendRegistry,
    OpenAICompatibleEmbeddingBackend,
    create_embedding_backend,
)
from triton_kernel_agent.platform.nvidia import NvidiaRAGPrescriber
from triton_kernel_agent.platform.registry import registry as platform_registry


_ROOT = Path(__file__).resolve().parent.parent
_RAG_CONFIG = _ROOT / "examples" / "configs" / "greedy_glm_rag.yaml"


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.25, 0.5, 0.75])]
        )


class _FakeEmbeddingBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]


def test_zhipu_config_uses_openai_compatible_embedding_request(monkeypatch):
    embeddings = _FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    client_kwargs = {}

    def fake_openai(**kwargs):
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setenv("ZAI_API_KEY", "test-zhipu-key")
    monkeypatch.setattr("openai.OpenAI", fake_openai)
    backend = create_embedding_backend(
        {
            "backend": "openai_compatible",
            "options": {
                "model": "embedding-3",
                "dimensions": 2048,
                "api_key_env": "ZAI_API_KEY",
                "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                "encoding_format": "float",
            },
        }
    )

    assert backend.embed("optimize tensor core utilization") == [0.25, 0.5, 0.75]
    assert client_kwargs == {
        "api_key": "test-zhipu-key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    }
    assert embeddings.calls == [
        {
            "input": "optimize tensor core utilization",
            "model": "embedding-3",
            "dimensions": 2048,
            "encoding_format": "float",
        }
    ]
    assert backend.base_url == "https://open.bigmodel.cn/api/paas/v4/"


def test_openai_defaults_remain_backward_compatible():
    backend = create_embedding_backend()

    assert isinstance(backend, OpenAICompatibleEmbeddingBackend)
    assert backend.model == "text-embedding-3-large"
    assert backend.api_key_env == "OPENAI_API_KEY"
    assert backend.base_url is None


def test_missing_embedding_api_key_is_actionable(monkeypatch):
    monkeypatch.delenv("MISSING_EMBEDDING_KEY", raising=False)
    backend = OpenAICompatibleEmbeddingBackend(
        api_key_env="MISSING_EMBEDDING_KEY"
    )

    with pytest.raises(RuntimeError, match="MISSING_EMBEDDING_KEY"):
        backend.embed("query")


def test_custom_embedding_backend_can_be_registered():
    custom = _FakeEmbeddingBackend()
    backends = EmbeddingBackendRegistry()
    backends.register("custom", lambda **_options: custom)

    assert backends.create("custom").embed("query") == [1.0, 0.0]
    assert custom.calls == ["query"]


def test_rag_prescriber_accepts_injected_backend():
    backend = _FakeEmbeddingBackend()
    prescriber = RAGPrescriber(embedding_backend=backend)

    node, scores = prescriber.retrieve("reduce register pressure")

    assert node is not None
    assert scores
    assert backend.calls[-1] == "reduce register pressure"
    assert len(backend.calls) > 1


def test_yaml_reaches_manager_worker_kwargs(tmp_path):
    manager = OptimizationManager(config=str(_RAG_CONFIG), log_dir=tmp_path)

    assert manager.worker_kwargs["rag_embedding"] == {
        "backend": "openai_compatible",
        "options": {
            "model": "embedding-3",
            "dimensions": 2048,
            "api_key_env": "ZAI_API_KEY",
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "encoding_format": "float",
        },
    }


def test_worker_forwards_embedding_config_to_platform(monkeypatch, tmp_path):
    config = {"backend": "custom", "options": {"value": 1}}
    captured = {}

    def fake_create_from_config(platform_config, **kwargs):
        captured["platform_config"] = platform_config
        captured["rag_embedding"] = kwargs["rag_embedding"]
        return {}

    monkeypatch.setattr(
        platform_registry,
        "create_from_config",
        fake_create_from_config,
    )
    worker = object.__new__(OptimizationWorker)
    worker._platform_config = {"rag_prescriber": "nvidia"}
    worker._platform = {}
    worker.logger = logging.getLogger("test_rag_embedding")
    worker.log_dir = tmp_path
    worker.artifact_dir = tmp_path
    worker.ncu_bin_path = None
    worker.profiling_semaphore = None
    worker.openai_model = "glm-5.2"
    worker.gpu_name = "NVIDIA H100 SXM 80GB"
    worker.roofline_config = None
    worker.rag_embedding = config

    worker._resolve_platform_config()

    assert captured == {
        "platform_config": {"rag_prescriber": "nvidia"},
        "rag_embedding": config,
    }


def test_nvidia_wrapper_passes_embedding_config_to_prescriber(monkeypatch):
    from triton_kernel_agent.opt_worker_component.prescribing import (
        RAG_based_prescriber as rag_module,
    )

    config = {"backend": "custom", "options": {"value": 1}}
    captured = {}

    class FakePrescriber:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def retrieve(self, query):
            return query, {}

    monkeypatch.setattr(rag_module, "RAGPrescriber", FakePrescriber)
    prescriber = NvidiaRAGPrescriber(rag_embedding=config)

    assert prescriber.retrieve("query") == ("query", {})
    assert captured["embedding_config"] == config
