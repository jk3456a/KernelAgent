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

"""Pluggable text-embedding backends for RAG retrieval."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol


class EmbeddingBackend(Protocol):
    """Minimal interface required by the RAG prescriber."""

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for *text*."""
        ...


EmbeddingFactory = Callable[..., EmbeddingBackend]


class EmbeddingBackendRegistry:
    """Map configuration names to embedding backend factories."""

    def __init__(self) -> None:
        self._factories: dict[str, EmbeddingFactory] = {}

    def register(self, name: str, factory: EmbeddingFactory) -> None:
        if not name:
            raise ValueError("embedding backend name must not be empty")
        self._factories[name] = factory

    def create(
        self,
        name: str,
        options: Mapping[str, Any] | None = None,
    ) -> EmbeddingBackend:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"Unknown embedding backend {name!r}; available: {available}"
            ) from exc
        return factory(**dict(options or {}))


class OpenAICompatibleEmbeddingBackend:
    """Embedding backend for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Embedding API key environment variable "
                    f"{self.api_key_env!r} is not set"
                )

            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed(self, text: str) -> list[float]:
        params: dict[str, Any] = {"input": text, "model": self.model}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        if self.encoding_format is not None:
            params["encoding_format"] = self.encoding_format

        response = self._get_client().embeddings.create(**params)
        if not response.data:
            raise RuntimeError("Embedding provider returned no vectors")
        return list(response.data[0].embedding)


embedding_backends = EmbeddingBackendRegistry()
embedding_backends.register(
    "openai_compatible",
    OpenAICompatibleEmbeddingBackend,
)


def create_embedding_backend(
    config: Mapping[str, Any] | None = None,
) -> EmbeddingBackend:
    """Create a backend from ``{backend, options}`` configuration."""
    config = config or {}
    name = str(config.get("backend", "openai_compatible"))
    options = config.get("options")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("rag_embedding.options must be a mapping")
    return embedding_backends.create(name, options)
