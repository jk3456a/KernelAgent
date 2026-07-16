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
"""Tests for the LLM-center (modelbest) GLM provider wiring.

GLM-5.2 is served over the OpenAI chat-completions protocol behind the
llm-center proxy, so it reuses ``OpenAICompatibleProvider`` with a pinned
``base_url`` and a dedicated key env var. These tests pin the configuration
contract without making real network calls.
"""

from __future__ import annotations

from utils.providers.llm_center_provider import LLMCenterProvider


class TestLLMCenterProvider:
    def test_base_url_pinned(self, monkeypatch):
        monkeypatch.setenv("LLM_CENTER_API_KEY", "sk-test")
        p = LLMCenterProvider()
        assert p.base_url.rstrip("/").endswith("/llm/v1")
        assert p.name == "llm_center"

    def test_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_CENTER_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_CENTER_BASE_URL", "https://example.test/llm/v1")
        p = LLMCenterProvider()
        assert p.base_url == "https://example.test/llm/v1"

    def test_uses_llm_center_key_env(self):
        # api_key_env wiring, independent of whether the key is set.
        p = LLMCenterProvider()
        assert p.api_key_env == "LLM_CENTER_API_KEY"

    def test_glm_uses_plain_max_tokens_and_temperature(self, monkeypatch):
        # GLM is not a gpt-5/o-series model, so it must get temperature + the
        # plain max_tokens param (not max_completion_tokens / reasoning_effort).
        monkeypatch.setenv("LLM_CENTER_API_KEY", "sk-test")
        monkeypatch.delenv("LLM_CENTER_GLM_THINKING", raising=False)
        p = LLMCenterProvider()
        params = p._build_api_params("glm-5.2", [{"role": "user", "content": "hi"}])
        assert "temperature" in params
        assert "max_tokens" in params
        assert "max_completion_tokens" not in params
        assert "reasoning_effort" not in params
        assert "extra_body" not in params

    def test_glm_thinking_disabled_via_env(self, monkeypatch):
        # llm-center's passthrough kills streams that stay chunk-silent >120s,
        # and the GLM reasoning phase can exceed that on hard prompts. The env
        # switch must disable the thinking phase via the Zhipu extra_body param.
        monkeypatch.setenv("LLM_CENTER_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_CENTER_GLM_THINKING", "disabled")
        p = LLMCenterProvider()
        params = p._build_api_params("glm-5.2", [{"role": "user", "content": "hi"}])
        assert params["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_thinking_env_ignored_for_non_glm_models(self, monkeypatch):
        monkeypatch.setenv("LLM_CENTER_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_CENTER_GLM_THINKING", "disabled")
        p = LLMCenterProvider()
        params = p._build_api_params("gpt-5", [{"role": "user", "content": "hi"}])
        assert "extra_body" not in params


class TestGLMModelRegistered:
    def test_glm_in_registry_routes_to_llm_center(self):
        from utils.providers.available_models import AVAILABLE_MODELS

        by_name = {m.name: m for m in AVAILABLE_MODELS}
        assert "glm-5.2" in by_name
        assert LLMCenterProvider in by_name["glm-5.2"].provider_classes
