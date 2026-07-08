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
"""The multi-completion seed path must honor the same hard timeout as the
single-response path.

agent1 generates its kernel seeds via ``get_multiple_responses`` (n>1). If that
call bypasses the wall-clock watchdog (``_create_with_hard_timeout``), a
silently half-open proxy connection hangs the whole pipeline forever (observed:
51 min with no output). These tests pin that the multi path funnels through the
same watchdog as the single path and that a stalled upstream surfaces as
``TimeoutError`` instead of blocking indefinitely.
"""

from __future__ import annotations

import pytest

from utils.providers.llm_center_provider import LLMCenterProvider


class _StubMessage:
    def __init__(self, content):
        self.content = content
        self.reasoning_content = None
        self.model_extra = None


class _StubChoice:
    def __init__(self, content):
        self.message = _StubMessage(content)
        self.finish_reason = "stop"


class _StubResponse:
    def __init__(self, n):
        self.choices = [_StubChoice(f"seed {i}") for i in range(n)]
        self.usage = None

    def model_dump(self):
        return {"choices": len(self.choices)}


def _make_provider(monkeypatch):
    monkeypatch.setenv("LLM_CENTER_API_KEY", "sk-test")
    return LLMCenterProvider()


def test_multiple_responses_routes_through_hard_timeout(monkeypatch):
    # The seed phase must go through the wall-clock watchdog, not a bare
    # client.chat.completions.create() that can hang forever.
    p = _make_provider(monkeypatch)
    seen = {}

    def _fake_hard_timeout(api_params):
        seen["n"] = api_params.get("n")
        return _StubResponse(api_params.get("n", 1))

    monkeypatch.setattr(p, "_create_with_hard_timeout", _fake_hard_timeout)
    out = p.get_multiple_responses(
        "glm-5.2", [{"role": "user", "content": "hi"}], n=4
    )
    assert seen["n"] == 4  # watchdog actually invoked with the multi request
    assert [r.content for r in out] == ["seed 0", "seed 1", "seed 2", "seed 3"]


def test_multiple_responses_propagates_timeout(monkeypatch):
    # A stalled upstream surfaces as TimeoutError to the caller (which then
    # records a failed attempt and retries) instead of blocking the pipeline.
    p = _make_provider(monkeypatch)

    def _raise_timeout(api_params):
        raise TimeoutError("LLM call exceeded hard timeout")

    monkeypatch.setattr(p, "_create_with_hard_timeout", _raise_timeout)
    with pytest.raises(TimeoutError):
        p.get_multiple_responses(
            "glm-5.2", [{"role": "user", "content": "hi"}], n=4
        )
