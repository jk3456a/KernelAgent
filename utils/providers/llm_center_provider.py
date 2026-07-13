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

"""LLM-center (modelbest) provider — OpenAI-compatible GLM endpoint.

The modelbest llm-center proxy exposes models like GLM-5.2 over the OpenAI
chat-completions protocol, e.g.

    curl https://llm-center.ali.modelbest.cn/llm/v1/chat/completions \\
      -H 'Authorization: Bearer sk-...' \\
      -d '{"model":"glm-5.2","messages":[...]}'

so it is a thin ``OpenAICompatibleProvider`` with a pinned ``base_url`` and a
dedicated key env var (``LLM_CENTER_API_KEY``) so it can coexist with a real
OpenAI key in the same process.
"""

import os

from .openai_base import OpenAICompatibleProvider

_DEFAULT_BASE_URL = "https://llm-center.ali.modelbest.cn/llm/v1"


class LLMCenterProvider(OpenAICompatibleProvider):
    """GLM (and other OpenAI-protocol models) via the modelbest llm-center proxy."""

    def __init__(self):
        base_url = os.environ.get("LLM_CENTER_BASE_URL", _DEFAULT_BASE_URL)
        super().__init__(api_key_env="LLM_CENTER_API_KEY", base_url=base_url)

    @property
    def name(self) -> str:
        return "llm_center"

    def get_max_tokens_limit(self, model_name: str) -> int:
        # GLM-5.2 is a reasoning model: the token budget is shared between
        # reasoning_content and the final content. llm-center currently rejects
        # max_tokens above 128K even though the context window is larger.
        return 131_072
