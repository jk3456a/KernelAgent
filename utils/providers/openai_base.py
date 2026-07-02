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

"""Base provider for OpenAI-compatible APIs."""

from typing import Any
import logging
from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible APIs."""

    def __init__(self, api_key_env: str, base_url: str | None = None):
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._original_proxy_env = None
        super().__init__()

    def _client_timeout(self) -> float:
        """Read timeout (seconds) for LLM calls.

        Reasoning models (GLM-5.2 with a 1M token budget) can spend many minutes
        emitting a long reasoning trace before any content, blowing past the
        SDK's 600s default and raising APITimeoutError mid-optimization. Default
        to 1 hour; override with LLM_TIMEOUT_S.
        """
        import os

        try:
            return float(os.environ.get("LLM_TIMEOUT_S", "3600"))
        except ValueError:
            return 3600.0

    def _create_with_hard_timeout(self, api_params: dict):
        """Run the chat-completions call under a wall-clock hard timeout.

        The SDK/httpx read timeout is unreliable when the connection is silently
        dropped by an intermediary (e.g. a tsh/ssh proxy going half-open): the
        read blocks forever and the optimization loop hangs. A watchdog thread
        with ``future.result(timeout=...)`` guarantees the call returns (or
        raises TimeoutError) so the caller can record a failure and move on.
        The hard cap is slightly above the client read timeout so a genuinely
        slow-but-alive response is not killed prematurely.
        """
        import concurrent.futures

        hard = self._client_timeout() + 120
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = ex.submit(self.client.chat.completions.create, **api_params)
        try:
            result = future.result(timeout=hard)
            ex.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError as exc:
            # Do NOT wait for the stuck worker thread (a blocked socket read can
            # never be interrupted from Python); abandon it so the loop proceeds.
            ex.shutdown(wait=False)
            logging.getLogger(__name__).error(
                "%s chat completion exceeded hard timeout of %.0fs "
                "(connection likely stalled); abandoning call.",
                self.name,
                hard,
            )
            raise TimeoutError(
                f"LLM call exceeded hard timeout of {hard:.0f}s"
            ) from exc

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            return

        api_key = self._get_api_key(self.api_key_env)
        if api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            # Initialize client (proxy configured via environment variables).
            # A generous read timeout keeps long reasoning-model calls from
            # failing as APITimeoutError.
            kwargs = {"api_key": api_key, "timeout": self._client_timeout()}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self._create_with_hard_timeout(api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (single): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        choice = response.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        # Reasoning models (GLM, o-series) expose their chain-of-thought
        # separately from the answer; the SDK surfaces it as reasoning_content
        # (sometimes only under model_extra). Capture it for the trajectory.
        reasoning = getattr(choice.message, "reasoning_content", None) or (
            getattr(choice.message, "model_extra", None) or {}
        ).get("reasoning_content")
        # Surface the two silent-failure modes that otherwise look like "the
        # model wrote nothing useful": a length-truncated response (raise
        # max_tokens) and an empty body (common with reasoning models when the
        # token budget is spent on reasoning before any content is emitted).
        if finish_reason == "length":
            logging.getLogger(__name__).warning(
                "%s response truncated (finish_reason=length): output hit "
                "max_tokens; raise max_tokens.",
                self.name,
            )
        elif not content:
            logging.getLogger(__name__).warning(
                "%s returned empty content (finish_reason=%s).",
                self.name,
                finish_reason,
            )

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.name,
            usage=response.usage.dict()
            if hasattr(response, "usage") and response.usage
            else None,
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        """Get multiple responses using n parameter."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, n=n, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (multi): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return [
            LLMResponse(
                content=choice.message.content,
                model=model_name,
                provider=self.name,
                usage=response.usage.dict()
                if hasattr(response, "usage") and response.usage
                else None,
                finish_reason=getattr(choice, "finish_reason", None),
            )
            for choice in response.choices
        ]

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 and o-series models pin their own sampling behaviour
        if not (model_name.startswith("gpt-5") or model_name.startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)

        # Use max_completion_tokens for newer models like GPT-5, fallback to max_tokens
        max_tokens_value = min(
            kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
        )
        if model_name.startswith("gpt-5") or model_name.startswith("o"):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]

        # Auto-enable high reasoning for GPT-5
        if model_name.startswith("gpt-5"):
            params["reasoning_effort"] = "high"
        elif kwargs.get("high_reasoning_effort") and model_name.startswith(
            ("o3", "o1")
        ):
            params["reasoning_effort"] = "high"

        return params

    def is_available(self) -> bool:
        """Check if provider is available."""
        return OPENAI_AVAILABLE and self.client is not None

    def supports_multiple_completions(self) -> bool:
        """OpenAI-compatible APIs support native multiple completions."""
        return True
