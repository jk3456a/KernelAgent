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

"""Streaming aggregation for OpenAI-compatible chat completions.

llm-center's GLM endpoint serves non-streaming requests through a pathological
slow path: a 6.4k-token conv-kernel completion took ~90 min non-streamed, but
~53 s (23 tok/s, first token < 1 s) with stream=True. So the provider must
stream and re-aggregate chunks into a response-shaped object. These tests pin
that aggregation: content, reasoning_content, finish_reason, usage, and n>1
grouping by choice index.
"""

from utils.providers.openai_base import _aggregate_stream


class _Delta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, index, content=None, reasoning_content=None, finish_reason=None):
        self.index = index
        self.delta = _Delta(content, reasoning_content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, **kw):
        self._d = kw

    def dict(self):
        return dict(self._d)


class _Chunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


def test_single_choice_content_and_finish():
    chunks = [
        _Chunk([_Choice(0, content="def ")]),
        _Chunk([_Choice(0, content="kernel_function(")]),
        _Chunk([_Choice(0, content="x): pass")]),
        _Chunk([_Choice(0, finish_reason="stop")]),
        _Chunk([], usage=_Usage(completion_tokens=7, total_tokens=20)),
    ]
    agg = _aggregate_stream(chunks)
    assert len(agg.choices) == 1
    assert agg.choices[0].message.content == "def kernel_function(x): pass"
    assert agg.choices[0].finish_reason == "stop"
    assert agg.usage.dict()["completion_tokens"] == 7


def test_reasoning_content_kept_separate():
    chunks = [
        _Chunk([_Choice(0, reasoning_content="think ")]),
        _Chunk([_Choice(0, reasoning_content="hard")]),
        _Chunk([_Choice(0, content="answer")]),
        _Chunk([_Choice(0, finish_reason="stop")]),
    ]
    agg = _aggregate_stream(chunks)
    msg = agg.choices[0].message
    assert msg.content == "answer"
    assert msg.reasoning_content == "think hard"


def test_multiple_choices_grouped_by_index():
    chunks = [
        _Chunk([_Choice(0, content="seed0-")]),
        _Chunk([_Choice(1, content="seed1-")]),
        _Chunk([_Choice(0, content="a"), _Choice(1, content="b")]),
        _Chunk([_Choice(0, finish_reason="stop"), _Choice(1, finish_reason="stop")]),
    ]
    agg = _aggregate_stream(chunks)
    assert len(agg.choices) == 2
    by_index = {c.index: c for c in agg.choices}
    assert by_index[0].message.content == "seed0-a"
    assert by_index[1].message.content == "seed1-b"


def test_empty_content_stays_empty_not_none_crash():
    # A stream that yields only a finish chunk (no content) must not crash and
    # must surface empty content so the caller's "empty response" path fires.
    chunks = [_Chunk([_Choice(0, finish_reason="length")])]
    agg = _aggregate_stream(chunks)
    assert agg.choices[0].message.content == ""
    assert agg.choices[0].finish_reason == "length"
