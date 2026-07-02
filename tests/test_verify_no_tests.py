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

"""Verification must NOT report success when there is nothing to run.

An empty (or all-missing) test set previously fell through to ``return True``,
so a kernel that never had its correctness checked was reported as verified —
a silent false pass. Verification with no runnable test is a failure.
"""

import logging
from pathlib import Path

from triton_kernel_agent.worker_util import _run_test_multiprocess


def test_no_runnable_tests_is_failure(tmp_path: Path):
    logger = logging.getLogger("test-no-tests")
    # No test files at all.
    success, _stdout, stderr = _run_test_multiprocess(logger, tmp_path, [])
    assert success is False
    assert "no" in stderr.lower() or "test" in stderr.lower()


def test_all_missing_tests_is_failure(tmp_path: Path):
    logger = logging.getLogger("test-missing-tests")
    missing = [tmp_path / "test_kernel.py"]  # path that does not exist
    success, _stdout, _stderr = _run_test_multiprocess(logger, tmp_path, missing)
    assert success is False
