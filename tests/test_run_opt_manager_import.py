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
"""run_opt_manager.py must import its package regardless of cwd / sys.path[0].

The conv/gemm pipeline runners launch this script via subprocess. Python sets
sys.path[0] to the script's own dir (examples/), not the repo root, so the
``from triton_kernel_agent...`` import fails with ModuleNotFoundError unless the
script prepends the repo root to sys.path BEFORE that import. This regression
test runs the script as a subprocess from a foreign cwd with a scrubbed
PYTHONPATH and asserts it gets past the import (argparse --help).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "examples" / "run_opt_manager.py"


def test_run_opt_manager_imports_from_foreign_cwd(tmp_path):
    # Scrub PYTHONPATH so success can only come from the script fixing sys.path
    # itself, and run from a dir that is neither the repo root nor examples/.
    env = {
        k: v
        for k, v in __import__("os").environ.items()
        if k != "PYTHONPATH"
    }
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr
    # argparse --help exits 0 after the imports succeed.
    assert proc.returncode == 0, proc.stderr
    assert "--kernel-dir" in proc.stdout
