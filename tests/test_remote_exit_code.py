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

"""Remote commands must propagate the child's real exit code.

``setsid`` starts a new session leader (so the remote process group can be
reaped as a unit on teardown) but WITHOUT ``--wait`` it returns 0 immediately,
regardless of whether the wrapped python/test process succeeded or failed. Over
ssh that turned every remote verification into a false pass: a kernel that fails
to compile exits 1, setsid returns 0, ssh returns 0, and the worker records the
kernel as verified. Every remote-command builder must use ``setsid --wait`` so
the true exit status flows back.
"""

from utils import remote_exec


def test_workdir_test_script_waits_for_exit_code():
    script = remote_exec._build_workdir_test_script("/tmp/wd", ["test_kernel.py"])
    assert "setsid --wait" in script


def test_command_script_waits_for_exit_code():
    script = remote_exec._build_command_script("/tmp/wd", "python3 -u bench.py")
    assert "setsid --wait" in script


def test_candidate_script_waits_for_exit_code():
    script = remote_exec._build_remote_script(
        "/tmp/wd", "kernel.py", isolated=False, deny_network=False
    )
    assert "setsid --wait" in script
