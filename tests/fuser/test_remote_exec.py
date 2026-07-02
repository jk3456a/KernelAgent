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
"""Tests for the SSH remote-execution stack (ported from Train_engine).

The remote stack mirrors Train_engine's ``remote_run.sh`` / ``remote_sync.py``:
a self-contained candidate file is rsync'd to a per-run remote workdir and then
executed over SSH with ``setsid`` + persistent Triton/Inductor caches. These
tests pin the pure command-construction and config-resolution contracts (no
real SSH), plus the local-fallback path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from utils import remote_config, remote_exec


class TestLoadRemoteConfig:
    def test_default_is_local(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for var in ("KERNEL_REMOTE_KIND", "KERNEL_REMOTE_HOST", "KERNEL_REMOTE_WORKSPACE", "KERNEL_REMOTE_CONFIG"):
            monkeypatch.delenv(var, raising=False)
        cfg = remote_config.load_remote_config()
        assert cfg["kind"] == "local"
        assert not remote_config.is_remote_enabled(cfg)

    def test_reads_toml(self, tmp_path, monkeypatch):
        toml = tmp_path / "remote.toml"
        toml.write_text(
            '[remote]\nkind = "ssh"\nhostname = "h100box"\nworkspace = "/data/ka"\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        for var in ("KERNEL_REMOTE_KIND", "KERNEL_REMOTE_HOST", "KERNEL_REMOTE_WORKSPACE", "KERNEL_REMOTE_CONFIG"):
            monkeypatch.delenv(var, raising=False)
        cfg = remote_config.load_remote_config()
        assert cfg["kind"] == "ssh"
        assert cfg["hostname"] == "h100box"
        assert cfg["workspace"] == "/data/ka"
        assert remote_config.is_remote_enabled(cfg)

    def test_env_overrides_toml(self, tmp_path, monkeypatch):
        toml = tmp_path / "remote.toml"
        toml.write_text('[remote]\nkind = "ssh"\nhostname = "from_toml"\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KERNEL_REMOTE_HOST", "from_env")
        cfg = remote_config.load_remote_config()
        assert cfg["hostname"] == "from_env"

    def test_explicit_config_path(self, tmp_path, monkeypatch):
        cfgfile = tmp_path / "custom.toml"
        cfgfile.write_text('[remote]\nkind = "ssh"\nhostname = "boxA"\n', encoding="utf-8")
        monkeypatch.setenv("KERNEL_REMOTE_CONFIG", str(cfgfile))
        for var in ("KERNEL_REMOTE_KIND", "KERNEL_REMOTE_HOST", "KERNEL_REMOTE_WORKSPACE"):
            monkeypatch.delenv(var, raising=False)
        cfg = remote_config.load_remote_config()
        assert cfg["hostname"] == "boxA"

    def test_ssh_without_hostname_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KERNEL_REMOTE_KIND", "ssh")
        monkeypatch.delenv("KERNEL_REMOTE_HOST", raising=False)
        monkeypatch.delenv("KERNEL_REMOTE_CONFIG", raising=False)
        try:
            remote_config.load_remote_config()
        except RuntimeError as exc:
            assert "hostname" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for ssh without hostname")


class TestRemoteWorkdir:
    def test_workspace_used_as_root(self):
        wd = remote_exec._remote_workdir("/data/ka", Path("/local/runs/attempt_123"))
        assert wd.startswith("/data/ka/.kernelagent_remote/")
        assert "attempt_123" in wd

    def test_empty_workspace_defaults_to_home(self):
        wd = remote_exec._remote_workdir("", Path("/local/runs/attempt_9"))
        assert wd.startswith("$HOME/.kernelagent_remote/")


class TestRemoteScript:
    def test_persistent_caches_and_setsid(self):
        script = remote_exec._build_remote_script(
            "/data/ka/.kernelagent_remote/r1", "candidate_main.py",
            isolated=False, deny_network=False,
        )
        assert "TRITON_CACHE_DIR" in script
        assert "TORCHINDUCTOR_CACHE_DIR" in script
        assert "exec setsid" in script
        assert "cd /data/ka/.kernelagent_remote/r1" in script
        assert "candidate_main.py" in script
        assert "PYTHONHASHSEED=0" in script

    def test_isolated_adds_flag(self):
        script = remote_exec._build_remote_script(
            "/r", "candidate_main.py", isolated=True, deny_network=False,
        )
        assert "python3 -u -I candidate_main.py" in script

    def test_deny_network_never_isolated(self):
        # -I would drop cwd from sys.path and defeat the sitecustomize block,
        # so deny_network must not pass -I even when isolated is requested.
        script = remote_exec._build_remote_script(
            "/r", "candidate_main.py", isolated=True, deny_network=True,
        )
        assert "-I" not in script
        assert "python3 -u candidate_main.py" in script


class TestSSHArgv:
    def test_serveralive_options(self):
        argv = remote_exec._build_ssh_argv("h100box", "echo hi")
        assert argv[0] == "ssh"
        assert "ServerAliveInterval=30" in argv
        assert "ServerAliveCountMax=4" in argv
        assert argv[-2] == "h100box"
        assert argv[-1] == "echo hi"


class TestPushCommands:
    def test_mkdir_and_rsync(self):
        mkdir = remote_exec._build_mkdir_argv("box", "/data/ka/.kernelagent_remote/r1")
        assert mkdir[0] == "ssh"
        assert mkdir[1] == "box"
        assert "mkdir -p" in mkdir[2]

        rsync = remote_exec._build_rsync_argv(
            "box", [Path("/l/candidate_main.py")], "/data/ka/.kernelagent_remote/r1"
        )
        assert rsync[0] == "rsync"
        assert "/l/candidate_main.py" in rsync
        assert rsync[-1] == "box:/data/ka/.kernelagent_remote/r1/"


class TestWorkdirTestScript:
    def test_chains_tests_with_and(self):
        # The TritonKernelAgent worker runs multiple test files sequentially
        # with && semantics inside one remote workdir (kernel.py sits alongside).
        script = remote_exec._build_workdir_test_script(
            "/data/ka/.kernelagent_remote/w1",
            ["test_kernel.py", "test_extra_1_kernel.py"],
        )
        assert "cd /data/ka/.kernelagent_remote/w1" in script
        assert "TRITON_CACHE_DIR" in script
        assert "exec setsid" in script
        # both tests, chained, run unbuffered
        assert "python3 -u test_kernel.py" in script
        assert "python3 -u test_extra_1_kernel.py" in script
        assert " && " in script

    def test_single_test(self):
        script = remote_exec._build_workdir_test_script("/w", ["test_kernel.py"])
        assert "python3 -u test_kernel.py" in script
        assert "exec setsid" in script


class TestRunWorkdirTests:
    def test_pushes_workdir_and_returns_contract(self, tmp_path):
        """run_workdir_tests rsyncs the whole workdir then returns (ok, out, err)."""
        workdir = tmp_path / "w"
        workdir.mkdir()
        (workdir / "kernel.py").write_text("# kernel\n", encoding="utf-8")
        (workdir / "test_kernel.py").write_text("print('PASS')\n", encoding="utf-8")

        captured = {}

        def fake_run(argv, capture_output, text, timeout=None):
            captured.setdefault("calls", []).append(argv)

            class R:
                returncode = 0
                stdout = "PASS\n"
                stderr = ""

            return R()

        cfg = {"kind": "ssh", "hostname": "h100box", "workspace": "/data/ka"}
        with patch.object(remote_exec.subprocess, "run", side_effect=fake_run):
            ok, out, err = remote_exec.run_workdir_tests(
                cfg, workdir, ["test_kernel.py"], timeout_s=60
            )

        assert ok is True
        assert "PASS" in out
        # first call mkdir, then rsync of the workdir, then ssh exec
        joined = [" ".join(c) for c in captured["calls"]]
        assert any(c.startswith("ssh") and "mkdir -p" in c for c in joined)
        assert any(c.startswith("rsync") for c in joined)
        assert any(c.startswith("ssh") and "test_kernel.py" in c for c in joined)

    def test_nonzero_rc_is_failure(self, tmp_path):
        workdir = tmp_path / "w"
        workdir.mkdir()
        (workdir / "test_kernel.py").write_text("raise SystemExit(1)\n", encoding="utf-8")

        def fake_run(argv, capture_output, text, timeout=None):
            joined = " ".join(argv)
            # mkdir-only ssh and rsync are the push phase (succeed); the test
            # exec is the ssh call that references the test file (fails here).
            is_exec = argv[0] == "ssh" and "test_kernel.py" in joined

            class R:
                returncode = 1 if is_exec else 0
                stdout = ""
                stderr = "boom" if is_exec else ""

            return R()

        cfg = {"kind": "ssh", "hostname": "box", "workspace": ""}
        with patch.object(remote_exec.subprocess, "run", side_effect=fake_run):
            ok, out, err = remote_exec.run_workdir_tests(
                cfg, workdir, ["test_kernel.py"], timeout_s=60
            )
        assert ok is False


class TestRunCommandWithArtifacts:
    def test_command_script_runs_in_workdir_with_caches(self):
        script = remote_exec._build_command_script(
            "/data/ka/.kernelagent_remote/p1", "python3 -u bench.py --json out.json"
        )
        assert "cd /data/ka/.kernelagent_remote/p1" in script
        assert "TRITON_CACHE_DIR" in script
        assert "exec setsid" in script
        assert "python3 -u bench.py --json out.json" in script

    def test_pull_argv_fetches_named_artifacts(self):
        argv = remote_exec._build_pull_argv(
            "box", "/data/ka/.kernelagent_remote/p1", "out.json", Path("/local/p1/out.json")
        )
        assert argv[0] == "rsync"
        assert argv[-2] == "box:/data/ka/.kernelagent_remote/p1/out.json"
        assert argv[-1] == "/local/p1/out.json"

    def test_runs_pushes_executes_and_pulls(self, tmp_path):
        workdir = tmp_path / "w"
        workdir.mkdir()
        (workdir / "bench.py").write_text("print('hi')\n", encoding="utf-8")
        (workdir / "kernel.py").write_text("# k\n", encoding="utf-8")

        calls = []

        def fake_run(argv, capture_output, text, timeout=None):
            calls.append(argv)
            joined = " ".join(argv)

            class R:
                returncode = 0
                stdout = "ok"
                stderr = ""

            # Simulate the artifact landing locally on the pull rsync.
            if argv[0] == "rsync" and joined.endswith("out.json"):
                Path(argv[-1]).write_text('{"time_ms": 1.5}', encoding="utf-8")
            return R()

        cfg = {"kind": "ssh", "hostname": "box", "workspace": "/data/ka"}
        with patch.object(remote_exec.subprocess, "run", side_effect=fake_run):
            rc, out, err = remote_exec.run_command_with_artifacts(
                cfg,
                workdir,
                "python3 -u bench.py --json out.json",
                artifacts=["out.json"],
                timeout_s=120,
            )

        assert rc == 0
        # the artifact was pulled back into the local workdir
        assert (workdir / "out.json").read_text() == '{"time_ms": 1.5}'
        joined_calls = [" ".join(c) for c in calls]
        assert any(c.startswith("ssh") and "mkdir -p" in c for c in joined_calls)
        assert any(c.startswith("rsync") and "bench.py" in c for c in joined_calls)
        assert any(c.startswith("ssh") and "bench.py --json" in c for c in joined_calls)
        assert any(c.startswith("rsync") and c.endswith("out.json") for c in joined_calls)

    def test_missing_artifact_is_tolerated(self, tmp_path):
        # A failed remote run may not produce the artifact; pull is best-effort
        # and the nonzero rc is surfaced to the caller for handling.
        workdir = tmp_path / "w"
        workdir.mkdir()
        (workdir / "bench.py").write_text("raise SystemExit(2)\n", encoding="utf-8")

        def fake_run(argv, capture_output, text, timeout=None):
            joined = " ".join(argv)
            is_exec = argv[0] == "ssh" and "bench.py" in joined

            class R:
                returncode = 2 if is_exec else 0
                stdout = ""
                stderr = "boom" if is_exec else ""

            return R()

        cfg = {"kind": "ssh", "hostname": "box", "workspace": ""}
        with patch.object(remote_exec.subprocess, "run", side_effect=fake_run):
            rc, out, err = remote_exec.run_command_with_artifacts(
                cfg, workdir, "python3 -u bench.py", artifacts=["out.json"], timeout_s=60
            )
        assert rc == 2  # surfaced, not raised


class TestRunCandidateRouting:
    def test_local_kind_runs_subprocess(self, tmp_path):
        """kind=local keeps the existing local subprocess path."""
        from Fuser import runner

        cand = tmp_path / "cand.py"
        cand.write_text("print('PASS')\n", encoding="utf-8")
        run_root = tmp_path / "runs"
        run_root.mkdir()

        with patch.object(remote_config, "load_remote_config", return_value={"kind": "local"}):
            res = runner.run_candidate(
                cand, run_root, timeout_s=30, isolated=True, deny_network=False
            )
        assert res.passed is True
        assert res.rc == 0

    def test_ssh_kind_routes_to_remote(self, tmp_path):
        """kind=ssh pushes the candidate and execs the ssh argv via _run_candidate."""
        from Fuser import runner

        cand = tmp_path / "cand.py"
        cand.write_text("print('PASS')\n", encoding="utf-8")
        run_root = tmp_path / "runs"
        run_root.mkdir()

        captured = {}

        def fake_run(run_dir, argv, env, stdout_path, stderr_path, t_started, timeout_s, cancel_event):
            captured["argv"] = argv
            Path(stdout_path).write_text("PASS\n", encoding="utf-8")
            Path(stderr_path).write_text("", encoding="utf-8")
            return 0, t_started + 1.0

        cfg = {"kind": "ssh", "hostname": "h100box", "workspace": "/data/ka"}
        with patch.object(remote_config, "load_remote_config", return_value=cfg), \
             patch.object(remote_exec, "_push_candidate", return_value=None) as push, \
             patch.object(runner, "_run_candidate", side_effect=fake_run):
            res = runner.run_candidate(
                cand, run_root, timeout_s=30, isolated=True, deny_network=False
            )

        assert push.called
        assert captured["argv"][0] == "ssh"
        assert "h100box" in captured["argv"]
        assert res.passed is True
        assert res.validator_used == "run_tests"
