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

"""Pin the GPU venv Dockerfile contract.

These are static text checks -- they run on the control machine (no docker,
no GPU) and guard the two files a remote `docker build` consumes:

* ``Dockerfile``            bakes the venv into the image (option A: clean
                            index config, Tsinghua mirrors for both apt and
                            pip, ``/root/venv-ka``).
* ``requirements-gpu.txt``  the pin manifest. Its index directives were
                            removed so a Tsinghua pip mirror can drive the
                            whole install; the header comment must not keep
                            describing an ordering that no longer exists.

A behavioral build test belongs where a GPU exists; these only fail when the
contract drifts, which is the only failure mode a text check can catch here.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCKERFILE = _REPO_ROOT / "Dockerfile"
_REQUIREMENTS = _REPO_ROOT / "requirements-gpu.txt"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    if not _DOCKERFILE.is_file():
        pytest.skip(f"{_DOCKERFILE} not present")
    return _DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def requirements_text() -> str:
    if not _REQUIREMENTS.is_file():
        pytest.skip(f"{_REQUIREMENTS} not present")
    return _REQUIREMENTS.read_text()


# --- requirements-gpu.txt: index config is gone ------------------------------


def test_requirements_has_no_index_url(requirements_text: str):
    assert "--index-url" not in requirements_text, (
        "requirements-gpu.txt must not declare --index-url; the Dockerfile "
        "drives pip through the Tsinghua mirror instead."
    )


def test_requirements_has_no_extra_index_url(requirements_text: str):
    assert "--extra-index-url" not in requirements_text, (
        "requirements-gpu.txt must not declare --extra-index-url; option A "
        "drops the cu126 channel and lets torch resolve from the mirror."
    )


def test_requirements_has_no_trusted_host(requirements_text: str):
    assert "--trusted-host" not in requirements_text, (
        "requirements-gpu.txt must not declare --trusted-host; the cluster "
        "self-signed-cert directive was removed with the index it served."
    )


# --- Dockerfile: base image, mirrors, packages, venv -------------------------


def test_dockerfile_base_image(dockerfile_text: str):
    assert (
        "modelbest-registry.cn-beijing.cr.aliyuncs.com/infra/nvidia-pytorch:latest"
        in dockerfile_text
    ), "Dockerfile must FROM the modelbest nvidia-pytorch:latest image."


def test_dockerfile_installs_required_apt_packages(dockerfile_text: str):
    for pkg in ("rsync", "tmux", "python3-venv"):
        assert pkg in dockerfile_text, (
            f"Dockerfile must apt-install {pkg} (requested system package)."
        )


def test_dockerfile_uses_aliyun_apt_mirror(dockerfile_text: str):
    assert "mirrors.cloud.aliyuncs.com" in dockerfile_text, (
        "Dockerfile must replace the apt sources with the Aliyun intranet "
        "mirror -- Tsinghua returns 403 from ACR build nodes."
    )


def test_dockerfile_apt_mirror_is_http(dockerfile_text: str):
    # ACR's build gateway MITMs https://mirrors.cloud.aliyuncs.com with a
    # self-signed cert (CN mismatch, IP 192.168.222.191), so apt's CA check
    # fails. The apt source MUST be http, not https, to sidestep that.
    assert "http://mirrors.cloud.aliyuncs.com/ubuntu/" in dockerfile_text, (
        "Dockerfile apt mirror must use http (not https) -- the ACR build "
        "gateway's TLS interception breaks https apt fetches."
    )
    assert "https://mirrors.cloud.aliyuncs.com/ubuntu/" not in dockerfile_text, (
        "Dockerfile apt mirror must NOT use https -- ACR gateway MITM breaks it."
    )


def test_dockerfile_apt_mirror_covers_security_host(dockerfile_text: str):
    # The sed must rewrite security.ubuntu.com too, not just archive. -- the
    # noble-security suite ships on security.ubuntu.com and an unmatched host
    # leaves apt fetching from the public Ubuntu mirror over the ACR egress.
    assert "security.ubuntu.com" in dockerfile_text, (
        "Dockerfile apt mirror sed must also cover security.ubuntu.com, the "
        "host for the noble-security suite."
    )


def test_dockerfile_uses_aliyun_pip_mirror(dockerfile_text: str):
    assert "mirrors.cloud.aliyuncs.com/pypi/simple" in dockerfile_text, (
        "Dockerfile must point pip at the Aliyun intranet PyPI mirror; the "
        "Tsinghua PyPI mirror is 403 from ACR build nodes too."
    )


def test_dockerfile_creates_venv_at_root(dockerfile_text: str):
    assert "/root/venv-ka" in dockerfile_text, (
        "Dockerfile must create the venv at /root/venv-ka."
    )


def test_dockerfile_prepends_venv_to_path(dockerfile_text: str):
    assert "/root/venv-ka/bin" in dockerfile_text and "ENV PATH" in dockerfile_text, (
        "Dockerfile must put /root/venv-ka/bin on PATH so a bare python3 "
        "resolves into the venv."
    )


def test_dockerfile_runs_python3_m_venv(dockerfile_text: str):
    assert "python3 -m venv" in dockerfile_text, (
        "Dockerfile must create the venv with stdlib `python3 -m venv`, not uv."
    )


def test_dockerfile_pip_installs_requirements(dockerfile_text: str):
    assert "requirements-gpu.txt" in dockerfile_text and "pip install" in dockerfile_text, (
        "Dockerfile must pip install -r requirements-gpu.txt into the venv."
    )
