# syntax=docker/dockerfile:1

# GPU image for KernelAgent: nvidia-pytorch base + rsync/tmux + a venv that
# bakes requirements-gpu.txt in.
#
# Option A (per the design that picked it): index configuration lives HERE,
# not in requirements-gpu.txt. Both apt and pip go through Tsinghua mirrors so
# a build inside the CN network does not depend on the cluster-local mirror or
# on download.pytorch.org. torch==2.13.0 therefore resolves from the Tsinghua
# PyPI mirror (generic / CUDA 13 runtime wheel) rather than a cu126 channel --
# acceptable because the base image ships a CUDA runtime matched to its torch,
# and the host driver is assumed to support it. requirements-gpu.txt pins the
# versions; see that file's header for the trade and the rollback path.
#
#   docker build -t kernelagent-gpu-venv .
#   docker run --rm -it --gpus all --cap-add=SYS_ADMIN \
#       -v "$PWD:/workspace" kernelagent-gpu-venv
#
# --cap-add=SYS_ADMIN lets ncu read GPU performance counters; without it every
# profile fails with ERR_NVGPUCTRPERM.

FROM modelbest-registry.cn-beijing.cr.aliyuncs.com/infra/nvidia-pytorch:latest

ARG DEBIAN_FRONTEND=noninteractive

# The base image is Ubuntu 24.04, whose apt sources live in deb822 format at
# /etc/apt/sources.list.d/ubuntu.sources (NOT the legacy /etc/apt/sources.list).
# Back up the original, then rewrite it to the Tsinghua mirror. Done before any
# apt-get so the install itself pulls from the mirror.
#
# sed across URIs only: the deb822 file uses `URIs: http://archive.ubuntu...`
# and `Signed-By` lines we must leave intact, so we swap just the two host
# prefixes (archive.ubuntu.com -> tsinghua, security.ubuntu.com -> tsinghua)
# rather than overwrite the whole file.
RUN set -Eeuo pipefail; \
    src=/etc/apt/sources.list.d/ubuntu.sources; \
    if [ -f "$src" ]; then \
        cp "$src" "${src}.bak"; \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
            -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
            -e 's|http://[a-z]*.archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
            "$src"; \
    fi

# rsync + tmux are the requested system tools; python3-venv backs `python3 -m
# venv` (the base image's python3 alone may ship without ensurepip's venv
# extras); python3-dev covers any future sdist in requirements-gpu.txt;
# ca-certificates is needed for the https Tsinghua mirrors.
RUN set -Eeuo pipefail; \
    apt-get update && apt-get install -y --no-install-recommends \
        rsync \
        tmux \
        python3-venv \
        python3-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# venv at /root/venv-ka (not /workspace -- a run-time bind-mount of the repo
# over /workspace would hide it). python3 -m venv, not uv: per the explicit
# tool choice for this image.
RUN python3 -m venv /root/venv-ka

# Put the venv first on PATH so a bare `python3` / `pip` resolves into it.
ENV PATH="/root/venv-ka/bin:${PATH}"

# pip through the Tsinghua PyPI mirror. requirements-gpu.txt carries no index
# directives of its own (option A), so the CLI flag is the single source of
# the index here.
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r /tmp/requirements-gpu.txt

# Fail the build, not the first run, if the venv did not actually take. A bare
# `python3` must be the venv interpreter and torch must import from it.
RUN set -Eeuo pipefail; \
    test "$(readlink -f "$(command -v python3)")" = "/root/venv-ka/bin/python3"; \
    python3 -c "import torch, triton, numpy; print('torch', torch.__version__, 'triton', triton.__version__)"

WORKDIR /workspace
CMD ["/bin/bash"]
