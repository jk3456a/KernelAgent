# syntax=docker/dockerfile:1

# GPU image for KernelAgent: nvidia-pytorch base + rsync/tmux + a venv that
# bakes requirements-gpu.txt in.
#
# Option A (per the design that picked it): index configuration lives HERE,
# not in requirements-gpu.txt. apt goes through the Aliyun intranet mirror
# (mirrors.cloud.aliyuncs.com, http) and pip through mirrors.aliyun.com
# (https) -- the two are different hosts on purpose: apt's
# mirrors.cloud.aliyuncs.com works over http on the ACR internal network,
# but its TLS cert does not cover that hostname so pip (which wants https)
# must use mirrors.aliyun.com instead. torch==2.13.0 resolves from that PyPI
# mirror (generic / CUDA 13 runtime wheel) rather than a cu126 channel --
# acceptable because the base image ships a CUDA runtime matched to its
# torch, and the host driver is assumed to support it. requirements-gpu.txt
# pins the versions; see that file's header for the trade and rollback path.
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
# Back up the original, then rewrite it to the Aliyun intranet mirror. Done
# before any apt-get so the install itself pulls from the mirror.
#
# http, NOT https: ACR's build gateway MITMs https://mirrors.cloud.aliyuncs.com
# with a self-signed certificate (CN mismatch, seen at IP 192.168.222.191), so
# apt's CA check rejects every https suite with "The certificate is NOT
# trusted." The Aliyun intranet mirror serves plain http too, and on the ACR
# internal network that is the path that works.
#
# Two hosts, both rewritten: archive.ubuntu.com carries noble / noble-updates
# / noble-backports, security.ubuntu.com carries noble-security. The previous
# sed matched only `archive.ubuntu.com` (and a phantom `security.archive...`),
# so noble-security kept hitting the public Ubuntu mirror -- fixed by matching
# each host explicitly. We swap only the host prefix, leaving the deb822
# `Signed-By:` lines intact; a legacy sources.list fallback covers a base that
# still uses the old format.
RUN set -Eeuo pipefail; \
    src=/etc/apt/sources.list.d/ubuntu.sources; \
    if [ -f "$src" ]; then \
        cp "$src" "${src}.bak"; \
        sed -i \
            -e 's|https\?://archive\.ubuntu\.com/ubuntu/|http://mirrors.cloud.aliyuncs.com/ubuntu/|g' \
            -e 's|https\?://security\.ubuntu\.com/ubuntu/|http://mirrors.cloud.aliyuncs.com/ubuntu/|g' \
            "$src"; \
    else \
        sed -i \
            -e 's|https\?://archive\.ubuntu\.com/ubuntu/|http://mirrors.cloud.aliyuncs.com/ubuntu/|g' \
            -e 's|https\?://security\.ubuntu\.com/ubuntu/|http://mirrors.cloud.aliyuncs.com/ubuntu/|g' \
            /etc/apt/sources.list; \
    fi

# rsync + tmux are the requested system tools; python3-venv backs `python3 -m
# venv` (the base image's python3 alone may ship without ensurepip's venv
# extras); python3-dev covers any future sdist in requirements-gpu.txt;
# ca-certificates stays for the https pip mirror (apt itself runs over http
# to the Aliyun mirror, see the note above).
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

# Clear the NGC base image's PIP_CONSTRAINT (/etc/pip/constraint.txt), which
# pins the NGC torch (2.7.0a0+nv25.4). Left set, pip honors it inside the
# venv too -- the torch==2.13.0 pin either conflicts or installs against a
# constraint that disagrees, and the import check then fails. ENV, not a
# per-shell `unset`, so it covers EVERY subsequent RUN (pip install AND the
# import verification), not just the one shell that sourced activate.
ENV PIP_CONSTRAINT=

# pip through the mirrors.aliyun.com PyPI mirror. requirements-gpu.txt
# carries no index directives of its own (option A), so the CLI flag is the
# single source of the index here.
#
# mirrors.aliyun.com, NOT mirrors.cloud.aliyuncs.com: the latter serves a TLS
# certificate whose SAN covers only mirrors-ssl.aliyuncs.com and
# mirrors.aliyun.com, so pip's hostname check rejects every request with
# "hostname 'mirrors.cloud.aliyuncs.com' doesn't match" and torch never
# downloads -- which then surfaces as a bogus ResolutionImpossible against
# the base image's preinstalled NGC torch (pip falls back to the local
# 2.7.0a0+nv25.4 when the remote fetch fails). mirrors.aliyun.com is in the
# cert's SAN and serves the same PyPI mirror.
#
# `source activate` in the SAME RUN as pip, deliberately, not just ENV PATH:
# the nvidia-pytorch base image preinstalls an NGC torch that pip's resolver
# could otherwise pick up as an installed constraint; activating the venv
# gives pip a clean interpreter whose site-packages do not see it. The
# activate is per-shell -- it must run in this RUN, an earlier one would not
# survive into it.
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN set -Eeuo pipefail; \
    source /root/venv-ka/bin/activate; \
    pip install --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple \
        -r /tmp/requirements-gpu.txt

# Fail the build, not the first run, if the venv install did not actually
# take. The check that matters is that torch loads FROM the venv's
# site-packages -- NOT that `python3` resolves to /root/venv-ka/bin/python3.
# On the NGC base image `command -v python3` returns /usr/bin/python3.12
# (the venv's python3 symlink is not what the shell lookup finds), yet torch
# still imports from /root/venv-ka/lib/python3.12/site-packages -- which is
# the property we care about. The old readlink test was a false negative that
# failed every build after the install itself succeeded.
RUN set -Eeuo pipefail; \
    python3 -c "import torch, triton, numpy; \
                assert torch.__file__.startswith('/root/venv-ka/'), \
                    'torch not from venv: ' + torch.__file__; \
                print('torch', torch.__version__, 'triton', triton.__version__, 'numpy', numpy.__version__)"

WORKDIR /workspace
CMD ["/bin/bash"]
