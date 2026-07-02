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
"""Remote-execution configuration (ported from Train_engine ``config/remote``).

SSOT for "where do candidate kernels run". Mirrors Train_engine's
``[remote]`` table with two kinds:

* ``local``  — execute candidates with a local subprocess (default, original
  behaviour; requires a local GPU + torch/triton).
* ``ssh``    — execute candidates on a remote host reached via plain
  ``ssh <hostname>``. The remote host owns the GPU and the torch/triton
  install; this machine only generates code via the LLM.

Resolution order (first wins per field):

1. Environment variables ``KERNEL_REMOTE_KIND`` / ``KERNEL_REMOTE_HOST`` /
   ``KERNEL_REMOTE_WORKSPACE`` (consistent with KernelAgent's ``.env`` style).
2. A TOML file: ``$KERNEL_REMOTE_CONFIG`` if set, else ``./remote.toml`` in the
   current working directory. The table is ``[remote]`` with keys
   ``kind`` / ``hostname`` / ``workspace`` — byte-for-byte the same schema as
   Train_engine's ``config/remote/ssh.toml``.

Pre-conditions for ``kind = "ssh"`` (identical to Train_engine):

* ``~/.ssh/config`` has the ``hostname`` alias configured (proxy / cert / port
  / identity) so plain ``ssh <hostname> <cmd>`` works without extra flags.
* ``rsync`` 3.x on the remote non-interactive ``$PATH``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib  # type: ignore[no-redefine]

__all__ = ["load_remote_config", "is_remote_enabled"]

_VALID_KINDS = ("local", "ssh")


def _read_toml_remote_table() -> dict[str, Any]:
    """Return the ``[remote]`` table from the resolved TOML file, or ``{}``."""
    explicit = os.environ.get("KERNEL_REMOTE_CONFIG", "").strip()
    path = Path(explicit) if explicit else Path.cwd() / "remote.toml"
    if not path.exists():
        if explicit:
            raise RuntimeError(f"KERNEL_REMOTE_CONFIG points at a missing file: {path}")
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    remote = data.get("remote", {})
    if not isinstance(remote, dict):
        raise RuntimeError(f"{path}: [remote] must be a table, got {type(remote).__name__}")
    return remote


def load_remote_config() -> dict[str, str]:
    """Resolve the effective remote config (env overrides TOML overrides default).

    Returns a dict with ``kind`` / ``hostname`` / ``workspace`` (always present).
    Raises ``RuntimeError`` on an invalid kind or an ``ssh`` kind without a
    hostname — fail fast rather than silently degrading to local.
    """
    table = _read_toml_remote_table()

    kind = (os.environ.get("KERNEL_REMOTE_KIND") or table.get("kind") or "local").strip()
    hostname = (os.environ.get("KERNEL_REMOTE_HOST") or table.get("hostname") or "").strip()
    workspace = (os.environ.get("KERNEL_REMOTE_WORKSPACE") or table.get("workspace") or "").strip()

    if kind not in _VALID_KINDS:
        raise RuntimeError(
            f"[remote].kind must be one of {_VALID_KINDS}, got {kind!r}"
        )
    if kind == "ssh" and not hostname:
        raise RuntimeError(
            'Remote execution kind="ssh" requires a hostname. Set '
            "KERNEL_REMOTE_HOST or [remote].hostname in remote.toml."
        )

    return {"kind": kind, "hostname": hostname, "workspace": workspace}


def is_remote_enabled(cfg: dict[str, str]) -> bool:
    return cfg.get("kind", "local") != "local"
