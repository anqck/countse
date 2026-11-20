"""Configure PyTorch CUDA caching allocator environment before torch import."""

import importlib.metadata
import os
import re
import sys

_VERSION_RE = re.compile(r"^\s*v?(\d+)(?:\.(\d+))?")
_ALLOC_CONF_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
_EXPANDABLE_SEGMENTS_TOKEN = "expandable_segments:True"


def _parse_major_minor(version_str):
    match = _VERSION_RE.match(str(version_str))
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor)


def _merge_alloc_conf(existing):
    tokens = [token.strip() for token in existing.split(",") if token.strip()]
    for token in tokens:
        key = token.split(":", 1)[0].strip()
        if key.lower() == "expandable_segments":
            return existing
    tokens.append(_EXPANDABLE_SEGMENTS_TOKEN)
    return ",".join(tokens)


def configure_cuda_alloc_conf(env=None, platform=None, torch_version=None):
    """Enable expandable_segments on Linux before PyTorch initializes CUDA."""
    target_env = os.environ if env is None else env
    target_platform = sys.platform if platform is None else platform
    if not target_platform.startswith("linux"):
        return target_env

    target_vars = [
        name for name in _ALLOC_CONF_VARS if target_env.get(name, "").strip()
    ]
    if not target_vars:
        if torch_version is None:
            try:
                torch_version = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                return target_env
        parsed = _parse_major_minor(torch_version)
        if parsed is None:
            return target_env
        target_vars = [
            "PYTORCH_ALLOC_CONF" if parsed >= (2, 9) else "PYTORCH_CUDA_ALLOC_CONF"
        ]

    for var_name in target_vars:
        target_env[var_name] = _merge_alloc_conf(target_env.get(var_name, ""))
    return target_env
