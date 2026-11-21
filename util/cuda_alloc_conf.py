"""Configure PyTorch CUDA caching allocator environment before torch import."""

import os
import sys

_ALLOC_CONF_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
_EXPANDABLE_SEGMENTS_TOKEN = "expandable_segments:True"


def _merge_alloc_conf(existing):
    tokens = [token.strip() for token in existing.split(",") if token.strip()]
    for token in tokens:
        if token.split(":", 1)[0].strip().lower() == "expandable_segments":
            return existing
    tokens.append(_EXPANDABLE_SEGMENTS_TOKEN)
    return ",".join(tokens)


def configure_cuda_alloc_conf(env=None, platform=None):
    """Enable expandable_segments on Linux before PyTorch initializes CUDA."""
    target_env = os.environ if env is None else env
    target_platform = sys.platform if platform is None else platform
    if not target_platform.startswith("linux"):
        return target_env

    target_vars = [
        name for name in _ALLOC_CONF_VARS if target_env.get(name, "").strip()
    ] or ["PYTORCH_ALLOC_CONF"]

    for var_name in target_vars:
        target_env[var_name] = _merge_alloc_conf(target_env.get(var_name, ""))
    return target_env
