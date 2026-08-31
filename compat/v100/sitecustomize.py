"""Runtime compatibility patches for SkyRL on V100 with vLLM 0.18."""

import importlib.util


_find_spec = importlib.util.find_spec


def _find_compatible_spec(name, package=None):
    if name in {"nixl_ep", "nixl_ep_cu12", "nixl_ep_cu13"}:
        return None
    return _find_spec(name, package)


importlib.util.find_spec = _find_compatible_spec

try:
    from vllm.model_executor.model_loader.reload import meta

    if not hasattr(meta, "CopyCounter"):
        meta.CopyCounter = meta.MetaCopyCounter
except (ImportError, AttributeError):
    pass
