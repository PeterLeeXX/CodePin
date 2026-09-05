"""Lightweight stage timing and opt-in NVTX without creating a CUDA context."""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from contextlib import contextmanager


class ProcessNvtx:
    """Process ranges can span async waits or different worker threads."""

    def __init__(self, enabled: bool):
        self.library = None
        if not enabled:
            return
        errors = []
        for name in ("libnvToolsExt.so.1", "libnvToolsExt.so"):
            try:
                self.library = ctypes.CDLL(name)
                break
            except OSError as exc:
                errors.append(str(exc))
        if self.library is None:
            raise RuntimeError(
                "NVTX requested but libnvToolsExt is unavailable: " + "; ".join(errors)
            )
        self.library.nvtxRangeStartA.argtypes = [ctypes.c_char_p]
        self.library.nvtxRangeStartA.restype = ctypes.c_uint64
        self.library.nvtxRangeEnd.argtypes = [ctypes.c_uint64]
        self.library.nvtxRangeEnd.restype = None
        self.library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self.library.nvtxRangePushA.restype = ctypes.c_int
        self.library.nvtxRangePop.argtypes = []
        self.library.nvtxRangePop.restype = ctypes.c_int

    @contextmanager
    def process_range(self, name: str):
        if self.library is None:
            yield
            return
        range_id = self.library.nvtxRangeStartA(name.encode())
        try:
            yield
        finally:
            self.library.nvtxRangeEnd(range_id)

    @contextmanager
    def capture_range(self):
        if self.library is None:
            yield
            return
        self.library.nvtxRangePushA(b"codepin.benchmark")
        try:
            yield
        finally:
            self.library.nvtxRangePop()


_nvtx = ProcessNvtx(os.environ.get("CODEPIN_PERF_NVTX") == "1")
nvtx_range = _nvtx.process_range


def issue_trace_id(issue: str) -> str:
    """Correlate MCP tasks and conversation exports without changing prompts."""
    return hashlib.sha256(issue.encode()).hexdigest()[:12]


@contextmanager
def measure_stage(stages: dict[str, float], name: str):
    started = time.monotonic()
    with nvtx_range("codepin." + name.removesuffix("_seconds")):
        try:
            yield
        finally:
            stages[name] = time.monotonic() - started
