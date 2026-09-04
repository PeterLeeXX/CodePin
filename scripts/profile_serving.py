"""Launch the API, worker and real MCP client as one Nsight process tree.

Run under nsys with capture-range=nvtx and nvtx-capture=codepin.benchmark.
Model loading and one full workload warmup happen before the capture range.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil


def stop_process_tree(process: subprocess.Popen, grace_seconds: float = 30) -> None:
    """Stop only our child and its descendants, including new-session MCP clients."""
    try:
        parent = psutil.Process(process.pid)
        targets = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for sig, timeout in (
        (signal.SIGINT, grace_seconds),
        (signal.SIGTERM, grace_seconds),
        (signal.SIGKILL, 10),
    ):
        for target in targets:
            try:
                target.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        _, targets = psutil.wait_procs(targets, timeout=timeout)
        if not targets:
            process.wait(timeout=1)
            return
    raise RuntimeError(f"profiling children did not exit: {[p.pid for p in targets]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="tuning")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--mcp-clients", type=int, default=1)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--minimum-duration", type=float, default=30)
    parser.add_argument("--warmup-duration", type=float, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--without-nvtx", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_performance",
        "e2e",
        "--tasks",
        str(args.tasks),
        "--repository-root",
        str(args.repository_root),
        "--split",
        args.split,
        "--client-concurrency",
        str(args.concurrency),
        "--service-concurrency",
        str(min(32, args.concurrency)),
        "--mcp-clients",
        str(args.mcp_clients),
        "--require-context",
        "--base-url",
        args.base_url,
    ]
    if args.continuous:
        command.append("--continuous")
    with (args.output / "server.log").open("w") as server_log:
        server = subprocess.Popen(
            ["bash", str(args.server_script)],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 360
            with httpx.Client(timeout=2) as client:
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"vLLM exited with {server.returncode}")
                    try:
                        ready = client.get(
                            args.base_url.removesuffix("/v1") + "/health"
                        )
                        if ready.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("vLLM did not become healthy in 360 seconds")
            for label, extra in (
                (
                    "warmup",
                    ["--cycles", "1", "--minimum-duration", str(args.warmup_duration)],
                ),
                (
                    "capture",
                    ["--minimum-duration", str(args.minimum_duration)]
                    + ([] if args.without_nvtx else ["--nvtx"]),
                ),
            ):
                with (args.output / f"{label}.log").open("w") as log:
                    benchmark = subprocess.Popen(
                        command + extra + ["--output", str(args.output / label)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "CODEPIN_PERF_TRACE_DIR": str(args.output / "trajectories"),
                        }
                        if label == "capture"
                        else None,
                    )
                    try:
                        while benchmark.poll() is None:
                            if server.poll() is not None:
                                raise RuntimeError(
                                    f"vLLM exited during {label}: {server.returncode}"
                                )
                            time.sleep(1)
                        if benchmark.returncode:
                            raise subprocess.CalledProcessError(
                                benchmark.returncode, benchmark.args
                            )
                    finally:
                        if benchmark.poll() is None:
                            stop_process_tree(benchmark)
        finally:
            if server.poll() is None:
                stop_process_tree(server)


if __name__ == "__main__":
    main()
