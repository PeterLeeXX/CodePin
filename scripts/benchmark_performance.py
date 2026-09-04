"""Benchmark fixed CodePin token replays and real MCP localization tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import tarfile
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from src.performance import (
    analyze_token_trajectories,
    build_replay_workload,
    has_runtime_exception,
    histogram_quantile_delta,
    load_trajectories,
    metric_delta,
    metric_value,
    parse_prometheus,
    summarize,
)
from src.profiling import ProcessNvtx, issue_trace_id
from src.rewards.file_localization.file_localization import (
    multilevel_localization_f1_reward,
)

COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
)
HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:e2e_request_latency_seconds",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def new_output(path: Path, service_root: Path | None = None) -> None:
    path.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = sorted(
        {
            Path(__file__).resolve(),
            *root.joinpath("src").rglob("*.py"),
            *root.joinpath("src").rglob("*.j2"),
        }
    )
    hashes = {}
    with tarfile.open(path / "implementation.tar.gz", "x:gz") as archive:
        for source in sources:
            name = source.relative_to(root).as_posix()
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            archive.add(source, arcname=name)
        if service_root is not None and service_root.resolve() != root:
            for source in sorted(
                [
                    *service_root.joinpath("src").rglob("*.py"),
                    *service_root.joinpath("src").rglob("*.j2"),
                ]
            ):
                name = "service/" + source.relative_to(service_root).as_posix()
                hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
                archive.add(source, arcname=name)
    (path / "implementation.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "alternate_service_root": str(service_root.resolve())
                if service_root
                else None,
                "source_sha256": hashes,
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "LITELLM_LOCAL_MODEL_COST_MAP",
                        "CODEPIN_PERF_NVTX",
                        "CODEPIN_PERF_TRACE_DIR",
                    )
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


async def scrape_metrics(session: httpx.AsyncClient, base_url: str) -> str:
    response = await session.get(base_url.removesuffix("/v1") + "/metrics")
    response.raise_for_status()
    return response.text


async def reset_prefix_cache(session: httpx.AsyncClient, base_url: str) -> None:
    url = base_url.removesuffix("/v1") + "/reset_prefix_cache"
    response = await session.post(url)
    if response.status_code != 200:
        raise RuntimeError(
            f"prefix reset failed ({response.status_code}): {response.text}"
        )


def prometheus_delta(before_text: str, after_text: str) -> dict[str, Any]:
    before = parse_prometheus(before_text)
    after = parse_prometheus(after_text)
    result: dict[str, Any] = {
        name: metric_delta(before, after, name) for name in COUNTERS
    }
    for name in HISTOGRAMS:
        count = metric_delta(before, after, name + "_count")
        total = metric_delta(before, after, name + "_sum")
        result[name] = {
            "count": count,
            "mean": total / count if count else None,
            "p50_bucket_upper_bound": histogram_quantile_delta(
                before, after, name, 0.50
            ),
            "p95_bucket_upper_bound": histogram_quantile_delta(
                before, after, name, 0.95
            ),
        }
    queries = result["vllm:prefix_cache_queries_total"]
    hits = result["vllm:prefix_cache_hits_total"]
    result["prefix_cache_hit_rate"] = hits / queries if queries else None
    result["prompt_tokens_by_source"] = {
        source: metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            {"source": source},
        )
        for source in ("local_compute", "local_cache_hit", "external_kv_transfer")
    }
    result["request_finished_reason"] = {
        reason: metric_delta(
            before,
            after,
            "vllm:request_success_total",
            {"finished_reason": reason},
        )
        for reason in ("stop", "length", "abort", "error", "repetition")
    }
    result["final_kv_cache_usage"] = metric_value(after, "vllm:kv_cache_usage_perc")
    return result


class ResourceSampler:
    def __init__(self, session: httpx.AsyncClient, base_url: str, interval: float):
        self.session = session
        self.base_url = base_url
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.stop = asyncio.Event()
        self.started = time.monotonic()
        self._last: dict[str, float] = {}
        self._process_roles = {os.getpid(): "client"}
        for process in Path("/proc").iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            try:
                command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            for pattern, role in (
                (b"src.mcp_server", "mcp"),
                (b"vllm.entrypoints.openai.api_server", "api"),
                (b"vllm.entrypoints.cli.main serve", "api"),
                (b"VLLM::APIServer", "api"),
                (b"VLLM::EngineCore", "engine"),
                (b"VLLM::Worker", "engine"),
            ):
                if command.startswith(pattern) or (
                    b"python" in command.split(b" ")[0] and pattern in command
                ):
                    self._process_roles[int(process.name)] = role
                    break
        self._nvml = None
        self._gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "GPU performance measurements require working NVML"
            ) from exc

    @staticmethod
    def _read_key_values(path: Path) -> dict[str, int]:
        values = {}
        for line in path.read_text().splitlines():
            key, value, *_ = line.split()
            values[key] = int(value)
        return values

    @staticmethod
    def _io_bytes() -> tuple[int, int]:
        read_bytes = 0
        write_bytes = 0
        for line in Path("/sys/fs/cgroup/io.stat").read_text().splitlines():
            values = dict(item.split("=") for item in line.split()[1:])
            read_bytes += int(values.get("rbytes", 0))
            write_bytes += int(values.get("wbytes", 0))
        return read_bytes, write_bytes

    @staticmethod
    def _network_bytes() -> tuple[int, int]:
        received = 0
        sent = 0
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            _, values = line.split(":", 1)
            fields = values.split()
            received += int(fields[0])
            sent += int(fields[8])
        return received, sent

    def _rates(self, now: float, values: dict[str, float]) -> dict[str, float]:
        rates = {}
        previous_time = self._last.get("time")
        if previous_time is not None and now > previous_time:
            elapsed = now - previous_time
            for key, value in values.items():
                previous = self._last.get(key)
                if previous is not None:
                    rates[key + "_per_second"] = max(0.0, value - previous) / elapsed
        self._last = {"time": now, **values}
        return rates

    async def run(self) -> None:
        while not self.stop.is_set():
            now = time.monotonic()
            cpu = self._read_key_values(Path("/sys/fs/cgroup/cpu.stat"))
            io_read, io_write = self._io_bytes()
            net_read, net_write = self._network_bytes()
            cumulative = {
                "cpu_seconds": cpu.get("usage_usec", 0) / 1_000_000,
                "cpu_throttled_seconds": cpu.get("throttled_usec", 0) / 1_000_000,
                "io_read_bytes": float(io_read),
                "io_write_bytes": float(io_write),
                "network_receive_bytes": float(net_read),
                "network_send_bytes": float(net_write),
            }
            for role in set(self._process_roles.values()):
                cumulative[f"{role}_cpu_seconds"] = 0.0
            process_gauges = {
                f"{role}_{metric}": 0
                for role in set(self._process_roles.values())
                for metric in ("rss_bytes", "threads")
            }
            for pid, role in self._process_roles.items():
                try:
                    fields = (
                        Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    )
                except (FileNotFoundError, ProcessLookupError):
                    continue
                cumulative[f"{role}_cpu_seconds"] += (
                    int(fields[11]) + int(fields[12])
                ) / os.sysconf("SC_CLK_TCK")
                # /proc/PID/stat fields 24 (RSS pages) and 20 (thread count).
                # Role RSS can double-count shared pages; cgroup memory is the
                # authoritative total. These gauges identify growth by role.
                process_gauges[f"{role}_rss_bytes"] += int(fields[21]) * os.sysconf(
                    "SC_PAGE_SIZE"
                )
                process_gauges[f"{role}_threads"] += int(fields[17])
            sample: dict[str, Any] = {
                "elapsed_seconds": now - self.started,
                "metrics_scrape_failed": False,
                "memory_current_bytes": int(
                    Path("/sys/fs/cgroup/memory.current").read_text()
                ),
                **self._rates(now, cumulative),
                **process_gauges,
            }
            memory_stat = self._read_key_values(Path("/sys/fs/cgroup/memory.stat"))
            sample.update(
                {
                    f"memory_{key}_bytes": memory_stat[key]
                    for key in ("anon", "file", "shmem")
                }
            )
            if "cpu_seconds_per_second" in sample:
                quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
                if quota != "max":
                    cores = int(quota) / int(period)
                    sample["cpu_quota_utilization_percent"] = (
                        sample["cpu_seconds_per_second"] / cores * 100
                    )
            if self._nvml is not None and self._gpu is not None:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu)
                clock_events = self._nvml.nvmlDeviceGetCurrentClocksEventReasons(
                    self._gpu
                )
                sample.update(
                    gpu_utilization_percent=utilization.gpu,
                    gpu_memory_utilization_percent=utilization.memory,
                    gpu_memory_used_bytes=memory.used,
                    gpu_power_watts=(
                        self._nvml.nvmlDeviceGetPowerUsage(self._gpu) / 1000
                    ),
                    gpu_temperature_c=self._nvml.nvmlDeviceGetTemperature(
                        self._gpu, self._nvml.NVML_TEMPERATURE_GPU
                    ),
                    gpu_sm_clock_mhz=self._nvml.nvmlDeviceGetClockInfo(
                        self._gpu, self._nvml.NVML_CLOCK_SM
                    ),
                    gpu_memory_clock_mhz=self._nvml.nvmlDeviceGetClockInfo(
                        self._gpu, self._nvml.NVML_CLOCK_MEM
                    ),
                    gpu_power_limited=bool(
                        clock_events & self._nvml.nvmlClocksEventReasonSwPowerCap
                    ),
                    gpu_thermal_limited=bool(
                        clock_events
                        & (
                            self._nvml.nvmlClocksEventReasonSwThermalSlowdown
                            | self._nvml.nvmlClocksEventReasonHwThermalSlowdown
                        )
                    ),
                )
            try:
                metrics = parse_prometheus(
                    await asyncio.wait_for(
                        scrape_metrics(self.session, self.base_url),
                        timeout=self.interval,
                    )
                )
                sample.update(
                    vllm_running_requests=metric_value(
                        metrics, "vllm:num_requests_running"
                    ),
                    vllm_waiting_requests=metric_value(
                        metrics, "vllm:num_requests_waiting"
                    ),
                    vllm_kv_cache_usage=metric_value(
                        metrics, "vllm:kv_cache_usage_perc"
                    ),
                )
            except (TimeoutError, httpx.HTTPError):
                sample["metrics_scrape_failed"] = True
            self.samples.append(sample)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.interval)
            except TimeoutError:
                pass
        if self._nvml is not None:
            self._nvml.nvmlShutdown()


def resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if key != "elapsed_seconds" and isinstance(value, int | float)
        }
    )
    return {
        key: summarize(
            float(sample[key])
            for sample in samples
            if isinstance(sample.get(key), int | float)
        )
        for key in keys
    }


def write_results(
    output: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    resources: list[dict[str, Any]],
    before: str,
    after: str,
) -> None:
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "resources.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in resources), encoding="utf-8"
    )
    (output / "metrics-before.prom").write_text(before, encoding="utf-8")
    (output / "metrics-after.prom").write_text(after, encoding="utf-8")


async def stream_completion(
    session: httpx.AsyncClient,
    base_url: str,
    request_id: str,
    prompt: list[int],
    max_tokens: int,
    nvtx: ProcessNvtx,
    cache_salt: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    first_token_at = None
    response_id = None
    finish_reason = None
    usage = {}
    text_chars = 0
    payload = {
        "model": "codepin",
        "prompt": prompt,
        "max_tokens": max(1, max_tokens),
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    try:
        label = f"codepin.request|{request_id}|prompt={len(prompt)}|output={max_tokens}"
        with nvtx.process_range(label):
            async with session.stream(
                "POST",
                base_url + "/completions",
                json=payload,
                headers={"X-Request-Id": request_id},
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    response_id = chunk.get("id") or response_id
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                        text = choice.get("text") or ""
                        if text and first_token_at is None:
                            first_token_at = time.monotonic()
                        text_chars += len(text)
        finished = time.monotonic()
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens != max_tokens or finish_reason != "length":
            raise RuntimeError(
                f"replay budget mismatch: {completion_tokens=}, {max_tokens=}, {finish_reason=}"
            )
        return {
            "status": "ok",
            "request_id": request_id,
            "server_request_id": response_id,
            "prompt_tokens": len(prompt),
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "text_chars": text_chars,
            "latency_seconds": finished - started,
            "ttft_seconds": (
                first_token_at - started if first_token_at is not None else None
            ),
            "tpot_seconds": (
                (finished - first_token_at) / max(1, completion_tokens - 1)
                if first_token_at is not None and completion_tokens > 1
                else None
            ),
        }
    except Exception as exc:  # noqa: BLE001 - every failed request is benchmark data.
        return {
            "status": "error",
            "request_id": request_id,
            "prompt_tokens": len(prompt),
            "completion_tokens": 0,
            "latency_seconds": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_replay(args: argparse.Namespace) -> None:
    new_output(args.output)
    if args.copies_per_cycle < 1:
        raise ValueError("copies-per-cycle must be positive")
    nvtx = ProcessNvtx(args.nvtx)
    trajectories = load_trajectories(args.trajectories)
    workload = build_replay_workload(trajectories)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=100)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as session:
        for _ in range(args.warmup_cycles):
            for task in workload:
                for round_ in task["rounds"]:
                    result = await stream_completion(
                        session,
                        args.base_url,
                        "warmup",
                        round_["prompt_token_ids"],
                        round_["response_tokens"],
                        nvtx,
                    )
                    if result["status"] != "ok":
                        raise RuntimeError(f"replay warmup failed: {result}")
        if args.reset_prefix_before:
            await reset_prefix_cache(session, args.base_url)
        before = await scrape_metrics(session, args.base_url)
        sampler = ResourceSampler(session, args.base_url, args.sample_interval)
        sampler_task = asyncio.create_task(sampler.run())
        records = []
        task_records = []
        benchmark_started = time.monotonic()
        cycle = 0
        try:
            with nvtx.capture_range():
                while cycle < args.cycles or (
                    args.minimum_duration
                    and time.monotonic() - benchmark_started < args.minimum_duration
                ):
                    if cycle and args.reset_prefix_between_cycles:
                        await reset_prefix_cache(session, args.base_url)
                    order = list(workload) * args.copies_per_cycle
                    random.Random(args.seed + cycle).shuffle(order)
                    semaphore = asyncio.Semaphore(args.concurrency)
                    cycle_started = time.monotonic()

                    async def one_task(
                        position: int,
                        task: dict[str, Any],
                        cycle_number: int = cycle,
                        cycle_origin: float = cycle_started,
                        limiter: asyncio.Semaphore = semaphore,
                    ) -> None:
                        due = (
                            cycle_origin + position / args.arrival_rate
                            if args.arrival_rate
                            else cycle_origin
                        )
                        await asyncio.sleep(max(0.0, due - time.monotonic()))
                        submitted = time.monotonic()
                        ok = True
                        with nvtx.process_range(
                            f"codepin.task|{cycle_number}|{task['instance_id']}"
                        ):
                            async with limiter:
                                admitted = time.monotonic()
                                for round_ in task["rounds"]:
                                    request_id = (
                                        f"replay-{cycle_number}-{position}-"
                                        f"{round_['turn']}-{task['instance_id']}"
                                    )
                                    result = await stream_completion(
                                        session,
                                        args.base_url,
                                        request_id,
                                        round_["prompt_token_ids"],
                                        round_["response_tokens"],
                                        nvtx,
                                        cache_salt=(
                                            f"{args.output.name}:task-{cycle_number}-{position}"
                                            if args.prefix_scope == "task"
                                            else f"{args.output.name}:{request_id}"
                                            if args.prefix_scope == "request"
                                            else None
                                        ),
                                    )
                                    result.update(
                                        cycle=cycle_number,
                                        instance_id=task["instance_id"],
                                        turn=round_["turn"],
                                    )
                                    records.append(result)
                                    ok &= result["status"] == "ok"
                                    if args.replay_tool_delays and round_[
                                        "turn"
                                    ] + 1 < len(task["rounds"]):
                                        await asyncio.sleep(
                                            round_["tool_duration_seconds"]
                                        )
                        task_records.append(
                            {
                                "cycle": cycle_number,
                                "instance_id": task["instance_id"],
                                "status": "ok" if ok else "error",
                                "queue_seconds": admitted - submitted,
                                "latency_seconds": time.monotonic() - submitted,
                                "turns": len(task["rounds"]),
                            }
                        )

                    await asyncio.gather(
                        *(
                            one_task(position, task)
                            for position, task in enumerate(order)
                        )
                    )
                    cycle += 1
                benchmark_duration = time.monotonic() - benchmark_started
        finally:
            sampler.stop.set()
            await sampler_task
        after = await scrape_metrics(session, args.base_url)
    completed = sum(row["status"] == "ok" for row in task_records)
    summary = {
        "mode": "fixed_token_replay",
        "config": vars(args)
        | {"output": str(args.output), "trajectories": str(args.trajectories)},
        "cycles": cycle,
        "tasks": len(task_records),
        "completed_tasks": completed,
        "errors": len(task_records) - completed,
        "duration_seconds": benchmark_duration,
        "completed_tasks_per_minute": completed / benchmark_duration * 60,
        "task_latency_seconds": summarize(
            row["latency_seconds"] for row in task_records
        ),
        "task_queue_seconds": summarize(row["queue_seconds"] for row in task_records),
        "request_latency_seconds": summarize(row["latency_seconds"] for row in records),
        "request_ttft_seconds": summarize(
            row["ttft_seconds"]
            for row in records
            if row.get("ttft_seconds") is not None
        ),
        "request_tpot_seconds": summarize(
            row["tpot_seconds"]
            for row in records
            if row.get("tpot_seconds") is not None
        ),
        "vllm_metrics": prometheus_delta(before, after),
        "resources": resource_summary(sampler.samples),
        "resource_processes": sampler._process_roles,
    }
    write_results(args.output, records, summary, sampler.samples, before, after)
    (args.output / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in task_records), encoding="utf-8"
    )
    print(json.dumps(summary))


@asynccontextmanager
async def mcp_session(args: argparse.Namespace):
    from mcp import ClientSession, StdioServerParameters

    if args.mcp_url:
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(args.mcp_url) as streams:
            read, write = streams[:2]
            async with ClientSession(read, write) as client:
                await client.initialize()
                yield client
    else:
        from mcp.client.stdio import stdio_client

        # Preserve the same declared environment for plain and profiled children,
        # including LiteLLM's explicit local cost map and profiler injection.
        environment = dict(os.environ)
        environment.pop("CODEPIN_PERF_NVTX", None)
        if args.nvtx:
            environment["CODEPIN_PERF_NVTX"] = "1"
        if args.save_trajectories:
            environment["CODEPIN_PERF_TRACE_DIR"] = str(args.output / "trajectories")
        parameters = StdioServerParameters(
            command=sys.executable,
            cwd=str(args.service_root) if args.service_root else None,
            env=environment or None,
            args=[
                "-m",
                "src.mcp_server",
                "--repository-root",
                str(args.repository_root),
                "--base-url",
                args.base_url,
                "--model",
                "openai/codepin",
                "--concurrency",
                str(args.service_concurrency),
                "--max-turns",
                str(args.max_turns),
                "--max-tokens",
                str(args.max_tokens),
                "--cache-size",
                "0",
            ],
        )
        async with (
            stdio_client(parameters) as (read, write),
            ClientSession(read, write) as client,
        ):
            await client.initialize()
            yield client


async def run_e2e(args: argparse.Namespace) -> None:
    if args.service_root:
        args.service_root = args.service_root.resolve()
        if args.mcp_url or not (args.service_root / "src/mcp_server.py").is_file():
            raise ValueError(
                "service-root must be a CodePin checkout for a spawned MCP server"
            )
    new_output(args.output, args.service_root)
    if args.copies_per_cycle < 1:
        raise ValueError("copies-per-cycle must be positive")
    if not 1 <= args.mcp_clients <= 64:
        raise ValueError("mcp-clients must be in 1..64")
    if args.client_concurrency < 1 or args.arrival_rate < 0 or args.max_pending < 0:
        raise ValueError("invalid concurrency, arrival rate or pending-task limit")
    if args.continuous and args.reset_prefix_between_cycles:
        raise ValueError(
            "continuous requests cannot reset shared cache while tasks run"
        )
    nvtx = ProcessNvtx(args.nvtx)
    tasks = [
        task
        for task in read_jsonl(args.tasks)
        if args.split == "all" or task.get("benchmark_split") == args.split
    ]
    if not tasks:
        raise ValueError("the selected task split is empty")
    async with (
        httpx.AsyncClient(timeout=args.timeout) as http,
        AsyncExitStack() as sessions,
    ):
        mcp_started = time.monotonic()
        clients = [
            await sessions.enter_async_context(mcp_session(args))
            for _ in range(args.mcp_clients)
        ]
        for client in clients:
            listed = await client.list_tools()
            if {tool.name for tool in listed.tools} != {
                "localize_code",
                "localize_batch",
            }:
                raise RuntimeError("unexpected MCP tool set")
        mcp_startup_seconds = time.monotonic() - mcp_started
        if args.reset_prefix_before:
            await reset_prefix_cache(http, args.base_url)
        before = await scrape_metrics(http, args.base_url)
        sampler = ResourceSampler(http, args.base_url, args.sample_interval)
        sampler_task = asyncio.create_task(sampler.run())
        records = []
        benchmark_started = time.monotonic()
        cycle = 0
        scheduled_count = 0
        pending: set[asyncio.Task] = set()
        pending_limit = args.max_pending or args.client_concurrency * (
            2 if args.arrival_rate else 1
        )
        semaphore = asyncio.Semaphore(args.client_concurrency)
        try:
            with nvtx.capture_range():
                while cycle < args.cycles or (
                    args.minimum_duration
                    and time.monotonic() - benchmark_started < args.minimum_duration
                ):
                    if cycle and args.reset_prefix_between_cycles:
                        await reset_prefix_cache(http, args.base_url)
                    order = list(tasks) * args.copies_per_cycle
                    random.Random(args.seed + cycle).shuffle(order)
                    cycle_started = (
                        benchmark_started + scheduled_count / args.arrival_rate
                        if args.continuous and args.arrival_rate
                        else time.monotonic()
                    )

                    async def one_task(
                        position: int,
                        task: dict[str, Any],
                        cycle_number: int = cycle,
                        cycle_origin: float = cycle_started,
                        limiter: asyncio.Semaphore = semaphore,
                        block_size: int = len(order),
                    ) -> None:
                        due = (
                            cycle_origin + position / args.arrival_rate
                            if args.arrival_rate
                            else time.monotonic()
                            if args.continuous
                            else cycle_origin
                        )
                        await asyncio.sleep(max(0.0, due - time.monotonic()))
                        submitted = time.monotonic()
                        record: dict[str, Any] = {
                            "cycle": cycle_number,
                            "position": position,
                            "instance_id": task["instance_id"],
                            "difficulty": task["difficulty"],
                            "problem_chars": len(task["problem_statement"]),
                            "submitted_offset_seconds": submitted - benchmark_started,
                            "submission_lag_seconds": max(0.0, submitted - due),
                        }
                        try:
                            issue_id = issue_trace_id(task["problem_statement"])
                            label = f"codepin.task|{cycle_number}|{task['instance_id']}|{issue_id}"
                            with nvtx.process_range(label):
                                async with limiter:
                                    admitted = time.monotonic()
                                    record["client_queue_seconds"] = (
                                        admitted - submitted
                                    )
                                    record["admitted_offset_seconds"] = (
                                        admitted - benchmark_started
                                    )
                                    response = await asyncio.wait_for(
                                        clients[
                                            (cycle_number * block_size + position)
                                            % len(clients)
                                        ].call_tool(
                                            "localize_code",
                                            {
                                                "request": {
                                                    "repository": task["repository"],
                                                    "issue": task["problem_statement"],
                                                    "max_context_chars": args.max_context_chars,
                                                    "max_context_lines": args.max_context_lines,
                                                }
                                            },
                                        ),
                                        timeout=args.timeout,
                                    )
                            result = response.structuredContent or {}
                            if response.isError:
                                raise RuntimeError(str(response.content))
                            quality, quality_metrics = (
                                multilevel_localization_f1_reward(
                                    instance=task,
                                    structured_locations=result.get("locations"),
                                )
                            )
                            metrics = result.get("metrics") or {}
                            context = result.get("context") or []
                            context_chars = sum(
                                len(item.get("text", "")) for item in context
                            )
                            context_lines = sum(
                                len(item.get("text", "").splitlines())
                                for item in context
                            )
                            effective = (
                                result.get("status") == "ok"
                                and not result.get("errors")
                                and not result.get("cache_hit")
                                and quality >= args.minimum_quality
                                and metrics.get("tool_errors", 0)
                                <= args.maximum_tool_errors
                                and context_chars <= args.max_context_chars
                                and context_lines <= args.max_context_lines
                                and (not args.require_context or bool(context))
                            )
                            record.update(
                                status=result.get("status", "error"),
                                effective=effective,
                                cache_hit=bool(result.get("cache_hit")),
                                execution_id=result.get("execution_id"),
                                context_chars=context_chars,
                                context_lines=context_lines,
                                locations=result.get("locations") or [],
                                errors=result.get("errors") or [],
                                quality=quality,
                                quality_metrics=quality_metrics,
                                metrics=metrics,
                            )
                        except Exception as exc:  # noqa: BLE001 - benchmark datum.
                            record.update(
                                status="error",
                                effective=False,
                                exception_type=type(exc).__name__,
                                errors=[f"{type(exc).__name__}: {exc}"],
                            )
                        record["latency_seconds"] = time.monotonic() - submitted
                        if "service_total_seconds" in record.get("metrics", {}):
                            record["service_transport_seconds"] = max(
                                0.0,
                                record["latency_seconds"]
                                - record.get("client_queue_seconds", 0.0)
                                - record["metrics"]["service_total_seconds"],
                            )
                        records.append(record)

                    if args.continuous:
                        for position, task in enumerate(order):
                            if args.arrival_rate:
                                due = cycle_started + position / args.arrival_rate
                                await asyncio.sleep(max(0.0, due - time.monotonic()))
                            done = {item for item in pending if item.done()}
                            for item in done:
                                item.result()
                            pending.difference_update(done)
                            scheduled_count += 1
                            if not args.arrival_rate and len(pending) >= pending_limit:
                                done, pending = await asyncio.wait(
                                    pending, return_when=asyncio.FIRST_COMPLETED
                                )
                                for item in done:
                                    item.result()
                            if args.arrival_rate and len(pending) >= pending_limit:
                                now = time.monotonic()
                                records.append(
                                    {
                                        "cycle": cycle,
                                        "position": position,
                                        "instance_id": task["instance_id"],
                                        "difficulty": task["difficulty"],
                                        "problem_chars": len(task["problem_statement"]),
                                        "submitted_offset_seconds": now
                                        - benchmark_started,
                                        "submission_lag_seconds": max(0.0, now - due),
                                        "status": "error",
                                        "effective": False,
                                        "admission_rejected": True,
                                        "errors": ["client_admission_limit"],
                                        "latency_seconds": 0.0,
                                    }
                                )
                            else:
                                pending.add(
                                    asyncio.create_task(one_task(position, task))
                                )
                            await asyncio.sleep(0)
                    else:
                        await asyncio.gather(
                            *(
                                one_task(position, task)
                                for position, task in enumerate(order)
                            )
                        )
                    cycle += 1
                await asyncio.gather(*pending)
                benchmark_duration = time.monotonic() - benchmark_started
        finally:
            sampler.stop.set()
            await sampler_task
        after = await scrape_metrics(http, args.base_url)
    successful = sum(row["status"] == "ok" for row in records)
    effective = sum(bool(row["effective"]) for row in records)
    timeout_count = sum(
        any(
            "timeout" in error.lower() or "timed out" in error.lower()
            for error in row.get("errors", [])
        )
        for row in records
    )
    tool_keys = (
        "num_tool_calls",
        "num_turns",
        "search_calls",
        "read_calls",
        "repeated_searches",
        "read_lines",
        "overlap_lines",
        "overlap_ratio",
        "output_chars",
        "excess_output_chars",
        "truncated_outputs",
        "tool_errors",
        "tool_efficiency_cost",
        "completion_tokens",
        "prompt_tokens",
        "max_prompt_tokens",
        "wall_clock_duration",
    )
    stage_keys = (
        "agent_setup_seconds",
        "conversation_run_seconds",
        "conversation_serialize_seconds",
        "conversation_close_seconds",
        "repository_digest_before_seconds",
        "cache_key_before_seconds",
        "rollout_seconds",
        "bounded_context_seconds",
        "repository_digest_after_seconds",
        "cache_key_after_seconds",
        "service_queue_seconds",
        "service_total_seconds",
    )
    summary = {
        "mode": "mcp_end_to_end",
        "config": vars(args)
        | {
            "output": str(args.output),
            "tasks": str(args.tasks),
            "repository_root": str(args.repository_root),
            "service_root": str(args.service_root) if args.service_root else None,
        },
        "diagnostics": {
            "nvtx": args.nvtx,
            "trajectory_directory": str(args.output / "trajectories")
            if args.save_trajectories
            else os.environ.get("CODEPIN_PERF_TRACE_DIR") or None,
        },
        "mcp_startup_seconds": mcp_startup_seconds,
        "cycles": cycle,
        "submitted_tasks": len(records),
        "successful_tasks": successful,
        "effective_tasks": effective,
        "duration_seconds": benchmark_duration,
        "terminal_tasks_per_minute": len(records) / benchmark_duration * 60,
        "admitted_terminal_tasks_per_minute": sum(
            not row.get("admission_rejected") for row in records
        )
        / benchmark_duration
        * 60,
        "successful_tasks_per_minute": successful / benchmark_duration * 60,
        "effective_tasks_per_minute": effective / benchmark_duration * 60,
        "success_rate": successful / len(records),
        "effective_rate": effective / len(records),
        "timeout_rate": timeout_count / len(records),
        "error_rate": (len(records) - successful) / len(records),
        "admission_rejection_rate": sum(
            bool(row.get("admission_rejected")) for row in records
        )
        / len(records),
        "infrastructure_error_rate": sum(has_runtime_exception(row) for row in records)
        / len(records),
        "latency_seconds": summarize(row["latency_seconds"] for row in records),
        "admitted_latency_seconds": summarize(
            row["latency_seconds"]
            for row in records
            if not row.get("admission_rejected")
        ),
        "successful_latency_seconds": summarize(
            row["latency_seconds"] for row in records if row["status"] == "ok"
        ),
        "client_queue_seconds": summarize(
            row.get("client_queue_seconds", 0.0) for row in records
        ),
        "submission_lag_seconds": summarize(
            row["submission_lag_seconds"] for row in records
        ),
        "service_transport_seconds": summarize(
            row["service_transport_seconds"]
            for row in records
            if "service_transport_seconds" in row
        ),
        "quality": summarize(float(row.get("quality", 0)) for row in records),
        "file_f1": summarize(
            float(row.get("quality_metrics", {}).get("file_f1", 0)) for row in records
        ),
        "class_f1": summarize(
            float(row.get("quality_metrics", {}).get("class_f1", 0)) for row in records
        ),
        "function_f1": summarize(
            float(row.get("quality_metrics", {}).get("function_f1", 0))
            for row in records
        ),
        "tool_metrics": {
            key: summarize(float(row.get("metrics", {}).get(key, 0)) for row in records)
            for key in tool_keys
        },
        "stage_metrics": {
            key: summarize(
                float(row["metrics"][key])
                for row in records
                if key in row.get("metrics", {})
            )
            for key in stage_keys
        },
        "vllm_metrics": prometheus_delta(before, after),
        "resources": resource_summary(sampler.samples),
        "resource_processes": sampler._process_roles,
    }
    write_results(args.output, records, summary, sampler.samples, before, after)
    print(json.dumps(summary))


def run_analyze(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze_token_trajectories(
        load_trajectories(args.trajectories), cache_block_size=args.cache_block_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def run_aggregate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    metrics = (
        "effective_tasks_per_minute",
        "successful_tasks_per_minute",
        "terminal_tasks_per_minute",
        "completed_tasks_per_minute",
        "success_rate",
        "effective_rate",
    )

    def latency(row: dict[str, Any]) -> dict[str, Any]:
        return row.get("latency_seconds") or row.get("task_latency_seconds") or {}

    report = {
        "runs": len(rows),
        "metrics": {
            key: summarize(float(row[key]) for row in rows if key in row)
            for key in metrics
        },
        "latency_p50_seconds": summarize(
            float(latency(row)["p50"])
            for row in rows
            if latency(row).get("p50") is not None
        ),
        "latency_p95_seconds": summarize(
            float(latency(row)["p95"])
            for row in rows
            if latency(row).get("p95") is not None
        ),
        "source_summaries": [str(path) for path in args.summaries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--copies-per-cycle", type=int, default=1)
    parser.add_argument("--minimum-duration", type=float, default=0)
    parser.add_argument("--arrival-rate", type=float, default=0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--sample-interval", type=float, default=1)
    parser.add_argument("--reset-prefix-before", action="store_true")
    parser.add_argument("--reset-prefix-between-cycles", action="store_true")
    parser.add_argument("--nvtx", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze-prefix")
    analyze.add_argument("--trajectories", type=Path, required=True)
    analyze.add_argument("--cache-block-size", type=int, required=True)
    analyze.add_argument("--output", type=Path, required=True)

    replay = subparsers.add_parser("replay")
    common_arguments(replay)
    replay.add_argument("--trajectories", type=Path, required=True)
    replay.add_argument("--concurrency", type=int, required=True)
    replay.add_argument("--warmup-cycles", type=int, default=0)
    replay.add_argument("--replay-tool-delays", action="store_true")
    replay.add_argument(
        "--prefix-scope",
        choices=["shared", "task", "request"],
        default="shared",
        help="Native cache_salt isolation for diagnosing cross-task versus within-task reuse.",
    )

    e2e = subparsers.add_parser("e2e")
    common_arguments(e2e)
    e2e.add_argument("--tasks", type=Path, required=True)
    e2e.add_argument("--repository-root", type=Path, required=True)
    e2e.add_argument(
        "--service-root",
        type=Path,
        help="Run the spawned MCP service from an isolated CodePin checkout; archive both driver and service sources.",
    )
    e2e.add_argument(
        "--split", choices=["tuning", "validation", "all"], default="tuning"
    )
    e2e.add_argument("--mcp-url")
    e2e.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Export actual token/tool events for diagnostics; keep off for formal timing.",
    )
    e2e.add_argument(
        "--mcp-clients",
        type=int,
        default=1,
        help="Independent Coding Agent MCP clients; stdio starts one server process per client.",
    )
    e2e.add_argument("--client-concurrency", type=int, required=True)
    e2e.add_argument(
        "--continuous",
        action="store_true",
        help="Refill completed tasks continuously; no wave barrier.",
    )
    e2e.add_argument(
        "--max-pending",
        type=int,
        default=0,
        help="Bound active plus queued requests; excess paced arrivals are recorded as rejections. Default: concurrency (closed loop), twice concurrency (paced).",
    )
    e2e.add_argument("--service-concurrency", type=int, required=True)
    e2e.add_argument("--max-turns", type=int, default=8)
    e2e.add_argument("--max-tokens", type=int, default=2048)
    e2e.add_argument("--max-context-chars", type=int, default=12000)
    e2e.add_argument("--max-context-lines", type=int, default=160)
    e2e.add_argument("--minimum-quality", type=float, default=0.5)
    e2e.add_argument("--maximum-tool-errors", type=float, default=0)
    e2e.add_argument("--require-context", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--summaries", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "analyze-prefix":
        run_analyze(args)
    elif args.command == "aggregate":
        run_aggregate(args)
    elif args.command == "replay":
        asyncio.run(run_replay(args))
    else:
        asyncio.run(run_e2e(args))


if __name__ == "__main__":
    main()
