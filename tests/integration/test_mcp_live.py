"""Live acceptance: requires the real SFT vLLM server; never mocks or skips it."""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import UUID

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_native_batch_and_mcp_delegation(tmp_path):
    base_url = os.environ.get("CODEPIN_TEST_BASE_URL", "http://127.0.0.1:8000/v1")
    deployment = Path(os.environ["CODEPIN_TEST_DEPLOYMENT_FILE"])
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "calculator.py").write_text(
        "def add(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n"
    )
    issue = "calculator.add(2, 3) returns -1 but should return 5. Locate the existing function that must change to fix addition."
    with httpx.Client(timeout=120) as client:
        response = client.post(
            base_url + "/completions",
            json={
                "model": "codepin",
                "prompt": ["def add(a, b):\n", "def multiply(a, b):\n"],
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        response.raise_for_status()
        assert len(response.json()["choices"]) == 2

    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable,
            env=dict(os.environ),
            args=[
                "-m",
                "src.mcp_server",
                "--repository-root",
                str(tmp_path),
                "--base-url",
                base_url,
                "--cache-size",
                "4",
                "--deployment-file",
                str(deployment),
                "--concurrency",
                "2",
            ],
        )
        async with stdio_client(parameters) as (read, write):  # noqa: SIM117 - session needs the transport streams.
            async with ClientSession(read, write) as client:
                await client.initialize()
                listed = await client.list_tools()
                assert {t.name for t in listed.tools} == {
                    "localize_code",
                    "localize_batch",
                }
                request = {
                    "repository": "repository",
                    "issue": issue,
                    "max_context_chars": 500,
                }
                first = await client.call_tool("localize_code", {"request": request})
                assert not first.isError
                result = first.structuredContent
                assert result["status"] == "ok", result
                assert any(
                    loc["file"] == "calculator.py" and loc.get("function_name") == "add"
                    for loc in result["locations"]
                ), result
                assert sum(len(c["text"]) for c in result["context"]) <= 500
                assert result["metrics"]["completion_tokens"] > 0
                assert str(UUID(result["execution_id"])) == result["execution_id"]
                repeat = await client.call_tool("localize_code", {"request": request})
                assert repeat.structuredContent["cache_hit"]
                assert (
                    repeat.structuredContent["execution_id"] == result["execution_id"]
                )
                cached_metrics = repeat.structuredContent["metrics"]
                assert cached_metrics["service_total_seconds"] >= 0
                assert "rollout_seconds" not in cached_metrics
                assert "wall_clock_duration" not in cached_metrics
                assert "repository_digest_after_seconds" not in cached_metrics
                (repo / "new_file.py").write_text("VALUE = 1\n")
                changed = await client.call_tool("localize_code", {"request": request})
                assert not changed.structuredContent["cache_hit"]
                assert changed.structuredContent["snapshot"] != result["snapshot"]
                assert (
                    changed.structuredContent["execution_id"] != result["execution_id"]
                )
                batch = await client.call_tool(
                    "localize_batch",
                    {
                        "requests": [
                            {**request, "issue": issue + " Verify its definition."},
                            {
                                **request,
                                "issue": issue
                                + " Inspect the function before submitting.",
                            },
                        ]
                    },
                )
                assert not batch.isError
                assert len(batch.structuredContent["results"]) == 2
                assert all(
                    r["status"] == "ok" for r in batch.structuredContent["results"]
                )
                invalid = await client.call_tool(
                    "localize_code", {"request": {**request, "repository": "../"}}
                )
                assert invalid.isError
                (tmp_path / "mcp_result.json").write_text(json.dumps(result, indent=2))

    asyncio.run(exercise())
