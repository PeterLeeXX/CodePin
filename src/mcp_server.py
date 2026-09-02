"""MCP delegation endpoint for Coding Agents (stdio or local Streamable HTTP)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from src.service import LocalizationRequest, LocalizationService, ServiceConfig
from src.tools.localization_finish import CodeLocation


class LocalizationResult(BaseModel):
    status: Literal["ok", "error"]
    locations: list[CodeLocation] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    snapshot: str = ""
    cache_hit: bool = False


class BatchResult(BaseModel):
    results: list[LocalizationResult]


def create_server(
    config: ServiceConfig, host: str = "127.0.0.1", port: int = 8001
) -> FastMCP:
    service = LocalizationService(config)
    server = FastMCP("CodePin", host=host, port=port)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    async def localize_code(request: LocalizationRequest) -> LocalizationResult:
        """Delegate a code-localization issue; return locations and bounded source context."""
        return LocalizationResult.model_validate(await service.localize(request))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    async def localize_batch(requests: list[LocalizationRequest]) -> BatchResult:
        """Localize 1..32 independent issues using bounded native serving concurrency."""
        return BatchResult.model_validate({"results": await service.batch(requests)})

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="openai/codepin")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--cache-ttl", type=float, default=300)
    parser.add_argument("--deployment-file", type=Path)
    parser.add_argument(
        "--transport", choices=["stdio", "streamable-http"], default="stdio"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = vars(parser.parse_args())
    transport, host, port = args.pop("transport"), args.pop("host"), args.pop("port")
    create_server(ServiceConfig(**args), host, port).run(transport=transport)


if __name__ == "__main__":
    main()
