"""Common tool interface, central registry, and result envelope helpers.

Every agent capability implements :class:`Tool` (name, description, typed JSON
Schema parameters, async ``execute``) and is registered centrally so the LLM
request always carries a single source of truth for callable capabilities
(FR-017). Failures are returned to the model as data, never raised (FR-016),
and successful results reach the model complete and well-formed with non-Latin
text in its native script (FR-023/FR-024) per
``specs/002-full-tool-results-and-delivery-reliability``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)
"""Recursive alias for any JSON-compatible value exchanged with the LLM."""


class ToolArgumentError(ValueError):
    """Raised when a tool argument is present but of an unusable type."""


@dataclass(frozen=True)
class ToolArguments:
    """Typed accessor wrapper over raw JSON arguments from the model.

    Attributes:
        raw: Mapping of parameter names to their decoded JSON values.
    """

    raw: dict[str, JSONValue] = field(default_factory=dict)

    def get_str(self, name: str) -> str | None:
        """Return an optional string argument.

        Args:
            name: Parameter name to look up.

        Returns:
            The value when present and a string; ``None`` when absent or null.

        Raises:
            ToolArgumentError: The value exists but is not a string.
        """
        value = self._get_optional(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ToolArgumentError(f"argument {name!r} must be a string")
        return value

    def get_int(self, name: str) -> int | None:
        """Return an optional integer argument.

        Args:
            name: Parameter name to look up.

        Returns:
            The value when present and an integer; ``None`` when absent or null.

        Raises:
            ToolArgumentError: The value exists but is not an integer.
        """
        value = self._get_optional(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolArgumentError(f"argument {name!r} must be an integer")
        return value

    def _get_optional(self, name: str) -> JSONValue | None:
        """Fetch a raw value, treating explicit nulls as absent."""
        return self.raw.get(name)


class JSONSchemaProperty(BaseModel):
    """A single property entry of a tool's JSON Schema object."""

    type: Literal["string", "integer", "number", "boolean"]
    description: str | None = None
    minimum: int | None = None
    maximum: int | None = None

    def to_wire(self) -> dict[str, JSONValue]:
        """Serialize to the OpenAI-compatible schema fragment."""
        wire: dict[str, JSONValue] = {"type": self.type}
        if self.description is not None:
            wire["description"] = self.description
        if self.minimum is not None:
            wire["minimum"] = self.minimum
        if self.maximum is not None:
            wire["maximum"] = self.maximum
        return wire


class JSONObjectSchema(BaseModel):
    """Top-level JSON Schema object describing a tool's parameters."""

    properties: dict[str, JSONSchemaProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    def to_wire(self) -> dict[str, JSONValue]:
        """Serialize to the OpenAI-compatible ``parameters`` object."""
        return {
            "type": "object",
            "properties": {name: prop.to_wire() for name, prop in self.properties.items()},
            "required": list(self.required),
        }


class ToolResult(BaseModel):
    """Uniform envelope returned by every tool execution.

    Attributes:
        ok: ``False`` when the capability failed; the error text then lives in
            :attr:`payload` as data so the model can adapt (FR-016).
        payload: JSON string of the tool-specific result, delivered complete
            and well-formed end to end with non-Latin text in native script
            (FR-023/FR-024); there is no size cap.
    """

    ok: bool = True
    payload: str = ""


def failure_result(reason: str) -> ToolResult:
    """Build the standard failure envelope for a capability error.

    Args:
        reason: Human-readable explanation handed to the model as data.

    Returns:
        A ``ToolResult`` with ``ok=False`` and ``{"error": ...}`` payload
        serialized with non-ASCII characters preserved (FR-024).
    """
    return ToolResult(ok=False, payload=json.dumps({"error": reason}, ensure_ascii=False))


MAX_PAGE_SIZE = 50


def bounded_limit(raw: int | None, default: int) -> int:
    """Clamp a model-supplied page size into the contract's ``1..MAX_PAGE_SIZE`` range.

    Args:
        raw: Requested limit or ``None`` when omitted.
        default: Fallback from ``TOOL_DEFAULT_LIMIT``.

    Returns:
        An integer within ``[1, MAX_PAGE_SIZE]``.
    """
    value = raw if raw is not None else default
    return max(1, min(MAX_PAGE_SIZE, value))


async def paginate_page[T](
    items: AsyncIterator[T], offset: int, limit: int
) -> tuple[list[T], bool]:
    """Collect one page from an async iterator and report whether more follow.

    Applies the N+1 pattern: consumes at most ``offset + limit + 1`` items so
    the caller learns a further page exists without draining the whole source.
    The probe item is detected by reaching another iteration after the page is
    full, so an exactly-full final page reports ``has_more=False`` correctly.

    Args:
        items: Async iterator over the full result sequence.
        offset: Number of leading items to skip.
        limit: Maximum number of items kept for the page.

    Returns:
        A ``(page, has_more)`` pair where ``has_more`` is ``True`` when an item
        exists beyond the requested page.
    """
    page: list[T] = []
    has_more = False
    index = 0
    async for item in items:
        if index < offset:
            index += 1
            continue
        if len(page) >= limit:
            has_more = True
            break
        page.append(item)
        index += 1
    return page, has_more


class Tool(ABC):
    """Abstract base for all agent capabilities (FR-017)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique registry key exposed to the model."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Model-facing purpose statement shown in the tools payload."""

    @property
    @abstractmethod
    def parameters_schema(self) -> JSONObjectSchema:
        """Typed JSON Schema describing accepted arguments."""

    @abstractmethod
    async def execute(self, arguments: ToolArguments) -> ToolResult:
        """Run the capability.

        Args:
            arguments: Typed accessor over the model-supplied JSON arguments.

        Returns:
            A :class:`ToolResult` envelope; failures must be returned as data,
            not raised (FR-016).
        """


class ToolRegistry:
    """Central registry producing the OpenAI tools payload and dispatching calls."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a capability under its unique name.

        Args:
            tool: The tool instance to add.

        Raises:
            ValueError: A tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool registration: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the registered tool ``name`` or ``None`` when unknown."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Return all registered tool names in registration order."""
        return list(self._tools)

    def to_openai_payload(self) -> list[dict[str, JSONValue]]:
        """Build the ``tools`` array sent with every chat-completions request.

        Returns:
            One function-definition object per registered tool, in
            registration order, per ``contracts/llm-tool-contracts.md``.
        """
        payload: list[dict[str, JSONValue]] = []
        for tool in self._tools.values():
            function: dict[str, JSONValue] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema.to_wire(),
            }
            payload.append({"type": "function", "function": function})
        return payload

    async def execute(self, name: str, arguments: dict[str, JSONValue]) -> ToolResult:
        """Dispatch a tool call with uniform error handling.

        Unknown tools, argument type errors, and any capability exception are
        converted into ``ok=False`` envelopes so the agent turn continues
        (FR-016). Successful results are returned exactly as produced by the
        capability: complete, well-formed JSON with no size cap (FR-023).

        Args:
            name: Registry key of the tool to run.
            arguments: Raw JSON arguments supplied by the model.

        Returns:
            The result envelope ready for a role-``tool`` message.
        """
        tool = self._tools.get(name)
        if tool is None:
            return failure_result(f"unknown tool: {name!r}")
        try:
            result = await tool.execute(ToolArguments(raw=arguments))
        except Exception as exc:  # noqa: BLE001 - errors are data to the model
            logger.exception("tool %s failed", name)
            return failure_result(f"tool {name!r} failed: {exc}")
        return result
