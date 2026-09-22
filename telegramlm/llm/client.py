"""Minimal async client for OpenAI-compatible chat-completions endpoints.

The client speaks exactly one request shape (``POST {base_url}/chat/completions``
with an optional ``tools`` array) so any compatible server — hosted or local —
works by changing configuration alone. It serializes typed message objects,
including image content parts for vision turns, and parses either a plain
assistant message or a list of tool calls (never both concerns are lost).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import SecretStr

from telegramlm.tools.base import JSONValue

logger = logging.getLogger(__name__)

# Wall-clock timeout for one chat-completions request; generous enough for
# reasoning models while bounding a hung upstream (plan performance goals).
DEFAULT_TIMEOUT_SECONDS = 90.0

_CONTEXT_OVERFLOW_MARKERS: tuple[str, ...] = (
    "exceed_context_size_error",
    "context_length_exceeded",
    "exceeds the available context size",
)


def _is_context_overflow(text: str) -> bool:
    """Detect a context-window rejection from the raw response body.

    Args:
        text: Untrusted response body of a failed request.

    Returns:
        ``True`` when any :data:`_CONTEXT_OVERFLOW_MARKERS` marker appears in
        ``text`` (case-insensitive); ``False`` otherwise.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


class LLMError(Exception):
    """Base class for language-service failures."""


class LLMTransportError(LLMError):
    """The endpoint could not be reached or timed out at the transport level."""


class LLMAPIError(LLMError):
    """The endpoint answered with a non-success status or a malformed body."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Record the failure description and optional HTTP status.

        Args:
            message: Human-readable error description (never contains secrets).
            status_code: HTTP status when the failure came from an answer.
        """
        super().__init__(message)
        self.status_code = status_code


class LLMContextOverflowError(LLMAPIError):
    """The request exceeded the model's context window.

    A specialization of :class:`LLMAPIError` carrying no extra state: the
    inherited ``status_code`` and message (which embeds the service's own
    wording) already describe the failure. It exists so callers can branch on
    an overflow without parsing error bodies or coupling to one server's field
    names — detection is by markers in the raw response text alone.
    """


@dataclass(frozen=True)
class TextPart:
    """A plain text content part of a request message."""

    text: str


@dataclass(frozen=True)
class ImagePart:
    """An image content part carried as a base64 ``data:`` URL.

    Encoding happens exclusively at request-build time; raw bytes never leave
    the unified message model (FR-008).
    """

    data_url: str


type ContentPart = TextPart | ImagePart
"""One element of a multimodal message content list."""

type Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCallRequest:
    """A previously issued tool call echoed back in an assistant message."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat-completions request.

    Invariant: at most one of ``content`` and ``parts`` may be set; both being
    ``None`` is legal (e.g. tool-call-only assistant turns).
    """

    role: Role
    content: str | None = None
    parts: list[ContentPart] | None = None
    tool_calls: list[ToolCallRequest] | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """Enforce the content/parts exclusivity invariant.

        Raises:
            ValueError: Both ``content`` and ``parts`` were provided.
        """
        if self.content is not None and self.parts is not None:
            raise ValueError("content and parts are mutually exclusive")


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class AssistantTurn:
    """The parsed assistant answer of one completion request.

    Attributes:
        text: Plain-text content when the model produced any.
        tool_calls: Requested tool invocations when the model issued any.
        finish_reason: Server-reported stop reason (``"stop"``,
            ``"tool_calls"``, ``"length"``, ...) or ``None`` when the endpoint
            omits it; key diagnostic for empty or truncated answers.
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


def _part_to_wire(part: ContentPart) -> dict[str, JSONValue]:
    """Serialize one content part to the OpenAI wire format."""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    return {"type": "image_url", "image_url": {"url": part.data_url}}


def _message_to_wire(message: ChatMessage) -> dict[str, JSONValue]:
    """Serialize one request message to the OpenAI wire format."""
    if message.role == "tool" or (message.parts is None and message.tool_calls is None):
        return {"role": message.role, "content": message.content}
    wire: dict[str, JSONValue] = {"role": message.role}
    if message.parts is not None:
        wire["content"] = [_part_to_wire(part) for part in message.parts]
    elif message.content is not None:
        wire["content"] = message.content
    if message.tool_calls is not None:
        calls: list[JSONValue] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in message.tool_calls
        ]
        wire["tool_calls"] = calls
    return wire


def _parse_tool_calls(raw: JSONValue) -> list[ToolCall]:
    """Validate and convert the raw ``tool_calls`` array from the response.

    Args:
        raw: Untrusted decoded JSON value found under ``message.tool_calls``.

    Returns:
        Parsed tool calls; an empty list when absent or not a non-empty list.

    Raises:
        LLMAPIError: An entry lacks the expected function-call shape.
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            raise LLMAPIError("malformed tool call entry in response")
        function = item.get("function")
        call_id = item.get("id")
        if not isinstance(call_id, str) or not isinstance(function, dict):
            raise LLMAPIError("malformed tool call entry in response")
        name = function.get("name")
        if not isinstance(name, str):
            raise LLMAPIError("malformed tool call entry in response")
        arguments = function.get("arguments")
        calls.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments_json=arguments if isinstance(arguments, str) else "",
            )
        )
    return calls


class OpenAICompatibleClient:
    """Async client issuing chat-completions requests via httpx."""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr,
        model: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Store connection parameters for future completion calls.

        Args:
            base_url: Server root, e.g. ``https://api.openai.com/v1``.
            api_key: Bearer token; empty string means no auth header.
            model: Model name passed unchanged to the endpoint.
            timeout_seconds: Per-request wall-clock timeout.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, JSONValue]] | None = None,
    ) -> AssistantTurn:
        """Send one chat-completions request and parse the assistant answer.

        Args:
            messages: Conversation context in wire order (system first).
            tools: Optional OpenAI-format tool definitions to advertise.

        Returns:
            Either plain text or a list of requested tool calls (or both when
            the model mixes them; the agent loop treats any call as a round).
            The server's ``finish_reason`` rides along on the turn.

        Raises:
            LLMTransportError: Connection failure or timeout.
            LLMContextOverflowError: The request exceeded the model context
                window (detected from markers in the response body).
            LLMAPIError: Non-success status or unparsable response body.
        """
        body: dict[str, JSONValue] = {
            "model": self._model,
            "messages": [_message_to_wire(message) for message in messages],
        }
        if tools:
            body["tools"] = list(tools)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Metadata only: message bodies (and base64 image parts) stay out of logs.
        logger.debug(
            "chat-completions request: model=%s messages=%d tools=%d",
            self._model,
            len(messages),
            len(tools) if tools else 0,
        )

        client_timeout = httpx.Timeout(self._timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMTransportError(f"language service unreachable: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:500]
            message = f"language service returned HTTP {response.status_code}: {detail}"
            if _is_context_overflow(response.text):
                raise LLMContextOverflowError(message, status_code=response.status_code)
            raise LLMAPIError(message, status_code=response.status_code)
        turn = self._parse_response(response)
        logger.debug(
            "chat-completions response: status=%d finish_reason=%s text_chars=%d tool_calls=%d",
            response.status_code,
            turn.finish_reason,
            len(turn.text) if turn.text else 0,
            len(turn.tool_calls) if turn.tool_calls else 0,
        )
        return turn

    def _parse_response(self, response: httpx.Response) -> AssistantTurn:
        """Extract the first-choice assistant message from a completion body.

        Args:
            response: Successful HTTP response to decode.

        Returns:
            Parsed assistant turn (text and/or tool calls).

        Raises:
            LLMAPIError: The JSON body lacks the expected shape.
        """
        try:
            data = json.loads(response.text)
        except ValueError as exc:
            raise LLMAPIError("language service returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise LLMAPIError("unexpected completion response shape")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMAPIError("completion response contains no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LLMAPIError("completion choice has no assistant message")
        raw_text = message.get("content")
        text = raw_text if isinstance(raw_text, str) and raw_text else None
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        raw_finish_reason = choices[0].get("finish_reason")
        finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else None
        return AssistantTurn(
            text=text, tool_calls=tool_calls or None, finish_reason=finish_reason
        )
