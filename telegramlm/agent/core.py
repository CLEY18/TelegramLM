"""Agent turn loop: context assembly, bounded tool rounds, delivery fallback.

:class:`AgentCore` owns one conversation turn end to end — it builds the model
request from stored history (bounded by ``MAX_HISTORY_MESSAGES``), advertises
the registered tools, executes requested calls as data-returning rounds up to
``AGENT_MAX_ITERATIONS``, keeps the typing indicator alive, and guarantees the
owner is never left hanging: a turn that produced no delivery falls back to
sending the last generated text automatically (FR-012).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

import hydrogram.types
from hydrogram.enums import ChatAction

from telegramlm.agent.prompts import SYSTEM_PROMPT
from telegramlm.config import Settings
from telegramlm.llm.client import (
    AssistantTurn,
    ChatMessage,
    ImagePart,
    OpenAICompatibleClient,
    TextPart,
    ToolCallRequest,
)
from telegramlm.messaging.models import Direction, MessageSource, UnifiedMessage
from telegramlm.storage.store import MessageStore
from telegramlm.tools.base import JSONValue, ToolRegistry

logger = logging.getLogger(__name__)

# Interval for refreshing the typing indicator so it never lapses mid-turn.
TYPING_REFRESH_SECONDS = 5.0

# Placeholder standing in for images from earlier turns (FR-015) and for
# current images when vision is disabled.
IMAGE_PLACEHOLDER = "[image]"

# Canned notice used when a turn produced neither a delivery nor any text.
_NO_OUTPUT_NOTICE = "Could not produce an answer, please try again."


def _encode_image_data_url(mime_type: str, data: bytes) -> str:
    """Encode raw image bytes as an OpenAI ``image_url`` data URL.

    Base64 exists only transiently here — never in storage (FR-008).

    Args:
        mime_type: Detected MIME type of the image.
        data: Raw image bytes.

    Returns:
        A ``data:<mime>;base64,<payload>`` URL string.
    """
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _preview(value: str | None, limit: int = 200) -> str:
    """Collapse a possibly long string into one bounded line for debug logs.

    Whitespace runs (newlines, base64 payloads) become single spaces so every
    log record stays on one line; ``None``/empty renders as ``<empty>`` and a
    cut tail reports how many characters were dropped.

    Args:
        value: The string to preview, possibly ``None``.
        limit: Maximum number of characters kept before the cut marker.

    Returns:
        A single-line preview suitable for a log record.
    """
    if not value:
        return "<empty>"
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}... [+{len(collapsed) - limit} chars]"


class AgentCore:
    """Runs bounded model/tool iterations for one conversation turn."""

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        registry: ToolRegistry,
        store: MessageStore,
        bot_client: hydrogram.Client,
        settings: Settings,
    ) -> None:
        """Wire the agent to its collaborators.

        Args:
            llm_client: Language-service client issuing chat-completions calls.
            registry: Central tool registry (payload + dispatch + size caps).
            store: Conversation history providing and receiving context.
            bot_client: Bot client for typing indicators and fallback delivery.
            settings: Tunables (history bound, iteration cap, vision flag).
        """
        self._llm = llm_client
        self._registry = registry
        self._store = store
        self._bot = bot_client
        self._settings = settings
        self._turn_counter: int = 0

    async def run_turn(self, batch: Sequence[UnifiedMessage]) -> None:
        """Run one consolidated agent turn for the queued messages.

        Args:
            batch: The unified messages (already appended to history) that
                make up this turn's input; answered together as one request.

        Raises:
            Exception: Language-service transport/API failures propagate after
                the typing indicator is stopped so the caller can notify the
                owner with a generic error notice.
        """
        if not batch:
            return
        chat_id = batch[0].source.chat_id
        self._turn_counter += 1
        turn_number = self._turn_counter
        logger.debug(
            "turn %d started for chat=%s with %d coalesced message(s)",
            turn_number,
            chat_id,
            len(batch),
        )
        typing_task = asyncio.create_task(self._typing_loop(chat_id))
        try:
            await self._run_loop(batch, turn_number)
        finally:
            typing_task.cancel()

    async def _run_loop(self, batch: Sequence[UnifiedMessage], turn_number: int) -> None:
        """Execute the bounded model/tool iterations and apply fallback.

        Args:
            batch: Current-turn messages (tail of history).
            turn_number: Monotonic id of this turn used to correlate log lines.
        """
        chat_id = batch[0].source.chat_id
        messages = self._build_context(batch)
        tools_payload = self._registry.to_openai_payload()
        delivered = False
        last_text: str | None = None
        iterations_run = 0
        failed_calls = 0

        for iteration in range(1, self._settings.agent_max_iterations + 1):
            iterations_run = iteration
            logger.debug(
                "turn %d iteration %d/%d: sending %d messages to the model",
                turn_number,
                iteration,
                self._settings.agent_max_iterations,
                len(messages),
            )
            turn: AssistantTurn = await self._llm.complete(messages, tools_payload)
            if turn.text:
                last_text = turn.text
            logger.debug(
                "turn %d iteration %d response: finish_reason=%s text=%s tool_calls=%d",
                turn_number,
                iteration,
                turn.finish_reason,
                _preview(turn.text),
                len(turn.tool_calls) if turn.tool_calls else 0,
            )
            if not turn.tool_calls:
                logger.debug("turn %d ends: model returned no tool calls", turn_number)
                break
            echo_calls = [
                ToolCallRequest(
                    call_id=call.call_id,
                    name=call.name,
                    arguments_json=call.arguments_json,
                )
                for call in turn.tool_calls
            ]
            messages.append(
                ChatMessage(role="assistant", content=turn.text, tool_calls=echo_calls)
            )
            for call in turn.tool_calls:
                logger.debug(
                    "turn %d calling %s with arguments %s",
                    turn_number,
                    call.name,
                    _preview(call.arguments_json),
                )
                arguments = _parse_arguments(call.arguments_json)
                result = await self._registry.execute(call.name, arguments)
                if call.name == "send_message" and result.ok:
                    delivered = True
                if not result.ok:
                    failed_calls += 1
                logger.debug(
                    "turn %d tool %s returned ok=%s payload=%s",
                    turn_number,
                    call.name,
                    result.ok,
                    _preview(result.payload),
                )
                envelope: dict[str, JSONValue] = {"ok": result.ok, "payload": result.payload}
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(envelope, ensure_ascii=False),
                        tool_call_id=call.call_id,
                    )
                )
        else:
            logger.debug(
                "turn %d ends: iteration cap (%d) reached without a final answer",
                turn_number,
                self._settings.agent_max_iterations,
            )

        if delivered:
            return
        if last_text:
            logger.info("turn %d produced no delivery; sending fallback text", turn_number)
            await self._bot.send_message(chat_id, last_text)
            self._record_out(chat_id, last_text)
        else:
            logger.warning(
                "turn %d produced neither a delivery nor any model text after %d iteration(s)"
                " (%d failed tool call(s)); sending the no-output notice",
                turn_number,
                iterations_run,
                failed_calls,
            )
            await self._bot.send_message(chat_id, _NO_OUTPUT_NOTICE)

    def _record_out(self, chat_id: int, text: str) -> None:
        """Append fallback-delivered assistant text to history as an OUT message.

        Args:
            chat_id: Owner's private bot-chat id.
            text: The assistant text that was sent automatically (FR-012).
        """
        self._store.append(
            UnifiedMessage(
                direction=Direction.OUT,
                text=text,
                source=MessageSource(
                    chat_id=chat_id,
                    message_id=0,  # fallback sends are not tied to a tool call id
                    sender_id=0,
                    sent_at=datetime.now(tz=UTC),
                ),
            )
        )

    def _build_context(self, batch: Sequence[UnifiedMessage]) -> list[ChatMessage]:
        """Assemble the request context from system prompt and stored history.

        Historical images become ``[image]`` placeholders; full image parts are
        attached only to current-turn messages when vision is enabled
        (FR-015). Forwarded content carries its origin note.

        Args:
            batch: Current-turn messages, expected to be the tail of history.

        Returns:
            The message list for one chat-completions request.
        """
        history = self._store.get_recent(self._settings.max_history_messages)
        current_start = max(0, len(history) - len(batch))
        context: list[ChatMessage] = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        for index, message in enumerate(history):
            is_current = index >= current_start
            if message.direction == Direction.OUT:
                if message.text:
                    context.append(ChatMessage(role="assistant", content=message.text))
                continue
            images = message.images if is_current and self._settings.vision_enabled else []
            text = _render_inbound_text(message, include_image_placeholder=not images)
            if images:
                parts: list[TextPart | ImagePart] = [TextPart(text=text)] if text else []
                parts.extend(
                    ImagePart(data_url=_encode_image_data_url(img.mime_type, img.data))
                    for img in images
                )
                context.append(ChatMessage(role="user", parts=parts))
            elif text:
                context.append(ChatMessage(role="user", content=text))
        return context

    async def _typing_loop(self, chat_id: int) -> None:
        """Refresh the typing indicator until the turn finishes.

        Args:
            chat_id: Owner's private bot-chat id.
        """
        while True:
            try:
                await self._bot.send_chat_action(chat_id, ChatAction.TYPING)
            except Exception:  # noqa: BLE001 - indicator is best-effort feedback
                logger.warning("typing indicator failed; continuing turn", exc_info=True)
                return
            await asyncio.sleep(TYPING_REFRESH_SECONDS)


def _render_inbound_text(
    message: UnifiedMessage, include_image_placeholder: bool = True
) -> str:
    """Render one inbound unified message as model-facing text.

    Includes the forward-origin note, the normalized body (which already
    carries any unsupported-file description line), embedded document
    contents, and optionally ``[image]`` placeholders for images that are
    not attached to the request in full.

    Args:
        message: The stored inbound message to render.
        include_image_placeholder: When ``False`` (images sent as real parts)
            no placeholder line is emitted.

    Returns:
        The combined text block for the user-role request entry.
    """
    blocks: list[str] = []
    if message.source.forwarded_from_name is not None:
        blocks.append(f"[forwarded from {message.source.forwarded_from_name}]")
    if message.text:
        blocks.append(message.text)
    for document in message.documents:
        blocks.append(f"Attached file {document.file_name}:\n{document.content}")
    if include_image_placeholder and message.images:
        blocks.append(" ".join(IMAGE_PLACEHOLDER for _ in message.images))
    return "\n\n".join(blocks)


def _parse_arguments(arguments_json: str) -> dict[str, JSONValue]:
    """Decode a tool call's raw JSON arguments defensively.

    Args:
        arguments_json: Raw string supplied by the model (may be empty).

    Returns:
        The decoded mapping, or an empty dict when it is not a JSON object —
        the registry turns missing/invalid arguments into failure envelopes.
    """
    if not arguments_json.strip():
        return {}
    try:
        parsed = json.loads(arguments_json)
    except ValueError:
        logger.warning("model produced invalid JSON arguments: %r", arguments_json[:200])
        return {}
    if isinstance(parsed, dict):
        return {key: _to_json_value(value) for key, value in parsed.items()}
    return {}


def _to_json_value(value: JSONValue) -> JSONValue:
    """Validate a decoded JSON member against the typed alias at runtime.

    ``json.loads`` returns untyped data; this function re-checks each member so
    everything downstream flows through the declared ``JSONValue`` shape.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    return str(value)
