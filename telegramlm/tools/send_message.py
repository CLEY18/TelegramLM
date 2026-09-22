"""The ``send_message`` delivery tool.

This is the only path through which assistant output becomes visible to the
owner (FR-011); every successful send is also recorded in history as an OUT
unified message so future turns can reference what was already said. The
contract lives in ``contracts/llm-tool-contracts.md``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import hydrogram.types

from telegramlm.messaging.models import Direction, MessageSource, UnifiedMessage
from telegramlm.storage.store import MessageStore
from telegramlm.tools.base import (
    JSONObjectSchema,
    JSONSchemaProperty,
    Tool,
    ToolArguments,
    ToolResult,
    failure_result,
)

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Delivers a text message to the owner's private bot chat."""

    def __init__(
        self, bot_client: hydrogram.Client, owner_chat_id: int, store: MessageStore
    ) -> None:
        """Bind the tool to the bot client, owner chat, and history store.

        Args:
            bot_client: Connected bot client used for sending.
            owner_chat_id: Chat id of the private bot conversation.
            store: History sink receiving each delivered message as OUT.
        """
        self._bot = bot_client
        self._owner_chat_id = owner_chat_id
        self._store = store

    @property
    def name(self) -> str:
        """Registry key exposed to the model."""
        return "send_message"

    @property
    def description(self) -> str:
        """Model-facing purpose statement (delivery is the only output path)."""
        return (
            "Delivers a text message to the owner in the private bot chat. "
            "This is the ONLY way your output becomes visible. Call it for every "
            "answer you intend to give; omit it only when no reply is warranted."
        )

    @property
    def parameters_schema(self) -> JSONObjectSchema:
        """Accepted arguments per ``contracts/llm-tool-contracts.md``."""
        return JSONObjectSchema(
            properties={
                "text": JSONSchemaProperty(
                    type="string", description="Message text, in the owner's language"
                ),
                "reply_to_message_id": JSONSchemaProperty(
                    type="integer",
                    description=(
                        "Optional: id of a message from this turn to answer "
                        "directly (Telegram reply quote)"
                    ),
                ),
            },
            required=["text"],
        )

    async def execute(self, arguments: ToolArguments) -> ToolResult:
        """Send one text message and record it as an OUT unified message.

        Args:
            arguments: ``text`` (required string) and optional
                ``reply_to_message_id`` integer.

        Returns:
            ``ok=True`` with the delivered message id, or a failure envelope
            when the argument is missing or Telegram rejects the send.
        """
        text = arguments.get_str("text")
        if not text:
            return failure_result("argument 'text' is required and must be non-empty")
        reply_to = arguments.get_int("reply_to_message_id")
        try:
            sent = await self._bot.send_message(
                self._owner_chat_id, text, reply_to_message_id=reply_to
            )
        except Exception as exc:  # noqa: BLE001 - failures are data to the model
            logger.exception("send_message delivery failed")
            return failure_result(f"failed to deliver message: {exc}")
        self._store.append(
            UnifiedMessage(
                direction=Direction.OUT,
                text=text,
                source=MessageSource(
                    chat_id=self._owner_chat_id,
                    message_id=sent.id,
                    sender_id=0,  # outbound bot messages carry no owner sender
                    sent_at=sent.date or datetime.now(tz=UTC),
                ),
            )
        )
        logger.info("delivered assistant message id=%s", sent.id)
        return ToolResult(
            ok=True,
            payload=json.dumps({"delivered": True, "message_id": sent.id}, ensure_ascii=False),
        )
