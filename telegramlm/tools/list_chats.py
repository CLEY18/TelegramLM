"""``list_chats`` capability: paged view over the owner's dialogs."""

from __future__ import annotations

import json
import logging

import hydrogram.types

from telegramlm.config import Settings
from telegramlm.tools.base import (
    JSONObjectSchema,
    JSONSchemaProperty,
    Tool,
    ToolArguments,
    ToolResult,
    bounded_limit,
    failure_result,
)

logger = logging.getLogger(__name__)


def _chat_projection(chat: hydrogram.types.Chat) -> dict[str, str | int | None]:
    """Project one dialog chat to the contract's ``ChatSummary`` shape.

    Args:
        chat: The dialog chat object from the user client.

    Returns:
        Mapping with ``chat_id``, human ``title``, optional ``username``,
        and Telegram ``type`` string per llm-tool-contracts.md.
    """
    title = chat.title or " ".join(
        part for part in (chat.first_name, chat.last_name) if part
    )
    return {
        "chat_id": chat.id,
        "title": title or None,
        "username": chat.username,
        "type": str(chat.type).split(".")[-1].lower(),
    }


class ListChatsTool(Tool):
    """Lists the owner's dialogs one page at a time via ``get_dialogs``."""

    def __init__(self, user_client: hydrogram.Client, settings: Settings) -> None:
        """Bind the tool to the authenticated user client and tunables.

        Args:
            user_client: Connected user-account client (reads as the owner).
            settings: Provides ``tool_default_limit`` for paging defaults.
        """
        self._user = user_client
        self._settings = settings

    @property
    def name(self) -> str:
        """Registry key exposed to the model."""
        return "list_chats"

    @property
    def description(self) -> str:
        """Model-facing purpose statement."""
        return (
            "Lists the owner's Telegram dialogs (private chats, groups, channels "
            "they participate in), one page at a time."
        )

    @property
    def parameters_schema(self) -> JSONObjectSchema:
        """Accepted arguments per ``contracts/llm-tool-contracts.md``."""
        return JSONObjectSchema(
            properties={
                "limit": JSONSchemaProperty(type="integer", minimum=1, maximum=100),
                "offset": JSONSchemaProperty(type="integer", minimum=0),
            }
        )

    async def execute(self, arguments: ToolArguments) -> ToolResult:
        """Fetch one page of dialogs as ``ChatSummary`` projections.

        Args:
            arguments: Optional ``limit`` (1..100) and ``offset`` (>= 0).

        Returns:
            ``ok=True`` with ``{"chats": [...], "offset", "returned"}``, or a
            failure envelope when the account read fails.
        """
        limit = bounded_limit(arguments.get_int("limit"), self._settings.tool_default_limit)
        offset = max(0, arguments.get_int("offset") or 0)
        chats: list[dict[str, str | int | None]] = []
        try:
            index = 0
            async for dialog in self._user.get_dialogs(limit=offset + limit):
                if index < offset:
                    index += 1
                    continue
                if len(chats) >= limit:
                    break
                chats.append(_chat_projection(dialog.chat))
                index += 1
        except Exception as exc:  # noqa: BLE001 - failures are data to the model
            logger.exception("list_chats failed")
            return failure_result(f"failed to list dialogs: {exc}")
        payload = {"chats": chats, "offset": offset, "returned": len(chats)}
        return ToolResult(ok=True, payload=json.dumps(payload, ensure_ascii=False))
