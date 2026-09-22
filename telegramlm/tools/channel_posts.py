"""``get_channel_posts`` capability: paged newest-first reads of a channel."""

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


def resolve_channel(raw: str) -> int | str:
    """Interpret a channel reference as numeric id or username.

    Args:
        raw: Model-supplied channel identifier (numeric string or username).

    Returns:
        An ``int`` id when the value parses as one, otherwise the trimmed
        username string accepted directly by hydrogram.
    """
    stripped = raw.strip().lstrip("@")
    try:
        return int(stripped)
    except ValueError:
        return stripped


def _describe_media(message: hydrogram.types.Message) -> str | None:
    """Summarize the media carried by a post, if any.

    Args:
        message: A channel post.

    Returns:
        A short lowercase kind label (``photo``, ``video``, ``document`` ...)
        or ``None`` for text-only posts.
    """
    kinds: tuple[tuple[str, str], ...] = (
        ("photo", "photo"),
        ("video", "video"),
        ("audio", "audio"),
        ("voice", "voice message"),
        ("document", "document"),
        ("sticker", "sticker"),
        ("animation", "animation"),
        ("poll", "poll"),
    )
    for attribute, label in kinds:
        if getattr(message, attribute, None) is not None:
            return label
    return None


class GetChannelPostsTool(Tool):
    """Reads recent posts of an accessible channel newest-first, paged."""

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
        return "get_channel_posts"

    @property
    def description(self) -> str:
        """Model-facing purpose statement."""
        return (
            "Reads recent posts from a channel the owner's account can access, "
            "newest first, one page at a time."
        )

    @property
    def parameters_schema(self) -> JSONObjectSchema:
        """Accepted arguments per ``contracts/llm-tool-contracts.md``."""
        return JSONObjectSchema(
            properties={
                "channel": JSONSchemaProperty(
                    type="string", description="Channel username or numeric id"
                ),
                "limit": JSONSchemaProperty(type="integer", minimum=1, maximum=100),
                "offset": JSONSchemaProperty(type="integer", minimum=0),
            },
            required=["channel"],
        )

    async def execute(self, arguments: ToolArguments) -> ToolResult:
        """Fetch one newest-first page of posts from the target channel.

        Args:
            arguments: Required ``channel`` plus optional ``limit``/``offset``.

        Returns:
            ``ok=True`` with ``{"posts": [...], "offset", "returned"}`` where
            each post carries id, ISO date, text, and media description;
            access failures become ``ok=False`` envelopes (FR-016).
        """
        channel_raw = arguments.get_str("channel")
        if not channel_raw:
            return failure_result("argument 'channel' is required")
        limit = bounded_limit(arguments.get_int("limit"), self._settings.tool_default_limit)
        offset = max(0, arguments.get_int("offset") or 0)
        posts: list[dict[str, str | int | None]] = []
        try:
            index = 0
            async for message in self._user.get_chat_history(
                resolve_channel(channel_raw), limit=offset + limit
            ):
                if index < offset:
                    index += 1
                    continue
                if len(posts) >= limit:
                    break
                posts.append(
                    {
                        "message_id": message.id,
                        "date": message.date.isoformat() if message.date else None,
                        "text": message.text or message.caption,
                        "media_description": _describe_media(message),
                    }
                )
                index += 1
        except Exception as exc:  # noqa: BLE001 - access errors are data to the model
            logger.exception("get_channel_posts failed for %r", channel_raw)
            return failure_result(f"failed to read posts of {channel_raw!r}: {exc}")
        payload = {"posts": posts, "offset": offset, "returned": len(posts)}
        return ToolResult(ok=True, payload=json.dumps(payload, ensure_ascii=False))
