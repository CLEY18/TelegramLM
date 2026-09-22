"""``get_post_comments`` capability: paged discussion replies under a post."""

from __future__ import annotations

import json
import logging

import hydrogram.types
from hydrogram.errors import RPCError

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
from telegramlm.tools.channel_posts import resolve_channel

logger = logging.getLogger(__name__)

# Telegram error ids that mean "this post simply has no discussion thread";
# per contract these yield an empty comment list rather than a failure.
_NO_DISCUSSION_ERROR_IDS = frozenset({"MESSAGE_ID_INVALID", "MESSAGE_NOT_FOUND"})


def _author_label(message: hydrogram.types.Message) -> str | None:
    """Human-readable author of a comment (user name or channel title)."""
    if message.from_user is not None:
        user = message.from_user
        full = " ".join(
            part for part in (_as_str(user.first_name), _as_str(user.last_name)) if part
        )
        return full or _as_str(user.username)
    if message.sender_chat is not None:
        title = _as_str(message.sender_chat.title) or _as_str(message.sender_chat.username)
        return title
    return None


def _as_str(value: str | None) -> str | None:
    """Pass through a hydrogram text field, coercing non-strings to ``None``."""
    return value if isinstance(value, str) else None


class GetPostCommentsTool(Tool):
    """Reads the comment thread under one channel post, paged."""

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
        return "get_post_comments"

    @property
    def description(self) -> str:
        """Model-facing purpose statement."""
        return (
            "Reads the comment thread (discussion replies) under a specific "
            "channel post, one page at a time."
        )

    @property
    def parameters_schema(self) -> JSONObjectSchema:
        """Accepted arguments per ``contracts/llm-tool-contracts.md``."""
        return JSONObjectSchema(
            properties={
                "channel": JSONSchemaProperty(type="string"),
                "message_id": JSONSchemaProperty(type="integer"),
                "limit": JSONSchemaProperty(type="integer", minimum=1, maximum=100),
                "offset": JSONSchemaProperty(type="integer", minimum=0),
            },
            required=["channel", "message_id"],
        )

    async def execute(self, arguments: ToolArguments) -> ToolResult:
        """Fetch one page of discussion replies for the given post.

        Args:
            arguments: Required ``channel`` and ``message_id`` plus optional
                ``limit``/``offset``.

        Returns:
            ``ok=True`` with ``{"comments": [...], "offset", "returned"}``;
            posts without any discussion yield an empty list rather than an
            error, while genuine access failures become ``ok=False`` envelopes.
        """
        channel_raw = arguments.get_str("channel")
        if not channel_raw:
            return failure_result("argument 'channel' is required")
        message_id = arguments.get_int("message_id")
        if message_id is None:
            return failure_result("argument 'message_id' is required")
        limit = bounded_limit(arguments.get_int("limit"), self._settings.tool_default_limit)
        offset = max(0, arguments.get_int("offset") or 0)

        comments: list[dict[str, str | int | None]] = []
        try:
            replies = await self._user.get_discussion_replies(
                resolve_channel(channel_raw), message_id, limit=offset + limit
            )
            index = 0
            if replies is not None:
                async for reply in replies:
                    if index < offset:
                        index += 1
                        continue
                    if len(comments) >= limit:
                        break
                    comments.append(
                        {
                            "author": _author_label(reply),
                            "date": reply.date.isoformat() if reply.date else None,
                            "text": reply.text or reply.caption,
                        }
                    )
                    index += 1
        except RPCError as exc:
            if exc.id in _NO_DISCUSSION_ERROR_IDS:
                logger.info(
                    "post %s in %r has no discussion (%s)", message_id, channel_raw, exc.id
                )
            else:
                logger.exception("get_post_comments failed for %r", channel_raw)
                return failure_result(f"failed to read comments of {channel_raw!r}: {exc}")
        except Exception as exc:  # noqa: BLE001 - failures are data to the model
            logger.exception("get_post_comments failed for %r", channel_raw)
            return failure_result(f"failed to read comments of {channel_raw!r}: {exc}")
        payload = {"comments": comments, "offset": offset, "returned": len(comments)}
        return ToolResult(ok=True, payload=json.dumps(payload, ensure_ascii=False))
