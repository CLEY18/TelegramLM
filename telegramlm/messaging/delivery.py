"""Outbound bot delivery: the single place that talks to Telegram for sending.

:class:`BotMessenger` wraps the raw hydrogram bot client so every text answer,
system notice, and fallback message goes through one path that respects
Telegram's 4096-character per-message limit by splitting long texts into
sequentially sent chunks. Inbound concerns (media downloads, update handling)
stay on the raw client — this class covers delivery only.
"""

from __future__ import annotations

import logging

import hydrogram
from hydrogram.enums import ChatAction

logger = logging.getLogger(__name__)

# Hard limit enforced by Telegram for a single message text.
MAX_TELEGRAM_MESSAGE = 4096


def _split_oversized_paragraph(paragraph: str) -> list[str]:
    """Break one paragraph longer than the limit into bounded pieces.

    Cuts prefer whitespace boundaries (spaces or newlines) nearest the end of
    each window so chunks stay readable; a run without any whitespace is cut
    hard at the limit. Separator whitespace consumed by a cut is dropped.

    Args:
        paragraph: Text that may exceed ``MAX_TELEGRAM_MESSAGE``.

    Returns:
        One or more pieces, each within the limit.
    """
    pieces: list[str] = []
    remaining = paragraph
    while len(remaining) > MAX_TELEGRAM_MESSAGE:
        window = remaining[:MAX_TELEGRAM_MESSAGE]
        cut = max(window.rfind(" "), window.rfind("\n"))
        if cut <= 0:
            cut = MAX_TELEGRAM_MESSAGE
        pieces.append(remaining[:cut])
        remaining = remaining[cut:].lstrip(" \n")
    if remaining:
        pieces.append(remaining)
    return pieces


def split_telegram_text(text: str) -> list[str]:
    """Split outgoing text into chunks that each fit one Telegram message.

    Paragraphs are packed greedily up to the limit and rejoined with blank
    lines; paragraphs alone exceeding the limit are word-wrapped first. The
    operation is lossless apart from separator whitespace at cut points.

    Args:
        text: Full outgoing message text.

    Returns:
        Non-empty chunks, each at most ``MAX_TELEGRAM_MESSAGE`` characters.
    """
    if len(text) <= MAX_TELEGRAM_MESSAGE:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        for piece in _split_oversized_paragraph(paragraph):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= MAX_TELEGRAM_MESSAGE:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


class BotMessenger:
    """Delivers bot text output, chunking anything longer than one message."""

    def __init__(self, bot_client: hydrogram.Client) -> None:
        """Bind the messenger to the connected bot client.

        Args:
            bot_client: Connected hydrogram bot client used for sending.
        """
        self._bot = bot_client

    async def send_text(
        self, chat_id: int, text: str, reply_to_message_id: int | None = None
    ) -> list[int]:
        """Send one logical text to ``chat_id``, splitting it into chunks.

        Chunks go out sequentially as separate messages; the optional reply
        quote is attached to the first chunk only.

        Args:
            chat_id: Target chat (the owner's private bot chat).
            text: Full message text, of any length.
            reply_to_message_id: Optional id of a message to answer directly.

        Returns:
            Telegram ids of every sent chunk, in send order.

        Raises:
            Exception: Transport or RPC failures from hydrogram propagate after
                the already-sent chunks; callers decide on user-facing notices.
        """
        chunks = split_telegram_text(text)
        if len(chunks) > 1:
            logger.info(
                "splitting %d-char message into %d chunks for chat=%s",
                len(text),
                len(chunks),
                chat_id,
            )
        sent_ids: list[int] = []
        for index, chunk in enumerate(chunks):
            sent = await self._bot.send_message(
                chat_id,
                chunk,
                reply_to_message_id=reply_to_message_id if index == 0 else None,
            )
            sent_ids.append(int(sent.id))
        return sent_ids

    async def send_typing(self, chat_id: int) -> None:
        """Refresh the typing indicator once for the target chat.

        Args:
            chat_id: Chat in which to show the bot as typing.
        """
        await self._bot.send_chat_action(chat_id, ChatAction.TYPING)
