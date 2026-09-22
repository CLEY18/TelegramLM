"""Bot update -> unified message normalization.

The processor is the only component that touches raw hydrogram message types:
it unwraps forwards, buffers media-group (album) fragments behind a
configurable idle window, downloads and classifies attachments, applies the
unsupported-content rules (FR-004..FR-007), and emits exactly one
:class:`~telegramlm.messaging.models.UnifiedMessage` per logical message — or
a single "not supported" notice instead. It knows nothing about the LLM;
accepted messages leave through the injected ``dispatch`` callback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Protocol

import hydrogram.types

from telegramlm.config import Settings
from telegramlm.messaging.models import (
    SUPPORTED_TEXT_EXTENSIONS,
    Direction,
    ImageAttachment,
    MessageSource,
    TextDocumentAttachment,
    UnifiedMessage,
    UnsupportedFileRef,
)

logger = logging.getLogger(__name__)

# How many flushed media-group ids to remember so late duplicate fragments of
# an already-processed album are discarded (FR-004) without unbounded growth.
_FLUSH_MEMORY_LIMIT = 256


class NoticeSender(Protocol):
    """Callback that delivers a short notice text to the owner's bot chat."""

    async def __call__(self, text: str) -> None: ...


def _sniff_image_mime(data: bytes) -> str:
    """Identify an image MIME type from its leading magic bytes.

    Args:
        data: Raw downloaded image bytes.

    Returns:
        A best-effort ``image/*`` type; Telegram photos are JPEG unless a more
        specific signature matches.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _forward_label(message: hydrogram.types.Message) -> tuple[int | None, str | None]:
    """Extract the forward-origin id and human-readable name of a message.

    Args:
        message: A (possibly forwarded) Telegram message.

    Returns:
        ``(chat_id, display_name)`` when the message is forwarded, otherwise
        ``(None, None)``. Prefers an original chat over an original user.
    """
    origin_chat = getattr(message, "forward_from_chat", None)
    if origin_chat is not None:
        name = origin_chat.title or origin_chat.username
        return origin_chat.id, name
    origin_user = message.forward_from
    if origin_user is not None:
        name = getattr(origin_user, "title", None) or getattr(origin_user, "username", None)
        if name is None and getattr(origin_user, "first_name", None):
            name = origin_user.first_name
        return origin_user.id, name
    return None, None


class _AlbumBuffer:
    """Mutable state for one media group being collected."""

    def __init__(self, chat_id: int, first_message_id: int) -> None:
        """Remember the group's owner chat and first fragment id.

        Args:
            chat_id: Chat the fragments arrive in.
            first_message_id: Id of the first fragment seen (becomes source).
        """
        self.chat_id = chat_id
        self.first_message_id = first_message_id


class MessageProcessor:
    """Normalizes authorized private updates into unified messages."""

    def __init__(
        self,
        bot_client: hydrogram.Client,
        settings: Settings,
        dispatch: Callable[[UnifiedMessage], Awaitable[None]],
        notify_unsupported: NoticeSender,
    ) -> None:
        """Wire the processor to its collaborators.

        Args:
            bot_client: Bot client used for media downloads and group fetches.
            settings: Tunables (idle window, file size limit).
            dispatch: Async sink receiving each accepted unified message.
            notify_unsupported: Sends the single "not supported" notice text.
        """
        self._bot = bot_client
        self._settings = settings
        self._dispatch = dispatch
        self._notify_unsupported = notify_unsupported
        self._buffers: dict[str, _AlbumBuffer] = {}
        self._flushed_ids: dict[str, float] = {}

    async def handle_message(self, message: hydrogram.types.Message) -> None:
        """Process one already-authorized private message.

        Album fragments enter the idle-window buffer; every other accepted
        update is normalized immediately into a single unified message or an
        unsupported notice (FR-004/FR-005).

        Args:
            message: The private in-service message from the owner.
        """
        group_id = message.media_group_id
        if group_id is not None:
            self._accept_fragment(group_id, message)
            return
        unified = await self._normalize([message])
        if unified is None:
            logger.info("unsupported single message id=%s", message.id)
            await self._notify_unsupported(_UNSUPPORTED_NOTICE_TEXT)
            return
        await self._dispatch(unified)

    # --- Media-group buffering (FR-004) ------------------------------------

    def _accept_fragment(self, group_id: str, message: hydrogram.types.Message) -> None:
        """Register an album fragment and (re)arm nothing if one is pending.

        Args:
            group_id: Telegram ``media_group_id`` of the album.
            message: The arriving fragment.
        """
        if group_id in self._flushed_ids or group_id in self._buffers:
            logger.debug("discarding duplicate/late fragment of group %s", group_id)
            return
        buffer = _AlbumBuffer(message.chat.id, message.id)
        self._buffers[group_id] = buffer
        asyncio.create_task(self._flush_after(group_id), name=f"album-flush-{group_id}")

    async def _flush_after(self, group_id: str) -> None:
        """Wait out the idle window, then normalize and emit the whole album.

        Args:
            group_id: Album identifier whose collection is being awaited.
        """
        try:
            await asyncio.sleep(self._settings.media_group_timeout_seconds)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        buffer = self._buffers.pop(group_id, None)
        if buffer is None:
            return
        try:
            fragments = await self._bot.get_media_group(buffer.chat_id, group_id)
        except Exception:
            logger.exception("failed to fetch media group %s", group_id)
            await self._notify_unsupported(_UNSUPPORTED_NOTICE_TEXT)
            self._remember_flushed(group_id)
            return
        unified = await self._normalize(list(fragments))
        self._remember_flushed(group_id)
        if unified is None:
            logger.info("album %s had no supported content", group_id)
            await self._notify_unsupported(_UNSUPPORTED_NOTICE_TEXT)
            return
        await self._dispatch(unified)

    def _remember_flushed(self, group_id: str) -> None:
        """Mark a group as flushed, pruning the oldest ids beyond the limit."""
        self._flushed_ids[group_id] = time.monotonic()
        while len(self._flushed_ids) > _FLUSH_MEMORY_LIMIT:
            oldest = min(self._flushed_ids.items(), key=lambda item: item[1])[0]
            del self._flushed_ids[oldest]

    # --- Attachment normalization (FR-005..FR-007, FR-003) ------------------

    async def _normalize(
        self, messages: Sequence[hydrogram.types.Message]
    ) -> UnifiedMessage | None:
        """Merge fragments into one unified message applying all content rules.

        Args:
            messages: One standalone message or all fragments of an album, in
                arrival order; the first provides provenance.

        Returns:
            The normalized :class:`UnifiedMessage`, or ``None`` when nothing
            supported was found (caller must send the unsupported notice).
        """
        if not messages:
            return None
        texts: list[str] = []
        images: list[ImageAttachment] = []
        documents: list[TextDocumentAttachment] = []
        unsupported: list[UnsupportedFileRef] = []
        unsupported_count = 0

        for fragment in messages:
            body = fragment.caption or fragment.text
            if body:
                texts.append(body)
            if fragment.photo is not None:
                data = await self._download_bytes(fragment)
                if data:
                    images.append(
                        ImageAttachment(data=data, mime_type=_sniff_image_mime(data))
                    )
                else:
                    unsupported_count += 1
            elif fragment.document is not None:
                document = fragment.document
                file_name = document.file_name or "document"
                size = document.file_size or 0
                suffix = PurePosixPath(file_name).suffix.lower()
                if suffix in SUPPORTED_TEXT_EXTENSIONS and size <= self._settings.max_file_size_bytes:
                    data = await self._download_bytes(fragment)
                    content = data.decode("utf-8", errors="replace") if data else ""
                    documents.append(
                        TextDocumentAttachment(
                            file_name=file_name, content=content, size_bytes=size
                        )
                    )
                else:
                    # Oversized or unsupported type: name + size only (FR-006/007).
                    unsupported.append(
                        UnsupportedFileRef(file_name=file_name, size_bytes=size)
                    )
            elif fragment.media is not None:
                # Media kinds without file-level metadata degrade to a count.
                unsupported_count += 1

        text = "\n\n".join(texts) if texts else None
        if not text and not images and not documents:
            return None
        description = _describe_unsupported(unsupported, unsupported_count)
        if description is not None:
            # FR-006: the description line becomes the first line of context.
            text = f"{description}\n{text}" if text else description

        origin_id, origin_name = _forward_label(messages[0])
        first = messages[0]
        source = MessageSource(
            chat_id=first.chat.id,
            message_id=first.id,
            sender_id=first.from_user.id if first.from_user is not None else 0,
            sent_at=first.date or datetime.now(tz=UTC),
            forwarded_from_chat_id=origin_id,
            forwarded_from_name=origin_name,
        )
        return UnifiedMessage(
            direction=Direction.IN,
            text=text,
            images=images,
            documents=documents,
            unsupported_files=unsupported,
            unsupported_file_count=unsupported_count,
            source=source,
        )

    async def _download_bytes(self, message: hydrogram.types.Message) -> bytes | None:
        """Download a message's media into memory.

        Args:
            message: Message whose single media item should be fetched.

        Returns:
            The raw bytes, or ``None`` when the download failed or produced
            an empty stream.
        """
        try:
            stream = await self._bot.download_media(message, in_memory=True)
        except Exception:
            logger.exception("media download failed for message id=%s", message.id)
            return None
        if stream is None:
            return None
        data: bytes = stream.read()
        stream.close()
        return data or None


# Shown verbatim once per unsupported album/message (FR-005).
_UNSUPPORTED_NOTICE_TEXT = "This message type is not supported."


def _describe_unsupported(
    refs: Sequence[UnsupportedFileRef], count_fallback: int
) -> str | None:
    """Build the FR-006 description line for unreadable files.

    Args:
        refs: File-level name/size descriptions collected during normalization.
        count_fallback: Number of items whose metadata required a download.

    Returns:
        A single bracketed line listing each file as ``name (size)`` plus any
        count-only remainder, or ``None`` when there is nothing to note.
    """
    if not refs and count_fallback <= 0:
        return None
    entries: list[str] = []
    for ref in refs:
        name = ref.file_name or "file"
        if ref.size_bytes is not None:
            entries.append(f"{name} ({_format_size(ref.size_bytes)})")
        else:
            entries.append(name)
    if count_fallback > 0:
        entries.append(f"+{count_fallback} unsupported item(s)")
    return f"[unsupported files: {'; '.join(entries)}]"


def _format_size(size_bytes: int | None) -> str:
    """Render a byte size compactly, e.g. ``1.2 MB``; ``?`` when unknown."""
    if size_bytes is None:
        return "?"
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
