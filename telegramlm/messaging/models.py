"""Unified message entities shared across messaging, storage, and agent layers.

Every accepted Telegram update is normalized into exactly one
:class:`UnifiedMessage` (the central entity of the feature); all downstream
components consume only these models and never raw library types. Field names
and validation rules follow ``specs/001-telegram-assistant-bot/data-model.md``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Extensions accepted as readable text documents (FR-005). Lower-case only;
# comparisons normalize the incoming file name first.
SUPPORTED_TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".csv"})


class Direction(str, Enum):
    """Who produced a unified message relative to the assistant."""

    IN = "in"
    """Message received from the owner."""

    OUT = "out"
    """Message delivered by the assistant via the delivery tool."""


class ImageAttachment(BaseModel):
    """A raw image downloaded from Telegram (base64 lives only in requests)."""

    model_config = ConfigDict(frozen=True)

    data: bytes
    """Raw image bytes as downloaded from Telegram."""

    mime_type: str
    """Detected MIME type, e.g. ``image/jpeg`` or ``image/png``."""

    @field_validator("data")
    @classmethod
    def _reject_empty_data(cls, value: bytes) -> bytes:
        """Reject zero-length payloads that would produce empty data URLs."""
        if not value:
            raise ValueError("image attachment requires non-empty bytes")
        return value


class TextDocumentAttachment(BaseModel):
    """A supported text file (.txt/.md/.csv) decoded for the model."""

    model_config = ConfigDict(frozen=True)

    file_name: str
    """Original file name as reported by Telegram."""

    content: str
    """Decoded text content of the document."""

    size_bytes: int
    """File size in bytes as reported by Telegram metadata."""

    @field_validator("file_name")
    @classmethod
    def _require_supported_extension(cls, value: str) -> str:
        """Ensure only documents with a supported text extension are stored."""
        suffix = PurePosixPath(value).suffix.lower()
        if suffix not in SUPPORTED_TEXT_EXTENSIONS:
            raise ValueError(f"unsupported document extension: {suffix!r}")
        return value


class UnsupportedFileRef(BaseModel):
    """Name/size description of a file the assistant cannot read.

    At least one of :attr:`file_name` / :attr:`size_bytes` must be populated
    whenever a file-level description is possible; the message-level
    ``unsupported_file_count`` covers the no-metadata case (FR-006).
    """

    model_config = ConfigDict(frozen=True)

    file_name: str | None = None
    """Original name, or ``None`` when unavailable without downloading."""

    size_bytes: int | None = None
    """Size in bytes, or ``None`` when unavailable without downloading."""

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> UnsupportedFileRef:
        """Reject refs that carry no descriptive information at all."""
        if self.file_name is None and self.size_bytes is None:
            raise ValueError("UnsupportedFileRef requires file_name or size_bytes")
        return self


class MessageSource(BaseModel):
    """Provenance of a unified message, built exclusively by trusted layers."""

    model_config = ConfigDict(frozen=True)

    chat_id: int
    """Owner's private bot-chat id (for ``IN`` messages)."""

    message_id: int
    """Telegram message id; for albums the first fragment seen."""

    sender_id: int
    """Always equals ``AUTHORIZED_USER_ID`` for accepted input."""

    sent_at: datetime
    """Timestamp reported by Telegram."""

    forwarded_from_chat_id: int | None = None
    """Origin chat id, present only for forwarded content (FR-003)."""

    forwarded_from_name: str | None = None
    """Human-readable origin label when available."""


class UnifiedMessage(BaseModel):
    """The single normalized representation of one inbound or outbound message.

    A message reaches the agent only if it carries at least one of: non-empty
    ``text``, images, or documents (FR-005). When unsupported files coexist
    with supported content, their description line is prepended to ``text``
    as its first line by the messaging layer (FR-006).
    """

    model_config = ConfigDict(frozen=True)

    direction: Direction
    """IN from the owner or OUT delivered by the assistant."""

    text: str | None = None
    """Body or caption after normalization rules."""

    images: list[ImageAttachment] = []
    """Raw image attachments; empty for text-only messages."""

    documents: list[TextDocumentAttachment] = []
    """Supported text-file attachments only."""

    unsupported_files: list[UnsupportedFileRef] = []
    """Name/size descriptions of files the assistant cannot read (FR-006)."""

    unsupported_file_count: int = 0
    """Count fallback for unsupported items whose metadata required download."""

    source: MessageSource
    """Provenance recorded by the messaging layer, never from user text."""
