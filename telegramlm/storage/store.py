"""Storage abstraction for the owner's conversation history.

The :class:`MessageStore` protocol is the single source of truth for agent
context (FR-014). v1 ships an in-process implementation; a database-backed
store can replace it without touching any consumer because everything depends
only on this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from telegramlm.messaging.models import UnifiedMessage


@runtime_checkable
class MessageStore(Protocol):
    """Append-only chronological history of unified messages."""

    def append(self, message: UnifiedMessage) -> None:
        """Add ``message`` to the tail of the history.

        Args:
            message: The unified message to persist.
        """
        ...

    def get_recent(self, limit: int) -> list[UnifiedMessage]:
        """Return up to ``limit`` most recent messages in chronological order.

        Args:
            limit: Maximum number of trailing messages to return; fewer are
                returned when the history is shorter.

        Returns:
            The last ``limit`` messages ordered oldest-first.
        """
        ...
