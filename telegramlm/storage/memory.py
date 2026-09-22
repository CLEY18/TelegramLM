"""In-process :class:`~telegramlm.storage.store.MessageStore` implementation."""

from __future__ import annotations

from telegramlm.messaging.models import UnifiedMessage
from telegramlm.storage.store import MessageStore


class InMemoryMessageStore(MessageStore):
    """Append-only list-backed history for the single owner (v1).

    The store is strictly chronological with no mutation or deletion; content
    is intentionally lost on process restart, which is a documented v1
    assumption. A future persistent implementation will satisfy the same
    ``MessageStore`` protocol.
    """

    def __init__(self) -> None:
        """Initialize an empty history."""
        self._messages: list[UnifiedMessage] = []

    def append(self, message: UnifiedMessage) -> None:
        """Add ``message`` to the tail of the history.

        Args:
            message: The unified message to persist.
        """
        self._messages.append(message)

    def get_recent(self, limit: int) -> list[UnifiedMessage]:
        """Return up to ``limit`` most recent messages in chronological order.

        Args:
            limit: Maximum number of trailing messages to return; fewer are
                returned when the history is shorter. A non-positive limit
                yields an empty list.

        Returns:
            The last ``limit`` messages ordered oldest-first.
        """
        if limit <= 0:
            return []
        return self._messages[-limit:]
