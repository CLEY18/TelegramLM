"""Sequential per-conversation worker with message coalescing.

One asyncio task serves the owner's single conversation: it takes the next
queued unified message, drains everything else that arrived while the previous
turn ran, appends the whole batch to history, and hands it to one consolidated
agent turn (FR-013). Ordering is preserved because turns never overlap; no
message is ever rejected or answered twice.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from telegramlm.agent.core import AgentCore
from telegramlm.messaging.models import UnifiedMessage
from telegramlm.storage.store import MessageStore

logger = logging.getLogger(__name__)

# Short generic notice sent when a turn dies on an unexpected failure; the
# full traceback goes to logs only (FR-021).
GENERIC_ERROR_NOTICE = "Something went wrong, please try again later."


class ConversationQueueWorker:
    """Serializes agent turns for one conversation with coalescing."""

    def __init__(
        self,
        agent_core: AgentCore,
        store: MessageStore,
        notify_error: Callable[[str], Awaitable[None]],
    ) -> None:
        """Wire the worker to the agent core, history, and error notifier.

        Args:
            agent_core: The turn runner invoked once per batch.
            store: History sink; accepted inbound messages are appended here
                before their turn starts so context slicing works.
            notify_error: Sends the short generic notice text to the owner.
        """
        self._agent_core = agent_core
        self._store = store
        self._notify_error = notify_error
        self._queue: asyncio.Queue[UnifiedMessage] = asyncio.Queue()

    async def submit(self, message: UnifiedMessage) -> None:
        """Enqueue an inbound unified message for the next turn.

        Args:
            message: Normalized owner message; processed alone or coalesced
                with neighbors that arrive during a running turn (FR-013).
        """
        self._queue.put_nowait(message)

    async def run(self) -> None:
        """Process batches forever until the task is cancelled.

        Each iteration pops one message, drains any additional queued messages
        into the same batch, appends all of them to history in arrival order,
        and runs a single consolidated agent turn. Unexpected exceptions are
        logged with traceback and answered by one generic notice so the loop
        itself never dies (FR-021).
        """
        while True:
            batch = [await self._queue.get()]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            for message in batch:
                self._store.append(message)
            logger.info("starting turn with %d coalesced message(s)", len(batch))
            try:
                await self._agent_core.run_turn(batch)
            except Exception:
                logger.exception("agent turn failed")
                try:
                    await self._notify_error(GENERIC_ERROR_NOTICE)
                except Exception:  # noqa: BLE001 - last-resort notice attempt
                    logger.exception("error notice delivery also failed")
