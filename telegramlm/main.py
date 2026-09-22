"""Application entry point: ``python -m telegramlm.main``.

Composes every module exactly once — settings, logging, the bot client with
its single-user authorization gate, the optional owner-account client that
backs the three capability tools, the message processor, the coalescing queue
worker, and the agent core — then runs until interrupted and shuts both
clients down cleanly. This is the only place where concrete implementations
are chosen (see plan.md dependency direction).
"""

from __future__ import annotations

import asyncio
import logging
import sys

import hydrogram
from hydrogram.enums import ChatType
from hydrogram.errors import AuthKeyUnregistered, SessionExpired, SessionRevoked
from pydantic import ValidationError

from telegramlm.agent.core import AgentCore
from telegramlm.agent.prompts import GREETING_TEXT
from telegramlm.agent.queue_worker import GENERIC_ERROR_NOTICE, ConversationQueueWorker
from telegramlm.config import Settings
from telegramlm.llm.client import OpenAICompatibleClient
from telegramlm.messaging.models import UnifiedMessage
from telegramlm.messaging.processor import MessageProcessor
from telegramlm.storage.memory import InMemoryMessageStore
from telegramlm.tools.base import ToolRegistry
from telegramlm.tools.channel_posts import GetChannelPostsTool
from telegramlm.tools.list_chats import ListChatsTool
from telegramlm.tools.post_comments import GetPostCommentsTool
from telegramlm.tools.send_message import SendMessageTool

logger = logging.getLogger(__name__)


def load_settings() -> Settings:
    """Load and validate configuration, failing fast with a readable report.

    Returns:
        The validated :class:`~telegramlm.config.Settings` instance.

    Exits:
        With code ``2`` and a field-level problem list (never echoing input
        values) when required configuration is missing or invalid (FR-020).
    """
    try:
        return Settings()
    except ValidationError as exc:
        print("Configuration error — fix .env and restart:", file=sys.stderr)
        for problem in exc.errors():
            location = ".".join(str(part) for part in problem["loc"])
            print(f"  {location}: {problem['msg']}", file=sys.stderr)
        sys.exit(2)


def configure_logging(level: int) -> None:
    """Set up console logging; secrets never enter log records (FR-020).

    Args:
        level: Requested application log level, resolved from the validated
            ``LOG_LEVEL`` setting.

    Only the ``telegramlm.*`` namespace is raised to the requested level; the
    root logger stays at INFO minimum so third-party libraries never go DEBUG.
    That matters because dependencies are noisy and unsafe at that verbosity:
    ``httpx``/``httpcore`` dump request headers (including the LLM bearer
    token), hydrogram dumps raw update payloads, and its session storage
    (aiosqlite) dumps SQL statements carrying account auth keys.
    """
    logging.basicConfig(
        level=max(level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("telegramlm").setLevel(level)


async def _start_user_client(user_client: hydrogram.Client) -> None:
    """Connect the owner-account client, prompting interactively on first run.

    A stored session file is reused silently for unattended restarts; a revoked
    or expired session ends the process with an explicit re-login instruction
    per ``contracts/telegram-interface.md`` (FR-019).

    Args:
        user_client: Configured but not yet started user-account client.
    """
    try:
        await user_client.start()
    except (SystemExit, EOFError):  # hydrogram prompts interactively on first run
        print(
            "No stored user-account session and no interactive terminal available.\n"
            "Start this application in an interactive terminal and complete the "
            "phone/code login once; later unattended runs reuse the saved session.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except (SessionRevoked, SessionExpired, AuthKeyUnregistered):
        print(
            "User-account session is revoked or expired.\n"
            "Run this application interactively in a terminal and complete the "
            "phone/code login again to restore account capabilities.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    logger.info("user client connected as %s", user_client.me.username or user_client.me.id)


async def _stop_quietly(client: hydrogram.Client) -> None:
    """Disconnect a client, ignoring failures from never-completed startups.

    Args:
        client: Either connected or partially initialized hydrogram client.
    """
    try:
        await client.stop()
    except Exception:  # noqa: BLE001 - shutdown must not mask the real outcome
        logger.debug("client stop ignored", exc_info=True)


async def run(settings: Settings) -> None:
    """Wire all components and serve until the process is interrupted.

    Args:
        settings: Validated application configuration.
    """
    store = InMemoryMessageStore()
    registry = ToolRegistry()
    llm_client = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )

    owner_chat_id = settings.authorized_user_id
    bot = hydrogram.Client(
        name="telegramlm-bot",
        api_id=settings.api_id,
        api_hash=settings.api_hash.get_secret_value(),
        bot_token=settings.bot_token.get_secret_value(),
        in_memory=True,
    )

    async def send_notice(text: str) -> None:
        """Deliver a short system notice (unsupported/error/greeting) to the owner."""
        await bot.send_message(owner_chat_id, text)

    agent_core = AgentCore(
        llm_client=llm_client,
        registry=registry,
        store=store,
        bot_client=bot,
        settings=settings,
    )
    worker = ConversationQueueWorker(agent_core=agent_core, store=store, notify_error=send_notice)

    async def dispatch(message: UnifiedMessage) -> None:
        """Route a normalized unified message into the sequential worker."""
        await worker.submit(message)

    processor = MessageProcessor(
        bot_client=bot,
        settings=settings,
        dispatch=dispatch,
        notify_unsupported=send_notice,
    )

    registry.register(SendMessageTool(bot_client=bot, owner_chat_id=owner_chat_id, store=store))

    @bot.on_message()
    async def handle_update(_client: hydrogram.Client, message: hydrogram.types.Message) -> None:
        """Authorization gate plus routing for every incoming message.

        Everything except private messages from the authorized owner is ignored
        with a log-only record (FR-001/FR-002); ``/start`` gets the capability
        greeting; all other accepted content goes to the processor, and any
        unexpected failure yields one generic notice while the traceback stays
        in logs (FR-021).
        """
        sender = message.from_user
        if message.chat.type != ChatType.PRIVATE or sender is None:
            logger.info("ignored non-private update id=%s chat=%s", message.id, message.chat.id)
            return
        if sender.id != settings.authorized_user_id:
            logger.info("ignored private update from unauthorized user id=%s", sender.id)
            return
        if message.text and message.text.split()[0].split("@")[0] == "/start":
            await send_notice(GREETING_TEXT)
            return
        try:
            await processor.handle_message(message)
        except Exception:
            logger.exception("unhandled failure processing message id=%s", message.id)
            try:
                await send_notice(GENERIC_ERROR_NOTICE)
            except Exception:  # noqa: BLE001 - last-resort notice attempt
                logger.exception("error notice delivery also failed")

    user_client = hydrogram.Client(
        name=settings.session_name,
        api_id=settings.api_id,
        api_hash=settings.api_hash.get_secret_value(),
    )

    bot_started = False
    user_started = False
    try:
        await bot.start()
        bot_started = True
        logger.info("bot client connected; serving owner id=%s", settings.authorized_user_id)
        await _start_user_client(user_client)
        user_started = True
        registry.register(ListChatsTool(user_client=user_client, settings=settings))
        registry.register(GetChannelPostsTool(user_client=user_client, settings=settings))
        registry.register(GetPostCommentsTool(user_client=user_client, settings=settings))

        worker_task = asyncio.create_task(worker.run(), name="conversation-worker")
        tool_names = ", ".join(sorted(registry.names()))
        logger.info("application ready; registered tools: %s", tool_names)
        try:
            await asyncio.Event().wait()
        finally:
            worker_task.cancel()
    finally:
        if user_started:
            await _stop_quietly(user_client)
        if bot_started:
            await _stop_quietly(bot)


def main() -> None:
    """Console entry point: load config, run the event loop, handle Ctrl+C."""
    settings = load_settings()
    configure_logging(logging.getLevelNamesMapping()[settings.log_level])
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down cleanly")


if __name__ == "__main__":
    main()
