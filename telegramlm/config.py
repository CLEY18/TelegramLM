"""Validated application settings loaded from the environment.

All credentials and tunables are defined here as a single ``Settings`` model so
that startup fails fast on missing or invalid values (FR-020). Secret fields
use :class:`~pydantic.SecretStr` so they never appear in logs, ``repr()``, or
tracebacks. The ``.env`` file is read-only for this application; the operator
fills it in themselves.
"""

from __future__ import annotations

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Console log levels accepted by ``LOG_LEVEL`` (validated at startup).
SUPPORTED_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")


class Settings(BaseSettings):
    """Runtime configuration: credentials (required) and behavior tunables.

    Values are read from environment variables (case-insensitive) with an
    optional ``.env`` file in the working directory. Credential fields have no
    defaults so that a missing value raises at startup; every tunable has a
    documented default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Credentials (no defaults: startup must fail without them) ---------

    bot_token: SecretStr
    """Bot token from @BotFather used to authenticate the bot client."""

    api_id: int
    """Telegram API id from my.telegram.org used by both clients."""

    api_hash: SecretStr
    """Telegram API hash from my.telegram.org (secret, never logged)."""

    authorized_user_id: int
    """Numeric Telegram user id of the single owner this bot serves."""

    llm_base_url: str
    """Base URL of an OpenAI-compatible API server, e.g. https://api.openai.com/v1."""

    llm_api_key: SecretStr
    """API key for the language service; may be empty for local servers."""

    llm_model: str
    """Model name passed to chat-completions requests (must support tools)."""

    # --- Behavior tunables (every field has a default) ----------------------

    llm_timeout_seconds: float = 300.0
    """Per-request wall-clock timeout for the language service; local models
    with tool calling often need well over a minute to answer."""

    vision_enabled: bool = False
    """Send current-turn images to the model; historical ones stay placeholders."""

    media_group_timeout_seconds: float = 1.5
    """Idle window in seconds used to collect album fragments before flushing."""

    max_file_size_bytes: int = 1_048_576
    """Largest text attachment (.txt/.md/.csv) that will be downloaded (bytes)."""

    max_history_messages: int = 30
    """Number of most recent stored messages included in each model request."""

    tool_default_limit: int = 20
    """Default page size for account-capability tools."""

    agent_max_iterations: int = 8
    """Maximum model/tool iterations within one agent turn."""

    session_name: str = "telegramlm_owner"
    """File name (without suffix) of the stored user-account session file."""

    log_level: str = "INFO"
    """Console log level for application (``telegramlm.*``) loggers; third-party
    libraries stay at INFO or stricter so secrets never reach the logs (FR-020)."""

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Uppercase and validate the configured console log level.

        Args:
            value: Raw ``LOG_LEVEL`` string from the environment.

        Returns:
            The normalized (upper-case) level name.

        Raises:
            ValueError: The value is not one of :data:`SUPPORTED_LOG_LEVELS`.
        """
        normalized = value.strip().upper()
        if normalized not in SUPPORTED_LOG_LEVELS:
            raise ValueError(f"log_level must be one of: {', '.join(SUPPORTED_LOG_LEVELS)}")
        return normalized
