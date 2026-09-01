"""Immutable authenticated provenance for gateway-originated plugin calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from gateway.session_context import get_session_env


logger = logging.getLogger(__name__)


def _reject(reason: str) -> None:
    """Log only the rejection class, never bound identity values."""

    logger.debug("authenticated gateway context rejected: reason=%s", reason)
    return None


@dataclass(frozen=True, slots=True)
class GatewayRequestContext:
    """A sanitized snapshot of host-owned gateway request metadata."""

    platform: str
    workspace_id: str
    actor_id: str
    channel_id: str
    thread_id: str
    message_id: str
    session_id: str


def current_gateway_request_context(
    *, session_id: str = ""
) -> Optional[GatewayRequestContext]:
    """Return the authenticated context or ``None`` for unsafe origins."""

    if get_session_env("HERMES_SESSION_SOURCE", "") != "gateway":
        return _reject("unauthenticated_source")

    bound_session_id = get_session_env("HERMES_SESSION_ID", "")
    if not isinstance(bound_session_id, str) or not bound_session_id:
        return _reject("incomplete_context")
    if session_id and session_id != bound_session_id:
        return _reject("session_id_mismatch")

    values = {
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "workspace_id": get_session_env("HERMES_SESSION_SCOPE_ID", ""),
        "actor_id": get_session_env("HERMES_SESSION_USER_ID", ""),
        "channel_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
        "message_id": get_session_env("HERMES_SESSION_MESSAGE_ID", ""),
        "session_id": bound_session_id,
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        return _reject("incomplete_context")
    return GatewayRequestContext(**values)


def gateway_request_context_from_event(
    event: Any, *, session_id: str
) -> Optional[GatewayRequestContext]:
    """Build a frozen context directly from an authenticated gateway event."""

    if event is None:
        return _reject("missing_event")
    if getattr(event, "internal", False):
        return _reject("internal_event")
    source = getattr(event, "source", None)
    platform_value = getattr(getattr(source, "platform", None), "value", "")
    values = {
        "platform": str(platform_value or ""),
        "workspace_id": str(getattr(source, "scope_id", "") or ""),
        "actor_id": str(getattr(source, "user_id", "") or ""),
        "channel_id": str(getattr(source, "chat_id", "") or ""),
        "thread_id": str(getattr(source, "thread_id", "") or ""),
        "message_id": str(
            getattr(source, "message_id", "")
            or getattr(event, "message_id", "")
            or ""
        ),
        "session_id": str(session_id or ""),
    }
    if not all(values.values()):
        return _reject("incomplete_event_context")
    return GatewayRequestContext(**values)
