import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.request_context import (
    GatewayRequestContext,
    current_gateway_request_context,
)
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.plugins import invoke_plugin_command_handler
from tools.thread_context import propagate_context_to_thread


def _bind(
    actor: str,
    channel: str,
    message: str,
    *,
    session_id: str = "session-1",
    source: str = "gateway",
):
    return set_session_vars(
        source=source,
        platform="slack",
        scope_id="T123",
        user_id=actor,
        chat_id=channel,
        thread_id="171.100",
        message_id=message,
        session_key="agent:main:slack:T123:C123:171.100",
        session_id=session_id,
    )


def _slack_root_turn(
    actor: str = "U123",
    channel: str = "C123",
    message: str = "171.101",
    session_id: str = "session-1",
):
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id=channel,
        chat_name="general",
        chat_type="group",
        user_id=actor,
        user_name="alice",
        thread_id=message,
        scope_id="T123",
    )
    event = MessageEvent(text="hello", source=source, message_id=message)
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key=f"agent:main:slack:T123:{channel}:{message}",
        session_id=session_id,
    )
    return event, context


def test_non_gateway_origin_has_no_authenticated_context():
    tokens = _bind("U123", "C123", "171.101", source="cli")
    try:
        assert current_gateway_request_context(session_id="session-1") is None
    finally:
        clear_session_vars(tokens)


def test_gateway_context_is_frozen_and_host_derived():
    tokens = _bind("U123", "C123", "171.101")
    try:
        context = current_gateway_request_context(session_id="session-1")
        assert context == GatewayRequestContext(
            platform="slack",
            workspace_id="T123",
            actor_id="U123",
            channel_id="C123",
            thread_id="171.100",
            message_id="171.101",
            session_id="session-1",
        )
        with pytest.raises(FrozenInstanceError):
            context.actor_id = "U999"
    finally:
        clear_session_vars(tokens)


def test_slack_root_event_produces_complete_authenticated_context():
    runner = object.__new__(GatewayRunner)
    event, session_context = _slack_root_turn()

    tokens = runner._set_session_env(
        session_context,
        message_id=event.message_id,
    )
    try:
        context = current_gateway_request_context(session_id="session-1")
    finally:
        runner._clear_session_env(tokens)

    assert context == GatewayRequestContext(
        platform="slack",
        workspace_id="T123",
        actor_id="U123",
        channel_id="C123",
        thread_id="171.101",
        message_id="171.101",
        session_id="session-1",
    )


def test_gateway_context_propagates_to_tool_worker_thread():
    tokens = _bind("U123", "C123", "171.101")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            context = pool.submit(
                propagate_context_to_thread(current_gateway_request_context),
                session_id="session-1",
            ).result()
        assert context is not None
        assert context.actor_id == "U123"
        assert context.message_id == "171.101"
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_concurrent_contexts_survive_shared_worker_thread_reuse():
    runner = object.__new__(GatewayRunner)
    executor = ThreadPoolExecutor(max_workers=1)
    runner._executor = executor

    async def resolve(actor: str, channel: str, message: str, session_id: str):
        event, session_context = _slack_root_turn(
            actor=actor,
            channel=channel,
            message=message,
            session_id=session_id,
        )
        tokens = runner._set_session_env(
            session_context,
            message_id=event.message_id,
        )
        try:
            await asyncio.sleep(0)
            return await runner._run_in_executor_with_context(
                lambda: current_gateway_request_context(session_id=session_id),
            )
        finally:
            runner._clear_session_env(tokens)

    try:
        first, second = await asyncio.gather(
            resolve("U111", "C111", "1", "session-111"),
            resolve("U222", "C222", "2", "session-222"),
        )
    finally:
        executor.shutdown(wait=True)

    assert (
        first.actor_id,
        first.channel_id,
        first.message_id,
        first.session_id,
    ) == ("U111", "C111", "1", "session-111")
    assert (
        second.actor_id,
        second.channel_id,
        second.message_id,
        second.session_id,
    ) == ("U222", "C222", "2", "session-222")


def test_caller_session_id_must_match_bound_session():
    tokens = _bind("U123", "C123", "171.101", session_id="session-bound")
    try:
        assert current_gateway_request_context(session_id="session-other") is None
    finally:
        clear_session_vars(tokens)


def test_internal_incomplete_and_cleaned_contexts_are_unavailable():
    internal_tokens = _bind(
        "U123",
        "C123",
        "171.101",
        source="gateway_internal",
    )
    try:
        assert current_gateway_request_context(session_id="session-1") is None
    finally:
        clear_session_vars(internal_tokens)

    incomplete_tokens = _bind("U123", "C123", "", session_id="session-1")
    try:
        assert current_gateway_request_context(session_id="session-1") is None
    finally:
        clear_session_vars(incomplete_tokens)

    complete_tokens = _bind("U123", "C123", "171.101", session_id="session-1")
    clear_session_vars(complete_tokens)
    assert current_gateway_request_context(session_id="session-1") is None


def test_rejected_context_diagnostics_never_include_identity_values(caplog):
    caplog.set_level(logging.DEBUG, logger="gateway.request_context")
    tokens = _bind(
        "SENSITIVE-ACTOR",
        "SENSITIVE-CHANNEL",
        "SENSITIVE-MESSAGE",
        session_id="SENSITIVE-BOUND-SESSION",
    )
    try:
        assert (
            current_gateway_request_context(session_id="SENSITIVE-CALLER-SESSION")
            is None
        )
    finally:
        clear_session_vars(tokens)

    assert "session_id_mismatch" in caplog.text
    assert "SENSITIVE" not in caplog.text


def test_model_tool_dispatch_passes_context_outside_model_arguments():
    from model_tools import handle_function_call
    from tools.registry import registry

    captured = {}

    def handler(args, **kwargs):
        captured["args"] = args
        captured["context"] = kwargs.get("gateway_context")
        return "ok"

    registry.register(
        name="test_gateway_context_tool",
        toolset="test_gateway_context",
        schema={
            "name": "test_gateway_context_tool",
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler,
        override=True,
    )
    tokens = _bind("U123", "C123", "171.101")
    try:
        assert handle_function_call(
            "test_gateway_context_tool", {}, session_id="session-1"
        ) == "ok"
    finally:
        clear_session_vars(tokens)

    assert captured["args"] == {}
    assert "gateway_context" not in captured["args"]
    assert captured["context"].actor_id == "U123"


def test_plugin_command_context_is_opt_in_and_backward_compatible():
    context = GatewayRequestContext(
        platform="slack",
        workspace_id="T123",
        actor_id="U123",
        channel_id="C123",
        thread_id="171.100",
        message_id="171.101",
        session_id="session-1",
    )

    def legacy(raw_args):
        return ("legacy", raw_args)

    def context_aware(raw_args, gateway_context=None):
        return (raw_args, gateway_context)

    assert invoke_plugin_command_handler(
        legacy, "status", gateway_context=context
    ) == ("legacy", "status")
    assert invoke_plugin_command_handler(
        context_aware, "approve", gateway_context=context
    ) == ("approve", context)
