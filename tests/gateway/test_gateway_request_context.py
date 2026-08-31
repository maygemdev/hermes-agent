from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from gateway.request_context import (
    GatewayRequestContext,
    current_gateway_request_context,
)
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.plugins import invoke_plugin_command_handler
from tools.thread_context import propagate_context_to_thread


def _bind(actor: str, channel: str, message: str, *, source: str = "gateway"):
    return set_session_vars(
        source=source,
        platform="slack",
        scope_id="T123",
        user_id=actor,
        chat_id=channel,
        thread_id="171.100",
        message_id=message,
        session_key="agent:main:slack:T123:C123:171.100",
    )


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


def test_concurrent_contexts_do_not_leak_between_workers():
    def resolve(actor: str, channel: str, message: str):
        tokens = _bind(actor, channel, message)
        try:
            return current_gateway_request_context(session_id=f"session-{actor}")
        finally:
            clear_session_vars(tokens)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(resolve, "U111", "C111", "1").result()
        second = pool.submit(resolve, "U222", "C222", "2").result()

    assert (first.actor_id, first.channel_id, first.message_id) == ("U111", "C111", "1")
    assert (second.actor_id, second.channel_id, second.message_id) == ("U222", "C222", "2")


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
