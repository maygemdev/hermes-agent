"""Request-auth provider behavior for identity-aware reverse proxies."""

from __future__ import annotations

import time
from collections.abc import Mapping

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    ProviderError,
    Session,
    clear_providers,
    list_request_providers,
    list_session_providers,
    register_provider,
)


class _HeaderProvider(DashboardAuthProvider):
    name = "header"
    display_name = "Header"
    supports_session = False
    supports_request_auth = True

    def __init__(self, *, unavailable: bool = False) -> None:
        self.unavailable = unavailable

    def verify_request(self, *, headers: Mapping[str, str]):
        if self.unavailable:
            raise ProviderError("identity proxy unavailable")
        assertion = headers.get("x-verified-identity", "")
        if assertion != "signed-alice":
            return None
        return Session(
            user_id="alice-id",
            email="alice@example.test",
            display_name="Alice",
            org_id="example",
            provider=self.name,
            expires_at=int(time.time()) + 300,
            access_token=assertion,
            refresh_token="",
        )

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        raise NotImplementedError

    def verify_session(self, *, access_token: str):
        return None

    def refresh_session(self, *, refresh_token: str):
        raise NotImplementedError

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


@pytest.fixture
def gated_client():
    clear_providers()
    previous = {
        name: getattr(web_server.app.state, name, None)
        for name in ("bound_host", "bound_port", "auth_required")
    }
    web_server.app.state.bound_host = "dashboard.example.test"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(
        web_server.app,
        base_url="https://dashboard.example.test",
    )
    yield client
    clear_providers()
    for name, value in previous.items():
        setattr(web_server.app.state, name, value)


def test_request_provider_is_separate_from_interactive_providers():
    provider = _HeaderProvider()
    register_provider(provider)

    assert list_request_providers() == [provider]
    assert list_session_providers() == []


def test_verified_request_header_authenticates_api_and_ticket(gated_client):
    register_provider(_HeaderProvider())
    headers = {"X-Verified-Identity": "signed-alice"}

    me = gated_client.get("/api/auth/me", headers=headers)
    ticket = gated_client.post("/api/auth/ws-ticket", headers=headers)

    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.test"
    assert me.json()["provider"] == "header"
    assert ticket.status_code == 200
    assert ticket.json()["ticket"]


def test_invalid_request_header_fails_closed(gated_client):
    register_provider(_HeaderProvider())

    response = gated_client.get(
        "/api/auth/me",
        headers={"X-Verified-Identity": "tampered"},
    )

    assert response.status_code == 401


def test_request_provider_outage_is_transient(gated_client):
    register_provider(_HeaderProvider(unavailable=True))

    response = gated_client.get(
        "/api/auth/me",
        headers={"X-Verified-Identity": "signed-alice"},
    )

    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
