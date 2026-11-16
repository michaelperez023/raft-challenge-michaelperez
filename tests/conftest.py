"""Every test runs offline: there is no API key, and no request leaves 127.0.0.1."""

import pytest

import main


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "ORDERS_API_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def api_url():
    """Raft's dummy API, served on 127.0.0.1 for the whole session."""
    with main.dummy_api() as url:
        yield url
