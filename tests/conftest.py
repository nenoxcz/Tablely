import pytest


@pytest.fixture(autouse=True)
def isolated_tablely_home(tmp_path, monkeypatch):
    """Keep tests away from the real ~/.tablely and from the caller's agent name."""
    home = tmp_path / "tablely-home"
    monkeypatch.setenv("TABLELY_HOME", str(home))
    monkeypatch.delenv("TABLELY_AGENT", raising=False)
    return home
