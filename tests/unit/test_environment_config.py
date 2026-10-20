"""
Tests for ConfigLoader.load_environment_config.

environment.yaml lists the environments and names the default; each entry's
definition lives in its own file. The loader must load the default's file as
the `environment` section, e.g. CURRENT_ENVIRONMENT=homebench gives the
homebench Yggdrasil URL, and refuse a default that is not listed.
"""

from pathlib import Path

import pytest

from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.shared.utils.config_resolver import resolve_yggdrasil_url

REPO_ENV_YAML = Path(__file__).resolve().parents[2] / "ami_agents" / "config" / "environment.yaml"


# -- helpers -----------------------------------------------------------------

def _write_listing(tmp_path: Path, default: str) -> Path:
    """Write a two-entry environment.yaml plus its definition files; return its path."""
    # 1. One definition file per entry, in a subfolder as in the repo.
    (tmp_path / "environments").mkdir()
    for env_id, port in (("home_a", 8080), ("home_b", 8090)):
        (tmp_path / "environments" / f"{env_id}.yaml").write_text(
            f'name: "{env_id}"\n'
            f"integration_engine:\n  yggdrasil:\n    url: \"http://localhost:{port}/\"\n"
        )

    # 2. The listing that names them and marks the default.
    listing = tmp_path / "environment.yaml"
    listing.write_text(
        "environments:\n"
        f'  default: "{default}"\n'
        "  entries:\n"
        '    home_a: "environments/home_a.yaml"\n'
        '    home_b: "environments/home_b.yaml"\n'
    )
    return listing


# -- tests -------------------------------------------------------------------

def test_default_entry_is_loaded_as_environment(tmp_path):
    """The default's definition becomes `environment`, and its URL is resolved."""
    config = ConfigLoader.load_environment_config(str(_write_listing(tmp_path, "home_b")))

    assert config["environment"]["name"] == "home_b"
    assert resolve_yggdrasil_url(config, env={}) == "http://localhost:8090/"


def test_unlisted_default_is_refused(tmp_path):
    """A default that names no entry fails loudly instead of picking another."""
    with pytest.raises(ValueError, match="not one of environments.entries"):
        ConfigLoader.load_environment_config(str(_write_listing(tmp_path, "home_c")))


def test_repo_environment_yaml_resolves_selected_entry(monkeypatch):
    """The repo's own file follows CURRENT_ENVIRONMENT to the matching definition."""
    # Clear overrides so the definition files' own defaults are what is read.
    monkeypatch.delenv("YGGDRASIL_URL", raising=False)
    monkeypatch.setenv("CURRENT_ENVIRONMENT", "homebench")

    config = ConfigLoader.load_environment_config(str(REPO_ENV_YAML))

    assert config["environment"]["name"] == "Home Bench"
    assert resolve_yggdrasil_url(config, env={}) == "http://localhost:8090/"


def test_missing_url_is_a_configuration_error():
    """No URL in config or YGGDRASIL_URL raises; BASE_WS_URI (HASP's setting) is not used."""
    with pytest.raises(ValueError, match="No Yggdrasil URL configured"):
        resolve_yggdrasil_url({}, env={"BASE_WS_URI": "http://localhost:8080/"})
