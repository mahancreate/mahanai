"""Persistent user config (API key) outside the project .env."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_CONFIG_NAME = "config.json"


def config_file_path() -> Path:
    override = os.environ.get("MAHANAI_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve() / _CONFIG_NAME
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home())
        return (base / "MahanAI" / _CONFIG_NAME).resolve()
    return (Path.home() / ".config" / "mahanai" / _CONFIG_NAME).resolve()


def _read_config() -> dict[str, Any]:
    path = config_file_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_config(data: dict[str, Any]) -> None:
    path = config_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    if os.name != "nt":
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass


def load_saved_api_key() -> str | None:
    key = (_read_config().get("api_key") or "").strip()
    return key or None


def save_api_key(api_key: str) -> None:
    data = _read_config()
    data["api_key"] = api_key.strip()
    _write_config(data)
    print("CONFIG PATH (SAVE):", config_file_path())


def clear_saved_api_key() -> None:
    data = _read_config()
    data.pop("api_key", None)
    if data:
        _write_config(data)
    else:
        path = config_file_path()
        try:
            path.unlink()
        except OSError:
            pass


def resolve_api_key():
    import json

    try:
        with open(config_file_path(), "r") as f:
            data = json.load(f)
            return data.get("api_key")
    except:
        return None


def save_nvidia_api_key(api_key: str) -> None:
    data = _read_config()
    data["nvidia_api_key"] = api_key.strip()
    _write_config(data)


def load_nvidia_api_key() -> str | None:
    key = (_read_config().get("nvidia_api_key") or "").strip()
    return key or None


def clear_nvidia_api_key() -> None:
    data = _read_config()
    data.pop("nvidia_api_key", None)
    if data:
        _write_config(data)
    else:
        path = config_file_path()
        try:
            path.unlink()
        except OSError:
            pass
