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


def save_codex_token(token_data: dict[str, Any]) -> None:
    data = _read_config()
    data["codex_token"] = token_data
    _write_config(data)


def load_codex_token() -> dict[str, Any] | None:
    return _read_config().get("codex_token") or None


def clear_codex_token() -> None:
    data = _read_config()
    data.pop("codex_token", None)
    if data:
        _write_config(data)
    else:
        path = config_file_path()
        try:
            path.unlink()
        except OSError:
            pass


def save_custom_endpoint(url: str, model: str, api_key: str) -> None:
    data = _read_config()
    data["custom_endpoint"] = {
        "url": url.strip(),
        "model": model.strip(),
        "api_key": api_key.strip(),
    }
    _write_config(data)


def load_custom_endpoint() -> dict[str, str] | None:
    entry = _read_config().get("custom_endpoint")
    if not entry or not entry.get("url"):
        return None
    return entry


def clear_custom_endpoint() -> None:
    data = _read_config()
    data.pop("custom_endpoint", None)
    if data:
        _write_config(data)
    else:
        path = config_file_path()
        try:
            path.unlink()
        except OSError:
            pass


def load_theme() -> str:
    return (_read_config().get("theme") or "midnight")


def save_theme(theme: str) -> None:
    data = _read_config()
    data["theme"] = theme
    _write_config(data)


def load_custom_theme_path() -> str | None:
    info = _read_config().get("custom_theme") or {}
    return (info.get("path") or _read_config().get("custom_theme_path") or "").strip() or None


def save_custom_theme_path(path: str) -> None:
    data = _read_config()
    data["custom_theme_path"] = path.strip()
    _write_config(data)


def clear_custom_theme_path() -> None:
    data = _read_config()
    data.pop("custom_theme_path", None)
    _write_config(data)


def save_custom_theme_info(slug: str, display: str, path: str) -> None:
    """Persist full .mai theme metadata so it can be re-registered on next startup."""
    data = _read_config()
    data["custom_theme"] = {"slug": slug, "display": display, "path": path}
    _write_config(data)


def load_custom_theme_info() -> dict | None:
    """Return {slug, display, path} for the saved .mai theme, or None."""
    return _read_config().get("custom_theme") or None


def clear_custom_theme() -> None:
    """Remove all saved .mai theme data from config."""
    data = _read_config()
    data.pop("custom_theme", None)
    data.pop("custom_theme_path", None)
    _write_config(data)


def save_ollama_provider(name: str, address: str, port: int, api_key: str, url: str = "") -> None:
    data = _read_config()
    providers = data.setdefault("ollama_providers", {})
    providers[name] = {
        "name": name,
        "address": address.strip(),
        "port": int(port),
        "api_key": api_key.strip(),
        "url": url,
    }
    _write_config(data)


def load_ollama_providers() -> dict[str, dict]:
    return _read_config().get("ollama_providers", {})


def remove_ollama_provider(name: str) -> None:
    data = _read_config()
    providers = data.get("ollama_providers", {})
    providers.pop(name, None)
    if providers:
        data["ollama_providers"] = providers
    else:
        data.pop("ollama_providers", None)
    _write_config(data)


def save_plugin(name: str, path: str, codename: str = "", reg_store: str = "", reg_name: str = "") -> None:
    data = _read_config()
    plugins = data.setdefault("plugins", {})
    plugins[name] = {"name": name, "path": path, "codename": codename, "reg_store": reg_store, "reg_name": reg_name}
    _write_config(data)


def load_plugins() -> dict[str, dict]:
    return _read_config().get("plugins", {})


def remove_plugin(name: str) -> None:
    data = _read_config()
    plugins = data.get("plugins", {})
    plugins.pop(name, None)
    if plugins:
        data["plugins"] = plugins
    else:
        data.pop("plugins", None)
    _write_config(data)


def load_always_allowed() -> dict:
    return _read_config().get("always_allowed", {})


def add_always_allowed_command(prefix: str) -> None:
    data = _read_config()
    aa = data.setdefault("always_allowed", {})
    prefixes: list = aa.setdefault("command_prefixes", [])
    if prefix not in prefixes:
        prefixes.append(prefix)
    _write_config(data)


def add_always_allowed_file_op(op: str) -> None:
    data = _read_config()
    aa = data.setdefault("always_allowed", {})
    ops: list = aa.setdefault("file_ops", [])
    if op not in ops:
        ops.append(op)
    _write_config(data)


# ── Memory ────────────────────────────────────────────────────────────────────

def save_memory(content: str) -> str:
    """Save a memory entry. Returns the generated ID."""
    import time as _time
    mid = str(int(_time.time() * 1000))
    data = _read_config()
    memories = data.setdefault("memories", {})
    memories[mid] = {"id": mid, "content": content.strip()}
    _write_config(data)
    return mid


def load_memories() -> dict[str, dict]:
    return _read_config().get("memories", {})


def remove_memory(mid: str) -> bool:
    data = _read_config()
    memories = data.get("memories", {})
    if mid not in memories:
        return False
    memories.pop(mid)
    data["memories"] = memories
    if not memories:
        data.pop("memories", None)
    _write_config(data)
    return True


# ── Prompt library ────────────────────────────────────────────────────────────

def save_prompt(name: str, content: str) -> None:
    data = _read_config()
    prompts = data.setdefault("prompts", {})
    prompts[name.strip()] = content.strip()
    _write_config(data)


def load_prompts() -> dict[str, str]:
    return _read_config().get("prompts", {})


def remove_prompt(name: str) -> bool:
    data = _read_config()
    prompts = data.get("prompts", {})
    if name not in prompts:
        return False
    prompts.pop(name)
    data["prompts"] = prompts
    if not prompts:
        data.pop("prompts", None)
    _write_config(data)
    return True


# ── Aliases ───────────────────────────────────────────────────────────────────

def save_alias(trigger: str, command: str) -> None:
    data = _read_config()
    aliases = data.setdefault("aliases", {})
    aliases[trigger.strip()] = command.strip()
    _write_config(data)


def load_aliases() -> dict[str, str]:
    return _read_config().get("aliases", {})


def remove_alias(trigger: str) -> bool:
    data = _read_config()
    aliases = data.get("aliases", {})
    if trigger not in aliases:
        return False
    aliases.pop(trigger)
    data["aliases"] = aliases
    if not aliases:
        data.pop("aliases", None)
    _write_config(data)
    return True


# ── Sessions ──────────────────────────────────────────────────────────────────

def sessions_dir() -> Path:
    return config_file_path().parent / "sessions"


def save_session(session_id: str, session_data: dict[str, Any]) -> None:
    d = sessions_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session_id}.json"
    path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")


def load_session(session_id: str) -> dict[str, Any] | None:
    path = sessions_dir() / f"{session_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_sessions() -> list[dict[str, Any]]:
    d = sessions_dir()
    if not d.is_dir():
        return []
    result = []
    for f in sorted(d.glob("*.json"), reverse=True)[:50]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append(data)
        except Exception:
            pass
    return result


# ── Token display setting ─────────────────────────────────────────────────────

def save_tokens_setting(enabled: bool) -> None:
    data = _read_config()
    data["show_tokens"] = enabled
    _write_config(data)


def load_tokens_setting() -> bool:
    return bool(_read_config().get("show_tokens", False))


# ── Document index ────────────────────────────────────────────────────────────

def save_index_documents(docs: list[dict[str, Any]]) -> None:
    data = _read_config()
    data["index_docs"] = docs
    _write_config(data)


def load_index_documents() -> list[dict[str, Any]]:
    return _read_config().get("index_docs", [])
