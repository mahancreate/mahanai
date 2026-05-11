"""Chat history storage and project management for MahanAI."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_filename(text: str, max_len: int = 60) -> str:
    """Convert arbitrary text to a safe filesystem filename stem."""
    text = str(text).strip()
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    text = text.strip("-")
    return text[:max_len] or "chat"


# ── Config I/O (lazy import to avoid circular deps) ───────────────────────────

def _read_cfg() -> dict[str, Any]:
    from mahanai.config import _read_config
    return _read_config()


def _write_cfg(data: dict[str, Any]) -> None:
    from mahanai.config import _write_config
    _write_config(data)


# ── Chat history setup ────────────────────────────────────────────────────────

def is_chat_history_setup() -> bool:
    return "chat_history" in _read_cfg()


def save_chat_history_config(universal_dir: str, project_folder_name: str) -> None:
    data = _read_cfg()
    data["chat_history"] = {
        "universal_dir": str(Path(universal_dir).expanduser().resolve()),
        "project_folder_name": (project_folder_name.strip() or "mahanai-chats"),
    }
    _write_cfg(data)


def load_chat_history_config() -> dict[str, str]:
    cfg = _read_cfg().get("chat_history", {})
    default_dir = str(Path.home() / "MahanAI-Chats")
    return {
        "universal_dir": cfg.get("universal_dir") or default_dir,
        "project_folder_name": cfg.get("project_folder_name") or "mahanai-chats",
    }


# ── Projects ──────────────────────────────────────────────────────────────────

def save_project(name: str, path: str, display: str) -> None:
    cfg = load_chat_history_config()
    folder = cfg["project_folder_name"]
    resolved = Path(path).expanduser().resolve()
    chats_dir = str(resolved / folder)
    data = _read_cfg()
    projects = data.setdefault("projects", {})
    projects[name] = {
        "name": name,
        "display": display,
        "path": str(resolved),
        "chats_dir": chats_dir,
    }
    _write_cfg(data)


def load_projects() -> dict[str, dict]:
    return _read_cfg().get("projects", {})


def remove_project(name: str) -> bool:
    data = _read_cfg()
    projects = data.get("projects", {})
    if name not in projects:
        return False
    projects.pop(name)
    if projects:
        data["projects"] = projects
    else:
        data.pop("projects", None)
    if data.get("active_project") == name:
        data.pop("active_project", None)
    _write_cfg(data)
    return True


def save_active_project(name: str | None) -> None:
    data = _read_cfg()
    if name is None:
        data.pop("active_project", None)
    else:
        data["active_project"] = name
    _write_cfg(data)


def load_active_project() -> str | None:
    return _read_cfg().get("active_project") or None


# ── Chat directory resolution ─────────────────────────────────────────────────

def get_chats_dir(project_name: str | None = None) -> Path:
    """Return the directory that holds chats for the given project (or universal)."""
    if project_name:
        projects = load_projects()
        p = projects.get(project_name)
        if p:
            return Path(p["chats_dir"])
    cfg = load_chat_history_config()
    return Path(cfg["universal_dir"]).expanduser()


# ── Saving / loading / renaming chats ────────────────────────────────────────

def save_chat(
    session_id: str,
    messages: list[dict[str, Any]],
    model_label: str,
    chat_name: str | None = None,
    project_name: str | None = None,
) -> str | None:
    """Persist chat messages. Returns the filename stem used, or None if empty."""
    user_msgs = [m for m in messages if m["role"] == "user"]
    if not user_msgs:
        return chat_name  # nothing to save yet

    if not chat_name:
        first_content = user_msgs[0].get("content", "")
        if isinstance(first_content, list):
            first_content = next(
                (
                    p.get("text", "")
                    for p in first_content
                    if isinstance(p, dict) and p.get("type") == "text"
                ),
                "",
            )
        chat_name = _safe_filename(str(first_content))

    chats_dir = get_chats_dir(project_name)
    chats_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "id": session_id,
        "name": chat_name,
        "created": datetime.now().isoformat(),
        "model": model_label,
        "messages": [m for m in messages if m["role"] != "system"],
    }
    (chats_dir / f"{chat_name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return chat_name


def list_chats(project_name: str | None = None) -> list[dict[str, Any]]:
    """Return up to 200 chats sorted newest-first."""
    chats_dir = get_chats_dir(project_name)
    if not chats_dir.is_dir():
        return []
    result = []
    for f in sorted(chats_dir.glob("*.json"), reverse=True)[:200]:
        try:
            result.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return result


def load_chat_by_name(name: str, project_name: str | None = None) -> dict[str, Any] | None:
    """Load a chat by exact filename stem, then falls back to substring search."""
    chats_dir = get_chats_dir(project_name)
    if not chats_dir.is_dir():
        return None
    exact = chats_dir / f"{name}.json"
    if exact.is_file():
        try:
            return json.loads(exact.read_text(encoding="utf-8"))
        except Exception:
            return None
    name_lower = name.lower()
    for f in sorted(chats_dir.glob("*.json"), reverse=True):
        if name_lower in f.stem.lower():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def rename_chat_file(
    old_name: str, new_name: str, project_name: str | None = None
) -> tuple[bool, str]:
    """Rename a chat file. Returns (success, new_safe_name)."""
    chats_dir = get_chats_dir(project_name)
    if not chats_dir.is_dir():
        return False, ""
    old_path = chats_dir / f"{old_name}.json"
    if not old_path.is_file():
        name_lower = old_name.lower()
        for f in chats_dir.glob("*.json"):
            if name_lower in f.stem.lower():
                old_path = f
                break
        else:
            return False, ""
    safe_new = _safe_filename(new_name)
    new_path = chats_dir / f"{safe_new}.json"
    try:
        data = json.loads(old_path.read_text(encoding="utf-8"))
        data["name"] = safe_new
        new_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        old_path.unlink()
        return True, safe_new
    except Exception:
        return False, ""


# ── Project selector UI ───────────────────────────────────────────────────────

def project_selector(current_name: str | None = None) -> str | None:
    """Interactive project picker. Returns chosen project name or None to cancel."""
    projects = load_projects()
    if not projects:
        return None

    items = list(projects.values())

    def _draw(idx: int) -> None:
        print("\033[H\033[J", end="", flush=True)
        print("\n  \033[1mSelect a project\033[0m  \033[2m(↑↓ move · Enter select · Esc cancel)\033[0m\n")
        for i, p in enumerate(items):
            cursor = "\033[36m▶\033[0m" if i == idx else " "
            active = "  \033[33m← active\033[0m" if p["name"] == current_name else ""
            path_note = f"  \033[2m{p['path']}\033[0m"
            print(f"  {cursor} {p['display']}{path_note}{active}")
        print()

    try:
        import msvcrt
        sel = 0
        _draw(sel)
        while True:
            key = msvcrt.getch()
            if key == b"\xe0":
                key2 = msvcrt.getch()
                if key2 == b"H":
                    sel = (sel - 1) % len(items)
                    _draw(sel)
                elif key2 == b"P":
                    sel = (sel + 1) % len(items)
                    _draw(sel)
            elif key == b"\r":
                print("\033[H\033[J", end="", flush=True)
                return items[sel]["name"]
            elif key == b"\x1b":
                print("\033[H\033[J", end="", flush=True)
                return None
    except ImportError:
        pass

    # Unix fallback: use tty raw mode for arrow keys
    try:
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        sel = 0
        _draw(sel)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        if ch3 == "A":   # up
                            sel = (sel - 1) % len(items)
                        elif ch3 == "B": # down
                            sel = (sel + 1) % len(items)
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                        _draw(sel)
                        tty.setraw(fd)
                    else:  # bare Esc
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                        print("\033[H\033[J", end="", flush=True)
                        return None
                elif ch in ("\r", "\n"):
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    print("\033[H\033[J", end="", flush=True)
                    return items[sel]["name"]
                elif ch == "\x03":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    print("\033[H\033[J", end="", flush=True)
                    return None
        except Exception:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        pass

    # Numbered fallback
    print("\n  Projects:\n")
    for i, p in enumerate(items):
        active = " (active)" if p["name"] == current_name else ""
        print(f"  {i + 1}. {p['display']}  ({p['name']}){active}")
    print()
    while True:
        try:
            raw = input("  Enter number (or press Enter to cancel): ").strip()
            if not raw:
                return None
            n = int(raw) - 1
            if 0 <= n < len(items):
                return items[n]["name"]
        except (ValueError, EOFError):
            pass
        print(f"  Please enter 1–{len(items)}")


# ── Chat selector UI ──────────────────────────────────────────────────────────

def chat_selector(project_name: str | None = None) -> str | None:
    """Interactive numbered list for picking a chat. Returns chat name or None."""
    chats = list_chats(project_name)
    if not chats:
        return None

    print(f"\n  Chats {'(project: ' + project_name + ')' if project_name else '(universal)'}:\n")
    for i, c in enumerate(chats):
        ts = c.get("created", "")[:10]
        msgs = len(c.get("messages", []))
        print(f"  {i + 1}. {c['name']}  \033[2m{ts}  {msgs} msgs\033[0m")
    print()

    while True:
        try:
            raw = input("  Enter number (or press Enter to cancel): ").strip()
            if not raw:
                return None
            n = int(raw) - 1
            if 0 <= n < len(chats):
                return chats[n]["name"]
        except (ValueError, EOFError):
            pass
        print(f"  Please enter 1–{len(chats)}")
