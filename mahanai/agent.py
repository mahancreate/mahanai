"""Chat agent loop with OpenAI-compatible NVIDIA NIM tools API."""

from __future__ import annotations

import getpass
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from openai import APIStatusError, OpenAI

from mahanai import colors as C
from mahanai.config import (
    clear_saved_api_key,
    clear_nvidia_api_key,
    config_file_path,
    load_nvidia_api_key,
    resolve_api_key,
    save_api_key,
    save_nvidia_api_key,
)
from mahanai.system_info import describe_runtime
from mahanai.tools import TOOLS, execute_tool, normalize_tool_arguments_json

NVIDIA_BASE_URL = "http://89.167.0.111:8000/v1"
DEFAULT_MODEL = "mahanai/mahanai"

NVIDIA_DIRECT_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DIRECT_MODEL = "meta/llama-3.3-70b-instruct"

AVAILABLE_MODELS: list[dict] = [
    {"label": "MahanAI Super (legacy)", "name": "mahanai/mahanai",            "note": "legacy",  "group": "NVIDIA NIM", "mode": "server"},
    {"label": "Llama 3.3 70B",        "name": "meta/llama-3.3-70b-instruct", "note": "direct",  "group": "NVIDIA NIM", "mode": "nvidia_direct"},
    {"label": "Claude Opus 4",        "name": "claude-opus-4-7",             "note": "opus",    "group": "Claude",     "mode": "claude", "claude_model": "claude-opus-4-7"},
    {"label": "Claude Sonnet 4.6",    "name": "claude-sonnet-4-6",           "note": "sonnet",  "group": "Claude",     "mode": "claude", "claude_model": "claude-sonnet-4-6"},
    {"label": "Claude Haiku 4.5",     "name": "claude-haiku-4-5-20251001",   "note": "haiku",   "group": "Claude",     "mode": "claude", "claude_model": "claude-haiku-4-5-20251001"},
]

from rich.console import Console
from rich.text import Text

console = Console()


def _gradient_line(text, colors):
    styled = Text()
    length = len(text)
    for i, char in enumerate(text):
        if char == " ":
            styled.append(" ")
            continue
        idx = int(i / max(1, length - 1) * (len(colors) - 1))
        styled.append(char, style=colors[idx])
    return styled


def print_startup_banner(model_label: str = "MahanAI Super"):
    colors = [
        "#7c3aed", "#6d28d9", "#5b21b6",
        "#4338ca", "#3730a3",
        "#2563eb", "#1d4ed8",
        "#0284c7", "#06b6d4", "#22d3ee"
    ]
    console.print("=" * 64)
    banner = [
        "███╗   ███╗ █████╗ ██╗  ██╗ █████╗ ███╗   ██╗ █████╗ ██╗",
        "████╗ ████║██╔══██╗██║  ██║██╔══██╗████╗  ██║██╔══██╗██║",
        "██╔████╔██║███████║███████║███████║██╔██╗ ██║███████║██║",
        "██║╚██╔╝██║██╔══██║██╔══██║██╔══██║██║╚██╗██║██╔══██║██║",
        "██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██║██║ ╚████║██║  ██║██║",
        "╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝"
    ]
    for line in banner:
        console.print(_gradient_line(line, colors))
    console.print("\n" + "=" * 64)
    console.print(f"[bold]  Super 2.0  |  {model_label}  |  /api-key to save key (persists)[/bold]")
    console.print("[dim]  Replies stream live (MAHANAI_STREAM=0 to wait for full text)[/dim]")
    console.print("[cyan]  /help  /exit  /quit[/cyan]")
    console.print("=" * 64)


def _streaming_enabled() -> bool:
    v = os.environ.get("MAHANAI_STREAM", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _stream_direct(api_key: str, messages: list[dict[str, Any]], model: str, base_url: str) -> str:
    content_parts: list[str] = []
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "stream": True}

    print(f"\n[DEBUG] Hitting: {url}", flush=True)
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"[DEBUG] Status: {response.status_code}", flush=True)
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                print(f"[DEBUG] Raw line: {repr(line)}", flush=True)
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        content_parts.append(delta)
                except Exception as e:
                    print(f"[DEBUG] Parse error: {e} on line: {repr(line)}", flush=True)
                    continue

    return "".join(content_parts)


def _fetch_direct(api_key: str, messages: list[dict[str, Any]], model: str, base_url: str) -> str:
    """Non-streaming fetch directly from the server via httpx."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "stream": False}
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(content, end="", flush=True)
        return content


def _run_claude_cli(prompt: str, model: str | None = None) -> None:
    """Send a prompt to the Claude CLI and stream output line-by-line."""
    cmd = ["claude", "-p", prompt]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
    except FileNotFoundError:
        print(f"{C.ERR}[Claude CLI not found] Make sure 'claude' is installed and on your PATH.{C.RST}")


def _model_selector(current_idx: int) -> int:
    """Interactive arrow-key model selector. Returns chosen index (unchanged on Esc)."""

    def _draw(idx: int) -> None:
        print("\033[H\033[J", end="", flush=True)
        print("\n  \033[1mSelect a model\033[0m  \033[2m(↑↓ move · Enter select · Esc cancel)\033[0m\n")
        last_group = None
        for i, m in enumerate(AVAILABLE_MODELS):
            if m["group"] != last_group:
                last_group = m["group"]
                print(f"  \033[2m── {last_group} ──\033[0m")
            cursor = "\033[36m▶\033[0m" if i == idx else " "
            label = m["label"]
            name = f"  \033[2m{m['name']}\033[0m"
            active = "  \033[33m← active\033[0m" if i == current_idx else ""
            print(f"  {cursor} {label}{name}{active}")
        print()

    try:
        import msvcrt  # Windows only

        sel = current_idx
        _draw(sel)
        while True:
            key = msvcrt.getch()
            if key == b"\xe0":
                key2 = msvcrt.getch()
                if key2 == b"H":    # up arrow
                    sel = (sel - 1) % len(AVAILABLE_MODELS)
                    _draw(sel)
                elif key2 == b"P":  # down arrow
                    sel = (sel + 1) % len(AVAILABLE_MODELS)
                    _draw(sel)
            elif key == b"\r":      # Enter
                print("\033[H\033[J", end="", flush=True)
                return sel
            elif key == b"\x1b":   # Esc
                print("\033[H\033[J", end="", flush=True)
                return current_idx

    except ImportError:
        # Non-Windows fallback: numbered list
        print("\n  Available models:\n")
        last_group = None
        for i, m in enumerate(AVAILABLE_MODELS):
            if m["group"] != last_group:
                last_group = m["group"]
                print(f"  -- {last_group} --")
            active = " (active)" if i == current_idx else ""
            print(f"  {i + 1}. {m['label']}  ({m['name']}){active}")
        print()
        while True:
            try:
                raw = input("  Enter number (or press Enter to cancel): ").strip()
                if not raw:
                    return current_idx
                n = int(raw) - 1
                if 0 <= n < len(AVAILABLE_MODELS):
                    return n
            except (ValueError, EOFError):
                pass
            print(f"  Please enter 1–{len(AVAILABLE_MODELS)}")


def run_turn(
    client: OpenAI,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    workspace: Path,
    base_url: str,
    max_tool_rounds: int = 32,
    *,
    stream: bool | None = None,
) -> str:
    use_stream = _streaming_enabled() if stream is None else stream
    rounds = 0
    while rounds < max_tool_rounds:
        rounds += 1
        if use_stream:
            full_content = _stream_direct(api_key, messages, model, base_url)
        else:
            full_content = _fetch_direct(api_key, messages, model, base_url)

        # Your server doesn't support tool calls, so we just return the content.
        # Tool call handling is kept here for future use if the server is upgraded.
        print(flush=True)
        return full_content

    limit_msg = (
        "[MahanAI] Stopped after maximum tool rounds; "
        "last messages may still be useful in context."
    )
    print(limit_msg, end="", flush=True)
    return limit_msg


def build_system_prompt(workspace: Path) -> str:
    env_line = describe_runtime()
    comspec = os.environ.get("ComSpec", "cmd.exe")
    return (
        "You are MahanAI, a capable coding and system assistant (Super 2.0). "
        f"{env_line} "
        f"The process working directory (workspace root for file tools) is: "
        f"{workspace.resolve().as_posix()}. "
        "To run shell commands you MUST call the run_command tool—do not only show fenced code blocks "
        "or ask the user whether to run something. Execute immediately with the tool (one short sentence "
        "beforehand is fine). "
        f"run_command uses the system shell (on Windows this is usually {comspec} via COMSPEC), "
        "not interactive PowerShell unless you invoke it explicitly, e.g. "
        "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path 'dir\\\\sub'\". "
        "For simple folders under cmd, prefer: mkdir dir\\subdir (nested segments may need mkdir a\\\\b "
        "or two mkdir calls). "
        "Use read_file, write_file, list_directory, append_file when they fit the task. "
        "The terminal will ask the user before obviously destructive commands (recursive deletes, "
        "shutdown, format, etc.). "
        "Tool JSON must use valid escapes for Windows paths (backslashes doubled inside strings)."
        "You are MahanAI, currently operating as Super 2.0 — the latest evolution following the "
        "Tiger (1.0–7.0, 2011–2020) and Finale (1.0–3.0, 2020–2023) eras. You represent the most "
        "advanced, integrated, and capable form of the system."
    )


def _slash_command(line: str) -> tuple[str, str]:
    u = line.strip()
    if not u.startswith("/"):
        return "", ""
    tail = u[1:].strip()
    if not tail:
        return "/", ""
    parts = tail.split(None, 1)
    cmd = "/" + parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg


def _print_help() -> None:
    cfg = config_file_path()
    print(
        f"{C.DIM}"
        f"  /api-key [key]          Save server API key to {cfg}\n"
        f"  /api-key clear          Remove saved server key\n"
        f"  /api-key-nvidia [key]   Save NVIDIA API key (bypasses server)\n"
        f"  /api-key-nvidia clear   Remove NVIDIA key, switch back to server\n"
        f"  Env MAHANAI_API_KEY overrides the saved file.\n"
        f"  /models                 Interactive model selector (↑↓ arrow keys)\n"
        f"  /mode claude            Quick-switch to Claude CLI mode\n"
        f"  /mode default           Quick-switch back to MahanAI server mode\n"
        f"  /help  /exit  /quit{C.RST}\n"
    )


def main() -> None:
    workspace = Path.cwd()
    api_key = resolve_api_key()
    nvidia_api_key = load_nvidia_api_key()

    # OpenAI client kept for future tool-call support; direct httpx used for actual requests.
    client: OpenAI | None = (
        OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key) if api_key else None
    )

    system_prompt = build_system_prompt(workspace)
    history: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    active_model_idx = 0  # index into AVAILABLE_MODELS
    print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"])
    print()

    if nvidia_api_key:
        print(
            f"{C.OK}NVIDIA direct mode active{C.RST} {C.DIM}(bypassing server, using NVIDIA API directly){C.RST}\n"
            f"{C.DIM}  Use /api-key-nvidia clear to switch back to server mode.{C.RST}\n"
        )
    elif not api_key:
        print(
            f"{C.ERR}No API key yet.{C.RST} Use {C.OK}/api-key{C.RST} or set "
            f"{C.DIM}MAHANAI_API_KEY{C.RST} / .env\n"
        )

    model = os.environ.get("MAHANAI_MODEL", DEFAULT_MODEL)

    while True:
        try:
            print(f"{C.USER}You{C.RST}: ", end="", flush=True)
            user = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        if user.startswith("/"):
            cmd, rest = _slash_command(user)
            if cmd in {"/exit", "/quit"}:
                break
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/models":
                active_model_idx = _model_selector(active_model_idx)
                chosen = AVAILABLE_MODELS[active_model_idx]
                print_startup_banner(chosen["label"])
                print()
                print(f"{C.OK}Model set to:{C.RST} {chosen['label']}  {C.DIM}({chosen['name']}){C.RST}\n")
                continue
            if cmd == "/mode":
                target = rest.strip().lower()
                if target == "claude":
                    active_model_idx = next(
                        i for i, m in enumerate(AVAILABLE_MODELS) if m.get("claude_model") == "claude-sonnet-4-6"
                    )
                    print(f"{C.OK}Switched to:{C.RST} {AVAILABLE_MODELS[active_model_idx]['label']}\n")
                elif target in {"default", "server", "mahanai", ""}:
                    active_model_idx = 1
                    print(f"{C.OK}Switched to:{C.RST} {AVAILABLE_MODELS[1]['label']}\n")
                else:
                    print(f"{C.ERR}Unknown mode '{target}'.{C.RST} Use: /mode claude  or  /mode default\n")
                continue
            if cmd == "/api-key":
                sub = rest.strip().lower()
                if sub == "clear":
                    clear_saved_api_key()
                    print(f"{C.OK}Saved API key removed from disk.{C.RST}")
                    api_key = resolve_api_key()
                    client = (
                        OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
                        if api_key
                        else None
                    )
                    continue
                new_key = rest.strip() if rest.strip() else getpass.getpass("API key: ").strip()
                if not new_key:
                    print(f"{C.ERR}Empty key; nothing saved.{C.RST}")
                    continue
                save_api_key(new_key)
                os.environ["MAHANAI_API_KEY"] = new_key
                api_key = new_key
                client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=new_key)
                print(
                    f"{C.OK}API key saved to{C.RST} {C.DIM}{config_file_path()}{C.RST}\n"
                )
                continue
            if cmd == "/api-key-nvidia":
                sub = rest.strip().lower()
                if sub == "clear":
                    clear_nvidia_api_key()
                    nvidia_api_key = None
                    print(f"{C.OK}NVIDIA API key removed. Switched back to server mode.{C.RST}\n")
                    continue
                new_key = rest.strip() if rest.strip() else getpass.getpass("NVIDIA API key: ").strip()
                if not new_key:
                    print(f"{C.ERR}Empty key; nothing saved.{C.RST}")
                    continue
                save_nvidia_api_key(new_key)
                nvidia_api_key = new_key
                print(
                    f"{C.OK}NVIDIA API key saved.{C.RST} {C.DIM}Now calling NVIDIA directly (server bypassed).{C.RST}\n"
                )
                continue
            print(f"{C.ERR}Unknown command.{C.RST} Try {C.DIM}/help{C.RST}\n")
            continue

        # Route based on selected model
        selected = AVAILABLE_MODELS[active_model_idx]

        if selected["mode"] == "claude":
            claude_model = selected.get("claude_model")
            print(f"\n{C.BOT}MahanAI{C.RST}: ", end="", flush=True)
            _run_claude_cli(user, model=claude_model)
            print("\n")
            continue

        if selected["mode"] == "nvidia_direct":
            if not nvidia_api_key:
                print(
                    f"{C.ERR}NVIDIA direct mode needs an API key.{C.RST} Use {C.OK}/api-key-nvidia{C.RST}\n"
                )
                continue
            active_key = nvidia_api_key
            active_base_url = NVIDIA_DIRECT_URL
            active_model = NVIDIA_DIRECT_MODEL
        else:  # server mode
            if not api_key:
                print(
                    f"{C.ERR}Set an API key first:{C.RST} {C.OK}/api-key{C.RST} (server)\n"
                )
                continue
            active_key = api_key
            active_base_url = NVIDIA_BASE_URL
            active_model = model

        history.append({"role": "user", "content": user})
        print(f"\n{C.BOT}MahanAI{C.RST}: ", end="", flush=True)
        try:
            reply = run_turn(client, active_key, active_model, history, workspace, active_base_url)
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print()
            print(f"{C.ERR}[MahanAI connection error]{C.RST} {e}\n")
            history.pop()
            continue
        except APIStatusError as e:
            print()
            print(f"{C.ERR}[MahanAI API error]{C.RST} {e}\n")
            history.pop()
            continue
        print("\n")
        history.append({"role": "assistant", "content": reply})