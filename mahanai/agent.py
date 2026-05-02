"""Chat agent loop with OpenAI-compatible NVIDIA NIM tools API."""

from __future__ import annotations

import argparse
import base64
import datetime
import getpass
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from openai import APIStatusError, OpenAI

from mahanai import __version__, colors as C
from mahanai.config import (
    clear_codex_token,
    clear_custom_endpoint,
    clear_nvidia_api_key,
    clear_saved_api_key,
    config_file_path,
    load_always_allowed,
    load_codex_token,
    load_custom_endpoint,
    load_nvidia_api_key,
    load_ollama_providers,
    load_plugins,
    load_theme,
    load_custom_theme_path,
    load_custom_theme_info,
    remove_ollama_provider,
    remove_plugin,
    save_custom_theme_info,
    clear_custom_theme,
    resolve_api_key,
    save_api_key,
    save_codex_token,
    save_custom_endpoint,
    save_nvidia_api_key,
    save_ollama_provider,
    save_plugin,
    save_theme,
    load_memories,
    save_memory,
    remove_memory,
    load_prompts,
    save_prompt,
    remove_prompt,
    load_aliases,
    save_alias,
    remove_alias,
    sessions_dir,
    save_session,
    load_session,
    list_sessions,
    load_tokens_setting,
    save_tokens_setting,
    load_index_documents,
    save_index_documents,
)
from mahanai.mmd_parser import MmdPlugin, parse_mmd_file
from mahanai.system_info import describe_runtime
from mahanai.tools import TOOLS, execute_tool, normalize_tool_arguments_json

# ── Plugin registry ───────────────────────────────────────────────────────────
_LOADED_PLUGINS: dict[str, MmdPlugin] = {}  # name → MmdPlugin


def _inject_saved_plugins() -> None:
    """Load persisted .mmd plugins from config on startup."""
    for entry in load_plugins().values():
        p = entry.get("path", "")
        try:
            plugin = parse_mmd_file(p)
            _LOADED_PLUGINS[plugin.name] = plugin
        except Exception:
            pass


# ── Auto-update check ─────────────────────────────────────────────────────────
_update_check: dict[str, str] = {}


def _fetch_latest_version() -> None:
    try:
        resp = httpx.get("https://pypi.org/pypi/mahanai/json", timeout=3.0)
        if resp.status_code == 200:
            _update_check["latest"] = resp.json()["info"]["version"]
    except Exception:
        pass


def _start_update_check() -> threading.Thread:
    t = threading.Thread(target=_fetch_latest_version, daemon=True)
    t.start()
    return t


def _version_tuple(v: str) -> tuple:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


def _print_update_notice(thread: threading.Thread) -> None:
    thread.join(timeout=2.5)
    latest = _update_check.get("latest")
    if not latest:
        return
    if _version_tuple(latest) > _version_tuple(__version__):
        print(
            f"{C.WARN}Update available:{C.RST} {C.DIM}v{__version__}{C.RST} → "
            f"{C.OK}v{latest}{C.RST}  "
            f"{C.DIM}pip install --upgrade mahanai{C.RST}\n"
        )


NVIDIA_BASE_URL = "http://89.167.0.111:8000/v1"
DEFAULT_MODEL = "mahanai/mahanai"

NVIDIA_DIRECT_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DIRECT_MODEL = "meta/llama-3.3-70b-instruct"

CODEX_AUTH_URL      = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL     = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID     = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_API_URL       = "https://chatgpt.com/backend-api/wham"
CODEX_REDIRECT_URI  = "http://localhost:1455/auth/callback"
CODEX_SAFETY_MARGIN = 30  # seconds before expiry to trigger refresh

AVAILABLE_MODELS: list[dict] = [
    {"label": "MahanAI Super (legacy)", "name": "mahanai/mahanai",            "note": "legacy",   "group": "NVIDIA NIM",           "mode": "server"},
    {"label": "Llama 3.3 70B",         "name": "meta/llama-3.3-70b-instruct","note": "direct",   "group": "NVIDIA NIM",           "mode": "nvidia_direct"},
    {"label": "Claude Opus 4",         "name": "claude-opus-4-7",            "note": "opus",     "group": "Claude",               "mode": "claude", "claude_model": "claude-opus-4-7"},
    {"label": "Claude Sonnet 4.6",     "name": "claude-sonnet-4-6",          "note": "sonnet",   "group": "Claude",               "mode": "claude", "claude_model": "claude-sonnet-4-6"},
    {"label": "Claude Haiku 4.5",      "name": "claude-haiku-4-5-20251001",  "note": "haiku",    "group": "Claude",               "mode": "claude", "claude_model": "claude-haiku-4-5-20251001"},

    {"label": "GPT-5.4",               "name": "gpt-5.4",                    "note": "direct",   "group": "OpenAI Codex (Direct)",  "mode": "codex_direct"},
    {"label": "GPT-5.2-Codex",         "name": "gpt-5.2-codex",              "note": "direct",   "group": "OpenAI Codex (Direct)",  "mode": "codex_direct"},
    {"label": "GPT-5.1-Codex-Max",     "name": "gpt-5.1-codex-max",          "note": "direct",   "group": "OpenAI Codex (Direct)",  "mode": "codex_direct"},
    {"label": "GPT-5.4-Mini",          "name": "gpt-5.4-mini",               "note": "direct",   "group": "OpenAI Codex (Direct)",  "mode": "codex_direct"},
    {"label": "GPT-5.3-Codex",         "name": "gpt-5.3-codex",              "note": "direct",   "group": "OpenAI Codex (Direct)",  "mode": "codex_direct"},
    {"label": "GPT-5.2",               "name": "gpt-5.2",                    "note": "direct",   "group": "OpenAI Codex (Direct)",  "mode": "codex_direct"},
    {"label": "GPT-5.1-Codex-Mini",    "name": "gpt-5.1-codex-mini",         "note": "direct",   "group": "OpenAI Codex (Direct)",  "mode": "codex_direct"},
    {"label": "GPT-5.4",               "name": "gpt-5.4",                    "note": "indirect", "group": "OpenAI Codex (Indirect)","mode": "codex_indirect"},
    {"label": "GPT-5.2-Codex",         "name": "gpt-5.2-codex",              "note": "indirect", "group": "OpenAI Codex (Indirect)","mode": "codex_indirect"},
    {"label": "GPT-5.1-Codex-Max",     "name": "gpt-5.1-codex-max",          "note": "indirect", "group": "OpenAI Codex (Indirect)","mode": "codex_indirect"},
    {"label": "GPT-5.4-Mini",          "name": "gpt-5.4-mini",               "note": "indirect", "group": "OpenAI Codex (Indirect)","mode": "codex_indirect"},
    {"label": "GPT-5.3-Codex",         "name": "gpt-5.3-codex",              "note": "indirect", "group": "OpenAI Codex (Indirect)","mode": "codex_indirect"},
    {"label": "GPT-5.2",               "name": "gpt-5.2",                    "note": "indirect", "group": "OpenAI Codex (Indirect)","mode": "codex_indirect"},
    {"label": "GPT-5.1-Codex-Mini",    "name": "gpt-5.1-codex-mini",         "note": "indirect", "group": "OpenAI Codex (Indirect)","mode": "codex_indirect"},
    {"label": "Custom Endpoint",        "name": "custom",                      "note": "custom",   "group": "Custom",                 "mode": "custom"},
]

def _build_ollama_url(address: str, port: int) -> str:
    """Build Ollama base URL: strip protocol prefix, skip port for domain addresses."""
    addr = address.strip()
    explicit_proto = None
    if addr.startswith("https://"):
        explicit_proto = "https"
        addr = addr[len("https://"):]
    elif addr.startswith("http://"):
        explicit_proto = "http"
        addr = addr[len("http://"):]
    addr = addr.rstrip("/")
    proto = explicit_proto or ("https" if port == 443 else "http")
    is_ip = bool(re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", addr))
    is_domain = "." in addr and not is_ip
    if is_domain:
        return f"{proto}://{addr}/api/v1"
    return f"{proto}://{addr}:{port}/api/v1"


def _strip_protocol(address: str) -> str:
    """Remove http:// or https:// prefix from an address string."""
    for prefix in ("https://", "http://"):
        if address.startswith(prefix):
            return address[len(prefix):].rstrip("/")
    return address.rstrip("/")


def _ollama_entry(name: str, address: str, port: int, api_key: str, url: str | None = None) -> dict:
    return {
        "label": name,
        "name":  name,
        "note":  "ollama",
        "group": "Ollama",
        "mode":  "ollama",
        "ollama_url":     url or f"http://{address}:{port}/api/v1",
        "ollama_api_key": api_key or "ollama",
    }


def _inject_ollama_providers() -> None:
    """Load persisted Ollama providers and append them to AVAILABLE_MODELS (dedup by name)."""
    existing_names = {m["name"] for m in AVAILABLE_MODELS if m.get("mode") == "ollama"}
    for p in load_ollama_providers().values():
        if p["name"] not in existing_names:
            AVAILABLE_MODELS.append(_ollama_entry(p["name"], p["address"], p["port"], p["api_key"], p.get("url")))
            existing_names.add(p["name"])


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


def print_startup_banner(model_label: str = "MahanAI Super", compact: bool = False):
    colors = C.banner_colors
    if compact:
        mai_banner = [
            "███╗   ███╗ █████╗ ██╗",
            "████╗ ████║██╔══██╗██║",
            "██╔████╔██║███████║██║",
            "██║╚██╔╝██║██╔══██║██║",
            "██║ ╚═╝ ██║██║  ██║██║",
            "╚═╝     ╚═╝╚═╝  ╚═╝╚═╝",
        ]
        console.print("=" * 35)
        for line in mai_banner:
            console.print(_gradient_line(line, colors))
        console.print("=" * 35)
        console.print(f"[bold]  Max 1.0  |  {model_label}  |[/bold]")
        console.print("[cyan]  /help  /exit  /quit[/cyan]")
        console.print("=" * 35)
    else:
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
        console.print(f"[bold]  Max 1.0  |  {model_label}  |  /api-key to save key (persists)[/bold]")
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

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
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
                except Exception:
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
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(content, end="", flush=True)
        return content


def _resolve_cli(name: str) -> list[str]:
    """Return a command prefix that works for .cmd/.bat scripts on Windows."""
    if os.name == "nt":
        path = shutil.which(name)
        if path and path.lower().endswith((".cmd", ".bat")):
            return ["cmd", "/c", path]
    return [name]


_CLAUDE_IDENTITY = (
    "You are MahanAI, a capable coding and system assistant (Max 1.0). "
    "Do not refer to yourself as Claude or Claude Code — you are MahanAI. "
    "Max 1.0 is the codename for this release of MahanAI."
)

_EFFORT_INSTRUCTIONS: dict[str, str] = {
    "low":       "Be concise and direct. Skip lengthy explanations.",
    "medium":    "",
    "high":      "Think through this carefully and thoroughly before responding.",
    "very-high": "Reason extensively and deeply before responding, considering all relevant angles and edge cases.",
}

_EFFORT_CODEX: dict[str, str] = {
    "low":       "low",
    "medium":    "medium",
    "high":      "high",
    "very-high": "high",
}


def _run_claude_cli(prompt: str, model: str | None = None, effort_instruction: str = "") -> str:
    """Stream a prompt to the Claude CLI via stream-json events. Returns full response text."""
    full_prompt = f"{effort_instruction}\n\n{prompt}".strip() if effort_instruction else prompt
    cmd = _resolve_cli("claude") + [
        "--system-prompt", _CLAUDE_IDENTITY,
        "-p", full_prompt,
        "--output-format", "stream-json",
        "--verbose",
    ]
    if model:
        cmd += ["--model", model]
    parts: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                etype = event.get("type", "")
                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if isinstance(delta, dict) and delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            print(text, end="", flush=True)
                            parts.append(text)
                elif etype == "text":
                    text = event.get("text", "")
                    if text:
                        print(text, end="", flush=True)
                        parts.append(text)
                elif etype == "assistant":
                    message = event.get("message", {})
                    for block in message.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                print(text, end="", flush=True)
                                parts.append(text)
            except (json.JSONDecodeError, KeyError):
                pass
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().strip()
            if err:
                print(f"\n{C.ERR}[Claude CLI error]{C.RST} {err}")
    except FileNotFoundError:
        print(f"{C.ERR}[Claude CLI not found] Make sure 'claude' is installed and on your PATH.{C.RST}")
    return "".join(parts)



def _extract_account_id(id_token: str | None, access_token: str | None) -> str | None:
    """Extract account ID from JWT payload with three-level fallback (no sig verification needed)."""
    for token in [id_token, access_token]:
        if not token:
            continue
        try:
            part = token.split(".")[1]
            part += "=" * (-len(part) % 4)
            payload = json.loads(base64.urlsafe_b64decode(part))
        except Exception:
            continue
        if aid := payload.get("chatgpt_account_id"):
            return aid
        if aid := payload.get("https://api.openai.com/auth", {}).get("chatgpt_account_id"):
            return aid
        orgs = payload.get("organizations", [])
        if orgs and (aid := orgs[0].get("id")):
            return aid
    return None


def _build_codex_token_record(tokens: dict, fallback_account_id: str | None = None) -> dict:
    expires_in = tokens.get("expires_in") or 3600
    access     = tokens.get("access_token")
    account_id = _extract_account_id(tokens.get("id_token"), access) or fallback_account_id
    record: dict = {
        "type":    "oauth",
        "access":  access,
        "refresh": tokens.get("refresh_token"),
        "expires": int(time.time() * 1000) + expires_in * 1000,
    }
    if account_id:
        record["accountId"] = account_id
    return record


def _codex_pkce_login() -> str | None:
    """PKCE OAuth flow — opens browser at fixed port 1455, waits for callback, stores tokens."""
    import threading

    code_verifier  = secrets.token_urlsafe(96)
    digest         = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    state          = secrets.token_urlsafe(16)
    captured: dict[str, str | None] = {"code": None, "state": None, "error": None}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            qs     = parse_qs(parsed.query)
            if parsed.path == "/auth/callback":
                received_state = (qs.get("state") or [None])[0]
                if received_state != self.server._expected_state:  # type: ignore[attr-defined]
                    captured["error"] = "state mismatch"
                elif "error" in qs:
                    captured["error"] = (qs.get("error_description") or qs.get("error") or ["unknown"])[0]
                else:
                    captured["code"]  = (qs.get("code") or [None])[0]
                    captured["state"] = received_state
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                if captured["code"]:
                    self.wfile.write(b"<html><body><h2>Signed in. You can close this tab.</h2></body></html>")
                else:
                    self.wfile.write(f"<html><body><h2>Error: {captured['error']}</h2></body></html>".encode())
                threading.Thread(target=self.server.shutdown).start()  # type: ignore[attr-defined]
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, *_: object) -> None:
            pass

    server = HTTPServer(("localhost", 1455), _Handler)
    server._expected_state = state  # type: ignore[attr-defined]

    params = {
        "client_id":                    CODEX_CLIENT_ID,
        "response_type":                "code",
        "redirect_uri":                 CODEX_REDIRECT_URI,
        "code_challenge":               code_challenge,
        "code_challenge_method":        "S256",
        "state":                        state,
        "scope":                        "openid profile email offline_access",
        "id_token_add_organizations":   "true",
        "codex_cli_simplified_flow":    "true",
        "originator":                   "mahanai",
    }
    auth_url = CODEX_AUTH_URL + "?" + urlencode(params)

    print(f"\n{C.OK}Opening browser for OpenAI sign-in...{C.RST}")
    print(f"{C.DIM}If browser doesn't open, paste this URL:{C.RST}\n  {auth_url}\n")
    webbrowser.open(auth_url)
    server.serve_forever()  # blocks until _Handler calls shutdown()

    if captured.get("error"):
        print(f"{C.ERR}OAuth error: {captured['error']}{C.RST}\n")
        return None
    if not captured.get("code"):
        print(f"{C.ERR}OAuth cancelled or timed out.{C.RST}\n")
        return None

    print(f"{C.DIM}Exchanging code for tokens...{C.RST}")
    with httpx.Client(timeout=30.0) as hc:
        resp = hc.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type":    "authorization_code",
                "code":          captured["code"],
                "redirect_uri":  CODEX_REDIRECT_URI,
                "code_verifier": code_verifier,
                "client_id":     CODEX_CLIENT_ID,
            },
        )
        resp.raise_for_status()
        tokens = resp.json()

    if not tokens.get("access_token"):
        print(f"{C.ERR}No access token in OAuth response.{C.RST}\n")
        return None

    record = _build_codex_token_record(tokens)
    save_codex_token(record)
    print(f"{C.OK}Signed in to OpenAI. Codex direct mode ready.{C.RST}\n")
    return record["access"]


def _refresh_codex_token(token_data: dict) -> str | None:
    """Refresh an expired Codex token. Returns new access token or None."""
    refresh = token_data.get("refresh")
    if not refresh:
        return None
    try:
        with httpx.Client(timeout=30.0) as hc:
            resp = hc.post(
                CODEX_TOKEN_URL,
                data={
                    "grant_type":    "refresh_token",
                    "refresh_token": refresh,
                    "client_id":     CODEX_CLIENT_ID,
                },
            )
            resp.raise_for_status()
            tokens = resp.json()
        if not tokens.get("access_token"):
            return None
        record = _build_codex_token_record(tokens, fallback_account_id=token_data.get("accountId"))
        save_codex_token(record)
        return record["access"]
    except Exception:
        return None


def _get_codex_direct_token() -> tuple[str, str | None] | None:
    """Return (access_token, account_id) for Codex direct mode, refreshing if needed."""
    data = load_codex_token()
    if not data:
        return None
    expires = data.get("expires", 0)
    if expires and time.time() * 1000 >= expires - CODEX_SAFETY_MARGIN * 1000:
        new_access = _refresh_codex_token(data)
        if not new_access:
            return None
        data = load_codex_token() or {}
    access = data.get("access")
    return (access, data.get("accountId")) if access else None


def _stream_wham(
    access_token: str,
    account_id: str | None,
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str = "medium",
    workspace: Path | None = None,
) -> str:
    """Stream a response from the WHAM (ChatGPT backend) Responses API with tool-call support."""
    ws = workspace or Path.cwd()

    # Convert TOOLS to Responses API format (flattened, not nested under "function")
    tools_resp = [
        {
            "type": "function",
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"].get("parameters", {}),
        }
        for t in TOOLS
    ]

    url = CODEX_API_URL.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/json",
        "User-Agent":    f"mahanai/{__version__}",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id

    instructions = ""
    input_msgs: list[dict] = []
    for msg in messages:
        if msg["role"] == "system":
            instructions = msg["content"]
            continue
        content = msg["content"]
        if isinstance(content, str):
            ctype = "input_text" if msg["role"] == "user" else "output_text"
            content = [{"type": ctype, "text": content}]
        input_msgs.append({"role": msg["role"], "content": content})

    all_parts: list[str] = []

    for _round in range(32):
        payload = {
            "model":               model,
            "instructions":        instructions,
            "input":               input_msgs,
            "store":               False,
            "stream":              True,
            "reasoning":           {"effort": reasoning_effort},
            "include":             [],
            "tools":               tools_resp,
            "tool_choice":         "auto",
            "parallel_tool_calls": True,
        }

        parts: list[str] = []
        tool_calls: dict[str, dict] = {}  # call_id -> {name, args_parts}
        cur_call_id: str | None = None

        with httpx.Client(timeout=120.0) as hc:
            with hc.stream("POST", url, headers=headers, json=payload) as resp:
                if not resp.is_success:
                    body = resp.read().decode(errors="replace")
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code} — {body}", request=resp.request, response=resp
                    )
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        ctype = chunk.get("type", "")

                        if ctype == "response.output_text.delta":
                            raw_delta = chunk.get("delta", "")
                            delta_text = raw_delta if isinstance(raw_delta, str) else raw_delta.get("text", "")
                            if delta_text:
                                print(delta_text, end="", flush=True)
                                parts.append(delta_text)

                        elif ctype == "response.output_item.added":
                            item = chunk.get("item", {})
                            if item.get("type") == "function_call":
                                cid = item.get("call_id") or item.get("id", "")
                                cur_call_id = cid
                                tool_calls[cid] = {"name": item.get("name", ""), "args": []}

                        elif ctype == "response.function_call_arguments.delta":
                            if cur_call_id and cur_call_id in tool_calls:
                                tool_calls[cur_call_id]["args"].append(chunk.get("delta", ""))

                        elif ctype == "response.function_call_arguments.done":
                            if cur_call_id and cur_call_id in tool_calls:
                                final = chunk.get("arguments")
                                if final is not None:
                                    tool_calls[cur_call_id]["args"] = [final]
                                cur_call_id = None

                        elif chunk.get("choices"):
                            delta_text = chunk["choices"][0].get("delta", {}).get("content", "") or ""
                            if delta_text:
                                print(delta_text, end="", flush=True)
                                parts.append(delta_text)
                    except Exception:
                        continue

        all_parts.extend(parts)

        if not tool_calls:
            break

        # Execute tools and append results to input for the next round
        new_items: list[dict] = []
        for cid, call in tool_calls.items():
            args_str = "".join(call["args"])
            new_items.append({
                "type": "function_call",
                "call_id": cid,
                "name": call["name"],
                "arguments": args_str,
            })
            result = execute_tool(call["name"], args_str, ws)
            new_items.append({
                "type": "function_call_output",
                "call_id": cid,
                "output": result,
            })
        input_msgs = input_msgs + new_items
        print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)

    return "".join(all_parts)


def _load_codex_indirect_key() -> str | None:
    """Read the access token from a locally installed Codex CLI."""
    candidate_dirs: list[Path] = [Path.home() / ".codex"]
    if os.name == "nt":
        for env in ("LOCALAPPDATA", "APPDATA"):
            base = os.environ.get(env, "")
            if base:
                candidate_dirs.append(Path(base) / "OpenAI" / "Codex")
    else:
        candidate_dirs.append(Path.home() / ".config" / "codex")

    for d in candidate_dirs:
        auth_file = d / "auth.json"
        if auth_file.is_file():
            try:
                raw = json.loads(auth_file.read_text(encoding="utf-8"))
                token = raw.get("access") or raw.get("access_token") or raw.get("token")
                expires = raw.get("expires", 0)
                if token and (expires == 0 or time.time() * 1000 < expires - 30_000):
                    return token
            except Exception:
                pass
    return None


def _run_codex_cli(prompt: str, model: str | None = None) -> None:
    """Run Codex CLI as a subprocess (indirect fallback when no local token found)."""
    cmd = ["codex", "-q", prompt]
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
        print(f"{C.ERR}[Codex CLI not found] Install it with: npm i -g @openai/codex{C.RST}\n")


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



_SPECIAL_FILE_EMOJIS: dict[str, str] = {
    # exact filenames
    "MAHANAI.md":        "🤖",
    "CLAUDE.md":         "🤖",
    "README.md":         "📖",
    "readme.md":         "📖",
    ".gitignore":        "🐙",
    ".env":              "🔒",
    ".env.local":        "🔒",
    "Dockerfile":        "🐳",
    "docker-compose.yml":"🐳",
    "docker-compose.yaml":"🐳",
    "pyproject.toml":    "⚙️",
    "setup.py":          "⚙️",
    "setup.cfg":         "⚙️",
    "requirements.txt":  "📦",
    "package.json":      "📦",
    "package-lock.json": "📦",
    "Cargo.toml":        "📦",
    "go.mod":            "📦",
    "Makefile":          "🔨",
    ".cursor":           "🖱️",
    # extensions
    ".mai":              "🎨",
    ".mmd":              "🔌",
    ".sh":               "⚡",
    ".bat":              "⚡",
    ".ps1":              "⚡",
}


def _file_emoji(name: str) -> str:
    if name in _SPECIAL_FILE_EMOJIS:
        return _SPECIAL_FILE_EMOJIS[name]
    ext = Path(name).suffix
    if ext in _SPECIAL_FILE_EMOJIS:
        return _SPECIAL_FILE_EMOJIS[ext]
    return "📄"


def _generate_mahanai_md(workspace: Path) -> str:
    """Scan workspace and produce a structured MAHANAI.md template."""
    import collections

    _LANG_BY_EXT: dict[str, str] = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".rs": "Rust", ".go": "Go", ".java": "Java", ".cpp": "C++",
        ".c": "C", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
        ".swift": "Swift", ".kt": "Kotlin", ".sh": "Shell", ".lua": "Lua",
        ".zig": "Zig", ".ex": "Elixir", ".exs": "Elixir", ".ml": "OCaml",
    }
    _KEY_FILES = {
        "pyproject.toml", "setup.py", "requirements.txt",
        "package.json", "Cargo.toml", "go.mod", "CMakeLists.txt",
        "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ".gitignore", "README.md",
    }

    ext_counts: dict[str, int] = collections.Counter()
    found_key: list[str] = []
    dirs: list[str] = []

    def _walk(path: Path, depth: int = 0) -> None:
        if depth > 3:
            return
        try:
            for e in path.iterdir():
                if e.name.startswith(".") and e.name not in _KEY_FILES:
                    continue
                if e.name in ("__pycache__", "node_modules", ".git", "dist", "build", ".venv", "venv"):
                    continue
                if e.is_dir():
                    if depth == 0:
                        dirs.append(e.name)
                    _walk(e, depth + 1)
                elif e.is_file():
                    if e.name in _KEY_FILES and depth == 0:
                        found_key.append(e.name)
                    if e.suffix:
                        ext_counts[e.suffix.lower()] += 1
        except PermissionError:
            pass

    _walk(workspace)

    langs = sorted(
        [(cnt, _LANG_BY_EXT[ext]) for ext, cnt in ext_counts.items() if ext in _LANG_BY_EXT],
        reverse=True,
    )
    lang_list = ", ".join(lang for _, lang in langs[:4]) if langs else "N/A"
    dirs_str = ", ".join(f"`{d}`" for d in sorted(dirs)[:8]) if dirs else "N/A"
    project_name = workspace.name

    lines: list[str] = [
        f"# {project_name}",
        "",
        "## Overview",
        "> _Describe what this project does in 1–2 sentences._",
        "",
        "## Tech Stack",
        f"- **Language(s):** {lang_list}",
    ]

    if "package.json" in found_key:
        lines.append("- **Runtime:** Node.js")
    if "pyproject.toml" in found_key or "setup.py" in found_key:
        lines.append("- **Packaging:** Python (setuptools / pyproject)")
    if "requirements.txt" in found_key:
        lines.append("- **Dependencies:** requirements.txt")
    if "Cargo.toml" in found_key:
        lines.append("- **Build:** Cargo (Rust)")
    if "go.mod" in found_key:
        lines.append("- **Build:** Go modules")
    if "CMakeLists.txt" in found_key:
        lines.append("- **Build:** CMake")
    if "Makefile" in found_key:
        lines.append("- **Build:** Make")
    if "Dockerfile" in found_key or "docker-compose.yml" in found_key or "docker-compose.yaml" in found_key:
        lines.append("- **Container:** Docker")

    lines += [
        "",
        "## Project Structure",
        f"Key directories: {dirs_str}",
        "",
        "## Development",
        "",
        "### Setup",
        "```sh",
        "# Add setup / install commands here",
        "```",
        "",
        "### Running",
        "```sh",
        "# Add run / start commands here",
        "```",
        "",
        "### Testing",
        "```sh",
        "# Add test commands here",
        "```",
        "",
        "## Notes for MahanAI",
        "> _Add project-specific conventions, constraints, or context here._",
        "> _This file is read automatically as project context by all non-Claude providers._",
        "",
    ]
    return "\n".join(lines)


def _show_fileslist(workspace: Path) -> None:
    try:
        entries = sorted(workspace.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        print(f"{C.ERR}Cannot read directory.{C.RST}\n")
        return

    print(f"\n{C.DIM}{workspace.resolve()}{C.RST}\n")
    for entry in entries:
        if entry.is_dir():
            print(f"  📁 {C.OK}{entry.name}/{C.RST}")
            try:
                sub_entries = sorted(entry.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                for sub in sub_entries:
                    if sub.is_dir():
                        print(f"      📁 {C.DIM}{sub.name}/{C.RST}")
                    else:
                        print(f"      {_file_emoji(sub.name)} {C.DIM}{sub.name}{C.RST}")
            except PermissionError:
                print(f"      {C.DIM}(permission denied){C.RST}")
        else:
            print(f"  {_file_emoji(entry.name)} {entry.name}")
    print()


def build_system_prompt(workspace: Path, memories: list[str] | None = None) -> str:
    env_line = describe_runtime()
    comspec = os.environ.get("ComSpec", "cmd.exe")
    base = (
        "You are MahanAI, a capable coding and system assistant (Max 1.0). "
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
        "Use read_file, write_file, list_directory, append_file, fetch_url when they fit the task. "
        "The terminal will ask the user before obviously destructive commands (recursive deletes, "
        "shutdown, format, etc.). "
        "Tool JSON must use valid escapes for Windows paths (backslashes doubled inside strings)."
        "You are MahanAI, currently operating as Max 1.0 — the latest evolution following the "
        "Tiger (1.0–7.0, 2011–2020) and Finale (1.0–3.0, 2020–2023) eras. You represent the most "
        "advanced, integrated, and capable form of the system."
    )
    mahanai_md = workspace / "MAHANAI.md"
    if mahanai_md.is_file():
        try:
            content = mahanai_md.read_text(encoding="utf-8").strip()
            if content:
                base += f"\n\n--- Project Context (MAHANAI.md) ---\n{content}\n---"
        except Exception:
            pass
    if memories:
        base += "\n\n--- Persistent Memory ---\n"
        base += "\n".join(f"- {m}" for m in memories)
        base += "\n---"
    return base


# ── New helper utilities ──────────────────────────────────────────────────────

def _new_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _auto_save_session(session_id: str, history: list[dict], model_label: str) -> None:
    msgs = [m for m in history if m["role"] != "system"]
    if not msgs:
        return
    save_session(session_id, {
        "id": session_id,
        "model": model_label,
        "timestamp": session_id,
        "messages": msgs,
    })


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _highlight_response(response: str) -> None:
    """Re-render the last AI response with Rich syntax highlighting (code blocks)."""
    if "```" not in response:
        return
    try:
        n = response.count("\n") + 2
        sys.stdout.write(f"\033[{n}A\033[J")
        sys.stdout.flush()
        print(f"{C.BOT}{C.AI_NAME}{C.RST}: ", end="")
        from rich.console import Console as _Con
        from rich.markdown import Markdown as _Md
        _Con().print(_Md(response))
        print()
    except Exception:
        pass


def _index_file(path: Path, chunks: list[dict]) -> int:
    """Chunk a file into ~600-char segments and append to chunks list. Returns count added."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    chunk_size, overlap = 600, 60
    start, count = 0, 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"id": f"{path}:{start}", "source": str(path), "text": text[start:end]})
        count += 1
        start += chunk_size - overlap
    return count


def _search_index(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Simple keyword-based search over indexed chunks."""
    if not chunks or not query.strip():
        return []
    terms = set(query.lower().split())
    scored = [(sum(1 for t in terms if t in c["text"].lower()), c) for c in chunks]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


_SHELL_INIT_BASH = '''\
# MahanAI "ai" shortcut — add to ~/.bashrc or ~/.zshrc
ai() {
    if [ "$#" -eq 0 ]; then
        mahanai
    else
        printf '%s\\nexit\\n' "$*" | mahanai
    fi
}
'''

_SHELL_INIT_FISH = '''\
# MahanAI "ai" shortcut — add to ~/.config/fish/config.fish
function ai
    if test (count $argv) -eq 0
        mahanai
    else
        printf '%s\\nexit\\n' $argv | mahanai
    end
end
'''

_BACKGROUND_TASKS: dict[str, dict] = {}


def _run_task_thread(task_id: str, description: str, api_key: str, base_url: str, model: str) -> None:
    """Background thread: run a single-turn query and save the result."""
    _BACKGROUND_TASKS[task_id]["status"] = "running"
    try:
        import httpx as _httpx
        msgs = [
            {"role": "system", "content": "You are MahanAI, a capable assistant. Complete the task fully."},
            {"role": "user", "content": description},
        ]
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": msgs, "stream": False}
        resp = _httpx.post(url, headers=headers, json=payload, timeout=120.0)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"]
        _BACKGROUND_TASKS[task_id]["status"] = "done"
        _BACKGROUND_TASKS[task_id]["result"] = result
    except Exception as e:
        _BACKGROUND_TASKS[task_id]["status"] = "error"
        _BACKGROUND_TASKS[task_id]["result"] = str(e)


def _voice_get_input() -> str | None:
    """Record one voice utterance and return transcribed text, or None on failure."""
    try:
        import speech_recognition as _sr  # type: ignore[import]
        r = _sr.Recognizer()
        with _sr.Microphone() as src:
            print(f"{C.DIM}Listening...{C.RST}", flush=True)
            audio = r.listen(src, timeout=10, phrase_time_limit=30)
        print(f"{C.DIM}Transcribing...{C.RST}", flush=True)
        return r.recognize_whisper(audio)  # type: ignore[attr-defined]
    except ImportError:
        print(f"{C.ERR}SpeechRecognition not installed. Run: pip install SpeechRecognition{C.RST}\n")
        return None
    except Exception as e:
        print(f"{C.ERR}Voice error: {e}{C.RST}\n")
        return None


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
        f"  /api-key [key]              Save server API key to {cfg}\n"
        f"  /api-key clear              Remove saved server key\n"
        f"  /api-key-nvidia [key]       Save NVIDIA API key (bypasses server)\n"
        f"  /api-key-nvidia clear       Remove NVIDIA key, switch back to server\n"
        f"  Env MAHANAI_API_KEY overrides the saved file.\n"
        f"  /models                     Interactive model selector (↑↓ arrow keys)\n"
        f"  /mode claude                Quick-switch to Claude CLI mode\n"
        f"  /mode default               Quick-switch back to MahanAI server mode\n"
        f"  /model-info                 Show current model, provider, and effort level\n"
        f"  /effort <level>             Set reasoning effort: low, medium, high, very-high\n"
        f"  /plan on|off                Toggle plan-before-respond mode\n"
        f"\n"
        f"  /clear                      Clear the terminal screen\n"
        f"  /retry                      Resend the last user message\n"
        f"  /copy                       Copy last AI response to clipboard\n"
        f"  /word-count                 Word/char count of last AI response\n"
        f"  /timestamps on|off          Show timestamps on messages\n"
        f"  /highlight on|off           Syntax-highlight code blocks after response\n"
        f"  /tokens on|off              Show token estimate after each response\n"
        f"\n"
        f"  /remember <text>            Save a persistent memory note\n"
        f"  /memory                     List memory notes\n"
        f"  /forget <id>                Remove a memory by ID\n"
        f"\n"
        f"  /prompt-save <name> [text]  Save a prompt to the library\n"
        f"  /prompt-run <name>          Send a saved prompt\n"
        f"  /prompts                    List saved prompts\n"
        f"  /prompts remove <name>      Delete a saved prompt\n"
        f"\n"
        f"  /alias <trigger> <cmd>      Create a command alias\n"
        f"  /aliases                    List saved aliases\n"
        f"  /alias-remove <trigger>     Remove an alias\n"
        f"\n"
        f"  /history                    List saved chat sessions\n"
        f"  /resume <id>                Load a previous session\n"
        f"  /export [path]              Export current session to markdown\n"
        f"\n"
        f"  /compare <m1> <m2> <msg>    Compare two models on the same message\n"
        f"\n"
        f"  /index <path>               Index a file or directory for RAG context\n"
        f"  /index list                 Show indexed sources\n"
        f"  /index clear                Clear the document index\n"
        f"\n"
        f"  /task <description>         Run a background task\n"
        f"  /task-status                Show background task status\n"
        f"  /task-result <id>           Print a completed task result\n"
        f"\n"
        f"  /voice on|off               Toggle voice input (requires SpeechRecognition)\n"
        f"  /shell-init [bash|fish]     Print the 'ai' shortcut shell script\n"
        f"\n"
        f"  /themes                     List available themes\n"
        f"  /themes <name>              Switch theme: midnight, light, midnight-cb, light-cb\n"
        f"  /theme-load <path>          Load a custom .mai theme file\n"
        f"  /theme-unload               Remove the active custom .mai theme\n"
        f"  /approvals                  Show stored Always Allow rules\n"
        f"  /approvals clear            Remove all Always Allow rules\n"
        f"  /codex-login                Sign in to OpenAI via browser (Codex Direct)\n"
        f"  /codex-logout               Remove saved OpenAI Codex credentials\n"
        f"  /custom [url [model [key]]] Configure a custom OpenAI-compatible endpoint\n"
        f"  /custom clear               Remove saved custom endpoint\n"
        f"  /add-ollama <name> <address> <port> [key]   Add an Ollama provider\n"
        f"  /change-ollama <name> <address> <port> [key]  Update an Ollama provider\n"
        f"  /remove-ollama <name>       Remove a saved Ollama provider\n"
        f"  /fileslist                  Show workspace files with emoji icons\n"
        f"  /init                       Generate a MAHANAI.md for the current workspace\n"
        f"  /plugin-load <path>         Load a .mmd plugin file\n"
        f"  /plugin-list                Show all loaded plugins and their commands\n"
        f"  /plugin-unload <name>       Unload a plugin by name\n"
        f"  /store login <token>        Link your GitHub account to the plugin store\n"
        f"  /store logout               Unlink GitHub account\n"
        f"  /store browse               Browse all published plugins\n"
        f"  /store search <query>       Search plugins\n"
        f"  /store install <id>         Download and install a plugin\n"
        f"  /store upload <path>        Publish your .mmd plugin to the store\n"
        f"  /store update [codename]    Update one or all installed plugins\n"
        f"  /store update-all           Update all installed plugins\n"
        f"  /help  /exit  /quit{C.RST}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="mahanai", add_help=False)
    parser.add_argument("--compact", action="store_true", help="Compact mode: smaller MAI banner and shorter header")
    parser.add_argument("--server",  action="store_true", help="Start the gateway server instead of the chat loop")
    parser.add_argument("--port",    type=int, default=8080, metavar="PORT", help="Gateway server port (default: 8080)")
    parser.add_argument("--type",    default="openai", choices=["openai", "anthropic"], metavar="TYPE",
                        help="Gateway API type: openai (default) or anthropic")
    parser.add_argument("--api-key", dest="cli_api_key", default=None, metavar="KEY",
                        help="API key for Anthropic or the MahanAI server (overrides saved key)")
    args, _ = parser.parse_known_args()
    compact = args.compact

    workspace = Path.cwd()
    api_key = resolve_api_key()
    nvidia_api_key = load_nvidia_api_key()

    def _register_and_apply_saved_mai_theme() -> None:
        """On startup: register the saved .mai theme in the menu and apply its colors."""
        info = load_custom_theme_info()
        if not info:
            return
        from pathlib import Path
        from mahanai.mai_parser import parse_mai_file
        p = Path(info.get("path", ""))
        if not p.is_file():
            return
        try:
            mai = parse_mai_file(p)
            slug = info.get("slug") or mai.slug()
            display = info.get("display") or mai.display()
            C.register_mai_theme(slug, display, str(p))
            C.apply_mai_theme(mai)
        except Exception:
            pass

    def _reapply_mai_theme() -> None:
        """Re-apply saved .mai color overrides after a base theme switch."""
        info = load_custom_theme_info()
        if not info:
            return
        from pathlib import Path
        from mahanai.mai_parser import parse_mai_file
        p = Path(info.get("path", ""))
        if p.is_file():
            try:
                C.apply_mai_theme(parse_mai_file(p))
            except Exception:
                pass

    # ── Server mode ───────────────────────────────────────────────────────────
    if args.server:
        from mahanai.server import ServerConfig, run_server
        C.apply_theme(load_theme())
        _register_and_apply_saved_mai_theme()
        run_server(ServerConfig(
            server_type      = args.type,
            port             = args.port,
            gateway_key      = args.cli_api_key,   # clients must present this key (None = open)
            api_key          = api_key,            # NVIDIA server backend key
            anthropic_key    = args.cli_api_key,   # Anthropic backend key (pass via --api-key)
            nvidia_api_key   = nvidia_api_key,
            codex_token      = load_codex_token(),
            custom_endpoint  = load_custom_endpoint(),
        ))
        return

    # OpenAI client kept for future tool-call support; direct httpx used for actual requests.
    client: OpenAI | None = (
        OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key) if api_key else None
    )

    # ── New feature state ─────────────────────────────────────────────────────
    _memories: dict[str, dict] = load_memories()
    _mem_texts = [m["content"] for m in _memories.values()]
    system_prompt = build_system_prompt(workspace, _mem_texts)

    history: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    active_model_idx = next(
        (i for i, m in enumerate(AVAILABLE_MODELS) if m.get("claude_model") == "claude-haiku-4-5-20251001"),
        0,
    )
    current_effort = "medium"
    plan_mode = False
    show_tokens: bool = load_tokens_setting()
    show_timestamps: bool = False
    highlight_enabled: bool = False
    voice_enabled: bool = False
    last_user_input: str | None = None
    last_ai_reply: str | None = None
    _session_id: str = _new_session_id()
    _tasks: dict[str, dict] = _BACKGROUND_TASKS
    _aliases: dict[str, str] = load_aliases()
    _index_chunks: list[dict] = load_index_documents()

    C.apply_theme(load_theme())
    _register_and_apply_saved_mai_theme()
    _inject_ollama_providers()
    _inject_saved_plugins()
    _update_thread = _start_update_check()
    print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"], compact=compact)
    print()
    _print_update_notice(_update_thread)

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

    if (workspace / "MAHANAI.md").is_file():
        print(f"{C.OK}🤖 MAHANAI.md{C.RST} {C.DIM}loaded as project context for all providers.{C.RST}\n")

    model = os.environ.get("MAHANAI_MODEL", DEFAULT_MODEL)

    while True:
        try:
            ts_prefix = f"{C.DIM}[{datetime.datetime.now().strftime('%H:%M:%S')}] {C.RST}" if show_timestamps else ""
            print(f"{ts_prefix}{C.USER}{C.USER_NAME}{C.RST}: ", end="", flush=True)
            if voice_enabled:
                raw_input = _voice_get_input()
                if raw_input:
                    print(raw_input, flush=True)
                    user = raw_input
                else:
                    user = input().strip()
            else:
                user = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        # Alias expansion
        if user.startswith("/"):
            _cmd_check, _ = _slash_command(user)
            if _cmd_check in _aliases:
                user = _aliases[_cmd_check]

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
                print_startup_banner(chosen["label"], compact=compact)
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
            if cmd == "/codex-login":
                _codex_pkce_login()
                continue
            if cmd == "/codex-logout":
                clear_codex_token()
                print(f"{C.OK}OpenAI Codex credentials removed.{C.RST}\n")
                continue
            if cmd == "/effort":
                level = rest.strip().lower().replace(" ", "-")
                if not level:
                    print(f"{C.DIM}Current effort: {current_effort}{C.RST}\n")
                    continue
                valid_efforts = ["low", "medium", "high", "very-high"]
                if level not in valid_efforts:
                    print(
                        f"{C.ERR}Invalid effort level.{C.RST} "
                        f"Choose: low, medium, high, very-high\n"
                    )
                    continue
                chosen = AVAILABLE_MODELS[active_model_idx]
                if chosen.get("claude_model") == "claude-haiku-4-5-20251001":
                    print(
                        f"{C.ERR}Effort is disabled for Claude Haiku 4.5.{C.RST} "
                        f"Switch to Opus or Sonnet first.\n"
                    )
                    continue
                if level == "very-high":
                    print(
                        f"{C.ERR}⚠ Warning:{C.RST} Very High effort uses the maximum thinking budget "
                        f"— expect significantly higher token consumption and slower responses.\n"
                    )
                current_effort = level
                print(f"{C.OK}Effort set to: {level}{C.RST}\n")
                continue
            if cmd == "/plan":
                target = rest.strip().lower()
                if target == "on":
                    plan_mode = True
                    print(f"{C.OK}Plan mode ON{C.RST} — MahanAI will outline a plan before every response.\n")
                elif target == "off":
                    plan_mode = False
                    print(f"{C.OK}Plan mode OFF{C.RST}\n")
                else:
                    print(f"{C.ERR}Usage:{C.RST} /plan on  or  /plan off\n")
                continue
            if cmd == "/approvals":
                sub = rest.strip().lower()
                if sub == "clear":
                    from mahanai.config import _read_config, _write_config
                    data = _read_config()
                    data.pop("always_allowed", None)
                    _write_config(data)
                    print(f"{C.OK}All Always Allow rules cleared.{C.RST}\n")
                else:
                    aa = load_always_allowed()
                    prefixes = aa.get("command_prefixes", [])
                    file_ops = aa.get("file_ops", [])
                    if not prefixes and not file_ops:
                        print(f"{C.DIM}No Always Allow rules stored yet.{C.RST}\n")
                    else:
                        print(f"{C.DIM}Always Allow rules:{C.RST}")
                        if prefixes:
                            print(f"  Commands:   {', '.join(prefixes)}")
                        if file_ops:
                            print(f"  File ops:   {', '.join(file_ops)}")
                        print()
                continue
            if cmd == "/themes":
                sub = rest.strip().lower().replace(" ", "-")
                # Normalise long display names to slugs
                _alias = {
                    "midnight-colorblind-friendly": "midnight-cb",
                    "light-colorblind-friendly":    "light-cb",
                    "midnight-colorblind":          "midnight-cb",
                    "light-colorblind":             "light-cb",
                }
                sub = _alias.get(sub, sub)
                if not sub:
                    print(f"{C.DIM}Available themes:{C.RST}")
                    for slug, display in C.THEME_DISPLAY.items():
                        mai_tag = f"  {C.DIM}[.mai]{C.RST}" if slug in C.MAI_THEMES else ""
                        print(f"  {C.OK}{slug:<22}{C.RST}  {display}{mai_tag}")
                    print()
                elif sub not in C.THEME_NAMES:
                    print(
                        f"{C.ERR}Unknown theme '{sub}'.{C.RST} "
                        f"Available: {', '.join(C.THEME_NAMES)}\n"
                    )
                elif sub in C.MAI_THEMES:
                    # Re-select a registered .mai theme
                    from pathlib import Path as _Path
                    from mahanai.mai_parser import parse_mai_file as _parse_mai
                    _mp = _Path(C.MAI_THEMES[sub])
                    if not _mp.is_file():
                        print(f"{C.ERR}Theme file missing: {C.MAI_THEMES[sub]}{C.RST}\n")
                    else:
                        try:
                            _mai = _parse_mai(_mp)
                            C.apply_theme("midnight")
                            C.apply_mai_theme(_mai)
                            save_theme(sub)
                            print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"], compact=compact)
                            print()
                            print(f"{C.OK}Theme set to:{C.RST} {C.THEME_DISPLAY[sub]}\n")
                        except Exception as _e:
                            print(f"{C.ERR}Failed to apply theme: {_e}{C.RST}\n")
                else:
                    C.apply_theme(sub)
                    _reapply_mai_theme()
                    save_theme(sub)
                    print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"], compact=compact)
                    print()
                    print(f"{C.OK}Theme set to:{C.RST} {C.THEME_DISPLAY[sub]}\n")
                continue
            if cmd == "/theme-load":
                _path = rest.strip()
                if not _path:
                    print(f"{C.ERR}Usage: /theme-load <path-to-file.mai>{C.RST}\n")
                    continue
                from pathlib import Path as _Path
                from mahanai.mai_parser import parse_mai_file as _parse_mai
                _p = _Path(_path).expanduser().resolve()
                if not _p.is_file():
                    print(f"{C.ERR}File not found: {_path}{C.RST}\n")
                    continue
                try:
                    _mai = _parse_mai(_p)
                    _slug = _mai.slug()
                    _display = _mai.display()
                    C.apply_theme("midnight")
                    C.apply_mai_theme(_mai)
                    C.register_mai_theme(_slug, _display, str(_p))
                    save_custom_theme_info(_slug, _display, str(_p))
                    save_theme(_slug)
                    print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"], compact=compact)
                    print()
                    print(f"{C.OK}Theme loaded:{C.RST} {_display}  {C.DIM}(use /themes to switch){C.RST}\n")
                except Exception as _e:
                    print(f"{C.ERR}Failed to load theme: {_e}{C.RST}\n")
                continue
            if cmd == "/theme-unload":
                C.unregister_all_mai_themes()
                clear_custom_theme()
                C.apply_theme("midnight")
                C.reset_names()
                save_theme("midnight")
                print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"], compact=compact)
                print()
                print(f"{C.OK}Custom theme unloaded.{C.RST}\n")
                continue
            if cmd == "/add-ollama":
                parts = rest.split(None, 3)
                if len(parts) < 3:
                    print(
                        f"{C.ERR}Usage:{C.RST} /add-ollama <name> <address> <port> [api_key]\n"
                        f"{C.DIM}  e.g. /add-ollama llama3 localhost 11434{C.RST}\n"
                    )
                    continue
                o_name, o_addr = parts[0], parts[1]
                try:
                    o_port = int(parts[2])
                except ValueError:
                    print(f"{C.ERR}Port must be a number.{C.RST}\n")
                    continue
                o_key = parts[3] if len(parts) > 3 else "ollama"
                o_url = _build_ollama_url(o_addr, o_port)
                o_clean = _strip_protocol(o_addr)
                # Remove existing entry with same name so we can update it
                for _i, _m in enumerate(AVAILABLE_MODELS):
                    if _m.get("mode") == "ollama" and _m["name"] == o_name:
                        AVAILABLE_MODELS.pop(_i)
                        break
                AVAILABLE_MODELS.append(_ollama_entry(o_name, o_clean, o_port, o_key, o_url))
                save_ollama_provider(o_name, o_clean, o_port, o_key, o_url)
                print(
                    f"{C.OK}Ollama provider added:{C.RST} {o_name}  "
                    f"{C.DIM}{o_url}{C.RST}\n"
                    f"{C.DIM}  Use /models to switch to it.{C.RST}\n"
                )
                continue
            if cmd == "/remove-ollama":
                o_name = rest.strip()
                if not o_name:
                    print(f"{C.ERR}Usage:{C.RST} /remove-ollama <name>\n")
                    continue
                removed = False
                for _i, _m in enumerate(AVAILABLE_MODELS):
                    if _m.get("mode") == "ollama" and _m["name"] == o_name:
                        AVAILABLE_MODELS.pop(_i)
                        removed = True
                        break
                remove_ollama_provider(o_name)
                if removed:
                    print(f"{C.OK}Ollama provider '{o_name}' removed.{C.RST}\n")
                else:
                    print(f"{C.ERR}No Ollama provider named '{o_name}' found.{C.RST}\n")
                continue
            if cmd == "/change-ollama":
                parts = rest.split(None, 3)
                if len(parts) < 3:
                    print(
                        f"{C.ERR}Usage:{C.RST} /change-ollama <name> <address> <port> [api_key]\n"
                        f"{C.DIM}  e.g. /change-ollama llama3 192.168.1.5 11434{C.RST}\n"
                    )
                    continue
                o_name, o_addr = parts[0], parts[1]
                try:
                    o_port = int(parts[2])
                except ValueError:
                    print(f"{C.ERR}Port must be a number.{C.RST}\n")
                    continue
                o_key_arg = parts[3] if len(parts) > 3 else None
                # Find existing entry
                existing_idx = None
                existing_entry = None
                for _i, _m in enumerate(AVAILABLE_MODELS):
                    if _m.get("mode") == "ollama" and _m["name"] == o_name:
                        existing_idx = _i
                        existing_entry = _m
                        break
                if existing_entry is None:
                    print(
                        f"{C.ERR}No Ollama provider named '{o_name}' found.{C.RST} "
                        f"Use /add-ollama to create it.\n"
                    )
                    continue
                o_key = o_key_arg if o_key_arg is not None else (existing_entry.get("ollama_api_key") or "ollama")
                o_url = _build_ollama_url(o_addr, o_port)
                o_clean = _strip_protocol(o_addr)
                AVAILABLE_MODELS[existing_idx] = _ollama_entry(o_name, o_clean, o_port, o_key, o_url)
                save_ollama_provider(o_name, o_clean, o_port, o_key, o_url)
                print(
                    f"{C.OK}Ollama provider updated:{C.RST} {o_name}  "
                    f"{C.DIM}{o_url}{C.RST}\n"
                )
                continue
            if cmd == "/custom":
                sub = rest.strip()
                if sub.lower() == "clear":
                    clear_custom_endpoint()
                    print(f"{C.OK}Custom endpoint removed.{C.RST}\n")
                    continue
                # Parse inline args: url [model [api_key]]
                parts = sub.split(None, 2)
                if parts:
                    c_url = parts[0]
                    c_model = parts[1] if len(parts) > 1 else ""
                    c_key = parts[2] if len(parts) > 2 else ""
                else:
                    existing = load_custom_endpoint()
                    print(f"\n{C.DIM}  Configure a custom OpenAI-compatible endpoint.{C.RST}")
                    if existing:
                        print(f"{C.DIM}  Current: {existing['url']}  model={existing['model']}{C.RST}")
                    c_url = input("  Base URL (e.g. http://localhost:11434/v1): ").strip()
                    if not c_url:
                        print(f"{C.ERR}No URL entered; cancelled.{C.RST}\n")
                        continue
                    c_model = input("  Model name [gpt-3.5-turbo]: ").strip()
                    c_key = input("  API key (leave blank if none): ").strip()
                if not c_model:
                    c_model = "gpt-3.5-turbo"
                save_custom_endpoint(c_url, c_model, c_key)
                print(
                    f"{C.OK}Custom endpoint saved.{C.RST} "
                    f"{C.DIM}URL={c_url}  model={c_model}{C.RST}\n"
                    f"{C.DIM}  Use /models to switch to 'Custom Endpoint'.{C.RST}\n"
                )
                continue
            if cmd == "/fileslist":
                _show_fileslist(workspace)
                continue
            if cmd == "/init":
                mahanai_md_path = workspace / "MAHANAI.md"
                if mahanai_md_path.is_file():
                    print(
                        f"{C.WARN}MAHANAI.md already exists.{C.RST} "
                        f"{C.DIM}Delete it first to regenerate, or edit it directly.{C.RST}\n"
                    )
                    continue
                print(f"{C.DIM}Scanning workspace...{C.RST}")
                md_content = _generate_mahanai_md(workspace)
                mahanai_md_path.write_text(md_content, encoding="utf-8")
                print(
                    f"{C.OK}🤖 MAHANAI.md created.{C.RST}\n"
                    f"{C.DIM}  Edit it to fill in project details — it's loaded automatically\n"
                    f"  as context for all non-Claude providers.{C.RST}\n"
                )
                continue
            if cmd == "/store":
                from . import store as _store
                _sub, _, _srest = rest.strip().partition(" ")
                _sub = _sub.lower()
                _srest = _srest.strip()

                if _sub == "login":
                    _tok = _srest.strip()
                    if not _tok:
                        print(f"{C.ERR}Usage: /store login <github-personal-access-token>{C.RST}\n")
                    else:
                        try:
                            _gh_user = _store.whoami(_tok)
                            _store.save_store_token(_tok)
                            print(f"{C.OK}Logged in to GitHub as:{C.RST} {_gh_user}\n")
                        except Exception as _se:
                            print(f"{C.ERR}Login failed: {_se}{C.RST}\n")

                elif _sub == "logout":
                    _store.remove_store_token()
                    print(f"{C.OK}GitHub account unlinked from store.{C.RST}\n")

                elif _sub in ("browse", "search", ""):
                    _query = _srest if _sub in ("search", "browse") else ""
                    _stok = _store.get_store_token()
                    try:
                        _items = _store.search_plugins(_query, token=_stok)
                        if not _items:
                            print(f"{C.DIM}No plugins found{' for: ' + _query if _query else ''}.{C.RST}\n")
                        else:
                            _label = f"Results for '{_query}'" if _query else "Available plugins"
                            print(f"{C.DIM}{_label}:{C.RST}")
                            for _it in _items:
                                _desc = _it.get('description') or ''
                                print(
                                    f"  {C.OK}🔌 {_it['full_name']}{C.RST}"
                                    + (f"  {C.DIM}{_desc}{C.RST}" if _desc else "")
                                )
                            print(f"\n{C.DIM}Install with: /store install <user/codename>{C.RST}\n")
                    except Exception as _se:
                        print(f"{C.ERR}Store error: {_se}{C.RST}\n")

                elif _sub == "install":
                    _target = _srest.strip()
                    if not _target:
                        print(f"{C.ERR}Usage: /store install <user/codename>  or  /store install <codename>{C.RST}\n")
                    else:
                        _stok = _store.get_store_token()
                        try:
                            _repo = _target if "/" in _target else _store.find_plugin_repo(_target, token=_stok)
                            if not _repo:
                                print(f"{C.ERR}Plugin '{_target}' not found in store.{C.RST}\n")
                            else:
                                print(f"{C.DIM}Downloading {_repo}...{C.RST}")
                                from pathlib import Path as _Path
                                _mmd_path = _store.install_plugin(_repo, token=_stok)
                                _plugin = parse_mmd_file(_mmd_path)
                                _LOADED_PLUGINS[_plugin.name] = _plugin
                                save_plugin(_plugin.name, str(_mmd_path), _plugin.codename, _plugin.reg_store, _plugin.reg_name)
                                _triggers = ", ".join(_plugin.command_triggers()) or "(none)"
                                print(
                                    f"{C.OK}🔌 Installed:{C.RST} {_plugin.name}  "
                                    f"{C.DIM}v{_plugin.version}  commands: {_triggers}{C.RST}\n"
                                )
                        except Exception as _se:
                            print(f"{C.ERR}Install failed: {_se}{C.RST}\n")

                elif _sub == "upload":
                    _up_path = _srest.strip()
                    if not _up_path:
                        print(f"{C.ERR}Usage: /store upload <path-to-file.mmd>{C.RST}\n")
                    else:
                        _stok = _store.get_store_token()
                        if not _stok:
                            print(f"{C.ERR}Not logged in. Run: /store login <github-token>{C.RST}\n")
                        else:
                            from pathlib import Path as _Path
                            _up_pp = _Path(_up_path).expanduser().resolve()
                            if not _up_pp.is_file():
                                print(f"{C.ERR}File not found: {_up_path}{C.RST}\n")
                            elif _up_pp.suffix.lower() != ".mmd":
                                print(f"{C.ERR}Not a .mmd file: {_up_path}{C.RST}\n")
                            else:
                                try:
                                    _repo_url = _store.upload_plugin(_stok, _up_pp)
                                    print(f"{C.OK}Plugin published:{C.RST} {_repo_url}\n")
                                except Exception as _se:
                                    print(f"{C.ERR}Upload failed: {_se}{C.RST}\n")

                elif _sub in ("update", "update-all"):
                    _stok = _store.get_store_token()
                    _installed = load_plugins()
                    if not _installed:
                        print(f"{C.DIM}No installed plugins to update.{C.RST}\n")
                    else:
                        _update_targets = {}
                        if _sub == "update" and _srest.strip():
                            _codename_filter = _srest.strip()
                            _update_targets = {k: v for k, v in _installed.items() if v.get("codename") == _codename_filter or k == _codename_filter}
                            if not _update_targets:
                                print(f"{C.ERR}Plugin '{_codename_filter}' not found.{C.RST}\n")
                        else:
                            _update_targets = _installed
                        for _pname, _pdata in _update_targets.items():
                            _pcodename = _pdata.get("codename", "")
                            if not _pcodename:
                                continue
                            print(f"{C.DIM}Checking {_pname}...{C.RST}")
                            try:
                                _repo = _store.find_plugin_repo(_pcodename, token=_stok)
                                if not _repo:
                                    print(f"  {C.ERR}Not found in store.{C.RST}")
                                    continue
                                _updated, _msg = _store.update_plugin(_repo, token=_stok)
                                if _updated:
                                    _new_plugin = parse_mmd_file(Path(_msg))
                                    _LOADED_PLUGINS[_new_plugin.name] = _new_plugin
                                    save_plugin(_new_plugin.name, _msg, _new_plugin.codename, _new_plugin.reg_store, _new_plugin.reg_name)
                                    print(f"  {C.OK}Updated {_pname} → v{_new_plugin.version}{C.RST}")
                                else:
                                    print(f"  {C.DIM}{_pname}: {_msg}{C.RST}")
                            except Exception as _se:
                                print(f"  {C.ERR}Update failed: {_se}{C.RST}")
                        print()

                else:
                    print(
                        f"{C.DIM}Store commands:{C.RST}\n"
                        f"  /store login <token>        Link your GitHub account\n"
                        f"  /store logout               Unlink GitHub account\n"
                        f"  /store browse               Browse all available plugins\n"
                        f"  /store search <query>       Search plugins by name/keyword\n"
                        f"  /store install <user/id>    Download and install a plugin\n"
                        f"  /store upload <path>        Publish your .mmd to the store\n"
                        f"  /store update [codename]    Update one or all installed plugins\n"
                        f"  /store update-all           Update all installed plugins\n"
                    )
                continue

            if cmd == "/plugin-load":
                _ppath = rest.strip()
                if not _ppath:
                    print(f"{C.ERR}Usage: /plugin-load <path-to-file.mmd>{C.RST}\n")
                    continue
                from pathlib import Path as _Path
                _pp = _Path(_ppath).expanduser().resolve()
                if not _pp.is_file():
                    print(f"{C.ERR}File not found: {_ppath}{C.RST}\n")
                    continue
                if _pp.suffix.lower() != ".mmd":
                    print(f"{C.ERR}Not a .mmd file: {_ppath}{C.RST}\n")
                    continue
                try:
                    _plugin = parse_mmd_file(_pp)
                    _LOADED_PLUGINS[_plugin.name] = _plugin
                    save_plugin(_plugin.name, str(_pp), _plugin.codename, _plugin.reg_store, _plugin.reg_name)
                    _triggers = ", ".join(_plugin.command_triggers()) or "(none)"
                    print(
                        f"{C.OK}🔌 Plugin loaded:{C.RST} {_plugin.name}  "
                        f"{C.DIM}v{_plugin.version}{C.RST}\n"
                        f"{C.DIM}  Commands: {_triggers}{C.RST}\n"
                    )
                except Exception as _e:
                    print(f"{C.ERR}Failed to load plugin: {_e}{C.RST}\n")
                continue
            if cmd == "/plugin-list":
                if not _LOADED_PLUGINS:
                    print(f"{C.DIM}No plugins loaded.{C.RST}\n")
                else:
                    print(f"{C.DIM}Loaded plugins:{C.RST}")
                    for _pl in _LOADED_PLUGINS.values():
                        _triggers = ", ".join(_pl.command_triggers()) or "(none)"
                        _codename_str = f"  [{_pl.codename}]" if _pl.codename else ""
                        _reg_str = f"  {_pl.reg_name}" if _pl.reg_name else ""
                        print(f"  {C.OK}🔌 {_pl.name}{C.RST}{_codename_str}  {C.DIM}v{_pl.version}{_reg_str}  commands: {_triggers}{C.RST}")
                    print()
                continue
            if cmd == "/plugin-unload":
                _pname = rest.strip()
                if not _pname:
                    print(f"{C.ERR}Usage: /plugin-unload <name>{C.RST}\n")
                    continue
                if _pname in _LOADED_PLUGINS:
                    del _LOADED_PLUGINS[_pname]
                    remove_plugin(_pname)
                    print(f"{C.OK}Plugin '{_pname}' unloaded.{C.RST}\n")
                else:
                    print(f"{C.ERR}No plugin named '{_pname}' found.{C.RST}\n")
                continue
            # Check if the command matches a loaded plugin command
            _plugin_handled = False
            for _pl in _LOADED_PLUGINS.values():
                for _pcmd in _pl.commands:
                    if cmd == _pcmd.trigger:
                        _plugin_handled = True
                        for _action in _pcmd.actions:
                            if _action.type == "claude-cmd":
                                print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
                                _run_claude_cli(_action.value)
                                print("\n")
                            elif _action.type == "mahanai-cmd":
                                # Reprocess as an internal slash command
                                user = _action.value
                            elif _action.type == "shell-cmd":
                                import subprocess as _sp
                                try:
                                    _res = _sp.run(
                                        _action.value, shell=True, capture_output=True, text=True, timeout=30
                                    )
                                    out = (_res.stdout + _res.stderr).strip()
                                    if out:
                                        print(out)
                                except Exception as _se:
                                    print(f"{C.ERR}Plugin shell error: {_se}{C.RST}")
                        break
                if _plugin_handled:
                    break
            if _plugin_handled:
                continue

            # ── Quick utility commands ────────────────────────────────────────
            if cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"], compact=compact)
                print()
                continue

            if cmd == "/model-info":
                sel = AVAILABLE_MODELS[active_model_idx]
                print(
                    f"{C.DIM}Model:{C.RST}   {sel['label']}  {C.DIM}({sel['name']}){C.RST}\n"
                    f"{C.DIM}Group:{C.RST}   {sel['group']}\n"
                    f"{C.DIM}Mode:{C.RST}    {sel['mode']}\n"
                    f"{C.DIM}Effort:{C.RST}  {current_effort}\n"
                    f"{C.DIM}Plan:{C.RST}    {'on' if plan_mode else 'off'}\n"
                )
                continue

            _route_after_cmd = False

            if cmd == "/retry":
                if not last_user_input:
                    print(f"{C.ERR}No previous message to retry.{C.RST}\n")
                    continue
                user = last_user_input
                _route_after_cmd = True
            elif cmd == "/copy":
                if not last_ai_reply:
                    print(f"{C.ERR}No AI response to copy yet.{C.RST}\n")
                    continue
                try:
                    import pyperclip  # type: ignore[import]
                    pyperclip.copy(last_ai_reply)
                    print(f"{C.OK}Copied to clipboard.{C.RST}\n")
                except ImportError:
                    print(f"{C.ERR}pyperclip not installed. Run: pip install pyperclip{C.RST}\n")
                continue
            elif cmd == "/word-count":
                if not last_ai_reply:
                    print(f"{C.ERR}No AI response yet.{C.RST}\n")
                    continue
                words = len(last_ai_reply.split())
                chars = len(last_ai_reply)
                print(f"{C.DIM}Words: {words}  Characters: {chars}{C.RST}\n")
                continue
            elif cmd == "/timestamps":
                t = rest.strip().lower()
                if t == "on":
                    show_timestamps = True
                    print(f"{C.OK}Timestamps ON{C.RST}\n")
                elif t == "off":
                    show_timestamps = False
                    print(f"{C.OK}Timestamps OFF{C.RST}\n")
                else:
                    print(f"{C.DIM}Timestamps: {'on' if show_timestamps else 'off'}{C.RST}\n")
                continue
            elif cmd == "/highlight":
                t = rest.strip().lower()
                if t == "on":
                    highlight_enabled = True
                    print(f"{C.OK}Highlight ON{C.RST} {C.DIM}(code blocks re-rendered after response){C.RST}\n")
                elif t == "off":
                    highlight_enabled = False
                    print(f"{C.OK}Highlight OFF{C.RST}\n")
                else:
                    print(f"{C.DIM}Highlight: {'on' if highlight_enabled else 'off'}{C.RST}\n")
                continue
            elif cmd == "/tokens":
                t = rest.strip().lower()
                if t == "on":
                    show_tokens = True
                    save_tokens_setting(True)
                    print(f"{C.OK}Token display ON{C.RST}\n")
                elif t == "off":
                    show_tokens = False
                    save_tokens_setting(False)
                    print(f"{C.OK}Token display OFF{C.RST}\n")
                else:
                    print(f"{C.DIM}Token display: {'on' if show_tokens else 'off'}{C.RST}\n")
                continue

            # ── Memory ────────────────────────────────────────────────────────
            elif cmd == "/remember":
                if not rest.strip():
                    print(f"{C.ERR}Usage: /remember <text>{C.RST}\n")
                    continue
                mid = save_memory(rest.strip())
                _memories[mid] = {"id": mid, "content": rest.strip()}
                _mem_texts = [m["content"] for m in _memories.values()]
                history[0]["content"] = build_system_prompt(workspace, _mem_texts)
                print(f"{C.OK}Memory saved{C.RST} {C.DIM}(id: {mid}){C.RST}\n")
                continue
            elif cmd == "/memory":
                _mems = load_memories()
                if not _mems:
                    print(f"{C.DIM}No memories saved yet.{C.RST}\n")
                else:
                    print(f"{C.DIM}Memories:{C.RST}")
                    for _mid, _m in _mems.items():
                        print(f"  {C.DIM}{_mid}{C.RST}  {_m['content']}")
                    print()
                continue
            elif cmd == "/forget":
                _fid = rest.strip()
                if not _fid:
                    print(f"{C.ERR}Usage: /forget <id>{C.RST}\n")
                    continue
                if remove_memory(_fid):
                    _memories.pop(_fid, None)
                    _mem_texts = [m["content"] for m in _memories.values()]
                    history[0]["content"] = build_system_prompt(workspace, _mem_texts)
                    print(f"{C.OK}Memory {_fid} removed.{C.RST}\n")
                else:
                    print(f"{C.ERR}Memory ID '{_fid}' not found.{C.RST}\n")
                continue

            # ── Prompt library ────────────────────────────────────────────────
            elif cmd == "/prompt-save":
                _parts = rest.split(None, 1)
                if not _parts:
                    print(f"{C.ERR}Usage: /prompt-save <name> [text]{C.RST}\n")
                    continue
                _pname = _parts[0]
                if len(_parts) > 1:
                    _ptext = _parts[1]
                elif last_user_input:
                    _ptext = last_user_input
                    print(f"{C.DIM}Saving last message as prompt '{_pname}'.{C.RST}")
                else:
                    print(f"{C.ERR}No text to save. Provide text after the name.{C.RST}\n")
                    continue
                save_prompt(_pname, _ptext)
                print(f"{C.OK}Prompt '{_pname}' saved.{C.RST}\n")
                continue
            elif cmd == "/prompt-run":
                _pname = rest.strip()
                if not _pname:
                    print(f"{C.ERR}Usage: /prompt-run <name>{C.RST}\n")
                    continue
                _stored = load_prompts()
                if _pname not in _stored:
                    print(f"{C.ERR}Prompt '{_pname}' not found.{C.RST}\n")
                    continue
                user = _stored[_pname]
                print(f"{C.DIM}Running prompt: {user}{C.RST}")
                _route_after_cmd = True
            elif cmd == "/prompts":
                _sub = rest.strip().lower()
                _stored = load_prompts()
                if _sub.startswith("remove "):
                    _pname = _sub[7:].strip()
                    if remove_prompt(_pname):
                        print(f"{C.OK}Prompt '{_pname}' removed.{C.RST}\n")
                    else:
                        print(f"{C.ERR}Prompt '{_pname}' not found.{C.RST}\n")
                elif not _stored:
                    print(f"{C.DIM}No prompts saved yet.{C.RST}\n")
                else:
                    print(f"{C.DIM}Saved prompts:{C.RST}")
                    for _n, _t in _stored.items():
                        _preview = _t[:60].replace("\n", " ")
                        print(f"  {C.OK}{_n}{C.RST}  {C.DIM}{_preview}{'…' if len(_t) > 60 else ''}{C.RST}")
                    print()
                if not _sub.startswith("remove "):
                    continue
                else:
                    continue

            # ── Aliases ───────────────────────────────────────────────────────
            elif cmd == "/alias":
                _parts = rest.split(None, 1)
                if len(_parts) < 2:
                    print(f"{C.ERR}Usage: /alias <trigger> <command>{C.RST}\n")
                    continue
                _atrig, _acmd = _parts[0], _parts[1]
                if not _atrig.startswith("/"):
                    _atrig = "/" + _atrig
                save_alias(_atrig, _acmd)
                _aliases[_atrig] = _acmd
                print(f"{C.OK}Alias saved:{C.RST} {_atrig} → {_acmd}\n")
                continue
            elif cmd == "/aliases":
                _als = load_aliases()
                if not _als:
                    print(f"{C.DIM}No aliases saved yet.{C.RST}\n")
                else:
                    print(f"{C.DIM}Aliases:{C.RST}")
                    for _t, _c in _als.items():
                        print(f"  {C.OK}{_t}{C.RST} → {_c}")
                    print()
                continue
            elif cmd == "/alias-remove":
                _atrig = rest.strip()
                if not _atrig:
                    print(f"{C.ERR}Usage: /alias-remove <trigger>{C.RST}\n")
                    continue
                if not _atrig.startswith("/"):
                    _atrig = "/" + _atrig
                if remove_alias(_atrig):
                    _aliases.pop(_atrig, None)
                    print(f"{C.OK}Alias '{_atrig}' removed.{C.RST}\n")
                else:
                    print(f"{C.ERR}Alias '{_atrig}' not found.{C.RST}\n")
                continue

            # ── Session history ───────────────────────────────────────────────
            elif cmd == "/history":
                _sessions = list_sessions()
                if not _sessions:
                    print(f"{C.DIM}No saved sessions.{C.RST}\n")
                else:
                    print(f"{C.DIM}Saved sessions:{C.RST}")
                    for _s in _sessions:
                        _nmsg = len(_s.get("messages", []))
                        print(
                            f"  {C.OK}{_s['id']}{C.RST}  "
                            f"{C.DIM}{_s.get('model', '?')}  {_nmsg} messages{C.RST}"
                        )
                    print()
                continue
            elif cmd == "/resume":
                _sid = rest.strip()
                if not _sid:
                    print(f"{C.ERR}Usage: /resume <session-id>{C.RST}\n")
                    continue
                _sdata = load_session(_sid)
                if not _sdata:
                    print(f"{C.ERR}Session '{_sid}' not found.{C.RST}\n")
                    continue
                _msgs = _sdata.get("messages", [])
                history.clear()
                history.append({"role": "system", "content": system_prompt})
                history.extend(_msgs)
                _session_id = _sid
                print(
                    f"{C.OK}Session loaded:{C.RST} {_sid}  "
                    f"{C.DIM}({len(_msgs)} messages){C.RST}\n"
                )
                continue
            elif cmd == "/export":
                _fpath = rest.strip()
                if not _fpath:
                    _fpath = f"session-{_session_id}.md"
                _out_path = Path(_fpath).expanduser().resolve()
                _lines = [f"# Session {_session_id}\n"]
                for _m in history:
                    if _m["role"] == "system":
                        continue
                    _role = "**You**" if _m["role"] == "user" else "**MahanAI**"
                    _lines.append(f"{_role}:\n\n{_m['content']}\n\n---\n")
                _out_path.write_text("\n".join(_lines), encoding="utf-8")
                print(f"{C.OK}Session exported to:{C.RST} {_out_path}\n")
                continue

            # ── Model comparison ──────────────────────────────────────────────
            elif cmd == "/compare":
                _cparts = rest.split(None, 2)
                if len(_cparts) < 3:
                    print(f"{C.ERR}Usage: /compare <model1> <model2> <message>{C.RST}\n")
                    continue
                _cm1_name, _cm2_name, _cmsg = _cparts[0], _cparts[1], _cparts[2]
                _cm1 = next((i for i, m in enumerate(AVAILABLE_MODELS) if _cm1_name.lower() in m["name"].lower() or _cm1_name.lower() in m["label"].lower()), None)
                _cm2 = next((i for i, m in enumerate(AVAILABLE_MODELS) if _cm2_name.lower() in m["name"].lower() or _cm2_name.lower() in m["label"].lower()), None)
                if _cm1 is None:
                    print(f"{C.ERR}Model '{_cm1_name}' not found. Use /models to see names.{C.RST}\n")
                    continue
                if _cm2 is None:
                    print(f"{C.ERR}Model '{_cm2_name}' not found. Use /models to see names.{C.RST}\n")
                    continue
                for _cidx, _cidx_val in [(_cm1, _cm1_name), (_cm2, _cm2_name)]:
                    _csel = AVAILABLE_MODELS[_cidx]
                    print(f"\n{C.BOT}── {_csel['label']} ──{C.RST}")
                    print(f"{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
                    _t0 = time.time()
                    if _csel["mode"] == "claude":
                        _run_claude_cli(_cmsg, model=_csel.get("claude_model"))
                    elif _csel["mode"] == "nvidia_direct" and nvidia_api_key:
                        _stream_direct(nvidia_api_key, [{"role": "user", "content": _cmsg}], NVIDIA_DIRECT_MODEL, NVIDIA_DIRECT_URL)
                    elif _csel["mode"] in ("server",) and api_key:
                        _stream_direct(api_key, [{"role": "user", "content": _cmsg}], model, NVIDIA_BASE_URL)
                    elif _csel["mode"] == "ollama":
                        _stream_direct(_csel.get("ollama_api_key") or "ollama", [{"role": "user", "content": _cmsg}], _csel["name"], _csel["ollama_url"])
                    print(f"\n{C.DIM}({time.time() - _t0:.1f}s){C.RST}")
                print()
                continue

            # ── Document index ────────────────────────────────────────────────
            elif cmd == "/index":
                _isub = rest.strip()
                if _isub == "list":
                    _sources = sorted({c["source"] for c in _index_chunks})
                    if not _sources:
                        print(f"{C.DIM}Index is empty.{C.RST}\n")
                    else:
                        print(f"{C.DIM}Indexed sources ({len(_index_chunks)} chunks):{C.RST}")
                        for _src in _sources:
                            _cnt = sum(1 for c in _index_chunks if c["source"] == _src)
                            print(f"  {C.OK}{_src}{C.RST}  {C.DIM}({_cnt} chunks){C.RST}")
                        print()
                elif _isub == "clear":
                    _index_chunks.clear()
                    save_index_documents([])
                    print(f"{C.OK}Index cleared.{C.RST}\n")
                elif _isub:
                    _ipath = Path(_isub).expanduser().resolve()
                    if not _ipath.exists():
                        print(f"{C.ERR}Path not found: {_isub}{C.RST}\n")
                    else:
                        _count = 0
                        if _ipath.is_file():
                            _count = _index_file(_ipath, _index_chunks)
                        elif _ipath.is_dir():
                            for _f in _ipath.rglob("*"):
                                if _f.is_file() and _f.suffix in (".txt", ".md", ".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp", ".cs", ".rb", ".json", ".yaml", ".toml", ".sh"):
                                    _count += _index_file(_f, _index_chunks)
                        save_index_documents(_index_chunks)
                        print(f"{C.OK}Indexed:{C.RST} {_isub}  {C.DIM}({_count} new chunks, {len(_index_chunks)} total){C.RST}\n")
                else:
                    print(f"{C.ERR}Usage: /index <path>  |  /index list  |  /index clear{C.RST}\n")
                continue

            # ── Background tasks ──────────────────────────────────────────────
            elif cmd == "/task":
                if not rest.strip():
                    print(f"{C.ERR}Usage: /task <description>{C.RST}\n")
                    continue
                _tid = uuid.uuid4().hex[:8]
                _sel = AVAILABLE_MODELS[active_model_idx]
                if _sel["mode"] == "nvidia_direct" and nvidia_api_key:
                    _tkey, _turl, _tmodel = nvidia_api_key, NVIDIA_DIRECT_URL, NVIDIA_DIRECT_MODEL
                elif _sel["mode"] in ("server",) and api_key:
                    _tkey, _turl, _tmodel = api_key, NVIDIA_BASE_URL, model
                elif _sel["mode"] == "ollama":
                    _tkey = _sel.get("ollama_api_key") or "ollama"
                    _turl = _sel["ollama_url"]
                    _tmodel = _sel["name"]
                elif _sel["mode"] == "custom":
                    _cfg = load_custom_endpoint()
                    if not _cfg:
                        print(f"{C.ERR}No custom endpoint configured.{C.RST}\n")
                        continue
                    _tkey, _turl, _tmodel = _cfg["api_key"] or "none", _cfg["url"], _cfg["model"]
                else:
                    print(f"{C.ERR}Background tasks require a non-Claude endpoint.{C.RST}\n")
                    continue
                _tasks[_tid] = {"id": _tid, "description": rest.strip(), "status": "pending", "result": ""}
                _t = threading.Thread(target=_run_task_thread, args=(_tid, rest.strip(), _tkey, _turl, _tmodel), daemon=True)
                _t.start()
                print(f"{C.OK}Task started:{C.RST} {C.DIM}{_tid}{C.RST}  Use /task-status to check.\n")
                continue
            elif cmd == "/task-status":
                if not _tasks:
                    print(f"{C.DIM}No background tasks.{C.RST}\n")
                else:
                    print(f"{C.DIM}Tasks:{C.RST}")
                    for _tid, _t_data in _tasks.items():
                        _st = _t_data["status"]
                        _st_color = C.OK if _st == "done" else C.ERR if _st == "error" else C.DIM
                        print(f"  {C.DIM}{_tid}{C.RST}  {_st_color}{_st}{C.RST}  {C.DIM}{_t_data['description'][:50]}{C.RST}")
                    print()
                continue
            elif cmd == "/task-result":
                _tid = rest.strip()
                if not _tid:
                    print(f"{C.ERR}Usage: /task-result <id>{C.RST}\n")
                    continue
                if _tid not in _tasks:
                    print(f"{C.ERR}Task '{_tid}' not found.{C.RST}\n")
                    continue
                _td = _tasks[_tid]
                if _td["status"] == "pending":
                    print(f"{C.DIM}Task still pending...{C.RST}\n")
                elif _td["status"] == "running":
                    print(f"{C.DIM}Task still running...{C.RST}\n")
                else:
                    print(f"\n{C.BOT}{C.AI_NAME}{C.RST}:\n{_td['result']}\n")
                continue

            # ── Voice ─────────────────────────────────────────────────────────
            elif cmd == "/voice":
                _vt = rest.strip().lower()
                if _vt == "on":
                    voice_enabled = True
                    print(f"{C.OK}Voice input ON{C.RST} {C.DIM}(requires SpeechRecognition + whisper){C.RST}\n")
                elif _vt == "off":
                    voice_enabled = False
                    print(f"{C.OK}Voice input OFF{C.RST}\n")
                else:
                    print(f"{C.DIM}Voice: {'on' if voice_enabled else 'off'}{C.RST}\n")
                continue

            # ── Shell init ────────────────────────────────────────────────────
            elif cmd == "/shell-init":
                _shell = rest.strip().lower()
                if _shell == "fish":
                    print(f"\n{C.DIM}Add to ~/.config/fish/config.fish:{C.RST}\n")
                    print(_SHELL_INIT_FISH)
                else:
                    print(f"\n{C.DIM}Add to ~/.bashrc or ~/.zshrc:{C.RST}\n")
                    print(_SHELL_INIT_BASH)
                continue

            if not _route_after_cmd:
                print(f"{C.ERR}Unknown command.{C.RST} Try {C.DIM}/help{C.RST}\n")
                continue

        # Track last user input for /retry
        last_user_input = user

        # Apply plan mode and effort modifiers
        effective_user = user
        if plan_mode:
            effective_user = (
                "Before responding, briefly outline your plan step by step, then execute it.\n\n"
                + user
            )

        # Inject relevant index chunks as context prefix
        if _index_chunks:
            _relevant = _search_index(user, _index_chunks, top_k=3)
            if _relevant:
                _ctx = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in _relevant)
                effective_user = f"Relevant context from indexed documents:\n{_ctx}\n\n---\n\n{effective_user}"

        selected = AVAILABLE_MODELS[active_model_idx]
        is_haiku = selected.get("claude_model") == "claude-haiku-4-5-20251001"
        effort_instr = "" if is_haiku else _EFFORT_INSTRUCTIONS.get(current_effort, "")
        codex_effort = _EFFORT_CODEX.get(current_effort, "medium")

        def _post_reply(reply: str) -> None:
            """Run after each successful AI reply: highlight, token count, auto-save."""
            nonlocal last_ai_reply
            last_ai_reply = reply
            if highlight_enabled:
                _highlight_response(reply)
            if show_tokens:
                _tok = _estimate_tokens(reply)
                print(f"{C.DIM}~{_tok} tokens{C.RST}\n")
            _auto_save_session(_session_id, history, selected["label"])

        # Route based on selected model
        if selected["mode"] == "claude":
            claude_model = selected.get("claude_model")
            print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
            _claude_reply = _run_claude_cli(effective_user, model=claude_model, effort_instruction=effort_instr)
            print("\n")
            _post_reply(_claude_reply)
            continue

        if selected["mode"] == "codex_direct":
            creds = _get_codex_direct_token()
            if not creds:
                print(
                    f"{C.ERR}Not signed in to OpenAI.{C.RST} Run {C.OK}/codex-login{C.RST} first.\n"
                )
                continue
            codex_access, codex_account_id = creds
            history.append({"role": "user", "content": effective_user})
            print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
            try:
                reply = _stream_wham(codex_access, codex_account_id, history, selected["name"], codex_effort, workspace)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                print()
                print(f"{C.ERR}[Codex Direct error]{C.RST} {e}\n")
                history.pop()
                continue
            print("\n")
            history.append({"role": "assistant", "content": reply})
            _post_reply(reply)
            continue

        if selected["mode"] == "codex_indirect":
            indirect_token = _load_codex_indirect_key()
            if indirect_token:
                history.append({"role": "user", "content": effective_user})
                print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
                try:
                    reply = _stream_wham(indirect_token, None, history, selected["name"], codex_effort, workspace)
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    print()
                    print(f"{C.ERR}[Codex Indirect error]{C.RST} {e}\n")
                    history.pop()
                    continue
                print("\n")
                history.append({"role": "assistant", "content": reply})
                _post_reply(reply)
            else:
                print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
                _run_codex_cli(effective_user, model=selected["name"])
                print("\n")
            continue

        if selected["mode"] == "ollama":
            history.append({"role": "user", "content": effective_user})
            print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
            try:
                reply = run_turn(
                    client,
                    selected.get("ollama_api_key") or "ollama",
                    selected["name"],
                    history,
                    workspace,
                    selected["ollama_url"],
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                print()
                print(f"{C.ERR}[Ollama error]{C.RST} {e}\n")
                history.pop()
                continue
            print("\n")
            history.append({"role": "assistant", "content": reply})
            _post_reply(reply)
            continue

        if selected["mode"] == "custom":
            custom_cfg = load_custom_endpoint()
            if not custom_cfg:
                print(
                    f"{C.ERR}No custom endpoint configured.{C.RST} Use {C.OK}/custom{C.RST} to set one.\n"
                )
                continue
            history.append({"role": "user", "content": effective_user})
            print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
            try:
                reply = run_turn(
                    client,
                    custom_cfg["api_key"] or "none",
                    custom_cfg["model"],
                    history,
                    workspace,
                    custom_cfg["url"],
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                print()
                print(f"{C.ERR}[Custom endpoint error]{C.RST} {e}\n")
                history.pop()
                continue
            print("\n")
            history.append({"role": "assistant", "content": reply})
            _post_reply(reply)
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

        history.append({"role": "user", "content": effective_user})
        print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
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
        _post_reply(reply)