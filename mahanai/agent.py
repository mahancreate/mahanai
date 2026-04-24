"""Chat agent loop with OpenAI-compatible NVIDIA NIM tools API."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
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
    load_theme,
    load_custom_theme_path,
    load_custom_theme_info,
    save_custom_theme_info,
    clear_custom_theme,
    resolve_api_key,
    save_api_key,
    save_codex_token,
    save_custom_endpoint,
    save_nvidia_api_key,
    save_theme,
)
from mahanai.system_info import describe_runtime
from mahanai.tools import TOOLS, execute_tool, normalize_tool_arguments_json

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
        console.print(f"[bold]  Super 2.0  |  {model_label}  |[/bold]")
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

    with httpx.Client(timeout=120.0) as client:
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
    with httpx.Client(timeout=120.0) as client:
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
    "You are MahanAI, a capable coding and system assistant (Super 2.0). "
    "Do not refer to yourself as Claude or Claude Code — you are MahanAI. "
    "Super 2.0 is the codename for this release of MahanAI."
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


def _run_claude_cli(prompt: str, model: str | None = None, effort_instruction: str = "") -> None:
    """Stream a prompt to the Claude CLI via stream-json events."""
    full_prompt = f"{effort_instruction}\n\n{prompt}".strip() if effort_instruction else prompt
    cmd = _resolve_cli("claude") + [
        "--system-prompt", _CLAUDE_IDENTITY,
        "-p", full_prompt,
        "--output-format", "stream-json",
        "--verbose",
    ]
    if model:
        cmd += ["--model", model]
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
                elif etype == "text":
                    text = event.get("text", "")
                    if text:
                        print(text, end="", flush=True)
                elif etype == "assistant":
                    message = event.get("message", {})
                    for block in message.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                print(text, end="", flush=True)
            except (json.JSONDecodeError, KeyError):
                pass
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().strip()
            if err:
                print(f"\n{C.ERR}[Claude CLI error]{C.RST} {err}")
    except FileNotFoundError:
        print(f"{C.ERR}[Claude CLI not found] Make sure 'claude' is installed and on your PATH.{C.RST}")



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
        f"  /effort <level>         Set reasoning effort: low, medium, high, very-high\n"
        f"                          (disabled for Claude Haiku 4.5)\n"
        f"  /plan on|off            Toggle plan-before-respond mode\n"
        f"  /themes                 List available themes\n"
        f"  /themes <name>          Switch theme: midnight, light,\n"
        f"                            midnight-cb, light-cb\n"
        f"  /theme-load <path>      Load a custom .mai theme file\n"
        f"  /theme-unload           Remove the active custom .mai theme\n"
        f"  /approvals              Show stored Always Allow rules\n"
        f"  /approvals clear        Remove all Always Allow rules\n"
        f"  /codex-login            Sign in to OpenAI via browser (Codex Direct)\n"
        f"  /codex-logout           Remove saved OpenAI Codex credentials\n"
        f"  /custom [url [model [key]]]  Configure a custom OpenAI-compatible endpoint\n"
        f"  /custom clear           Remove saved custom endpoint\n"
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

    system_prompt = build_system_prompt(workspace)

    history: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    active_model_idx = next(
        (i for i, m in enumerate(AVAILABLE_MODELS) if m.get("claude_model") == "claude-haiku-4-5-20251001"),
        0,
    )
    current_effort = "medium"
    plan_mode = False
    C.apply_theme(load_theme())
    _register_and_apply_saved_mai_theme()
    print_startup_banner(AVAILABLE_MODELS[active_model_idx]["label"], compact=compact)
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
            print(f"{C.USER}{C.USER_NAME}{C.RST}: ", end="", flush=True)
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
            print(f"{C.ERR}Unknown command.{C.RST} Try {C.DIM}/help{C.RST}\n")
            continue

        # Apply plan mode and effort modifiers
        effective_user = user
        if plan_mode:
            effective_user = (
                "Before responding, briefly outline your plan step by step, then execute it.\n\n"
                + user
            )
        selected = AVAILABLE_MODELS[active_model_idx]
        is_haiku = selected.get("claude_model") == "claude-haiku-4-5-20251001"
        effort_instr = "" if is_haiku else _EFFORT_INSTRUCTIONS.get(current_effort, "")
        codex_effort = _EFFORT_CODEX.get(current_effort, "medium")

        # Route based on selected model
        if selected["mode"] == "claude":
            claude_model = selected.get("claude_model")
            print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
            _run_claude_cli(effective_user, model=claude_model, effort_instruction=effort_instr)
            print("\n")
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
            else:
                print(f"\n{C.BOT}{C.AI_NAME}{C.RST}: ", end="", flush=True)
                _run_codex_cli(effective_user, model=selected["name"])
                print("\n")
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