"""
MahanAI Gateway Server
Unified local endpoint that aggregates all configured providers and exposes
them as either an OpenAI-compatible or Anthropic-compatible API.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx

from mahanai import colors as C

# ── External API constants ────────────────────────────────────────────────────

ANTHROPIC_API_URL  = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION  = "2023-06-01"
NVIDIA_BASE_URL    = "http://89.167.0.111:8000/v1"
NVIDIA_DIRECT_URL  = "https://integrate.api.nvidia.com/v1"
WHAM_URL           = "https://chatgpt.com/backend-api/wham/responses"

# ── Model → backend routing table ────────────────────────────────────────────
# Each entry: model_id → (mode, backend_model_name)
# Modes: "server" | "nvidia_direct" | "claude" | "codex_direct" | "custom"

_ROUTES: dict[str, tuple[str, str]] = {
    "mahanai/mahanai":             ("server",        "mahanai/mahanai"),
    "meta/llama-3.3-70b-instruct": ("nvidia_direct", "meta/llama-3.3-70b-instruct"),
    "claude-opus-4-7":             ("claude",        "claude-opus-4-7"),
    "claude-sonnet-4-6":           ("claude",        "claude-sonnet-4-6"),
    "claude-haiku-4-5-20251001":   ("claude",        "claude-haiku-4-5-20251001"),
    "gpt-5.4":                     ("codex_direct",  "gpt-5.4"),
    "gpt-5.2-codex":               ("codex_direct",  "gpt-5.2-codex"),
    "gpt-5.1-codex-max":           ("codex_direct",  "gpt-5.1-codex-max"),
    "gpt-5.4-mini":                ("codex_direct",  "gpt-5.4-mini"),
    "gpt-5.3-codex":               ("codex_direct",  "gpt-5.3-codex"),
    "gpt-5.2":                     ("codex_direct",  "gpt-5.2"),
    "gpt-5.1-codex-mini":          ("codex_direct",  "gpt-5.1-codex-mini"),
}

_MODEL_DISPLAY = {
    "mahanai/mahanai":             ("MahanAI Super (legacy)", "NVIDIA NIM"),
    "meta/llama-3.3-70b-instruct": ("Llama 3.3 70B",          "NVIDIA NIM"),
    "claude-opus-4-7":             ("Claude Opus 4",           "Anthropic"),
    "claude-sonnet-4-6":           ("Claude Sonnet 4.6",       "Anthropic"),
    "claude-haiku-4-5-20251001":   ("Claude Haiku 4.5",        "Anthropic"),
    "gpt-5.4":                     ("GPT-5.4",                 "OpenAI Codex"),
    "gpt-5.2-codex":               ("GPT-5.2-Codex",           "OpenAI Codex"),
    "gpt-5.1-codex-max":           ("GPT-5.1-Codex-Max",       "OpenAI Codex"),
    "gpt-5.4-mini":                ("GPT-5.4-Mini",            "OpenAI Codex"),
    "gpt-5.3-codex":               ("GPT-5.3-Codex",           "OpenAI Codex"),
    "gpt-5.2":                     ("GPT-5.2",                 "OpenAI Codex"),
    "gpt-5.1-codex-mini":          ("GPT-5.1-Codex-Mini",      "OpenAI Codex"),
}


# ── Server configuration ──────────────────────────────────────────────────────

class ServerConfig:
    def __init__(
        self,
        server_type: str,           # "openai" or "anthropic"
        port: int,
        gateway_key: str | None,    # Bearer key clients must send (None = open access)
        api_key: str | None,        # NVIDIA server-mode backend key
        anthropic_key: str | None,  # Anthropic API key (x-api-key header)
        nvidia_api_key: str | None,
        codex_token: dict | None,
        custom_endpoint: dict | None,
    ):
        self.server_type      = server_type
        self.port             = port
        self.gateway_key      = gateway_key
        self.api_key          = api_key
        self.anthropic_key    = anthropic_key  # not inherited from api_key
        self.nvidia_api_key   = nvidia_api_key
        self.codex_token      = codex_token
        self.custom_endpoint  = custom_endpoint


# ── Format converters ─────────────────────────────────────────────────────────

def _oai_to_anth_body(body: dict, model: str) -> dict:
    """Convert an OpenAI chat/completions request body → Anthropic messages body."""
    system_parts: list[str] = []
    messages: list[dict] = []
    for msg in body.get("messages", []):
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
        else:
            messages.append({"role": role, "content": content})

    result: dict = {
        "model":      model,
        "messages":   messages,
        "max_tokens": body.get("max_tokens") or 8096,
    }
    if system_parts:
        result["system"] = "\n".join(system_parts)
    if body.get("stream"):
        result["stream"] = True
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    return result


def _anth_to_oai_body(body: dict, model: str) -> dict:
    """Convert an Anthropic messages request body → OpenAI chat/completions body."""
    messages: list[dict] = []
    if "system" in body:
        messages.append({"role": "system", "content": body["system"]})
    messages.extend(body.get("messages", []))

    result: dict = {
        "model":      model,
        "messages":   messages,
        "max_tokens": body.get("max_tokens") or 8096,
    }
    if body.get("stream"):
        result["stream"] = True
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    return result


def _anth_resp_to_oai(anth: dict, model: str) -> dict:
    content = "".join(
        b.get("text", "") for b in anth.get("content", []) if b.get("type") == "text"
    )
    usage = anth.get("usage", {})
    return {
        "id":      anth.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": anth.get("stop_reason", "stop")}],
        "usage": {
            "prompt_tokens":     usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens":      usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def _oai_resp_to_anth(oai: dict, model: str) -> dict:
    choice  = oai.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage   = oai.get("usage", {})
    return {
        "id":            f"msg_{uuid.uuid4().hex[:24]}",
        "type":          "message",
        "role":          "assistant",
        "model":         model,
        "content":       [{"type": "text", "text": content}],
        "stop_reason":   choice.get("finish_reason", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── HTTP handler factory ──────────────────────────────────────────────────────

def _make_handler(cfg: ServerConfig) -> type:

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *_: object) -> None:
            pass  # suppress default Apache-style logging

        # ── Auth ──────────────────────────────────────────────────────────────

        def _check_auth(self) -> bool:
            if not cfg.gateway_key:
                return True  # open access
            auth  = self.headers.get("Authorization", "")
            token = auth[7:].strip() if auth.startswith("Bearer ") else ""
            return token == cfg.gateway_key

        def _auth_error(self) -> None:
            if cfg.server_type == "openai":
                self._json(401, {"error": {
                    "message": "Invalid API key.",
                    "type":    "invalid_request_error",
                    "code":    "invalid_api_key",
                }})
            else:
                self._json(401, {"type": "error", "error": {
                    "type":    "authentication_error",
                    "message": "Invalid API key.",
                }})

        # ── Routing ───────────────────────────────────────────────────────────

        def do_GET(self) -> None:
            clean = self.path.split("?")[0].rstrip("/")
            # Web UI: no auth required for the HTML page itself
            if clean in ("", "/", "/ui"):
                self._serve_web_ui()
                return
            if not self._check_auth():
                self._auth_error()
                return
            if clean in ("/v1/models", "/models"):
                self._handle_models()
            else:
                self._json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

        def do_POST(self) -> None:
            if not self._check_auth():
                self._auth_error()
                return
            path = self.path.split("?")[0].rstrip("/")
            if cfg.server_type == "openai":
                if path in ("/v1/chat/completions", "/chat/completions"):
                    self._handle_oai_chat()
                else:
                    self._json(404, {"error": {"message": f"Unknown endpoint: {self.path}", "type": "invalid_request_error"}})
            else:  # anthropic
                if path in ("/v1/messages", "/messages"):
                    self._handle_anth_messages()
                else:
                    self._json(404, {"error": {"type": "not_found_error", "message": f"Unknown endpoint: {self.path}"}})

        # ── /v1/models ────────────────────────────────────────────────────────

        def _handle_models(self) -> None:
            created = int(time.time())
            if cfg.server_type == "openai":
                data = [
                    {"id": mid, "object": "model", "created": created, "owned_by": "mahanai"}
                    for mid in _ROUTES
                ]
                if cfg.custom_endpoint:
                    data.append({"id": cfg.custom_endpoint.get("model", "custom"),
                                 "object": "model", "created": created, "owned_by": "custom"})
                self._json(200, {"object": "list", "data": data})
            else:
                data_anth = [
                    {
                        "id":           mid,
                        "type":         "model",
                        "display_name": _MODEL_DISPLAY.get(mid, (mid, ""))[0],
                        "created_at":   "2025-01-01T00:00:00Z",
                    }
                    for mid in _ROUTES
                ]
                self._json(200, {"data": data_anth, "has_more": False,
                                 "first_id": data_anth[0]["id"] if data_anth else None,
                                 "last_id":  data_anth[-1]["id"] if data_anth else None})

        # ── Web UI ────────────────────────────────────────────────────────────

        def _serve_web_ui(self) -> None:
            port = cfg.port
            _CLAUDE_FIRST = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"]
            _ordered_models = _CLAUDE_FIRST + [m for m in _ROUTES if m not in _CLAUDE_FIRST]
            model_options = "\n".join(
                f'<option value="{mid}">{_MODEL_DISPLAY.get(mid, (mid,""))[0]}'
                f'  —  {_MODEL_DISPLAY.get(mid, ("","?"))[1]}</option>'
                for mid in _ordered_models
            )
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MahanAI</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#08080f;
  --bg2:#0e0e1a;
  --surface:#13131f;
  --surface2:#1a1a2e;
  --border:#1e1e35;
  --border2:#2a2a4a;
  --text:#e8e8f0;
  --text2:#a0a0c0;
  --dim:#5a5a80;
  --accent1:#7c3aed;
  --accent2:#06b6d4;
  --user-bg:linear-gradient(135deg,#7c3aed,#5b21b6);
  --ai-bg:var(--surface);
  --green:#10b981;
  --red:#ef4444;
  --yellow:#f59e0b;
  --code-bg:#0a0a18;
  --radius:14px;
  --shadow:0 4px 24px rgba(0,0,0,.5);
}}
html,body{{height:100%;overflow:hidden}}
body{{
  background:var(--bg);
  color:var(--text);
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
  display:flex;flex-direction:column;
  -webkit-font-smoothing:antialiased;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:99px}}
::-webkit-scrollbar-thumb:hover{{background:var(--dim)}}

/* ── Header ── */
#header{{
  display:flex;align-items:center;gap:12px;
  padding:0 20px;height:56px;
  background:rgba(8,8,15,.85);
  backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  flex-shrink:0;z-index:10;
  position:relative;
}}
#header::after{{
  content:'';position:absolute;bottom:-1px;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent1),var(--accent2),transparent);
  opacity:.4;
}}
.logo{{display:flex;align-items:center;gap:8px;text-decoration:none}}
.logo-gem{{
  width:28px;height:28px;
  background:linear-gradient(135deg,var(--accent1),var(--accent2));
  border-radius:7px;
  display:flex;align-items:center;justify-content:center;
  font-size:14px;box-shadow:0 0 12px rgba(124,58,237,.4);
  flex-shrink:0;
}}
.logo-text{{font-size:.95rem;font-weight:700;letter-spacing:-.01em;
  background:linear-gradient(120deg,#c4b5fd,var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.logo-version{{font-size:.65rem;font-weight:500;color:var(--dim);
  -webkit-text-fill-color:var(--dim);margin-left:2px;align-self:flex-end;padding-bottom:2px}}

.header-spacer{{flex:1}}

.model-wrap{{
  display:flex;align-items:center;gap:8px;
  background:var(--surface);border:1px solid var(--border2);
  border-radius:8px;padding:5px 10px;
}}
.model-wrap label{{font-size:.7rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.08em;color:var(--dim);white-space:nowrap}}
#model{{
  background:transparent;color:var(--text);border:none;
  font-size:.8rem;font-family:inherit;cursor:pointer;outline:none;
  max-width:220px;
}}
#model option{{background:#1a1a2e}}

.btn-new{{
  display:flex;align-items:center;gap:5px;
  background:transparent;border:1px solid var(--border2);
  color:var(--text2);border-radius:8px;padding:5px 10px;
  font-size:.78rem;font-family:inherit;cursor:pointer;
  transition:all .15s;white-space:nowrap;
}}
.btn-new:hover{{border-color:var(--accent1);color:var(--text);background:rgba(124,58,237,.12)}}
.btn-new svg{{width:13px;height:13px;stroke:currentColor;fill:none;stroke-width:2}}

#status-dot{{
  width:7px;height:7px;border-radius:50%;
  background:var(--dim);flex-shrink:0;transition:background .3s;
}}
#status-dot.active{{background:var(--green);box-shadow:0 0 6px var(--green)}}

/* ── Main ── */
#main{{flex:1;overflow:hidden;display:flex;flex-direction:column}}
#chat{{
  flex:1;overflow-y:auto;
  padding:32px 0 16px;
  display:flex;flex-direction:column;
  scroll-behavior:smooth;
}}
#chat-inner{{
  width:100%;max-width:780px;margin:0 auto;
  padding:0 20px;
  display:flex;flex-direction:column;gap:6px;
}}

/* ── Welcome ── */
#welcome{{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;flex:1;gap:16px;padding:40px 20px;
  animation:fadeUp .4s ease;
}}
.welcome-gem{{
  width:60px;height:60px;border-radius:16px;
  background:linear-gradient(135deg,var(--accent1),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:28px;
  box-shadow:0 0 30px rgba(124,58,237,.35),0 0 60px rgba(6,182,212,.15);
}}
.welcome-title{{
  font-size:1.6rem;font-weight:700;letter-spacing:-.02em;
  background:linear-gradient(120deg,#c4b5fd,var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.welcome-sub{{color:var(--text2);font-size:.9rem;text-align:center;max-width:380px;line-height:1.6}}
.welcome-chips{{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:4px}}
.chip{{
  background:var(--surface);border:1px solid var(--border2);
  border-radius:8px;padding:7px 12px;font-size:.78rem;color:var(--text2);
  cursor:pointer;transition:all .15s;
}}
.chip:hover{{border-color:var(--accent1);color:var(--text);background:rgba(124,58,237,.1)}}

/* ── Messages ── */
.msg-row{{
  display:flex;gap:10px;align-items:flex-start;
  animation:fadeUp .22s ease;max-width:100%;
}}
.msg-row.user{{flex-direction:row-reverse}}

.avatar{{
  width:30px;height:30px;border-radius:8px;
  display:flex;align-items:center;justify-content:center;
  font-size:13px;flex-shrink:0;margin-top:2px;font-weight:700;
}}
.avatar.ai{{
  background:linear-gradient(135deg,var(--accent1),var(--accent2));
  box-shadow:0 0 10px rgba(124,58,237,.3);
}}
.avatar.user{{background:var(--surface2);border:1px solid var(--border2);color:var(--text2)}}

.msg-body{{display:flex;flex-direction:column;gap:3px;max-width:calc(100% - 44px)}}
.msg-meta{{
  font-size:.68rem;color:var(--dim);
  padding:0 4px;display:flex;gap:8px;align-items:center;
}}
.msg-row.user .msg-meta{{justify-content:flex-end}}

.bubble{{
  padding:11px 15px;border-radius:var(--radius);
  line-height:1.65;font-size:.88rem;word-wrap:break-word;
  position:relative;
}}
.bubble.user{{
  background:linear-gradient(135deg,#7c3aed,#5b21b6);
  color:#fff;border-radius:var(--radius) var(--radius) 4px var(--radius);
  box-shadow:0 2px 12px rgba(124,58,237,.35);
  white-space:pre-wrap;
}}
.bubble.ai{{
  background:var(--surface);border:1px solid var(--border2);
  color:var(--text);border-radius:4px var(--radius) var(--radius) var(--radius);
}}

/* copy button on hover */
.bubble{{cursor:default}}
.copy-btn{{
  align-self:flex-end;
  background:var(--surface2);border:1px solid var(--border2);
  color:var(--dim);border-radius:5px;padding:3px 9px;
  font-size:.68rem;cursor:pointer;opacity:0;transition:opacity .15s;
  font-family:inherit;margin-top:2px;
}}
.msg-body:hover .copy-btn{{opacity:1}}
.copy-btn:hover{{color:var(--text);border-color:var(--accent1)}}

/* ── Markdown ── */
.bubble.ai p{{margin-bottom:.6em}}
.bubble.ai p:last-child{{margin-bottom:0}}
.bubble.ai strong{{color:#fff;font-weight:600}}
.bubble.ai em{{color:#c4b5fd}}
.bubble.ai h1,.bubble.ai h2,.bubble.ai h3{{
  font-size:1rem;font-weight:600;color:#fff;
  margin:.7em 0 .3em;border-bottom:1px solid var(--border2);padding-bottom:.25em;
}}
.bubble.ai h1{{font-size:1.15rem}}
.bubble.ai ul,.bubble.ai ol{{padding-left:1.4em;margin:.3em 0}}
.bubble.ai li{{margin:.15em 0}}
.bubble.ai a{{color:var(--accent2);text-decoration:none}}
.bubble.ai a:hover{{text-decoration:underline}}
.bubble.ai hr{{border:none;border-top:1px solid var(--border2);margin:.6em 0}}
.bubble.ai blockquote{{
  border-left:3px solid var(--accent1);margin:.4em 0;
  padding:.2em .8em;color:var(--text2);font-style:italic;
}}
.inline-code{{
  font-family:'Cascadia Code','Fira Code','JetBrains Mono',monospace;
  background:var(--code-bg);border:1px solid var(--border2);
  border-radius:4px;padding:1px 5px;font-size:.82em;color:#a5f3fc;
}}
.code-block{{
  background:var(--code-bg);border:1px solid var(--border2);
  border-radius:10px;margin:.5em 0;overflow:hidden;
}}
.code-header{{
  display:flex;align-items:center;justify-content:space-between;
  padding:6px 12px;background:rgba(255,255,255,.03);
  border-bottom:1px solid var(--border2);
}}
.code-lang{{font-size:.7rem;color:var(--dim);font-family:monospace;text-transform:uppercase;letter-spacing:.06em}}
.code-copy{{
  background:transparent;border:1px solid var(--border2);
  color:var(--dim);border-radius:4px;padding:2px 8px;
  font-size:.68rem;cursor:pointer;font-family:inherit;transition:all .15s;
}}
.code-copy:hover{{color:var(--text);border-color:var(--accent2)}}
.code-copy.copied{{color:var(--green);border-color:var(--green)}}
.code-block pre{{
  padding:14px;overflow-x:auto;font-size:.82rem;
  font-family:'Cascadia Code','Fira Code','JetBrains Mono',monospace;
  line-height:1.6;color:#e2e8f0;
}}

/* ── Typing indicator ── */
.typing-indicator{{display:flex;gap:4px;padding:14px 16px;align-items:center}}
.typing-dot{{
  width:7px;height:7px;border-radius:50%;
  background:var(--dim);
  animation:bounce .9s ease-in-out infinite;
}}
.typing-dot:nth-child(2){{animation-delay:.15s}}
.typing-dot:nth-child(3){{animation-delay:.3s}}
@keyframes bounce{{
  0%,60%,100%{{transform:translateY(0)}}
  30%{{transform:translateY(-6px);background:var(--accent1)}}
}}

/* ── Divider ── */
.session-divider{{
  display:flex;align-items:center;gap:10px;
  color:var(--dim);font-size:.72rem;padding:4px 0;
}}
.session-divider::before,.session-divider::after{{
  content:'';flex:1;height:1px;background:var(--border);
}}

/* ── Composer ── */
#composer{{
  padding:12px 20px 16px;flex-shrink:0;
  background:linear-gradient(transparent,var(--bg) 20%);
}}
.composer-box{{
  max-width:780px;margin:0 auto;
  background:var(--surface);border:1px solid var(--border2);
  border-radius:14px;padding:10px 12px;
  display:flex;align-items:flex-end;gap:8px;
  box-shadow:0 4px 24px rgba(0,0,0,.4);
  transition:border-color .2s;
}}
.composer-box:focus-within{{border-color:rgba(124,58,237,.6);box-shadow:0 4px 24px rgba(124,58,237,.15)}}
#input{{
  flex:1;background:transparent;color:var(--text);
  border:none;outline:none;resize:none;
  font-size:.88rem;font-family:inherit;line-height:1.6;
  max-height:200px;padding:2px 0;
}}
#input::placeholder{{color:var(--dim)}}
.composer-actions{{display:flex;align-items:center;gap:6px;flex-shrink:0}}
#send{{
  width:34px;height:34px;border-radius:9px;border:none;
  background:linear-gradient(135deg,var(--accent1),#5b21b6);
  color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;transition:all .15s;box-shadow:0 2px 8px rgba(124,58,237,.4);
}}
#send:hover{{filter:brightness(1.15);box-shadow:0 4px 14px rgba(124,58,237,.5)}}
#send:disabled{{opacity:.4;cursor:default;filter:none;box-shadow:none}}
#send svg{{width:16px;height:16px;stroke:#fff;fill:none;stroke-width:2.5}}
.composer-hint{{
  text-align:center;font-size:.68rem;color:var(--dim);margin-top:6px;
}}
.composer-hint kbd{{
  background:var(--surface2);border:1px solid var(--border2);
  border-radius:3px;padding:1px 4px;font-size:.65rem;font-family:monospace;
}}

@keyframes fadeUp{{
  from{{opacity:0;transform:translateY(6px)}}
  to{{opacity:1;transform:translateY(0)}}
}}
</style>
</head>
<body>

<div id="header">
  <a class="logo" href="#">
    <div class="logo-gem">◈</div>
    <span class="logo-text">MahanAI<span class="logo-version">Max 2.0</span></span>
  </a>
  <div class="header-spacer"></div>
  <div class="model-wrap">
    <label for="model">Model</label>
    <select id="model">{model_options}</select>
  </div>
  <button class="btn-new" id="btn-new" title="Start a new conversation">
    <svg viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
    New chat
  </button>
  <div id="status-dot" title="Idle"></div>
</div>

<div id="main">
  <div id="chat">
    <div id="chat-inner">
      <div id="welcome">
        <div class="welcome-gem">◈</div>
        <div class="welcome-title">MahanAI Max 2.0</div>
        <div class="welcome-sub">Your AI assistant is ready. Ask anything or try one of these prompts.</div>
        <div class="welcome-chips">
          <div class="chip" data-prompt="Explain how async/await works in Python">Explain async/await</div>
          <div class="chip" data-prompt="Write a bash script to find the 10 largest files in a directory">Find large files script</div>
          <div class="chip" data-prompt="What are the key differences between REST and GraphQL?">REST vs GraphQL</div>
          <div class="chip" data-prompt="Give me a concise git cheat sheet">Git cheat sheet</div>
        </div>
      </div>
    </div>
  </div>

  <div id="composer">
    <div class="composer-box">
      <textarea id="input" rows="1" placeholder="Message MahanAI…"></textarea>
      <div class="composer-actions">
        <button id="send" title="Send (Enter)">
          <svg viewBox="0 0 24 24"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
        </button>
      </div>
    </div>
    <div class="composer-hint"><kbd>Enter</kbd> to send &nbsp;·&nbsp; <kbd>Shift+Enter</kbd> for newline</div>
  </div>
</div>

<script>
(function(){{
'use strict';
const chatInner = document.getElementById('chat-inner');
const chatEl    = document.getElementById('chat');
const inputEl   = document.getElementById('input');
const sendBtn   = document.getElementById('send');
const modelSel  = document.getElementById('model');
const statusDot = document.getElementById('status-dot');
const welcome   = document.getElementById('welcome');
let history = [];
let streaming = false;

// ── Markdown renderer ────────────────────────────────────────────────────────
function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}}

function renderMarkdown(raw){{
  let s = raw;
  // fenced code blocks
  s = s.replace(/```([\\w.-]*)\\n?([\\s\\S]*?)```/g, (_,lang,code)=>{{
    const l = esc(lang.trim());
    const c = esc(code.trimEnd());
    return `<div class="code-block"><div class="code-header"><span class="code-lang">${{l||'code'}}</span><button class="code-copy" onclick="copyCode(this)">Copy</button></div><pre>${{c}}</pre></div>`;
  }});
  // inline code
  s = s.replace(/`([^`\\n]+)`/g, (_,c)=>`<span class="inline-code">${{esc(c)}}</span>`);
  // headings
  s = s.replace(/^### (.+)$/gm, (_,t)=>`<h3>${{t}}</h3>`);
  s = s.replace(/^## (.+)$/gm,  (_,t)=>`<h2>${{t}}</h2>`);
  s = s.replace(/^# (.+)$/gm,   (_,t)=>`<h1>${{t}}</h1>`);
  // blockquote
  s = s.replace(/^> (.+)$/gm, (_,t)=>`<blockquote>${{t}}</blockquote>`);
  // hr
  s = s.replace(/^(-{{3,}}|\\*{{3,}})$/gm,'<hr>');
  // bold / italic
  s = s.replace(/\\*\\*\\*(.+?)\\*\\*\\*/g, (_,t)=>`<strong><em>${{t}}</em></strong>`);
  s = s.replace(/\\*\\*(.+?)\\*\\*/g,       (_,t)=>`<strong>${{t}}</strong>`);
  s = s.replace(/\\*(.+?)\\*/g,             (_,t)=>`<em>${{t}}</em>`);
  s = s.replace(/__(.+?)__/g,               (_,t)=>`<strong>${{t}}</strong>`);
  s = s.replace(/_([^_\\n]+)_/g,            (_,t)=>`<em>${{t}}</em>`);
  // links
  s = s.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, (_,txt,url)=>`<a href="${{esc(url)}}" target="_blank" rel="noopener">${{esc(txt)}}</a>`);
  // unordered lists
  s = s.replace(/((?:^[ \\t]*[-*+] .+\\n?)+)/gm, block=>{{
    const items = block.match(/^[ \\t]*[-*+] (.+)$/gm)||[];
    return '<ul>'+items.map(l=>`<li>${{l.replace(/^[ \\t]*[-*+] /,'')}}</li>`).join('')+'</ul>';
  }});
  // ordered lists
  s = s.replace(/((?:^\\d+\\. .+\\n?)+)/gm, block=>{{
    const items = block.match(/^\\d+\\. (.+)$/gm)||[];
    return '<ol>'+items.map(l=>`<li>${{l.replace(/^\\d+\\. /,'')}}</li>`).join('')+'</ol>';
  }});
  // paragraphs: double newline → <p>
  s = s.split(/\\n\\n+/).map(para=>{{
    const t = para.trim();
    if(!t) return '';
    if(/^<(h[1-6]|ul|ol|blockquote|pre|hr|div)/.test(t)) return t;
    return `<p>${{t.replace(/\\n/g,'<br>')}}</p>`;
  }}).join('');
  return s;
}}

window.copyCode = function(btn){{
  const pre = btn.closest('.code-block').querySelector('pre');
  navigator.clipboard.writeText(pre.textContent).then(()=>{{
    btn.textContent='Copied!'; btn.classList.add('copied');
    setTimeout(()=>{{btn.textContent='Copy';btn.classList.remove('copied');}},1800);
  }});
}};

// ── Time ─────────────────────────────────────────────────────────────────────
function timeStr(){{
  const d=new Date(); return d.getHours().toString().padStart(2,'0')+':'+d.getMinutes().toString().padStart(2,'0');
}}

// ── Add message row ───────────────────────────────────────────────────────────
function addRow(role, html, raw){{
  welcome && (welcome.style.display='none');
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;

  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + (role==='user'?'user':'ai');
  avatar.textContent = role==='user' ? 'U' : '◈';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = (role==='user'?'You':'MahanAI') + ' · ' + timeStr();

  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + (role==='user'?'user':'ai');

  if(role==='user'){{
    bubble.textContent = raw;
    const cb = document.createElement('button');
    cb.className='copy-btn';cb.textContent='Copy';
    cb.onclick=()=>{{navigator.clipboard.writeText(raw);cb.textContent='✓';setTimeout(()=>cb.textContent='Copy',1500);}};
    body.appendChild(meta);
    body.appendChild(bubble);
    body.appendChild(cb);
  }} else {{
    bubble.innerHTML = html;
    const cb = document.createElement('button');
    cb.className='copy-btn';cb.textContent='Copy';
    cb.onclick=()=>{{navigator.clipboard.writeText(bubble.innerText);cb.textContent='✓';setTimeout(()=>cb.textContent='Copy',1500);}};
    body.appendChild(meta);
    body.appendChild(bubble);
    body.appendChild(cb);
  }}
  row.appendChild(avatar);
  row.appendChild(body);
  chatInner.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}}

function addTyping(){{
  welcome && (welcome.style.display='none');
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.id = 'typing-row';
  const avatar = document.createElement('div');
  avatar.className = 'avatar ai'; avatar.textContent='◈';
  const body = document.createElement('div');
  body.className='msg-body';
  const bubble = document.createElement('div');
  bubble.className='bubble ai';
  bubble.innerHTML='<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
  body.appendChild(bubble);
  row.appendChild(avatar); row.appendChild(body);
  chatInner.appendChild(row);
  chatEl.scrollTop=chatEl.scrollHeight;
  return row;
}}

// ── Send ──────────────────────────────────────────────────────────────────────
async function sendMessage(){{
  const text = inputEl.value.trim();
  if(!text||streaming) return;
  inputEl.value=''; inputEl.style.height='';
  streaming=true; sendBtn.disabled=true;
  statusDot.classList.add('active'); statusDot.title='Thinking…';

  addRow('user', null, text);
  history.push({{role:'user',content:text}});

  const model = modelSel.value;
  const isAnth = model.startsWith('claude-');
  const endpoint = isAnth ? '/anthropic/v1/messages' : '/v1/chat/completions';
  const body = isAnth
    ? {{model, messages:history.filter(m=>m.role!=='system'), stream:true, max_tokens:8096}}
    : {{model, messages:history, stream:true}};

  const typingRow = addTyping();
  let fullText = '';
  let aiBubble = null;

  try{{
    const resp = await fetch(endpoint,{{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify(body)
    }});
    if(!resp.ok) throw new Error(`HTTP ${{resp.status}}`);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf='';
    outer:
    while(true){{
      const {{done,value}} = await reader.read();
      if(done) break;
      buf += dec.decode(value,{{stream:true}});
      const lines = buf.split('\\n'); buf=lines.pop();
      for(const line of lines){{
        if(!line.startsWith('data: ')) continue;
        const d=line.slice(6).trim();
        if(d==='[DONE]') break outer;
        try{{
          const chunk=JSON.parse(d);
          let delta='';
          if(isAnth){{ if(chunk.type==='content_block_delta') delta=chunk.delta?.text||''; }}
          else{{ delta=chunk.choices?.[0]?.delta?.content||''; }}
          if(delta){{
            fullText+=delta;
            if(!aiBubble){{
              typingRow.remove();
              aiBubble=addRow('assistant',renderMarkdown(fullText),fullText);
            }} else {{
              aiBubble.innerHTML=renderMarkdown(fullText);
            }}
            chatEl.scrollTop=chatEl.scrollHeight;
          }}
        }}catch(_){{}}
      }}
    }}
  }}catch(e){{
    typingRow.remove();
    const msg = e?.message||String(e);
    let hint='';
    if(msg.includes('401')) hint=' — API key missing or invalid';
    else if(msg.includes('403')) hint=' — check your gateway key';
    else if(msg.includes('500')) hint=' — server-side error';
    else if(msg.includes('Failed to fetch')||msg.includes('NetworkError')||msg.includes('input stream')) hint=' — connection dropped';
    if(aiBubble){{
      const errDiv=document.createElement('div');
      errDiv.style.cssText='color:var(--red);font-size:.8rem;margin-top:.5em;padding-top:.4em;border-top:1px solid rgba(239,68,68,.3)';
      errDiv.textContent='⚠ '+msg+hint;
      aiBubble.appendChild(errDiv);
      aiBubble.style.borderColor='rgba(239,68,68,.4)';
    }}else{{
      const errBubble=addRow('assistant',`<span style="color:var(--red)">Error: ${{esc(msg)}}${{hint}}</span>`,'');
      errBubble.style.borderColor='rgba(239,68,68,.4)';
    }}
  }}

  if(fullText) history.push({{role:'assistant',content:fullText}});
  streaming=false; sendBtn.disabled=false;
  statusDot.classList.remove('active'); statusDot.title='Idle';
  inputEl.focus();
}}

// ── Events ────────────────────────────────────────────────────────────────────
sendBtn.addEventListener('click',sendMessage);
inputEl.addEventListener('keydown',e=>{{
  if(e.key==='Enter'&&!e.shiftKey){{e.preventDefault();sendMessage();}}
  setTimeout(()=>{{inputEl.style.height='';inputEl.style.height=Math.min(inputEl.scrollHeight,200)+'px';}},0);
}});

document.getElementById('btn-new').addEventListener('click',()=>{{
  history=[];
  chatInner.innerHTML='';
  chatInner.appendChild(welcome);
  welcome.style.display='';
}});

document.querySelectorAll('.chip').forEach(c=>{{
  c.addEventListener('click',()=>{{
    inputEl.value=c.dataset.prompt||'';
    inputEl.focus();
    inputEl.style.height='';
    inputEl.style.height=Math.min(inputEl.scrollHeight,200)+'px';
  }});
}});

inputEl.focus();
}})();
</script>
</body>
</html>"""
            body_bytes = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        # ── POST /v1/chat/completions (OpenAI server type) ────────────────────

        def _handle_oai_chat(self) -> None:
            body   = self._read_body()
            model  = body.get("model", "")
            stream = bool(body.get("stream", False))
            mode, bmodel = self._resolve_model(model)
            if mode is None:
                self._json(404, {"error": {"message": f"Model '{model}' not found", "type": "invalid_request_error"}})
                return
            if mode == "claude":
                self._oai_via_claude(body, bmodel, stream)
            elif mode == "codex_direct":
                self._oai_via_codex(body, bmodel, stream)
            else:
                self._oai_proxy(body, mode, bmodel, stream)

        # ── POST /v1/messages (Anthropic server type) ─────────────────────────

        def _handle_anth_messages(self) -> None:
            body   = self._read_body()
            model  = body.get("model", "")
            stream = bool(body.get("stream", False))
            mode, bmodel = self._resolve_model(model)
            if mode is None:
                self._json(404, {"error": {"type": "not_found_error", "message": f"Model '{model}' not found"}})
                return
            if mode == "claude":
                self._anth_proxy(body, bmodel, stream)
            elif mode == "codex_direct":
                self._anth_via_codex(body, bmodel, stream)
            else:
                self._anth_via_oai(body, mode, bmodel, stream)

        # ── Model resolver ────────────────────────────────────────────────────

        def _resolve_model(self, model: str) -> tuple[str | None, str]:
            if model in _ROUTES:
                return _ROUTES[model]
            if cfg.custom_endpoint and (not model or model == "custom"):
                return ("custom", cfg.custom_endpoint.get("model", "custom"))
            if cfg.custom_endpoint and model == cfg.custom_endpoint.get("model"):
                return ("custom", model)
            return None, model

        # ── Backend: Claude / Anthropic API ───────────────────────────────────

        def _anth_headers(self) -> dict[str, str]:
            return {
                "x-api-key":        cfg.anthropic_key or "",
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type":     "application/json",
            }

        def _oai_via_claude(self, body: dict, model: str, stream: bool) -> None:
            """OpenAI server + Claude backend: convert OAI→Anthropic, proxy, convert back."""
            anth_body = _oai_to_anth_body(body, model)
            url = f"{ANTHROPIC_API_URL}/messages"

            if stream:
                anth_body["stream"] = True
                msg_id  = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created = int(time.time())
                self._start_sse()
                # Send role delta first
                self._sse_data(json.dumps({
                    "id": msg_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }))
                with httpx.Client(timeout=120.0) as hc:
                    with hc.stream("POST", url, headers=self._anth_headers(), json=anth_body) as resp:
                        for line in resp.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            try:
                                evt = json.loads(raw)
                            except Exception:
                                continue
                            etype = evt.get("type", "")
                            if etype == "content_block_delta":
                                text = evt.get("delta", {}).get("text", "")
                                if text:
                                    self._sse_data(json.dumps({
                                        "id": msg_id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                    }))
                            elif etype == "message_stop":
                                self._sse_data(json.dumps({
                                    "id": msg_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                }))
                self._sse_done()
            else:
                with httpx.Client(timeout=120.0) as hc:
                    resp = hc.post(url, headers=self._anth_headers(), json=anth_body)
                if not resp.is_success:
                    self._json(resp.status_code, resp.json())
                    return
                self._json(200, _anth_resp_to_oai(resp.json(), model))

        def _anth_proxy(self, body: dict, model: str, stream: bool) -> None:
            """Anthropic server + Claude backend: forward directly to Anthropic API."""
            body = dict(body)
            body["model"] = model
            url = f"{ANTHROPIC_API_URL}/messages"

            if stream:
                body["stream"] = True
                self._start_sse()
                with httpx.Client(timeout=120.0) as hc:
                    with hc.stream("POST", url, headers=self._anth_headers(), json=body) as resp:
                        for chunk in resp.iter_bytes(chunk_size=4096):
                            self.wfile.write(chunk)
                            self.wfile.flush()
            else:
                with httpx.Client(timeout=120.0) as hc:
                    resp = hc.post(url, headers=self._anth_headers(), json=body)
                self._json(resp.status_code, resp.json())

        # ── Backend: OpenAI-compatible (NVIDIA server / NVIDIA direct / custom) ──

        def _oai_backend_url(self, mode: str) -> str:
            if mode == "server":
                return f"{NVIDIA_BASE_URL}/chat/completions"
            if mode == "nvidia_direct":
                return f"{NVIDIA_DIRECT_URL}/chat/completions"
            if mode == "custom" and cfg.custom_endpoint:
                base = cfg.custom_endpoint.get("url", "").rstrip("/")
                return f"{base}/chat/completions"
            return f"{NVIDIA_BASE_URL}/chat/completions"

        def _oai_backend_key(self, mode: str) -> str:
            if mode == "nvidia_direct" and cfg.nvidia_api_key:
                return cfg.nvidia_api_key
            if mode == "custom" and cfg.custom_endpoint:
                return cfg.custom_endpoint.get("api_key") or "none"
            return cfg.api_key or "none"

        def _oai_headers(self, mode: str) -> dict[str, str]:
            return {
                "Authorization": f"Bearer {self._oai_backend_key(mode)}",
                "Content-Type":  "application/json",
            }

        def _oai_proxy(self, body: dict, mode: str, model: str, stream: bool) -> None:
            """OpenAI server + OpenAI-compat backend: transparent proxy."""
            body = dict(body)
            body["model"] = model
            url = self._oai_backend_url(mode)

            if stream:
                body["stream"] = True
                self._start_sse()
                with httpx.Client(timeout=120.0) as hc:
                    with hc.stream("POST", url, headers=self._oai_headers(mode), json=body) as resp:
                        for chunk in resp.iter_bytes(chunk_size=4096):
                            self.wfile.write(chunk)
                            self.wfile.flush()
            else:
                with httpx.Client(timeout=120.0) as hc:
                    resp = hc.post(url, headers=self._oai_headers(mode), json=body)
                self._json(resp.status_code, resp.json())

        def _anth_via_oai(self, body: dict, mode: str, model: str, stream: bool) -> None:
            """Anthropic server + OpenAI-compat backend: convert Anthropic→OAI, proxy, convert back."""
            oai_body = _anth_to_oai_body(body, model)
            url      = self._oai_backend_url(mode)

            if stream:
                oai_body["stream"] = True
                msg_id  = f"msg_{uuid.uuid4().hex[:24]}"
                self._start_sse()
                # Anthropic SSE preamble
                self._sse_event("message_start", {
                    "type": "message_start",
                    "message": {"id": msg_id, "type": "message", "role": "assistant",
                                "content": [], "model": model, "stop_reason": None,
                                "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 1}},
                })
                self._sse_event("content_block_start", {
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                self._sse_event("ping", {"type": "ping"})

                with httpx.Client(timeout=120.0) as hc:
                    with hc.stream("POST", url, headers=self._oai_headers(mode), json=oai_body) as resp:
                        for line in resp.iter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            raw = line[6:] if line.startswith("data: ") else line
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                                text  = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if text:
                                    self._sse_event("content_block_delta", {
                                        "type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": text},
                                    })
                            except Exception:
                                continue

                self._sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                self._sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                })
                self._sse_event("message_stop", {"type": "message_stop"})
            else:
                with httpx.Client(timeout=120.0) as hc:
                    resp = hc.post(url, headers=self._oai_headers(mode), json=oai_body)
                if not resp.is_success:
                    self._json(resp.status_code, resp.json())
                    return
                self._json(200, _oai_resp_to_anth(resp.json(), model))

        # ── Backend: OpenAI Codex (WHAM) ──────────────────────────────────────

        def _codex_creds(self) -> tuple[str, str | None] | None:
            data = cfg.codex_token
            if not data:
                return None
            expires = data.get("expires", 0)
            if expires and time.time() * 1000 >= expires - 30_000:
                return None
            access = data.get("access")
            return (access, data.get("accountId")) if access else None

        def _wham_headers(self, access: str, account_id: str | None) -> dict[str, str]:
            h = {"Authorization": f"Bearer {access}", "Content-Type": "application/json"}
            if account_id:
                h["ChatGPT-Account-Id"] = account_id
            return h

        def _wham_payload(self, messages: list[dict], model: str) -> dict:
            instructions = ""
            input_msgs = []
            for msg in messages:
                role    = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    instructions = content if isinstance(content, str) else ""
                    continue
                ctype = "input_text" if role == "user" else "output_text"
                text  = content if isinstance(content, str) else ""
                input_msgs.append({"role": role, "content": [{"type": ctype, "text": text}]})
            return {
                "model":               model,
                "input":               input_msgs,
                "instructions":        instructions,
                "store":               False,
                "stream":              True,
                "reasoning":           {"effort": "medium"},
                "include":             [],
                "tools":               [],
                "tool_choice":         "auto",
                "parallel_tool_calls": True,
            }

        def _stream_wham_to_oai(
            self, access: str, account_id: str | None, payload: dict, model: str, msg_id: str, created: int
        ) -> str | None:
            """Stream WHAM response, emitting OpenAI SSE chunks. Returns error string on failure."""
            try:
                with httpx.Client(timeout=120.0) as hc:
                    with hc.stream("POST", WHAM_URL, headers=self._wham_headers(access, account_id), json=payload) as resp:
                        if not resp.is_success:
                            body = resp.read().decode("utf-8", errors="replace")
                            return f"[WHAM {resp.status_code}] {body[:300]}"
                        for line in resp.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                evt  = json.loads(raw)
                                text = evt.get("delta", "")
                                if evt.get("type") == "response.output_text.delta":
                                    if isinstance(text, dict):
                                        text = text.get("text", "")
                                    if text:
                                        self._sse_data(json.dumps({
                                            "id": msg_id, "object": "chat.completion.chunk",
                                            "created": created, "model": model,
                                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                        }))
                            except Exception:
                                continue
            except Exception as exc:
                return str(exc)
            return None

        def _collect_wham(
            self, access: str, account_id: str | None, payload: dict
        ) -> tuple[str, str | None]:
            """Collect all WHAM output text. Returns (content, error_message)."""
            parts: list[str] = []
            try:
                with httpx.Client(timeout=120.0) as hc:
                    with hc.stream("POST", WHAM_URL, headers=self._wham_headers(access, account_id), json=payload) as resp:
                        if not resp.is_success:
                            body = resp.read().decode("utf-8", errors="replace")
                            return "", f"[WHAM {resp.status_code}] {body[:300]}"
                        for line in resp.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                evt  = json.loads(raw)
                                text = evt.get("delta", "")
                                if evt.get("type") == "response.output_text.delta":
                                    if isinstance(text, dict):
                                        text = text.get("text", "")
                                    if text:
                                        parts.append(text)
                            except Exception:
                                continue
            except Exception as exc:
                return "".join(parts), str(exc)
            return "".join(parts), None

        def _oai_via_codex(self, body: dict, model: str, stream: bool) -> None:
            creds = self._codex_creds()
            if not creds:
                self._json(401, {"error": {"message": "Codex not authenticated — run /codex-login in the MahanAI CLI", "type": "authentication_error"}})
                return
            access, account_id = creds
            payload = self._wham_payload(body.get("messages", []), model)
            msg_id  = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            if stream:
                self._start_sse()
                self._sse_data(json.dumps({
                    "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }))
                err = self._stream_wham_to_oai(access, account_id, payload, model, msg_id, created)
                if err:
                    self._sse_data(json.dumps({
                        "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": err}, "finish_reason": "stop"}],
                    }))
                else:
                    self._sse_data(json.dumps({
                        "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }))
                self._sse_done()
            else:
                content, err = self._collect_wham(access, account_id, payload)
                if err:
                    self._json(502, {"error": {"message": err, "type": "upstream_error"}})
                    return
                self._json(200, {
                    "id": msg_id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })

        def _anth_via_codex(self, body: dict, model: str, stream: bool) -> None:
            creds = self._codex_creds()
            if not creds:
                self._json(401, {"error": {"type": "authentication_error", "message": "Codex not authenticated — run /codex-login in the MahanAI CLI"}})
                return
            access, account_id = creds
            oai_body = _anth_to_oai_body(body, model)
            payload  = self._wham_payload(oai_body.get("messages", []), model)
            msg_id   = f"msg_{uuid.uuid4().hex[:24]}"

            if stream:
                self._start_sse()
                self._sse_event("message_start", {
                    "type": "message_start",
                    "message": {"id": msg_id, "type": "message", "role": "assistant",
                                "content": [], "model": model, "stop_reason": None,
                                "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 1}},
                })
                self._sse_event("content_block_start", {
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                self._sse_event("ping", {"type": "ping"})

                wham_err: str | None = None
                with httpx.Client(timeout=120.0) as hc:
                    with hc.stream("POST", WHAM_URL, headers=self._wham_headers(access, account_id), json=payload) as resp:
                        if not resp.is_success:
                            body_bytes = resp.read().decode("utf-8", errors="replace")
                            wham_err = f"[WHAM {resp.status_code}] {body_bytes[:300]}"
                        else:
                            for line in resp.iter_lines():
                                if not line.startswith("data:"):
                                    continue
                                raw = line[5:].strip()
                                if raw == "[DONE]":
                                    break
                                try:
                                    evt  = json.loads(raw)
                                    text = evt.get("delta", "")
                                    if evt.get("type") == "response.output_text.delta":
                                        if isinstance(text, dict):
                                            text = text.get("text", "")
                                        if text:
                                            self._sse_event("content_block_delta", {
                                                "type": "content_block_delta", "index": 0,
                                                "delta": {"type": "text_delta", "text": text},
                                            })
                                except Exception:
                                    continue

                if wham_err:
                    self._sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": wham_err},
                    })
                self._sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                self._sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                })
                self._sse_event("message_stop", {"type": "message_stop"})
            else:
                content, err = self._collect_wham(access, account_id, payload)
                if err:
                    self._json(502, {"type": "error", "error": {"type": "upstream_error", "message": err}})
                    return
                self._json(200, {
                    "id": msg_id, "type": "message", "role": "assistant", "model": model,
                    "content": [{"type": "text", "text": content}],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                })

        # ── Response helpers ──────────────────────────────────────────────────

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            raw    = self.rfile.read(length) if length else b""
            try:
                return json.loads(raw) if raw else {}
            except Exception:
                return {}

        def _json(self, status: int, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _start_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type",  "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection",    "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

        def _sse_data(self, payload: str) -> None:
            """Emit a bare `data: …\n\n` SSE event (OpenAI style)."""
            try:
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except _DROPPED:
                pass

        def _sse_event(self, event: str, payload: Any) -> None:
            """Emit a named `event: …\ndata: …\n\n` SSE event (Anthropic style)."""
            try:
                self.wfile.write(
                    f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
            except _DROPPED:
                pass

        def _sse_done(self) -> None:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except _DROPPED:
                pass

    return Handler


# ── Quiet server — suppresses dropped-connection noise on Windows ─────────────

_DROPPED = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)


class _QuietHTTPServer(HTTPServer):
    def handle_error(self, request: object, client_address: tuple) -> None:
        import sys
        if isinstance(sys.exc_info()[1], _DROPPED):
            return  # client disconnected — not an error worth printing
        super().handle_error(request, client_address)


# ── Public entry point ────────────────────────────────────────────────────────

def run_server(cfg: ServerConfig) -> None:
    """Start the gateway and block until Ctrl+C."""
    Handler = _make_handler(cfg)
    httpd   = _QuietHTTPServer(("0.0.0.0", cfg.port), Handler)

    type_label   = "Anthropic" if cfg.server_type == "anthropic" else "OpenAI"
    chat_path    = "/v1/messages" if cfg.server_type == "anthropic" else "/v1/chat/completions"
    local        = f"http://localhost:{cfg.port}"

    print(f"\n{C.OK}MahanAI Gateway Server{C.RST}  {C.DIM}(Ctrl+C to stop){C.RST}")
    print(f"  {C.DIM}API type :{C.RST}  {type_label}-compatible")
    print(f"  {C.DIM}Listening:{C.RST}  {local}")
    print(f"  {C.DIM}Web UI   :{C.RST}  {C.OK}{local}/{C.RST}  {C.DIM}(open in browser){C.RST}")
    print(f"  {C.DIM}Chat     :{C.RST}  {local}{chat_path}")
    print(f"  {C.DIM}Models   :{C.RST}  {local}/v1/models")
    print()

    # Print available providers
    providers: dict[str, list[str]] = {}
    for mid, (mode, _) in _ROUTES.items():
        label = _MODEL_DISPLAY.get(mid, (mid, "Unknown"))[1]
        providers.setdefault(label, []).append(mid)
    if cfg.custom_endpoint:
        providers.setdefault("Custom", []).append(cfg.custom_endpoint.get("model", "custom"))

    for provider, models in providers.items():
        avail = C.OK if _provider_ready(provider, cfg) else C.ERR
        status = "ready" if _provider_ready(provider, cfg) else "no credentials"
        print(f"  {avail}●{C.RST} {provider:<24} {C.DIM}{status}{C.RST}")
        for mid in models:
            print(f"      {C.DIM}{mid}{C.RST}")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Server stopped.{C.RST}\n")
    finally:
        httpd.server_close()


def _provider_ready(provider: str, cfg: ServerConfig) -> bool:
    if provider == "Anthropic":
        return bool(cfg.anthropic_key)
    if provider == "NVIDIA NIM":
        return bool(cfg.api_key or cfg.nvidia_api_key)
    if provider == "OpenAI Codex":
        return bool(cfg.codex_token)
    if provider == "Custom":
        return bool(cfg.custom_endpoint)
    return False
