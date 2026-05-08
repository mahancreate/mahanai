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
from pathlib import Path
from typing import Any

# Path to the custom web UI file (next to this package)
_WEBUI_PATH = Path(__file__).parent.parent / "webui.html"

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
            try:
                body_bytes = _WEBUI_PATH.read_bytes()
            except OSError:
                body_bytes = b"<h1>MahanAI Web UI</h1><p>webui.html not found next to the package.</p>"
            self.send_response(200)
            self._cors_headers()
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

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors_headers()
            self.end_headers()

        def _json(self, status: int, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _start_sse(self) -> None:
            self.send_response(200)
            self._cors_headers()
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
