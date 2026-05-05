"""Tool definitions and execution for the MahanAI agent."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mahanai import colors as C

_JSON_BAD_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})')


def repair_invalid_json_escapes(raw: str) -> str:
    s = raw
    for _ in range(128):
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            nxt = _JSON_BAD_BACKSLASH.sub(r"\\\\", s)
            if nxt == s:
                return s
            s = nxt
    return s


def normalize_tool_arguments_json(arguments: str) -> str:
    """Parse model tool arguments and re-serialize as valid JSON for the API and executor."""
    raw = (arguments or "").strip() or "{}"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = json.loads(repair_invalid_json_escapes(raw))
        except json.JSONDecodeError:
            return "{}"
    if not isinstance(obj, dict):
        return "{}"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command now (the user sees ⚡Running: in the terminal). "
                "Do not only show commands in chat—this tool must be called to run them. "
                "On Windows the default shell is usually cmd.exe (COMSPEC), not PowerShell; "
                "use cmd syntax or call powershell -NoProfile -Command \"...\" explicitly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full command line to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory (absolute or relative to cwd).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Max seconds to wait (default 120).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full text of a file (UTF-8).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with UTF-8 text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a path (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: current working directory).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append UTF-8 text to a file (creates the file if missing).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the text content of a URL (HTML is stripped to plain text).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_repl",
            "description": "Execute Python code in an isolated subprocess and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Max seconds to wait (default 30).",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets for top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ── Approval helpers ───────────────────────────────────────────────────────────

_HIGH_RISK_PATTERNS = [
    re.compile(r"\brm\s+.*(-\s*rf\b|-\s*fr\b|--no-preserve-root)"),
    re.compile(r"\brmdir(\.exe)?\b.*\s/s\b"),
    re.compile(r"\brd(\.exe)?\b.*\s/s\b"),
    re.compile(r"\bdel(\.exe)?\b.*\s/s\b"),
    re.compile(r"\berase(\.exe)?\b.*\s/s\b"),
    re.compile(r"\bformat\s+"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\blogoff\b"),
    re.compile(r"\bhalt\b"),
    re.compile(r"\bpoweroff\b"),
    re.compile(r"\binit\s+0\b"),
    re.compile(r"\bdiskpart\b"),
    re.compile(r"\bmkfs"),
    re.compile(r"\bdd\s+if="),
    re.compile(r":\(\)\s*\{\s*:"),
    re.compile(r"remove-item.*-recurse.*-force"),
]


def _is_high_risk(cmd: str) -> bool:
    low = cmd.strip().lower()
    return any(p.search(low) for p in _HIGH_RISK_PATTERNS)


def _command_category(cmd: str) -> str:
    first = cmd.strip().split()[0].lower().rstrip(".exe") if cmd.strip() else ""
    if first == "git":
        return "git"
    if first == "gh":
        return "github"
    return "normal"


def _command_prefix(cmd: str) -> str:
    return cmd.strip().split()[0].lower() if cmd.strip() else ""


def _read_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _approve_command(cmd: str) -> tuple[bool, str]:
    """
    Show an approval prompt for a shell command.
    Returns (approved, denial_message_for_ai).
    """
    from mahanai.config import load_always_allowed, add_always_allowed_command

    category = _command_category(cmd)
    prefix = _command_prefix(cmd)
    high_risk = _is_high_risk(cmd)

    # Always-Allow only applies to normal (non-git, non-gh) commands
    if category == "normal" and not high_risk:
        always = load_always_allowed()
        if prefix in always.get("command_prefixes", []):
            return True, ""

    # ── Build prompt ──────────────────────────────────────────────────────────
    cat_label = {
        "git":    "Git Command",
        "github": "GitHub Command",
        "normal": "Shell Command",
    }[category]

    risk_tag = f"  {C.ERR}[DESTRUCTIVE]{C.RST}" if high_risk else ""
    print(f"\n{C.WARN}  {cat_label}{C.RST}{risk_tag}")
    print(f"  {C.DIM}{cmd}{C.RST}")

    if category in ("git", "github"):
        print(f"  {C.OK}[A]{C.RST} Allow    {C.ERR}[D]{C.RST} Deny")
        ans = _read_input("  > ")
        approved = ans.lower() in ("a", "allow")
    else:
        # Normal: Allow / Always Allow / Deny
        always_label = f"Always Allow ({prefix})" if not high_risk else "Always Allow (disabled for destructive)"
        if high_risk:
            print(f"  {C.OK}[A]{C.RST} Allow    {C.ERR}[D]{C.RST} Deny")
            ans = _read_input("  > ")
            approved = ans.lower() in ("a", "allow")
        else:
            print(f"  {C.OK}[A]{C.RST} Allow    {C.DIM}[W] {always_label}{C.RST}    {C.ERR}[D]{C.RST} Deny")
            ans = _read_input("  > ").lower()
            if ans in ("w", "always allow", "always"):
                add_always_allowed_command(prefix)
                print(f"  {C.OK}'{prefix}' commands will always be allowed.{C.RST}")
                return True, ""
            approved = ans in ("a", "allow")

    if approved:
        return True, ""

    # Denied — let user send a message to the AI
    msg = _read_input(f"  {C.DIM}Instruction for AI (Enter to skip):{C.RST} ")
    return False, msg or "Command was denied by the user."


def _approve_file_op(op: str, display_path: str) -> tuple[bool, str]:
    """
    Show an approval prompt for a file operation.
    Returns (approved, denial_message_for_ai).
    """
    from mahanai.config import load_always_allowed, add_always_allowed_file_op

    always = load_always_allowed()
    if op in always.get("file_ops", []):
        return True, ""

    op_labels = {
        "read_file":      "Read File",
        "write_file":     "Write / Create File",
        "append_file":    "Append to File",
        "list_directory": "List Directory",
    }
    label = op_labels.get(op, op)

    print(f"\n{C.WARN}  {label}{C.RST}")
    print(f"  {C.DIM}{display_path}{C.RST}")
    print(f"  {C.OK}[A]{C.RST} Allow    {C.DIM}[W] Always Allow ({label}){C.RST}    {C.ERR}[D]{C.RST} Deny")

    ans = _read_input("  > ").lower()

    if ans in ("w", "always allow", "always"):
        add_always_allowed_file_op(op)
        print(f"  {C.OK}'{label}' will always be allowed.{C.RST}")
        return True, ""

    if ans in ("a", "allow"):
        return True, ""

    msg = _read_input(f"  {C.DIM}Instruction for AI (Enter to skip):{C.RST} ")
    return False, msg or "File operation was denied by the user."


# ── Tool implementations ───────────────────────────────────────────────────────

def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def run_command(base: Path, args: dict[str, object]) -> str:
    cmd = str(args.get("command", "")).strip()
    if not cmd:
        return json.dumps({"error": "empty command"})
    cwd_raw = args.get("cwd")
    timeout = int(args.get("timeout_seconds") or 120)
    cwd = base
    if isinstance(cwd_raw, str) and cwd_raw.strip():
        cwd = _resolve_path(base, cwd_raw)

    approved, denial_msg = _approve_command(cmd)
    if not approved:
        return json.dumps({
            "exit_code": -1,
            "error": "user_denied",
            "message": denial_msg,
            "command": cmd,
            "output": denial_msg,
            "cwd": str(cwd),
        })

    print(f"\n{C.OK}⚡Running:{C.RST} {cmd}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=max(1, timeout),
            env=os.environ.copy(),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > 100_000:
            out = out[:100_000] + "\n… [truncated]"
        return json.dumps({"exit_code": proc.returncode, "output": out, "cwd": str(cwd)})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"timed out after {timeout}s", "command": cmd})
    except OSError as e:
        return json.dumps({"error": str(e), "command": cmd})


def read_file(base: Path, args: dict[str, object]) -> str:
    raw_path = str(args.get("path", ""))
    path = _resolve_path(base, raw_path)

    approved, denial_msg = _approve_file_op("read_file", str(path))
    if not approved:
        return json.dumps({"error": "user_denied", "message": denial_msg, "path": str(path)})

    if not path.is_file():
        return json.dumps({"error": "not a file", "path": str(path)})
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 200_000:
            text = text[:200_000] + "\n… [truncated]"
        return json.dumps({"path": str(path), "content": text})
    except OSError as e:
        return json.dumps({"error": str(e), "path": str(path)})


def write_file(base: Path, args: dict[str, object]) -> str:
    raw_path = str(args.get("path", ""))
    path = _resolve_path(base, raw_path)
    content = str(args.get("content", ""))

    approved, denial_msg = _approve_file_op("write_file", str(path))
    if not approved:
        return json.dumps({"error": "user_denied", "message": denial_msg, "path": str(path)})

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": str(path), "bytes": len(content.encode("utf-8"))})
    except OSError as e:
        return json.dumps({"error": str(e), "path": str(path)})


def append_file(base: Path, args: dict[str, object]) -> str:
    raw_path = str(args.get("path", ""))
    path = _resolve_path(base, raw_path)
    content = str(args.get("content", ""))

    approved, denial_msg = _approve_file_op("append_file", str(path))
    if not approved:
        return json.dumps({"error": "user_denied", "message": denial_msg, "path": str(path)})

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"ok": True, "path": str(path), "appended_bytes": len(content.encode("utf-8"))})
    except OSError as e:
        return json.dumps({"error": str(e), "path": str(path)})


def list_directory(base: Path, args: dict[str, object]) -> str:
    raw = args.get("path")
    path = base if not isinstance(raw, str) or not raw.strip() else _resolve_path(base, raw)

    approved, denial_msg = _approve_file_op("list_directory", str(path))
    if not approved:
        return json.dumps({"error": "user_denied", "message": denial_msg, "path": str(path)})

    if not path.is_dir():
        return json.dumps({"error": "not a directory", "path": str(path)})
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        truncated = len(entries) > 500
        rows = [
            {"name": p.name, "type": "dir" if p.is_dir() else "file"}
            for p in entries[:500]
        ]
        return json.dumps({"path": str(path), "entries": rows, "truncated": truncated})
    except OSError as e:
        return json.dumps({"error": str(e), "path": str(path)})


def fetch_url(base: Path, args: dict[str, object]) -> str:
    url = str(args.get("url", "")).strip()
    if not url:
        return json.dumps({"error": "empty url"})

    print(f"\n{C.WARN}  Fetch URL{C.RST}")
    print(f"  {C.DIM}{url}{C.RST}")
    print(f"  {C.OK}[A]{C.RST} Allow    {C.ERR}[D]{C.RST} Deny")
    ans = _read_input("  > ").lower()
    if ans not in ("a", "allow"):
        return json.dumps({"error": "user_denied", "url": url})

    try:
        import httpx as _httpx
        from html.parser import HTMLParser as _HP

        resp = _httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" in ct.lower():
            class _Stripper(_HP):
                def __init__(self) -> None:
                    super().__init__()
                    self.parts: list[str] = []
                    self._skip = False
                def handle_starttag(self, tag: str, attrs: object) -> None:
                    if tag in ("script", "style"):
                        self._skip = True
                def handle_endtag(self, tag: str) -> None:
                    if tag in ("script", "style"):
                        self._skip = False
                def handle_data(self, d: str) -> None:
                    if not self._skip:
                        self.parts.append(d)
            s = _Stripper()
            s.feed(resp.text)
            text = re.sub(r"\s+", " ", " ".join(s.parts)).strip()
        else:
            text = resp.text
        if len(text) > 50_000:
            text = text[:50_000] + "\n… [truncated]"
        return json.dumps({"url": url, "content": text, "status": resp.status_code})
    except Exception as e:
        return json.dumps({"error": str(e), "url": url})


def python_repl(base: Path, args: dict[str, object]) -> str:
    code = str(args.get("code", "")).strip()
    if not code:
        return json.dumps({"error": "empty code"})
    timeout = int(args.get("timeout_seconds") or 30)

    print(f"\n{C.WARN}  Python REPL{C.RST}")
    lines = code.split("\n")
    preview = "\n  ".join(lines[:5])
    if len(lines) > 5:
        preview += f"\n  … ({len(lines)} lines total)"
    print(f"  {C.DIM}{preview}{C.RST}")
    print(f"  {C.OK}[A]{C.RST} Allow    {C.ERR}[D]{C.RST} Deny")
    ans = _read_input("  > ").lower()
    if ans not in ("a", "allow"):
        return json.dumps({"error": "user_denied"})

    print(f"\n{C.OK}🐍 Running Python...{C.RST}", flush=True)
    fname: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            fname = f.name
        proc = subprocess.run(
            [sys.executable, fname],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=max(1, timeout),
            env=os.environ.copy(),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > 50_000:
            out = out[:50_000] + "\n… [truncated]"
        return json.dumps({"exit_code": proc.returncode, "output": out})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if fname:
            try:
                Path(fname).unlink(missing_ok=True)
            except Exception:
                pass


def web_search(base: Path, args: dict[str, object]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"error": "empty query"})
    max_results = int(args.get("max_results") or 5)

    print(f"\n{C.WARN}  Web Search{C.RST}")
    print(f"  {C.DIM}{query}{C.RST}")
    print(f"  {C.OK}[A]{C.RST} Allow    {C.ERR}[D]{C.RST} Deny")
    ans = _read_input("  > ").lower()
    if ans not in ("a", "allow"):
        return json.dumps({"error": "user_denied", "query": query})

    try:
        import urllib.parse
        import httpx as _httpx
        from html.parser import HTMLParser as _HP

        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = _httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()

        class _DDGParser(_HP):
            def __init__(self) -> None:
                super().__init__()
                self._in_title = False
                self._in_snippet = False
                self._cur_url: str | None = None
                self._cur_title: list[str] = []
                self._cur_snippet: list[str] = []
                self.results: list[dict] = []

            def handle_starttag(self, tag: str, attrs: list) -> None:
                d = dict(attrs)
                cls = d.get("class", "")
                if tag == "a" and "result__a" in cls:
                    self._in_title = True
                    self._cur_title = []
                    href = d.get("href", "")
                    if "uddg=" in href:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        self._cur_url = parsed.get("uddg", [href])[0]
                    else:
                        self._cur_url = href
                if "result__snippet" in cls:
                    self._in_snippet = True
                    self._cur_snippet = []

            def handle_endtag(self, tag: str) -> None:
                if self._in_title and tag == "a":
                    self._in_title = False
                    if self._cur_url and self._cur_title:
                        if not any(r["url"] == self._cur_url for r in self.results):
                            self.results.append({
                                "title": "".join(self._cur_title).strip(),
                                "url": self._cur_url,
                                "snippet": "",
                            })
                if self._in_snippet and tag in ("a", "div", "span"):
                    self._in_snippet = False
                    if self.results:
                        self.results[-1]["snippet"] = "".join(self._cur_snippet).strip()
                    self._cur_snippet = []

            def handle_data(self, d: str) -> None:
                if self._in_title:
                    self._cur_title.append(d)
                elif self._in_snippet:
                    self._cur_snippet.append(d)

        parser = _DDGParser()
        parser.feed(resp.text)
        results = parser.results[:max_results]
        return json.dumps({"query": query, "results": results})
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


def execute_tool(name: str, arguments_json: str, workspace: Path) -> str:
    canon = normalize_tool_arguments_json(arguments_json)
    if canon == "{}" and (arguments_json or "").strip() not in ("", "{}"):
        return json.dumps(
            {
                "error": "could not parse tool arguments as JSON",
                "raw": (arguments_json or "")[:500],
            }
        )
    try:
        args = json.loads(canon)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid JSON arguments: {e}"})

    if name == "run_command":
        result = run_command(workspace, args)
    elif name == "read_file":
        result = read_file(workspace, args)
    elif name == "write_file":
        result = write_file(workspace, args)
    elif name == "append_file":
        result = append_file(workspace, args)
    elif name == "list_directory":
        result = list_directory(workspace, args)
    elif name == "fetch_url":
        result = fetch_url(workspace, args)
    elif name == "python_repl":
        result = python_repl(workspace, args)
    elif name == "web_search":
        result = web_search(workspace, args)
    else:
        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        from mahanai.config import audit_log_path
        import datetime as _dt
        _log = audit_log_path()
        _log.parent.mkdir(parents=True, exist_ok=True)
        _ts = _dt.datetime.now().isoformat(timespec="seconds")
        _ap = (arguments_json or "")[:120].replace("\n", " ")
        _rp = (result or "")[:120].replace("\n", " ")
        with _log.open("a", encoding="utf-8") as _f:
            _f.write(f"{_ts} | {name} | {_ap} | {_rp}\n")
    except Exception:
        pass

    return result
