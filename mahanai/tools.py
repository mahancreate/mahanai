"""Tool definitions and execution for the MahanAI agent."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from mahanai import colors as C

# Backslashes in model-generated JSON often break parsing (e.g. C:\Users — invalid \U escape).
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
]


def _high_risk_shell_command(cmd: str) -> bool:
    """Heuristic: recursive delete, disk/system shutdown, etc."""
    low = cmd.strip().lower()
    if not low:
        return False
    if re.search(r"\brm\s+.*(-\s*rf\b|-\s*fr\b|--no-preserve-root)", low):
        return True
    if re.search(r"\brmdir(\.exe)?\b.*\s/s\b", low):
        return True
    if re.search(r"\brd(\.exe)?\b.*\s/s\b", low):
        return True
    if re.search(r"\bdel(\.exe)?\b.*\s/s\b", low):
        return True
    if re.search(r"\berase(\.exe)?\b.*\s/s\b", low):
        return True
    if re.search(r"\bformat\s+", low):
        return True
    if re.search(r"\bshutdown\b", low):
        return True
    if re.search(r"\breboot\b", low):
        return True
    if re.search(r"\blogoff\b", low):
        return True
    if re.search(r"\bhalt\b", low):
        return True
    if re.search(r"\bpoweroff\b", low):
        return True
    if re.search(r"\binit\s+0\b", low):
        return True
    if re.search(r"\bdiskpart\b", low):
        return True
    if re.search(r"\bmkfs", low):
        return True
    if re.search(r"\bdd\s+if=", low):
        return True
    if re.search(r":\(\)\s*\{\s*:", low):
        return True
    if "remove-item" in low and "-recurse" in low and "-force" in low:
        return True
    return False


def _confirm_high_risk_command(cmd: str) -> bool:
    print(f"\n{C.WARN}High-risk command — approval required{C.RST}")
    print(f"{C.DIM}  {cmd}{C.RST}")
    try:
        ans = input("Allow? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def run_command(
    base: Path, args: dict[str, object]
) -> str:
    cmd = str(args.get("command", "")).strip()
    if not cmd:
        return json.dumps({"error": "empty command"})
    cwd_raw = args.get("cwd")
    timeout = int(args.get("timeout_seconds") or 120)
    cwd = base
    if isinstance(cwd_raw, str) and cwd_raw.strip():
        cwd = _resolve_path(base, cwd_raw)

    if _high_risk_shell_command(cmd):
        if not _confirm_high_risk_command(cmd):
            return json.dumps(
                {
                    "exit_code": -1,
                    "error": "user_denied_high_risk_command",
                    "command": cmd,
                    "output": "",
                    "cwd": str(cwd),
                }
            )

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
        return json.dumps(
            {
                "exit_code": proc.returncode,
                "output": out,
                "cwd": str(cwd),
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"timed out after {timeout}s", "command": cmd})
    except OSError as e:
        return json.dumps({"error": str(e), "command": cmd})


def read_file(base: Path, args: dict[str, object]) -> str:
    path = _resolve_path(base, str(args.get("path", "")))
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
    path = _resolve_path(base, str(args.get("path", "")))
    content = str(args.get("content", ""))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": str(path), "bytes": len(content.encode("utf-8"))})
    except OSError as e:
        return json.dumps({"error": str(e), "path": str(path)})


def append_file(base: Path, args: dict[str, object]) -> str:
    path = _resolve_path(base, str(args.get("path", "")))
    content = str(args.get("content", ""))
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
        return run_command(workspace, args)
    if name == "read_file":
        return read_file(workspace, args)
    if name == "write_file":
        return write_file(workspace, args)
    if name == "append_file":
        return append_file(workspace, args)
    if name == "list_directory":
        return list_directory(workspace, args)
    return json.dumps({"error": f"unknown tool: {name}"})
