# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MahanAI is a Python terminal AI agent (CLI) published to PyPI as `mahanai-finale`. It integrates with NVIDIA NIM (NVIDIA Inference Microservices) via an OpenAI-compatible API. Users interact with it through a streaming chat interface in the terminal.

## Running and Development

```bash
# Install in editable mode for development
pip install -e .

# Run the app
python -m mahanai
# or after install:
mahanai
```

No test suite or linter is currently configured.

## Publishing to PyPI

Before publishing:
1. Bump version in **both** `pyproject.toml` and `mahanai/__init__.py` (they must stay in sync).

```bash
pip install build twine
python -m build
python -m twine check dist/*

# Windows PowerShell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-YOUR_TOKEN_HERE"
python -m twine upload dist/mahanai_finale-X.Y.Z*
```

## Architecture

**Entry point:** `mahanai.agent:main` (configured in `pyproject.toml`)

```
mahanai/
├── agent.py        # Main chat loop, API routing, slash command handling
├── tools.py        # Tool definitions (run_command, read/write/list/append_file) and execution
├── config.py       # Persistent API key storage per platform
├── system_info.py  # OS/shell detection for system prompt context
├── colors.py       # colorama-based terminal color helpers
├── __main__.py     # Calls agent.main()
└── __init__.py     # Package version string
```

### API Request Flow

`agent.py` routes requests to one of two backends:
- **Server mode** (default): Local NVIDIA NIM at `http://89.167.0.111:8000/v1`, model `mahanai/mahanai`
- **NVIDIA direct mode**: `https://integrate.api.nvidia.com/v1`, model `meta/llama-3.3-70b-instruct`

Responses are streamed via `httpx` (not the OpenAI SDK) by parsing SSE (Server-Sent Events) directly in `_stream_direct()`.

### Tools System

`tools.py` defines 5 tools and their execution logic, including a safety confirmation step before high-risk shell commands (recursive delete, format, shutdown, etc.). Tool call handling is currently stubbed out in `agent.py`'s `run_turn()` — the infrastructure exists but the agent runs in direct-chat-only mode.

### Configuration

`config.py` stores API keys in platform-specific paths:
- Windows: `%APPDATA%\MahanAI\config.json`
- Linux/macOS: `~/.config/mahanai/config.json`

Keys can also be set via environment variables (`MAHANAI_API_KEY`) or a `.env` file.

## Environment Variables

| Variable | Purpose |
|---|---|
| `MAHANAI_API_KEY` | Override saved server API key |
| `MAHANAI_MODEL` | Override default model ID |
| `MAHANAI_STREAM` | Set to `0`/`false`/`no`/`off` to disable streaming |
| `MAHANAI_CONFIG_DIR` | Override config file directory |
| `NO_COLOR` | Disable terminal colors |
