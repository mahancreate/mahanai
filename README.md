<img width="600" alt="icon" src="https://github.com/user-attachments/assets/07c20ad0-61ef-47a6-b973-1e86b22be201" />

# MahanAI Super

Terminal AI agent (Super 2.0) with multi-model support, streaming chat, tools, and a built-in Claude CLI mode. Docs: [MahanAI](https://mahancreate.github.io/mahanai).

## Install

```bash
pip install mahanai
mahanai
```

## Models

MahanAI Super supports multiple backends selectable at runtime:

| Pretty Name       | Model ID                        | Backend             |
|-------------------|---------------------------------|---------------------|
| Llama 3.3 70B     | meta/llama-3.3-70b-instruct     | NVIDIA NIM (direct) |
| Claude Opus 4     | claude-opus-4-7                 | Claude CLI          |
| Claude Sonnet 4.6 | claude-sonnet-4-6               | Claude CLI          |
| Claude Haiku 4.5  | claude-haiku-4-5-20251001       | Claude CLI          |

> **Note:** A legacy server mode (`mahanai/mahanai`) exists in the model selector but is undocumented and not recommended for use.

Switch models interactively with `/models` (arrow-key selector) or quick-switch with `/mode claude` / `/mode default`.

## Commands

| Command | Description |
|---|---|
| `/models` | Interactive model selector (↑↓ arrows, Enter to confirm, Esc to cancel) |
| `/mode claude` | Quick-switch to Claude Sonnet 4.6 |
| `/mode default` | Quick-switch back to MahanAI Super (server) |
| `/api-key [key]` | Save server API key (omit key for hidden prompt) |
| `/api-key clear` | Remove saved server key |
| `/api-key-nvidia [key]` | Save NVIDIA direct API key |
| `/api-key-nvidia clear` | Remove NVIDIA key, switch back to server |
| `/help` | Show help |
| `/exit` or `/quit` | Leave |

## API Keys

### Server / NVIDIA NIM
1. **Environment:** `MAHANAI_API_KEY=...`
2. **Project `.env`:** `MAHANAI_API_KEY=...`
3. **In-app:** `/api-key your-key`

Keys are stored under `%APPDATA%\MahanAI\config.json` on Windows or `~/.config/mahanai/config.json` on Linux/macOS.

### Claude CLI mode
Claude models use your local `claude` CLI installation. Make sure [Claude Code](https://claude.ai/code) is installed and on your PATH. No extra API key configuration needed inside MahanAI — it uses whatever account Claude CLI is authenticated with.

## Environment Variables

| Variable | Purpose |
|---|---|
| `MAHANAI_API_KEY` | Override saved server API key |
| `MAHANAI_MODEL` | Override default model ID |
| `MAHANAI_STREAM` | Set to `0`/`false`/`no`/`off` to disable streaming |
| `MAHANAI_CONFIG_DIR` | Override config file directory |
| `NO_COLOR` | Disable terminal colors |

## Tools

MahanAI can execute tools on your behalf:

- **run_command** — run shell commands (asks confirmation before destructive ops)
- **read_file** — read a file
- **write_file** — write a file
- **append_file** — append to a file
- **list_directory** — list directory contents

## Develop

```bash
pip install -e .
python -m mahanai
```

## Publish to PyPI

Bump `version` in both `pyproject.toml` and `mahanai/__init__.py`, then:

```bash
pip install build twine
python -m build
python -m twine check dist/*
```

**Windows (PowerShell):**

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-YOUR_TOKEN_HERE"
python -m twine upload dist/mahanai-4.0.0*
```

**macOS / Linux:**

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TOKEN_HERE
python -m twine upload dist/mahanai-4.0.0*
```

`twine` cannot publish without your token; keep it out of git.
