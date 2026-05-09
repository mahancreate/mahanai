<div align="center">

<img width="700" height="300" alt="(M MahanAI)" src="https://github.com/user-attachments/assets/fc20edd6-601f-4740-9ac2-e2db61c8f49f" />

# MahanAI

**A terminal AI agent with a plugin ecosystem, gateway server, and full multi-model support.**

[![PyPI version](https://img.shields.io/pypi/v/mahanai?color=blueviolet)](https://pypi.org/project/mahanai/)
[![Python](https://img.shields.io/pypi/pyversions/mahanai)](https://pypi.org/project/mahanai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/mahanai)](https://pypi.org/project/mahanai/)

[📖 Docs](https://mahancreate.github.io/mahanai) · [🐛 Issues](https://github.com/maifoundation/mahanai/issues) · [🔌 Plugin Store](#plugin-store)

</div>

---

## What is MahanAI?

MahanAI is a terminal AI agent you install with pip. It gives you a powerful chat interface, agentic tool use (run commands, read/write files, search the web), and a **local gateway server** that makes all your configured AI providers available as a single unified API endpoint — so any tool that speaks OpenAI or Anthropic format can point at it.

On top of that: a **plugin system** (`.mmd` files) with a GitHub-backed store, **custom themes** (`.mai` files), conversation branching, effort levels, plan mode, and 60+ slash commands.

---

## Install

```bash
pip install mahanai
mahanai
```

That's it. An onboarding wizard runs on first launch to help you pick a model and set up API keys.

---

## Highlights

- **Gateway server** — expose all your providers behind one OpenAI/Anthropic-compatible endpoint; works with Cursor, Continue, LM Studio, Claude Code, and more
- **Plugin store** — install community `.mmd` plugins with `/store install`, publish your own with `/store upload`
- **Custom themes** — write `.mai` theme files with gradients, color aliases, and display name overrides
- **Multi-model** — Claude (via Claude Code), NVIDIA NIM, OpenAI Codex, Ollama, and any OpenAI-compatible endpoint
- **Agentic tools** — shell commands, file read/write/edit, web search, Python REPL, URL fetch — all with approval prompts and inline diffs
- **Plan mode & effort levels** — `/plan on` to outline before acting; `/effort high` for deeper reasoning
- **Conversation branching** — save and restore conversation states to explore multiple paths
- **Shell history awareness** — inject your recent bash/zsh history into context with `/shell-history inject`
- **Cost tracking** — `/cost` shows session token usage and estimated spend
- **Desktop notifications** — get pinged when a long generation finishes

---

## Gateway Server

Start a local HTTP server that routes requests to the right backend automatically:

```bash
mahanai --server                          # OpenAI-compatible on port 8080
mahanai --server --type anthropic         # Anthropic-compatible
mahanai --server --port 9000 --api-key sk-gaming
```

Point any OpenAI-compatible client at `http://localhost:8080` and use model IDs like `claude-sonnet-4-6`, `meta/llama-3.3-70b-instruct`, or your Ollama model name — MahanAI routes and converts formats automatically, including SSE streaming end-to-end.

A browser-based chat UI is also available at `http://localhost:8080` when the server is running.

---

## Models

Switch models interactively with `/models` or quick-switch with `/mode claude` / `/mode default`.

| Provider | How to connect |
|---|---|
| **Claude** (Opus, Sonnet, Haiku) | Install [Claude Code](https://claude.ai/code) and sign in |
| **NVIDIA NIM** (Llama 3.3 70B) | `/api-key-nvidia your-key` |
| **OpenAI Codex** | `/codex-login` (browser OAuth, no API key needed) |
| **Ollama** | `/add-ollama name localhost 11434` |
| **Any OpenAI-compatible API** | `/custom http://your-server/v1 model-name` |

> Default model on first launch: **Claude Haiku 4.5**

---

## Plugins

Plugins are `.mmd` files that register new slash commands. They can delegate to Claude Code, MahanAI itself, or the shell.

```bash
# Install from the store
/store install mahancreate/maifoundation.example.mahmod

# Load a local plugin
/plugin-load path/to/my-plugin.mmd

# Browse the store
/store browse
/store search compact
```

Publishing your own plugin is one command:

```bash
/store upload path/to/my-plugin.mmd
```

This creates a public GitHub repo, pushes the `.mmd` file, and tags it so it shows up in `/store browse`.

---

## Themes

Four built-in themes (`midnight`, `light`, `midnight-cb`, `light-cb`) plus full custom theme support via `.mai` files:

```bash
/themes midnight          # switch theme
/theme-load mytheme.mai   # load a custom theme
```

A `.mai` theme file looks like this:

```
theme.name        = my-theme
theme.pretty.name = My Custom Theme

blue = #5B8DEF
gold = #F5C842

ascii-art.default.color = gradient("blue -> gold")
message.ai.color        = color("blue")
message.user.color      = color("gold")
message.ai.name         = text("assistant")
```

---

## Project Config (`.mahanairc`)

Place a `.mahanairc` in any project directory and MahanAI loads it automatically — pre-loading context files, auto-installing plugins, and activating dev kit extras scoped to that workspace.

```
load(location="context.md" type=context)
load(location="plugins/my-plugin.mmd" type=mmd)
load(python-dev-kit)
```

---

## Key Commands

| Command | Description |
|---|---|
| `/models` | Interactive model picker |
| `/effort <low\|medium\|high\|very-high>` | Set reasoning depth |
| `/plan on\|off` | Outline approach before every response |
| `/auto on\|off` | Autonomous mode (skip approval prompts) |
| `/branch save <name>` | Snapshot conversation state |
| `/branch load <name>` | Restore a snapshot |
| `/cost` | Show session token usage and cost |
| `/memory add <text>` | Save a persistent memory |
| `/shell-history inject` | Add recent shell history to context |
| `/store browse` | Browse the plugin store |
| `/cmd` | Fuzzy-search all 60+ commands |
| `/init` | Generate a `MAHANAI.md` workspace context file |

Full command reference: [mahancreate.github.io/mahanai](https://mahancreate.github.io/mahanai)

---

## API Keys

| Provider | How |
|---|---|
| NVIDIA NIM | `/api-key your-key` or `MAHANAI_API_KEY=...` |
| NVIDIA direct | `/api-key-nvidia your-key` |
| Claude | Handled by Claude Code — no extra config |
| OpenAI Codex | `/codex-login` (browser OAuth) |

Keys are stored in `~/.config/mahanai/config.json` (Linux/macOS) or `%APPDATA%\MahanAI\config.json` (Windows).

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `MAHANAI_API_KEY` | Override saved server API key |
| `MAHANAI_MODEL` | Override default model |
| `MAHANAI_STREAM` | Set to `0` to disable streaming |
| `MAHANAI_CONFIG_DIR` | Override config directory |
| `NO_COLOR` | Disable terminal colors |

---

## License

MIT © The MahanAI Foundation