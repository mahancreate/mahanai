<img width="700" height="300" alt="(M MahanAI)" src="https://github.com/user-attachments/assets/fc20edd6-601f-4740-9ac2-e2db61c8f49f" />

# MahanAI Super

Terminal AI agent (Super 2.0) with multi-model support, streaming chat, tools, and a built-in Claude CLI mode. Docs: [MahanAI](https://mahancreate.github.io/mahanai).

## Install

```bash
pip install mahanai
mahanai
```

## Launch Options

| Flag | Description |
|---|---|
| `--compact` | Compact mode: renders a small **MAI** ASCII logo and a trimmed header |
| `--server` | Start the gateway server instead of the chat loop |
| `--port PORT` | Gateway server port (default: `8080`) |
| `--type TYPE` | Gateway API type: `openai` (default) or `anthropic` |
| `--api-key KEY` | API key clients must send to the gateway; also used as the Anthropic backend key |

```bash
mahanai --compact
mahanai --server --port 8080 --type openai --api-key my-secret
```

## Gateway Server

`--server` starts a local HTTP gateway that exposes **all configured providers** behind a single endpoint. Any tool that speaks the OpenAI or Anthropic wire format (Cursor, Continue, LM Studio, Claude Code, etc.) can point at it.

### Starting the server

```bash
# OpenAI-compatible gateway on port 8080 (default)
mahanai --server

# Custom port
mahanai --server --port 9000

# Anthropic-compatible gateway
mahanai --server --type anthropic

# With an API key (clients must send Authorization: Bearer <key>)
mahanai --server --port 4343 --api-key sk-gaming

# Anthropic gateway with your Anthropic API key
mahanai --server --type anthropic --api-key sk-ant-...
```

### Endpoints

| Server type | Endpoint | Purpose |
|---|---|---|
| `openai` | `POST /v1/chat/completions` | Chat completions |
| `anthropic` | `POST /v1/messages` | Messages API |
| both | `GET /v1/models` | List all available models |

### Model routing

The gateway automatically routes requests to the right backend based on the `model` field in the request:

| Model ID | Provider | Credentials needed |
|---|---|---|
| `mahanai/mahanai` | NVIDIA NIM (server) | `/api-key` |
| `meta/llama-3.3-70b-instruct` | NVIDIA NIM (direct) | `/api-key-nvidia` |
| `claude-opus-4-7` | Anthropic | `--api-key sk-ant-...` |
| `claude-sonnet-4-6` | Anthropic | `--api-key sk-ant-...` |
| `claude-haiku-4-5-20251001` | Anthropic | `--api-key sk-ant-...` |
| `gpt-5.4`, `gpt-5.2`, `gpt-5.2-codex` … | OpenAI Codex | `/codex-login` |
| custom model ID | Custom endpoint | `/custom` |

### Format conversion

Requests are automatically converted between formats:

- **OpenAI gateway + Claude model** → converts OpenAI→Anthropic, calls Anthropic API, converts response back
- **Anthropic gateway + NVIDIA/Codex model** → converts Anthropic→OpenAI, calls backend, converts response back
- **Same-format routes** (OpenAI gateway + NVIDIA/Codex) → transparent proxy, no conversion overhead

Streaming SSE is preserved end-to-end for all routes.

### Authentication

If `--api-key` is supplied, the server validates every incoming request:

```
Authorization: Bearer <your-key>
```

Requests with a missing or wrong key receive HTTP 401. Omit `--api-key` to run with open access (local use only).

### Example curl

```bash
curl http://localhost:4343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-gaming" \
  -d '{
    "model": "gpt-5.2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Hello!"}
    ]
  }'
```

---

## Themes

MahanAI supports four built-in terminal color themes, including two designed for colorblind accessibility, plus fully custom themes written in the `.mai` theme language.

### Built-in themes

| Theme | Description |
|---|---|
| `midnight` | Dark terminal — purple→cyan gradient banner (default) |
| `light` | Light terminal — navy→teal gradient banner |
| `midnight-cb` | Dark + colorblind-friendly — blue replaces green, yellow replaces red |
| `light-cb` | Light + colorblind-friendly |

```
/themes                   # list all themes
/themes midnight
/themes light
/themes midnight-cb
/themes light-cb
```

Themes persist across sessions (saved to `config.json`). The banner gradient, prompt colors, and status colors all update when you switch.

### Custom themes (.mai files)

You can create fully custom themes using `.mai` files — a simple domain-specific language designed for MahanAI theming.

```
/theme-load path/to/mytheme.mai   # load and apply a custom theme
/theme-unload                     # remove the custom theme, revert to midnight
```

Once loaded, the custom theme appears as a named entry in `/themes` alongside the built-in themes and persists across sessions. You can switch away and back to it just like any other theme.

#### .mai file syntax

```
# Comments start with #

# Declare the theme's identity
theme.name        = my-theme          # slug used in /themes
theme.pretty.name = My Custom Theme   # display name shown in /themes list
theme.code.name   = author.my-theme   # optional qualified identifier
theme.version     = 1.0.0

# Import the default built-in themes as a base
import mahanai-themes from requirements

# Define named color aliases (hex codes or CSS color names)
blue   = #0000FF
green  = #00FF00
red    = #FF0000
yellow = #FFFF00

# Banner ASCII art gradient (start color -> end color)
ascii-art.default.color = gradient("blue -> red")

# Chat message colors
message.user.color = color("green")     # "You:" prompt color
message.ai.color   = color("yellow")    # "MahanAI:" response color

# Display name overrides
message.ai.name   = text("my little ai")   # replaces "MahanAI" in chat
message.user.name = text("Boss")            # replaces "You" in chat

# Additional color slots
message.err.color    = color("red")
message.warn.color   = color("yellow")
message.ok.color     = color("green")
message.banner.color = color("purple")
```

#### Color values

Colors can be specified anywhere as:

| Format | Example |
|---|---|
| Hex code | `#FF0000`, `#F00` |
| Named color | `red`, `green`, `blue`, `cyan`, `magenta`, `yellow`, `orange`, `purple`, `teal`, `gold`, `navy`, `pink`, `lime`, … |
| Variable reference | `blue` (a name defined earlier in the same file) |

#### Gradient syntax

`gradient("start -> end")` interpolates 10 evenly-spaced hex colors between two colors and applies them across the banner ASCII art. Both endpoints accept hex codes or named colors.

```
ascii-art.default.color = gradient("purple -> cyan")
ascii-art.default.color = gradient("#ff0000 -> #0000ff")
```

---

## Command Approvals

Every tool action MahanAI takes on your behalf requires explicit approval before it runs. The prompt style depends on the action type:

### Shell commands

```
  Shell Command
  npm install react

  [A] Allow    [W] Always Allow (npm)    [D] Deny
  >
```

### Git commands

```
  Git Command
  git push origin main

  [A] Allow    [D] Deny
  >
```

### GitHub CLI commands

```
  GitHub Command
  gh pr create --title "..."

  [A] Allow    [D] Deny
  >
```

### File operations

```
  Write / Create File
  C:\Users\Mahan\project\main.py

  [A] Allow    [W] Always Allow (Write / Create File)    [D] Deny
  >
```

### Approval options

| Key | Effect |
|---|---|
| `A` | Allow once |
| `W` | Always Allow — stored in `config.json`, never asked again for that command prefix or file op |
| `D` | Deny — optionally type an instruction the AI will receive as the tool result |

**Always Allow** is available for shell commands (stored by command prefix, e.g. `npm`) and file operations (stored by operation type). It is **not** available for git or GitHub commands — those always prompt.

**Destructive commands** (`rm -rf`, `format`, `shutdown`, etc.) are flagged `[DESTRUCTIVE]` and Always Allow is disabled for them regardless.

### Managing stored rules

```
/approvals          # list all Always Allow rules
/approvals clear    # remove all Always Allow rules
```

---

## Models

MahanAI Super supports multiple backends selectable at runtime via `/models`.

### NVIDIA NIM

| Pretty Name       | Model ID                        | Backend             |
|-------------------|---------------------------------|---------------------|
| Llama 3.3 70B     | meta/llama-3.3-70b-instruct     | NVIDIA NIM (direct) |

> **Note:** A legacy server mode (`mahanai/mahanai`) exists in the model selector but is undocumented and not recommended for use.

### Claude

| Pretty Name       | Model ID                  | Backend    |
|-------------------|---------------------------|------------|
| Claude Opus 4     | claude-opus-4-7           | Claude CLI |
| Claude Sonnet 4.6 | claude-sonnet-4-6         | Claude CLI |
| Claude Haiku 4.5  | claude-haiku-4-5-20251001 | Claude CLI |

### OpenAI Codex

Seven models available, each accessible in **Direct** and **Indirect** mode (see [OpenAI Codex](#openai-codex-1) below):

| Pretty Name        | Model ID            |
|--------------------|---------------------|
| GPT-5.4            | gpt-5.4             |
| GPT-5.2-Codex      | gpt-5.2-codex       |
| GPT-5.1-Codex-Max  | gpt-5.1-codex-max   |
| GPT-5.4-Mini       | gpt-5.4-mini        |
| GPT-5.3-Codex      | gpt-5.3-codex       |
| GPT-5.2            | gpt-5.2             |
| GPT-5.1-Codex-Mini | gpt-5.1-codex-mini  |

Switch models interactively with `/models` (arrow-key selector) or quick-switch with `/mode claude` / `/mode default`.

> **Default model:** MahanAI starts on **Claude Haiku 4.5** out of the box.

### Ollama

Run local models via any Ollama-compatible server. Each provider is saved to config and appears as its own named entry in `/models`.

```
/add-ollama <name> <address> <port> [api_key]
```

Examples:
```
/add-ollama llama3   localhost        11434
/add-ollama mistral  192.168.1.100    11434
/add-ollama cloud    https://my.server.com  443
```

**Smart URL rules applied automatically:**
- `http://` / `https://` prefixes are stripped from the address and the correct scheme is re-applied
- Port `443` with no explicit scheme → `https://` is used automatically
- Domain addresses (e.g. `my.server.com`) → port is omitted from the URL; local IPs and `localhost` keep the port

Resulting base URL is always `<scheme>://<host>[:<port>]/api/v1`.

**Update an existing provider** (keeps the current API key if none given):
```
/change-ollama <name> <new-address> <new-port> [new_api_key]
```

**Remove a provider:**
```
/remove-ollama <name>
```

After adding, open `/models` and select the provider by name to start chatting. Providers persist across sessions.

### Custom Endpoint

Point MahanAI at any OpenAI-compatible API (LM Studio, vLLM, OpenRouter, etc.):

```
/custom http://localhost:11434/v1 llama3 [optional-api-key]
```

Once saved, select **Custom Endpoint** from `/models` to start using it. The config persists across sessions.

---

## Commands

| Command | Description |
|---|---|
| `/models` | Interactive model selector (↑↓ arrows, Enter to confirm, Esc to cancel) |
| `/mode claude` | Quick-switch to Claude Sonnet 4.6 |
| `/mode default` | Quick-switch back to MahanAI Super (server) |
| `/effort <level>` | Set reasoning effort: `low`, `medium`, `high`, `very-high` |
| `/plan on` | Enable plan mode — MahanAI outlines a plan before every response |
| `/plan off` | Disable plan mode |
| `/themes` | List available color themes (built-in and loaded .mai themes) |
| `/themes <name>` | Switch theme by slug — built-in or custom .mai |
| `/theme-load <path>` | Load a `.mai` theme file and add it to the themes menu |
| `/theme-unload` | Remove the active custom theme and revert to midnight |
| `/approvals` | Show stored Always Allow rules |
| `/approvals clear` | Remove all Always Allow rules |
| `/api-key [key]` | Save server API key (omit key for hidden prompt) |
| `/api-key clear` | Remove saved server key |
| `/api-key-nvidia [key]` | Save NVIDIA direct API key |
| `/api-key-nvidia clear` | Remove NVIDIA key, switch back to server |
| `/codex-login` | Sign in to OpenAI via browser (Codex Direct mode) |
| `/codex-logout` | Remove saved OpenAI Codex credentials |
| `/custom [url [model [key]]]` | Configure a custom OpenAI-compatible endpoint |
| `/custom clear` | Remove saved custom endpoint |
| `/add-ollama <name> <addr> <port> [key]` | Add an Ollama provider to the model list |
| `/change-ollama <name> <addr> <port> [key]` | Update address/port/key of an existing Ollama provider |
| `/remove-ollama <name>` | Remove a saved Ollama provider |
| `/help` | Show help |
| `/exit` or `/quit` | Leave |

---

## Effort Levels

`/effort` controls how much reasoning the model applies before responding.

| Level | Effect |
|---|---|
| `low` | Concise and fast. Minimal reasoning. |
| `medium` | Balanced (default). |
| `high` | Careful, thorough reasoning before responding. |
| `very-high` | Maximum reasoning depth. ⚠️ Significantly higher token usage and slower responses. |

```
/effort high
/effort very-high
```

> **Note:** Effort is disabled for **Claude Haiku 4.5** — it does not support extended thinking. Switch to Opus or Sonnet to use effort levels.

For **OpenAI Codex** models, effort maps to the `reasoning.effort` parameter (`low` / `medium` / `high`).
For **Claude** models (Opus, Sonnet), the effort instruction is prepended to your prompt to guide reasoning depth.

## Plan Mode

Plan mode instructs MahanAI to outline its approach before taking action on every message — useful for complex multi-step tasks where you want visibility into the reasoning before execution.

```
/plan on    # enable
/plan off   # disable
```

Plan mode works across all model backends.

---

## API Keys

### Server / NVIDIA NIM
1. **Environment:** `MAHANAI_API_KEY=...`
2. **Project `.env`:** `MAHANAI_API_KEY=...`
3. **In-app:** `/api-key your-key`

Keys are stored under `%APPDATA%\MahanAI\config.json` on Windows or `~/.config/mahanai/config.json` on Linux/macOS.

### Claude CLI mode
Claude models use your local `claude` CLI installation. Make sure [Claude Code](https://claude.ai/code) is installed and on your PATH. No extra API key configuration needed inside MahanAI — it uses whatever account Claude CLI is authenticated with.

### OpenAI Codex

MahanAI supports two Codex authentication modes:

#### Direct mode
Signs in to your OpenAI account via a browser-based OAuth PKCE flow — no API key needed.

```
/codex-login
```

This opens your browser to `auth.openai.com`. After you approve, MahanAI receives and stores the access token automatically. Tokens are refreshed silently before they expire (saved to the same `config.json` as other keys).

#### Indirect mode
Reads credentials from a locally installed and signed-in [OpenAI Codex CLI](https://github.com/openai/codex). MahanAI looks for `auth.json` in these locations:

| Platform | Paths checked |
|---|---|
| Windows | `%LOCALAPPDATA%\OpenAI\Codex\auth.json`, `~\.codex\auth.json` |
| macOS / Linux | `~/.codex/auth.json`, `~/.config/codex/auth.json` |

If no token file is found, MahanAI falls back to running the `codex` CLI as a subprocess (requires Codex CLI on your PATH).

To use indirect mode, install and sign in to the Codex CLI first:

```bash
npm i -g @openai/codex
codex login
```

Then select any **OpenAI Codex (Indirect)** model from `/models`.

### Custom Endpoint

Use `/custom` to connect to any OpenAI-compatible server — Ollama, LM Studio, vLLM, OpenRouter, or your own deployment.

**Interactive setup** (prompts for each field):
```
/custom
```

**One-liner:**
```
/custom <base-url> [model] [api-key]
```

Examples:
```
/custom http://localhost:11434/v1 llama3
/custom http://localhost:1234/v1 mistral-7b
/custom https://openrouter.ai/api/v1 openai/gpt-4o sk-or-...
```

- `base-url` — the `/v1` base URL of the server
- `model` — model ID to send in requests (defaults to `gpt-3.5-turbo` if omitted)
- `api-key` — leave blank if the server doesn't require one

After saving, run `/models` and select **Custom Endpoint**. To remove:

```
/custom clear
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `MAHANAI_API_KEY` | Override saved server API key |
| `MAHANAI_MODEL` | Override default model ID |
| `MAHANAI_STREAM` | Set to `0`/`false`/`no`/`off` to disable streaming |
| `MAHANAI_CONFIG_DIR` | Override config file directory |
| `NO_COLOR` | Disable terminal colors |

---

## Tools

Every tool action is shown to you for approval before it runs (see [Command Approvals](#command-approvals) above).

| Tool | Description |
|---|---|
| `run_command` | Run a shell command |
| `read_file` | Read a file |
| `write_file` | Create or overwrite a file |
| `append_file` | Append to a file |
| `list_directory` | List directory contents |

---

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
