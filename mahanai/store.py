"""MahanAI plugin store — GitHub-backed plugin registry."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .mmd_parser import MmdPlugin, parse_mmd_file

GH_API = "https://api.github.com"
STORE_TOPIC = "mahanai-plugin"
_UA = "MahanAI-Store/1.0"


# ---------------------------------------------------------------------------
# Low-level GitHub API helper
# ---------------------------------------------------------------------------

def _gh(method: str, endpoint: str, token: str | None = None, body: dict | None = None) -> object:
    url = endpoint if endpoint.startswith("http") else f"{GH_API}{endpoint}"
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _UA,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub {exc.code}: {exc.read().decode(errors='replace')}") from exc


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------

def get_store_token() -> str | None:
    from .config import _read_config
    return _read_config().get("store_token") or None


def save_store_token(token: str) -> None:
    from .config import _read_config, _write_config
    data = _read_config()
    data["store_token"] = token
    _write_config(data)


def remove_store_token() -> None:
    from .config import _read_config, _write_config
    data = _read_config()
    data.pop("store_token", None)
    _write_config(data)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def whoami(token: str) -> str:
    """Return the authenticated GitHub username."""
    return _gh("GET", "/user", token=token)["login"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_plugin(token: str, mmd_path: Path) -> str:
    """
    Create/update a GitHub repo <user>/<plugin.codename> and push the .mmd file.
    Returns the repo HTML URL.
    """
    plugin = parse_mmd_file(mmd_path)

    if not plugin.name or plugin.name == mmd_path.stem:
        raise ValueError("plugin.name is required for store upload")
    if not plugin.codename:
        raise ValueError("plugin.codename is required for store upload")
    if not plugin.reg_store:
        raise ValueError("plugin.reg.store is required for store upload")

    gh_user = whoami(token)
    repo_name = plugin.codename

    # Create repo (ignore 422 = already exists)
    try:
        _gh("POST", "/user/repos", token=token, body={
            "name": repo_name,
            "description": f"MahanAI plugin: {plugin.name}",
            "private": False,
            "auto_init": False,
        })
    except RuntimeError as exc:
        if "422" not in str(exc) and "already exists" not in str(exc).lower():
            raise

    # Tag with topics so the store can find it
    topics = list({STORE_TOPIC, plugin.reg_store.replace(" ", "-").lower()})
    _gh("PUT", f"/repos/{gh_user}/{repo_name}/topics", token=token, body={"names": topics})

    # Push the .mmd file (create or update)
    file_name = mmd_path.name
    content_b64 = base64.b64encode(mmd_path.read_bytes()).decode()

    sha: str | None = None
    try:
        existing = _gh("GET", f"/repos/{gh_user}/{repo_name}/contents/{file_name}", token=token)
        sha = existing.get("sha")  # type: ignore[union-attr]
    except RuntimeError:
        pass

    put_body: dict = {
        "message": f"publish {plugin.name} v{plugin.version}",
        "content": content_b64,
    }
    if sha:
        put_body["sha"] = sha

    _gh("PUT", f"/repos/{gh_user}/{repo_name}/contents/{file_name}", token=token, body=put_body)

    return f"https://github.com/{gh_user}/{repo_name}"


# ---------------------------------------------------------------------------
# Browse / search
# ---------------------------------------------------------------------------

def search_plugins(query: str = "", token: str | None = None) -> list[dict]:
    """Return a list of GitHub repo dicts tagged as mahanai plugins."""
    q = f"topic:{STORE_TOPIC}"
    if query.strip():
        q += f" {query.strip()}"
    enc = urllib.parse.quote(q)
    result = _gh("GET", f"/search/repositories?q={enc}&per_page=30&sort=updated", token=token)
    return result.get("items", [])  # type: ignore[union-attr,return-value]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def _plugins_dir() -> Path:
    """Local directory where downloaded store plugins are stored."""
    from .config import config_file_path
    return config_file_path().parent / "store-plugins"


def install_plugin(repo_full_name: str, token: str | None = None) -> Path:
    """
    Download the .mmd file from <user>/<repo> and save it to the local store
    cache. Returns the saved file path.
    """
    contents = _gh("GET", f"/repos/{repo_full_name}/contents/", token=token)
    mmd_files = [f for f in contents if isinstance(f, dict) and f["name"].endswith(".mmd")]  # type: ignore[union-attr]
    if not mmd_files:
        raise RuntimeError(f"No .mmd file found in {repo_full_name}")

    meta = mmd_files[0]
    raw_url = meta["download_url"]

    req = urllib.request.Request(raw_url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read()

    dest = _plugins_dir()
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / meta["name"]
    out.write_bytes(content)
    return out


def find_plugin_repo(codename: str, token: str | None = None) -> str | None:
    """
    Search the store for a plugin whose repo name matches codename.
    Returns '<user>/<repo>' or None.
    """
    enc = urllib.parse.quote(f"topic:{STORE_TOPIC} {codename} in:name")
    result = _gh("GET", f"/search/repositories?q={enc}&per_page=5", token=token)
    items = result.get("items", [])  # type: ignore[union-attr]
    for item in items:
        if item.get("name") == codename:
            return item["full_name"]
    return items[0]["full_name"] if items else None
