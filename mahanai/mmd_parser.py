"""Parser for .mmd MahanAI plugin (mahmod) files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MmdAction:
    type: str   # "claude-cmd" | "mahanai-cmd" | "shell-cmd"
    value: str  # the command string to execute


@dataclass
class MmdCommand:
    trigger: str                           # e.g. "/compact"
    actions: list[MmdAction] = field(default_factory=list)


@dataclass
class MmdPlugin:
    name: str
    path: str
    version: str = "1.0"
    codename: str = ""
    reg_store: str = ""
    reg_name: str = ""
    commands: list[MmdCommand] = field(default_factory=list)

    def command_triggers(self) -> list[str]:
        return [c.trigger for c in self.commands]


def _derive_name(stem: str) -> str:
    """Extract plugin name from filename stem like 'example-mahanai-mahmod'."""
    for prefix in ("example-mahanai-", "mahanai-", "plugin-"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def parse_mmd_file(path: str | Path) -> MmdPlugin:
    """Parse a .mmd plugin file and return an MmdPlugin."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    plugin = MmdPlugin(name=_derive_name(path.stem), path=str(path))

    def _str_val(pattern: str) -> str:
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            return ""
        return m.group(1).strip().strip('"').strip("'")

    # plugin.name overrides the filename-derived name
    explicit_name = _str_val(r'^plugin\.name\s*=\s*(.+)$')
    if explicit_name:
        plugin.name = explicit_name

    # Extract version if declared: plugin.version = 1.2.3
    ver_m = re.search(r'^plugin\.version\s*=\s*(.+)$', text, re.MULTILINE)
    if ver_m:
        plugin.version = ver_m.group(1).strip()

    plugin.codename = _str_val(r'^plugin\.codename\s*=\s*(.+)$')
    plugin.reg_store = _str_val(r'^plugin\.reg\.store\s*=\s*(.+)$')
    plugin.reg_name  = _str_val(r'^plugin\.reg\.name\s*=\s*(.+)$')

    # Parse: add command("/trigger", ...) { ... }
    # The arg list may contain nested parens like `if fail create(status = 1)`,
    # so we allow one level of inner parens in the args.
    cmd_pattern = re.compile(
        r'add\s+command\s*\(\s*"([^"]+)"(?:[^()]|\([^()]*\))*\)\s*\{([^}]*)\}',
        re.DOTALL,
    )
    for m in cmd_pattern.finditer(text):
        trigger = m.group(1)
        body = m.group(2)
        cmd = MmdCommand(trigger=trigger)

        # pvd(claude-code)[ use-claude-cmd("/cmd") ]
        cc = re.search(
            r'pvd\s*\(\s*claude-code\s*\)\s*\[\s*use-claude-cmd\s*\(\s*"([^"]*)"\s*\)\s*\]',
            body,
        )
        if cc:
            cmd.actions.append(MmdAction(type="claude-cmd", value=cc.group(1)))

        # pvd(mahanai)[ run("/cmd") ]
        mc = re.search(
            r'pvd\s*\(\s*mahanai\s*\)\s*\[\s*run\s*\(\s*"([^"]*)"\s*\)\s*\]',
            body,
        )
        if mc:
            cmd.actions.append(MmdAction(type="mahanai-cmd", value=mc.group(1)))

        # shell("command")
        sc = re.search(r'shell\s*\(\s*"([^"]*)"\s*\)', body)
        if sc:
            cmd.actions.append(MmdAction(type="shell-cmd", value=sc.group(1)))

        plugin.commands.append(cmd)

    return plugin
