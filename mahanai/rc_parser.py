"""Parser for .mahanairc project configuration files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RcConfig:
    context_files: list[str] = field(default_factory=list)
    mmd_plugins: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    system_extras: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_BUILTIN_PACKAGE_EXTRAS: dict[str, str] = {
    "python-dev-kit": (
        "The project uses Python. Prefer idiomatic Python 3.10+ style with type hints, "
        "dataclasses, and pathlib where appropriate."
    ),
    "csc": (
        "Before finalizing any answer, perform a quick sanity check: "
        "is the response correct, complete, and consistent with the request?"
    ),
    "web-dev-kit": (
        "The project uses web technologies (HTML/CSS/JS). "
        "Prefer modern ES2022+ JavaScript, semantic HTML, and CSS custom properties."
    ),
    "rust-dev-kit": (
        "The project uses Rust. Follow idiomatic Rust patterns: ownership, borrowing, "
        "Result/Option chaining, and clippy lint compliance."
    ),
    "go-dev-kit": (
        "The project uses Go. Follow idiomatic Go: explicit error handling, "
        "goroutines/channels where appropriate, and gofmt-formatted output."
    ),
}

_NOOP_PACKAGES = {
    "mahanai", "default", "defaults", "mahanairc", "crt",
}


def _expand_env(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def parse_rc_file(path: str | Path) -> RcConfig:
    """Parse a .mahanairc file and return an RcConfig.

    Supported directives:
      import X from Y
      load(location="..." type=context)
      load(location="..." type=mmd)
      load(pkg1, pkg2, ...)
      defaults(def) / start(mahanai)   — recognized, no-op
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rc = RcConfig()

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # import X from Y
        if re.match(r'^import\s+\w[\w-]*\s+from\s+[\w-]+', line):
            continue

        # load(location="..." type=context)
        m_ctx = re.match(
            r'^load\s*\(\s*location\s*=\s*"([^"]+)"\s+type\s*=\s*context\s*\)', line
        )
        if m_ctx:
            rc.context_files.append(_expand_env(m_ctx.group(1)))
            continue

        # load(location="..." type=mmd)
        m_mmd = re.match(
            r'^load\s*\(\s*location\s*=\s*"([^"]+)"\s+type\s*=\s*mmd\s*\)', line
        )
        if m_mmd:
            rc.mmd_plugins.append(_expand_env(m_mmd.group(1)))
            continue

        # load(pkg1, pkg2, ...) — named packages
        m_pkg = re.match(r'^load\s*\(([^)]+)\)', line)
        if m_pkg:
            pkg_names = [p.strip() for p in m_pkg.group(1).split(",") if p.strip()]
            for pkg in pkg_names:
                if pkg in _NOOP_PACKAGES:
                    continue
                if pkg in _BUILTIN_PACKAGE_EXTRAS:
                    rc.system_extras.append(_BUILTIN_PACKAGE_EXTRAS[pkg])
                else:
                    rc.packages.append(pkg)
                    rc.warnings.append(f"line {lineno}: unknown package '{pkg}' — ignored")
            continue

        # defaults(...) / start(...) — recognized no-ops
        if re.match(r'^(defaults|start)\s*\(', line):
            continue

    return rc
