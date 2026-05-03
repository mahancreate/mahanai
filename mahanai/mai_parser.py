"""Parser and interpreter for .mai custom theme files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MaiTheme:
    """Parsed representation of a .mai theme file."""
    name: str = "custom"
    # Theme metadata (from theme.* lines)
    theme_name: str | None = None        # slug used in /themes menu
    theme_pretty_name: str | None = None # human-readable display name
    theme_code_name: str | None = None   # qualified/package identifier
    theme_version: str | None = None
    # Color overrides (hex strings e.g. "#00ff00")
    user_color: str | None = None
    ai_color: str | None = None
    err_color: str | None = None
    banner_color: str | None = None
    ok_color: str | None = None
    warn_color: str | None = None
    banner_gradient: list[str] = field(default_factory=list)
    # Display name overrides
    ai_name: str | None = None
    user_name: str | None = None

    def slug(self) -> str:
        """Return the slug to use in the /themes menu."""
        return self.theme_name or self.name or "custom"

    def display(self) -> str:
        """Return the human-readable display name for the /themes menu."""
        return self.theme_pretty_name or self.slug()


_NAMED_COLORS: dict[str, str] = {
    "red":     "#ff0000",
    "green":   "#00ff00",
    "blue":    "#0000ff",
    "yellow":  "#ffff00",
    "cyan":    "#00ffff",
    "magenta": "#ff00ff",
    "white":   "#ffffff",
    "black":   "#000000",
    "orange":  "#ff8800",
    "pink":    "#ff69b4",
    "purple":  "#8800ff",
    "violet":  "#ee82ee",
    "indigo":  "#4b0082",
    "teal":    "#008080",
    "lime":    "#00ff80",
    "gold":    "#ffd700",
    "silver":  "#c0c0c0",
    "navy":    "#000080",
    "coral":   "#ff7f50",
    "salmon":  "#fa8072",
    "crimson": "#dc143c",
    "maroon":  "#800000",
    "olive":   "#808000",
    "aqua":    "#00ffff",
    "fuchsia": "#ff00ff",
    "gray":    "#808080",
    "grey":    "#808080",
}


def _expand_short_hex(hex_str: str) -> str:
    h = hex_str.lstrip("#")
    return "#" + "".join(c * 2 for c in h)


_HEX3 = re.compile(r'^#[0-9a-fA-F]{3}$')
_HEX6 = re.compile(r'^#[0-9a-fA-F]{6}$')


def _resolve_color(token: str, variables: dict[str, str]) -> str:
    """Resolve a color name, variable reference, or hex literal to a hex string.

    Raises ValueError for unrecognized or malformed color values.
    """
    token = token.strip()
    lower = token.lower()
    if lower in variables:
        return variables[lower]
    if token.startswith("#"):
        if _HEX3.match(token):
            return _expand_short_hex(token)
        if _HEX6.match(token):
            return token.lower()
        raise ValueError(f"invalid hex color {token!r} (expected #RGB or #RRGGBB)")
    if lower in _NAMED_COLORS:
        return _NAMED_COLORS[lower]
    valid = ", ".join(sorted(_NAMED_COLORS))
    raise ValueError(f"unknown color name {token!r} — valid names: {valid}")


def _interpolate_gradient(start_hex: str, end_hex: str, steps: int = 10) -> list[str]:
    def parse(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    sr, sg, sb = parse(start_hex)
    er, eg, eb = parse(end_hex)
    result = []
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0.0
        r = int(sr + (er - sr) * t)
        g = int(sg + (eg - sg) * t)
        b = int(sb + (eb - sb) * t)
        result.append(f"#{r:02x}{g:02x}{b:02x}")
    return result


def _parse_gradient_arg(arg: str, variables: dict[str, str]) -> list[str]:
    parts = [p.strip() for p in arg.split("->")]
    if len(parts) != 2:
        raise ValueError(f"gradient requires 'color1 -> color2' syntax, got {arg!r}")
    start = _resolve_color(parts[0], variables)
    end = _resolve_color(parts[1], variables)
    return _interpolate_gradient(start, end, steps=10)


def _extract_function_call(value: str) -> tuple[str, str] | None:
    """Return (func_name, arg) from 'func("arg")' syntax, or None."""
    m = re.match(r'^(\w[\w-]*)\s*\(\s*"([^"]*)"\s*\)$', value.strip())
    if m:
        return m.group(1), m.group(2)
    return None


# Bare metadata properties (plain string values, no function call)
_META_MAP: dict[str, str] = {
    "theme.name":        "theme_name",
    "theme.pretty.name": "theme_pretty_name",
    "theme.code.name":   "theme_code_name",
    "theme.version":     "theme_version",
}

# Function-call properties
_PROPERTY_MAP: dict[str, str] = {
    "ascii-art.default.color": "banner_gradient",
    "message.user.color":      "user_color",
    "message.ai.color":        "ai_color",
    "message.ai.name":         "ai_name",
    "message.user.name":       "user_name",
    "message.err.color":       "err_color",
    "message.warn.color":      "warn_color",
    "message.ok.color":        "ok_color",
    "message.banner.color":    "banner_color",
}


def parse_mai_file(path: str | Path) -> MaiTheme:
    """Parse a .mai file and return a MaiTheme.

    Raises ValueError for empty files or invalid color values.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    content_lines = [
        l.strip() for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("import ")
    ]
    if not content_lines:
        raise ValueError(f"{path.name!r} is empty — no theme properties found")

    theme = MaiTheme(name=path.stem)
    variables: dict[str, str] = {}

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("import "):
            continue
        if "=" not in line:
            continue

        lhs, _, rhs = line.partition("=")
        lhs = lhs.strip()
        rhs = rhs.strip()

        # Variable declaration: identifier = color_value
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', lhs):
            try:
                variables[lhs.lower()] = _resolve_color(rhs, variables)
            except ValueError as exc:
                raise ValueError(f"line {lineno}: {exc}") from exc
            continue

        # Bare metadata: theme.name = some value (no function call)
        if lhs in _META_MAP:
            setattr(theme, _META_MAP[lhs], rhs)
            continue

        # Property assignment: object.sub.prop = func("arg")
        if lhs not in _PROPERTY_MAP:
            continue
        fn_call = _extract_function_call(rhs)
        if fn_call is None:
            continue
        fn_name, fn_arg = fn_call
        target = _PROPERTY_MAP[lhs]

        try:
            if target == "banner_gradient":
                if fn_name == "gradient":
                    theme.banner_gradient = _parse_gradient_arg(fn_arg, variables)
            elif target in ("user_color", "ai_color", "err_color", "banner_color", "ok_color", "warn_color"):
                if fn_name == "color":
                    setattr(theme, target, _resolve_color(fn_arg, variables))
            elif target in ("ai_name", "user_name"):
                if fn_name == "text":
                    setattr(theme, target, fn_arg)
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc

    return theme
