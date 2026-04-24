"""Terminal colors (Windows-safe via colorama)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import colorama

if TYPE_CHECKING:
    from mahanai.mai_parser import MaiTheme

_THEMES: dict[str, dict] = {
    "midnight": {
        "USER":   lambda: colorama.Fore.CYAN + colorama.Style.BRIGHT,
        "BOT":    lambda: colorama.Fore.GREEN + colorama.Style.BRIGHT,
        "ERR":    lambda: colorama.Fore.RED + colorama.Style.BRIGHT,
        "BANNER": lambda: colorama.Fore.MAGENTA + colorama.Style.BRIGHT,
        "OK":     lambda: colorama.Fore.GREEN,
        "DIM":    lambda: colorama.Style.DIM,
        "WARN":   lambda: colorama.Fore.YELLOW + colorama.Style.BRIGHT,
        "banner_colors": [
            "#7c3aed", "#6d28d9", "#5b21b6",
            "#4338ca", "#3730a3",
            "#2563eb", "#1d4ed8",
            "#0284c7", "#06b6d4", "#22d3ee",
        ],
    },
    "light": {
        "USER":   lambda: colorama.Fore.BLUE + colorama.Style.BRIGHT,
        "BOT":    lambda: colorama.Fore.GREEN,
        "ERR":    lambda: colorama.Fore.RED + colorama.Style.BRIGHT,
        "BANNER": lambda: colorama.Fore.MAGENTA,
        "OK":     lambda: colorama.Fore.GREEN,
        "DIM":    lambda: colorama.Style.DIM,
        "WARN":   lambda: colorama.Fore.YELLOW,
        "banner_colors": [
            "#1e3a8a", "#1d4ed8", "#2563eb",
            "#0284c7", "#0369a1",
            "#0891b2", "#06b6d4", "#14b8a6",
        ],
    },
    "midnight-cb": {
        "USER":   lambda: colorama.Fore.CYAN + colorama.Style.BRIGHT,
        "BOT":    lambda: colorama.Fore.BLUE + colorama.Style.BRIGHT,
        "ERR":    lambda: colorama.Fore.YELLOW + colorama.Style.BRIGHT,
        "BANNER": lambda: colorama.Fore.CYAN + colorama.Style.BRIGHT,
        "OK":     lambda: colorama.Fore.BLUE + colorama.Style.BRIGHT,
        "DIM":    lambda: colorama.Style.DIM,
        "WARN":   lambda: colorama.Fore.YELLOW + colorama.Style.BRIGHT,
        "banner_colors": [
            "#1e40af", "#1d4ed8", "#2563eb",
            "#3b82f6", "#60a5fa",
            "#0284c7", "#0ea5e9", "#22d3ee",
        ],
    },
    "light-cb": {
        "USER":   lambda: colorama.Fore.BLUE + colorama.Style.BRIGHT,
        "BOT":    lambda: colorama.Fore.BLUE,
        "ERR":    lambda: colorama.Fore.YELLOW,
        "BANNER": lambda: colorama.Fore.BLUE,
        "OK":     lambda: colorama.Fore.BLUE,
        "DIM":    lambda: colorama.Style.DIM,
        "WARN":   lambda: colorama.Fore.YELLOW,
        "banner_colors": [
            "#1e3a8a", "#1e40af", "#1d4ed8",
            "#2563eb", "#3b82f6",
            "#0369a1", "#0284c7", "#0891b2",
        ],
    },
}

THEME_NAMES = list(_THEMES.keys())
THEME_DISPLAY = {
    "midnight":    "Midnight",
    "light":       "Light",
    "midnight-cb": "Midnight (Colorblind Friendly)",
    "light-cb":    "Light (Colorblind Friendly)",
}

# Module-level color variables — mutated by apply_theme() / apply_mai_theme()
RST: str = ""
USER: str = ""
BOT: str = ""
ERR: str = ""
BANNER: str = ""
OK: str = ""
DIM: str = ""
WARN: str = ""
banner_colors: list[str] = []

# Display names — mutated by apply_mai_theme()
AI_NAME: str = "MahanAI"
USER_NAME: str = "You"

# Registered .mai themes: slug -> file path (populated at runtime)
MAI_THEMES: dict[str, str] = {}


def apply_theme(name: str) -> None:
    """Apply a named theme, updating module-level color variables."""
    global RST, USER, BOT, ERR, BANNER, OK, DIM, WARN, banner_colors
    if os.environ.get("NO_COLOR", "").strip():
        RST = USER = BOT = ERR = BANNER = OK = DIM = WARN = ""
        banner_colors = ["#ffffff"] * 10
        return
    theme = _THEMES.get(name) or _THEMES["midnight"]
    RST    = colorama.Style.RESET_ALL
    USER   = theme["USER"]()
    BOT    = theme["BOT"]()
    ERR    = theme["ERR"]()
    BANNER = theme["BANNER"]()
    OK     = theme["OK"]()
    DIM    = theme["DIM"]()
    WARN   = theme["WARN"]()
    banner_colors = list(theme["banner_colors"])


def _hex_to_ansi(hex_color: str, bright: bool = False) -> str:
    """Convert a #RRGGBB hex color to an ANSI 24-bit foreground escape sequence."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    code = f"\x1b[38;2;{r};{g};{b}m"
    if bright:
        code += "\x1b[1m"
    return code


def apply_mai_theme(mai_theme: "MaiTheme") -> None:
    """Apply a parsed MaiTheme on top of the currently active base theme."""
    global USER, BOT, ERR, BANNER, OK, WARN, banner_colors, AI_NAME, USER_NAME
    if os.environ.get("NO_COLOR", "").strip():
        return
    if mai_theme.user_color:
        USER = _hex_to_ansi(mai_theme.user_color, bright=True)
    if mai_theme.ai_color:
        BOT = _hex_to_ansi(mai_theme.ai_color, bright=True)
    if mai_theme.err_color:
        ERR = _hex_to_ansi(mai_theme.err_color, bright=True)
    if mai_theme.banner_color:
        BANNER = _hex_to_ansi(mai_theme.banner_color, bright=True)
    if mai_theme.ok_color:
        OK = _hex_to_ansi(mai_theme.ok_color)
    if mai_theme.warn_color:
        WARN = _hex_to_ansi(mai_theme.warn_color, bright=True)
    if mai_theme.banner_gradient:
        banner_colors = list(mai_theme.banner_gradient)
    if mai_theme.ai_name:
        AI_NAME = mai_theme.ai_name
    if mai_theme.user_name:
        USER_NAME = mai_theme.user_name


def reset_names() -> None:
    """Reset display names to their defaults (used when unloading a custom theme)."""
    global AI_NAME, USER_NAME
    AI_NAME = "MahanAI"
    USER_NAME = "You"


def register_mai_theme(slug: str, display: str, path: str) -> None:
    """Add a .mai theme to the THEME_NAMES / THEME_DISPLAY / MAI_THEMES tables."""
    global MAI_THEMES
    if slug not in THEME_NAMES:
        THEME_NAMES.append(slug)
    THEME_DISPLAY[slug] = display
    MAI_THEMES[slug] = path


def unregister_all_mai_themes() -> None:
    """Remove every .mai theme from the in-memory theme tables."""
    global MAI_THEMES
    for slug in list(MAI_THEMES.keys()):
        if slug in THEME_NAMES:
            THEME_NAMES.remove(slug)
        THEME_DISPLAY.pop(slug, None)
    MAI_THEMES.clear()


if not os.environ.get("NO_COLOR", "").strip():
    colorama.init()
apply_theme("midnight")
