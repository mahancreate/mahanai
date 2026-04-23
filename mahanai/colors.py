"""Terminal colors (Windows-safe via colorama)."""

from __future__ import annotations

import os

import colorama

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

# Module-level color variables — mutated by apply_theme()
RST: str = ""
USER: str = ""
BOT: str = ""
ERR: str = ""
BANNER: str = ""
OK: str = ""
DIM: str = ""
WARN: str = ""
banner_colors: list[str] = []


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


if not os.environ.get("NO_COLOR", "").strip():
    colorama.init()
apply_theme("midnight")
