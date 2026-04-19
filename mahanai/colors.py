"""Terminal colors (Windows-safe via colorama)."""

from __future__ import annotations

import os

import colorama

if os.environ.get("NO_COLOR", "").strip():
    RST = USER = BOT = ERR = BANNER = OK = DIM = WARN = ""
else:
    colorama.init()
    RST = colorama.Style.RESET_ALL
    USER = colorama.Fore.CYAN + colorama.Style.BRIGHT
    BOT = colorama.Fore.GREEN + colorama.Style.BRIGHT
    ERR = colorama.Fore.RED + colorama.Style.BRIGHT
    BANNER = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
    OK = colorama.Fore.GREEN
    DIM = colorama.Style.DIM
    WARN = colorama.Fore.YELLOW + colorama.Style.BRIGHT
