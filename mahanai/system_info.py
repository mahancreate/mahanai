"""Runtime OS and shell detection for agent context."""

from __future__ import annotations

import os
import platform


def describe_runtime() -> str:
    system = platform.system()
    release = platform.release()
    machine = platform.machine()

    if system == "Windows":
        if os.environ.get("PSModulePath"):
            shell = "Windows PowerShell"
        elif os.environ.get("MSYSTEM") or os.environ.get("SHELL", "").lower().endswith(
            ("bash.exe", "sh.exe", "zsh.exe")
        ):
            shell = "Git Bash / MSYS / POSIX-like shell on Windows"
        else:
            shell = "Windows CMD"
    else:
        shell = os.environ.get("SHELL") or "unknown shell"

    return (
        f"Operating system: {system} {release} ({machine}). "
        f"Interactive shell context for this CLI process: {shell}. "
        f"When running commands, prefer syntax appropriate for this shell."
    )
