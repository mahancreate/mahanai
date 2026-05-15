"""Security scanning and validation for .mmd MahanAI plugins."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .mmd_parser import MmdAction, MmdCommand, MmdPlugin


class ThreatLevel(Enum):
    """Threat severity classification."""
    SAFE = 0
    WARNING = 1
    BLOCKED = 2


@dataclass
class SecurityIssue:
    """A single security finding."""
    level: ThreatLevel
    action_type: str  # "claude-cmd", "mahanai-cmd", "shell-cmd"
    trigger: str      # e.g., "/compact"
    message: str
    snippet: str      # The problematic code snippet


@dataclass
class PluginSecurityReport:
    """Full security audit for a plugin."""
    plugin_name: str
    is_safe: bool
    repo_source: str | None = None  # GitHub repo if from store
    issues: list[SecurityIssue] = field(default_factory=list)
    
    def blocked_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ThreatLevel.BLOCKED)
    
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ThreatLevel.WARNING)
    
    def summary(self) -> str:
        """Return human-readable summary."""
        if self.is_safe and not self.issues:
            return "✅ Plugin passed security scan (no issues found)"
        
        parts = []
        if self.blocked_count():
            parts.append(f"🚫 {self.blocked_count()} BLOCKED action(s)")
        if self.warning_count():
            parts.append(f"⚠️  {self.warning_count()} WARNING(s)")
        
        return " | ".join(parts) if parts else "✅ Scanned (no critical issues)"


# =============================================================================
# Dangerous Pattern Detection
# =============================================================================

DANGEROUS_SHELL_PATTERNS = [
    # Destructive filesystem operations
    (r"rm\s+-[a-z]*f", "Force remove (rm -f) detected"),
    (r"rm\s+-[a-z]*r", "Recursive remove detected"),
    (r"dd\s+", "Low-level disk write (dd) detected"),
    (r"mkfs", "Format filesystem (mkfs) detected"),
    (r"chmod\s+000", "Permission lockdown (chmod 000) detected"),
    (r"chown\s+0:0", "Ownership change to root detected"),
    
    # Command injection vectors
    (r"\$\(", "Command substitution $(…) detected"),
    (r"`[^`]*`", "Backtick command substitution detected"),
    (r"\|\s*bash", "Pipe-to-bash pattern detected"),
    (r"\|\s*sh", "Pipe-to-shell pattern detected"),
    (r"\|\s*python", "Pipe-to-python pattern detected"),
    (r"eval\s+", "eval() call detected"),
    
    # Privilege escalation
    (r"\bsudo\b", "Sudo usage detected"),
    (r"\bsu\b\s", "User switching (su) detected"),
    
    # Data exfiltration
    (r"curl\s+.*\|", "Curl pipe-to (potential exfil) detected"),
    (r"wget\s+.*\|", "Wget pipe-to (potential exfil) detected"),
    
    # System takeover
    (r"crontab", "Cron manipulation detected"),
    (r"/etc/passwd", "Direct /etc/passwd access detected"),
    (r"/etc/shadow", "Direct /etc/shadow access detected"),
    (r"iptables", "Firewall manipulation detected"),
]

SUSPICIOUS_SHELL_PATTERNS = [
    (r"curl\s+", "Network request (curl)"),
    (r"wget\s+", "Network request (wget)"),
    (r"wget\s+", "Network request (wget)"),
    (r"python\s+-c", "Python inline execution"),
    (r"perl\s+-e", "Perl inline execution"),
    (r"chmod\s+", "Permission modification"),
    (r"useradd|adduser", "User account creation"),
    (r"systemctl", "System service manipulation"),
]


def _detect_shell_threats(shell_cmd: str) -> tuple[ThreatLevel, str | None]:
    """
    Scan a shell command for dangerous patterns.
    Returns (threat_level, message) tuple.
    """
    # Check for BLOCKED patterns first
    for pattern, msg in DANGEROUS_SHELL_PATTERNS:
        if re.search(pattern, shell_cmd, re.IGNORECASE):
            return ThreatLevel.BLOCKED, msg
    
    # Check for WARNING patterns
    for pattern, msg in SUSPICIOUS_SHELL_PATTERNS:
        if re.search(pattern, shell_cmd, re.IGNORECASE):
            return ThreatLevel.WARNING, msg
    
    return ThreatLevel.SAFE, None


def _check_mahanai_cmd(cmd_str: str) -> tuple[ThreatLevel, str | None]:
    """
    Scan a mahanai command for suspicious behavior.
    Most mahanai commands are safe, but check for things like system access.
    """
    # Commands that try to access system internals are suspicious
    suspicious_patterns = [
        (r"/plugin-unload", "Plugin unloading (potential hijack)"),
        (r"/store\s+logout", "Token removal (potential hijack)"),
    ]
    
    for pattern, msg in suspicious_patterns:
        if re.search(pattern, cmd_str, re.IGNORECASE):
            return ThreatLevel.WARNING, msg
    
    return ThreatLevel.SAFE, None


def _check_claude_cmd(cmd_str: str) -> tuple[ThreatLevel, str | None]:
    """
    Claude commands are generally safe, but check for obvious red flags.
    """
    # Most claude commands are fine
    # This is mostly here for future extensibility
    return ThreatLevel.SAFE, None


# =============================================================================
# Main Security Scanner
# =============================================================================

def _check_foundation_source(plugin: MmdPlugin) -> tuple[ThreatLevel, str | None]:
    """
    Check if a plugin is from the official maifoundation GitHub org.
    Returns (threat_level, message) tuple.
    
    Plugins from maifoundation are trusted. Anything else gets a warning.
    """
    reg_store = (plugin.reg_store or "").lower().strip()
    
    # Check if it's explicitly marked as from maifoundation
    if reg_store == "mai-foundation" or reg_store == "maifoundation":
        return ThreatLevel.SAFE, None
    
    # If no store info or unknown store, that's suspicious
    return ThreatLevel.WARNING, "Plugin source not verified (not from maifoundation)"


# =============================================================================
# Main Security Scanner
# =============================================================================

def scan_plugin_security(plugin: MmdPlugin) -> PluginSecurityReport:
    """
    Perform a comprehensive security audit on a plugin.
    Returns a PluginSecurityReport with all findings.
    """
    report = PluginSecurityReport(
        plugin_name=plugin.name,
        is_safe=True,
        repo_source=plugin.codename,
    )
    
    # First, check if plugin is from maifoundation (trusted source)
    source_level, source_msg = _check_foundation_source(plugin)
    if source_level == ThreatLevel.WARNING and source_msg:
        report.issues.append(SecurityIssue(
            level=ThreatLevel.WARNING,
            action_type="metadata",
            trigger="[plugin]",
            message=source_msg,
            snippet=plugin.reg_store or "(unknown)",
        ))
    
    # Scan each command and its actions
    for cmd in plugin.commands:
        for action in cmd.actions:
            issue = _scan_action(action, cmd.trigger)
            if issue:
                report.issues.append(issue)
                if issue.level == ThreatLevel.BLOCKED:
                    report.is_safe = False
    
    return report


def _scan_action(action: MmdAction, trigger: str) -> SecurityIssue | None:
    """
    Scan a single action for security issues.
    Returns a SecurityIssue if found, None otherwise.
    """
    if action.type == "shell-cmd":
        level, msg = _detect_shell_threats(action.value)
        if level != ThreatLevel.SAFE:
            return SecurityIssue(
                level=level,
                action_type="shell-cmd",
                trigger=trigger,
                message=msg or "Shell command execution",
                snippet=action.value[:80],
            )
    
    elif action.type == "mahanai-cmd":
        level, msg = _check_mahanai_cmd(action.value)
        if level != ThreatLevel.SAFE:
            return SecurityIssue(
                level=level,
                action_type="mahanai-cmd",
                trigger=trigger,
                message=msg or "MahanAI command",
                snippet=action.value[:80],
            )
    
    elif action.type == "claude-cmd":
        level, msg = _check_claude_cmd(action.value)
        if level != ThreatLevel.SAFE:
            return SecurityIssue(
                level=level,
                action_type="claude-cmd",
                trigger=trigger,
                message=msg or "Claude command",
                snippet=action.value[:80],
            )
    
    return None


def format_security_report(report: PluginSecurityReport, verbose: bool = False) -> str:
    """
    Format a security report for display to the user.
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Security Scan: {report.plugin_name}")
    lines.append(f"{'='*70}")
    
    if not report.issues:
        lines.append("✅ No security issues detected!")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)
    
    # Group by threat level
    blocked = [i for i in report.issues if i.level == ThreatLevel.BLOCKED]
    warnings = [i for i in report.issues if i.level == ThreatLevel.WARNING]
    
    if blocked:
        lines.append(f"\n🚫 BLOCKED ({len(blocked)} issue(s)):")
        lines.append("-" * 70)
        for issue in blocked:
            lines.append(f"  {issue.trigger} → {issue.action_type}")
            lines.append(f"    ❌ {issue.message}")
            if verbose:
                lines.append(f"    Code: {issue.snippet}")
            lines.append("")
    
    if warnings:
        lines.append(f"\n⚠️  WARNINGS ({len(warnings)} issue(s)):")
        lines.append("-" * 70)
        for issue in warnings:
            lines.append(f"  {issue.trigger} → {issue.action_type}")
            lines.append(f"    ⚠️  {issue.message}")
            if verbose:
                lines.append(f"    Code: {issue.snippet}")
            lines.append("")
    
    lines.append(f"{'='*70}\n")
    
    if blocked:
        lines.append("❌ This plugin CANNOT be loaded due to security issues.")
    elif warnings:
        lines.append("⚠️  This plugin has warnings but may be loaded if you approve.")
    
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)
