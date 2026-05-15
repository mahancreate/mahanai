"""Audit logging for MahanAI plugin security events."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class AuditEventType(Enum):
    """Types of security-related events to log."""
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"
    PLUGIN_EXECUTED = "plugin_executed"
    SECURITY_BLOCKED = "security_blocked"
    SECURITY_WARNING = "security_warning"
    SIGNATURE_VERIFIED = "signature_verified"
    SIGNATURE_FAILED = "signature_failed"
    FOUNDATION_CHECK = "foundation_check"


@dataclass
class AuditEvent:
    """A single audit log entry."""
    timestamp: str
    event_type: str
    plugin_name: str
    plugin_version: str
    details: dict
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class AuditLogger:
    """
    Logs all plugin security-related events.
    Maintains a persistent audit trail for debugging and transparency.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        """Initialize the audit logger."""
        if log_path is None:
            config_dir = Path.home() / ".config" / "mahanai"
            config_dir.mkdir(parents=True, exist_ok=True)
            log_path = config_dir / "plugin-audit.log"
        
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(
        self,
        event_type: AuditEventType | str,
        plugin_name: str,
        plugin_version: str = "unknown",
        details: Optional[dict] = None
    ) -> None:
        """Log a security event."""
        if isinstance(event_type, AuditEventType):
            event_type = event_type.value
        
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            details=details or {},
        )
        
        # Append to log file
        try:
            with open(self.log_path, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception:
            # Logging failures shouldn't crash the system
            pass
    
    def plugin_loaded(self, name: str, version: str, source: str = "local") -> None:
        """Log when a plugin is loaded."""
        self.log_event(
            AuditEventType.PLUGIN_LOADED,
            name,
            version,
            {"source": source}
        )
    
    def plugin_unloaded(self, name: str) -> None:
        """Log when a plugin is unloaded."""
        self.log_event(AuditEventType.PLUGIN_UNLOADED, name)
    
    def plugin_executed(self, name: str, trigger: str, action_count: int) -> None:
        """Log when a plugin command is executed."""
        self.log_event(
            AuditEventType.PLUGIN_EXECUTED,
            name,
            details={"trigger": trigger, "actions": action_count}
        )
    
    def security_blocked(
        self,
        name: str,
        reason: str,
        blocked_count: int,
        details: Optional[dict] = None
    ) -> None:
        """Log when a plugin is blocked for security reasons."""
        d = {"reason": reason, "issues_count": blocked_count}
        if details:
            d.update(details)
        self.log_event(AuditEventType.SECURITY_BLOCKED, name, details=d)
    
    def security_warning(
        self,
        name: str,
        warning: str,
        warning_count: int,
        details: Optional[dict] = None
    ) -> None:
        """Log security warnings."""
        d = {"warning": warning, "issues_count": warning_count}
        if details:
            d.update(details)
        self.log_event(AuditEventType.SECURITY_WARNING, name, details=d)
    
    def signature_verified(self, name: str, signed_at: str) -> None:
        """Log successful signature verification."""
        self.log_event(
            AuditEventType.SIGNATURE_VERIFIED,
            name,
            details={"signed_at": signed_at}
        )
    
    def signature_failed(self, name: str, reason: str) -> None:
        """Log signature verification failure."""
        self.log_event(
            AuditEventType.SIGNATURE_FAILED,
            name,
            details={"reason": reason}
        )
    
    def foundation_check(self, name: str, is_trusted: bool, source: str) -> None:
        """Log foundation source verification."""
        self.log_event(
            AuditEventType.FOUNDATION_CHECK,
            name,
            details={"trusted": is_trusted, "source": source}
        )
    
    def read_log(self, limit: Optional[int] = None) -> list[AuditEvent]:
        """Read audit log entries."""
        if not self.log_path.exists():
            return []
        
        events = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            event = AuditEvent(**data)
                            events.append(event)
                        except Exception:
                            pass
        except Exception:
            pass
        
        if limit:
            return events[-limit:]
        return events
    
    def format_log_summary(self, limit: int = 20) -> str:
        """Format the audit log for display."""
        events = self.read_log(limit)
        
        if not events:
            return "📝 Audit log is empty\n"
        
        lines = [
            f"\n{'='*70}",
            f"📝 Plugin Audit Log (last {min(limit, len(events))} entries)",
            f"{'='*70}",
        ]
        
        for event in events:
            event_emoji = self._emoji_for_event(event.event_type)
            lines.append(f"\n{event_emoji} {event.event_type.upper()}")
            lines.append(f"  Plugin: {event.plugin_name} (v{event.plugin_version})")
            lines.append(f"  Time:   {event.timestamp}")
            
            if event.details:
                for key, val in event.details.items():
                    lines.append(f"  {key}: {val}")
        
        lines.append(f"\n{'='*70}\n")
        return "\n".join(lines)
    
    @staticmethod
    def _emoji_for_event(event_type: str) -> str:
        """Get emoji for event type."""
        emojis = {
            "plugin_loaded": "✅",
            "plugin_unloaded": "🔌",
            "plugin_executed": "⚙️",
            "security_blocked": "🚫",
            "security_warning": "⚠️",
            "signature_verified": "🔐",
            "signature_failed": "❌",
            "foundation_check": "🔗",
        }
        return emojis.get(event_type, "📝")


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
