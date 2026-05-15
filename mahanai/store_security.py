"""Plugin store security integration — Scans downloaded plugins before installation."""

from __future__ import annotations

from pathlib import Path
from .store import install_plugin
from .plugin_security import scan_plugin_security, format_security_report
from .mmd_parser import parse_mmd_file


def install_plugin_with_security(
    repo_full_name: str,
    token: str | None = None,
    require_approval: bool = True,
) -> tuple[bool, str, Path | None]:
    """
    Download and scan a plugin from the store before installation.
    
    Args:
        repo_full_name: GitHub repo in format "user/repo"
        token: Optional GitHub auth token
        require_approval: If True, require user approval for warnings
    
    Returns:
        (success, message, installed_path)
    """
    try:
        # Download the plugin
        plugin_path = install_plugin(repo_full_name, token)
        plugin = parse_mmd_file(plugin_path)
        
        # Run security scan
        report = scan_plugin_security(plugin)
        
        # Display results
        print(f"\n📦 Plugin: {plugin.name} v{plugin.version}")
        print(f"   From: {repo_full_name}")
        print(format_security_report(report, verbose=False))
        
        # Block if critical issues
        if report.blocked_count() > 0:
            # Delete the downloaded file
            plugin_path.unlink()
            msg = f"❌ Installation blocked: {report.blocked_count()} security issue(s) detected"
            return False, msg, None
        
        # Warn if suspicious
        if report.warning_count() > 0 and require_approval:
            response = input(f"⚠️  Continue with installation? (yes/no): ")
            if response.lower() not in ("yes", "y"):
                plugin_path.unlink()
                return False, "Installation cancelled by user", None
        
        return True, f"✅ Plugin installed: {plugin_path}", plugin_path
    
    except Exception as e:
        return False, f"❌ Error: {str(e)}", None
