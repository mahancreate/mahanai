#!/usr/bin/env python3
"""MahanAI Plugin Signature Tool — Sign your plugins for distribution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from mahanai.plugin_signer import PluginSigner, PluginVerificationManager
from mahanai.mmd_parser import parse_mmd_file


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mahanai-signer",
        description="Sign and verify MahanAI plugin (.mmd) files",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ========== SIGN command ==========
    sign_parser = subparsers.add_parser(
        "sign",
        help="Sign a plugin file"
    )
    sign_parser.add_argument(
        "file",
        help="Path to .mmd plugin file to sign"
    )
    sign_parser.add_argument(
        "-k", "--key",
        help="Path to signing key (auto-created if missing)"
    )
    sign_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing signature"
    )
    
    # ========== VERIFY command ==========
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a plugin's signature"
    )
    verify_parser.add_argument(
        "file",
        help="Path to .mmd plugin file to verify"
    )
    verify_parser.add_argument(
        "-k", "--key",
        help="Path to signing key"
    )
    
    # ========== KEY command ==========
    key_parser = subparsers.add_parser(
        "key",
        help="Manage signing keys"
    )
    key_parser.add_argument(
        "action",
        choices=["show", "create", "rotate"],
        help="Key action"
    )
    key_parser.add_argument(
        "-p", "--path",
        help="Key file path"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == "sign":
            return sign_plugin(args.file, args.key, args.force)
        elif args.command == "verify":
            return verify_plugin(args.file, args.key)
        elif args.command == "key":
            return manage_key(args.action, args.path)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def sign_plugin(file_path: str, key_path: Optional[str] = None, force: bool = False) -> int:
    """Sign a plugin file."""
    mmd_path = Path(file_path).expanduser().resolve()
    
    if not mmd_path.exists():
        print(f"❌ File not found: {file_path}", file=sys.stderr)
        return 1
    
    if mmd_path.suffix.lower() != ".mmd":
        print(f"❌ Not a .mmd file: {file_path}", file=sys.stderr)
        return 1
    
    # Parse to validate
    try:
        plugin = parse_mmd_file(mmd_path)
    except Exception as e:
        print(f"❌ Invalid plugin file: {e}", file=sys.stderr)
        return 1
    
    # Check for existing signature
    sig_path = Path(str(mmd_path) + ".sig")
    if sig_path.exists() and not force:
        print(
            f"⚠️  Signature already exists: {sig_path.name}\n"
            f"   Use --force to overwrite"
        )
        return 0
    
    # Initialize signer
    signer = PluginSigner()
    
    # Sign
    print(f"📝 Signing plugin: {plugin.name} v{plugin.version}")
    metadata = signer.sign_plugin(mmd_path)
    
    # Save metadata
    sig_path = signer.save_metadata(metadata, mmd_path)
    
    print(f"✅ Signed successfully!")
    print(f"   Signature: {sig_path.name}")
    print(f"   Signed at: {metadata['signed_at']}")
    print(f"   Content hash: {metadata['content_hash'][:16]}...")
    
    return 0


def verify_plugin(file_path: str, key_path: Optional[str] = None) -> int:
    """Verify a plugin's signature."""
    mmd_path = Path(file_path).expanduser().resolve()
    
    if not mmd_path.exists():
        print(f"❌ File not found: {file_path}", file=sys.stderr)
        return 1
    
    # Parse to get info
    try:
        plugin = parse_mmd_file(mmd_path)
    except Exception as e:
        print(f"❌ Invalid plugin file: {e}", file=sys.stderr)
        return 1
    
    # Initialize verifier
    manager = PluginVerificationManager()
    
    # Verify
    print(f"🔍 Verifying plugin: {plugin.name} v{plugin.version}")
    is_valid, msg, metadata = manager.verify_plugin_signature(mmd_path)
    
    if is_valid and metadata:
        print(f"✅ {msg}")
        print(f"   Signed by: {metadata.get('signer', 'unknown')}")
        print(f"   Signed at: {metadata.get('signed_at', 'unknown')}")
        print(f"   Content hash: {metadata.get('content_hash', 'unknown')[:16]}...")
        return 0
    else:
        print(f"❌ {msg}", file=sys.stderr)
        return 1


def manage_key(action: str, path: Optional[str] = None) -> int:
    """Manage signing keys."""
    signer = PluginSigner()
    
    if action == "show":
        key_path = Path(path) if path else signer._key_path()
        if key_path.exists():
            print(f"🔐 Signing key: {key_path}")
            print(f"   Size: 256-bit (32 bytes)")
            print(f"   Protected: {oct(key_path.stat().st_mode)[-3:]}")
        else:
            print(f"❌ Key not found: {key_path}", file=sys.stderr)
            return 1
    
    elif action == "create":
        key_path = Path(path) if path else signer._key_path()
        if key_path.exists():
            print(f"⚠️  Key already exists: {key_path}")
            return 0
        
        key = signer._load_or_create_key()
        print(f"✅ Key created: {key_path}")
        print(f"   Size: 256-bit (32 bytes)")
        print(f"   Permissions: 0600 (read/write by owner only)")
    
    elif action == "rotate":
        import secrets
        import base64
        key_path = Path(path) if path else signer._key_path()
        
        # Create backup
        if key_path.exists():
            backup_path = key_path.with_suffix(".key.backup")
            backup_path.write_text(key_path.read_text())
            print(f"📦 Backed up old key: {backup_path.name}")
        
        # Create new key
        new_key = secrets.token_bytes(32)
        key_path.write_text(base64.b64encode(new_key).decode())
        key_path.chmod(0o600)
        print(f"✅ Key rotated: {key_path}")
        print(f"   Old key backed up to: {backup_path.name if key_path.exists() else 'N/A'}")
        print(f"   ⚠️  WARNING: Old plugins must be re-signed with the new key")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
