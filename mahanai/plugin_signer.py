"""Plugin signing and cryptographic verification for MahanAI plugins."""

from __future__ import annotations

import hmac
import hashlib
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional


class PluginSigner:
    """
    Handles cryptographic signing and verification of .mmd plugins.
    Uses HMAC-SHA256 for fast, simple verification.
    """
    
    SIGNATURE_VERSION = "1.0"
    METADATA_SUFFIX = ".sig"  # Plugin metadata file: foo.mmd.sig
    
    def __init__(self, signing_key: Optional[bytes] = None):
        """
        Initialize the signer.
        If signing_key is None, load from default location.
        """
        if signing_key is None:
            self.signing_key = self._load_or_create_key()
        else:
            self.signing_key = signing_key
    
    def _key_path(self) -> Path:
        """Get the default key file path."""
        config_dir = Path.home() / ".config" / "mahanai"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / ".mahanai-plugin-signer.key"
    
    def _load_or_create_key(self) -> bytes:
        """Load signing key from disk, or create one if it doesn't exist."""
        key_path = self._key_path()
        
        if key_path.exists():
            try:
                return base64.b64decode(key_path.read_text().strip())
            except Exception:
                # If key is corrupted, create a new one
                pass
        
        # Create new key
        import secrets
        key = secrets.token_bytes(32)  # 256-bit key
        key_path.write_text(base64.b64encode(key).decode())
        key_path.chmod(0o600)  # Protect key file
        return key
    
    def sign_plugin(self, mmd_path: Path) -> dict:
        """
        Sign a .mmd file and return metadata.
        Returns a dict with signature info to be stored in .sig file.
        """
        mmd_path = Path(mmd_path)
        content = mmd_path.read_bytes()
        
        # Create HMAC-SHA256 signature
        signature = hmac.new(self.signing_key, content, hashlib.sha256).hexdigest()
        
        # Create SHA256 hash of the content (for integrity verification)
        content_hash = hashlib.sha256(content).hexdigest()
        
        metadata = {
            "version": self.SIGNATURE_VERSION,
            "filename": mmd_path.name,
            "signed_at": datetime.utcnow().isoformat(),
            "signature": signature,
            "content_hash": content_hash,
            "signer": "mahanai-official",  # You can customize this
        }
        
        return metadata
    
    def verify_plugin(self, mmd_path: Path, metadata: dict) -> tuple[bool, str]:
        """
        Verify a plugin's signature and integrity.
        Returns (is_valid, message) tuple.
        """
        mmd_path = Path(mmd_path)
        
        if not mmd_path.exists():
            return False, f"Plugin file not found: {mmd_path}"
        
        # Verify version
        if metadata.get("version") != self.SIGNATURE_VERSION:
            return False, f"Unsupported signature version: {metadata.get('version')}"
        
        content = mmd_path.read_bytes()
        
        # Verify content hash first (fast check)
        content_hash = hashlib.sha256(content).hexdigest()
        if content_hash != metadata.get("content_hash"):
            return False, "Plugin content has been modified (hash mismatch)"
        
        # Verify HMAC signature
        expected_signature = hmac.new(self.signing_key, content, hashlib.sha256).hexdigest()
        provided_signature = metadata.get("signature", "")
        
        if not hmac.compare_digest(expected_signature, provided_signature):
            return False, "Invalid plugin signature (not signed by mahanai-official)"
        
        return True, "Plugin signature verified successfully"
    
    def save_metadata(self, metadata: dict, mmd_path: Path) -> Path:
        """Save signature metadata to a .sig file."""
        mmd_path = Path(mmd_path)
        sig_path = mmd_path.parent / (mmd_path.name + self.METADATA_SUFFIX)
        sig_path.write_text(json.dumps(metadata, indent=2))
        return sig_path
    
    def load_metadata(self, mmd_path: Path) -> Optional[dict]:
        """Load signature metadata from a .sig file."""
        mmd_path = Path(mmd_path)
        sig_path = mmd_path.parent / (mmd_path.name + self.METADATA_SUFFIX)
        
        if not sig_path.exists():
            return None
        
        try:
            return json.loads(sig_path.read_text())
        except Exception:
            return None


class PluginVerificationManager:
    """
    High-level manager for plugin verification.
    Handles verification workflow and caching.
    """
    
    def __init__(self):
        self.signer = PluginSigner()
        self._verification_cache: dict[str, bool] = {}
    
    def verify_plugin_signature(
        self,
        mmd_path: Path,
        require_signature: bool = True
    ) -> tuple[bool, str, dict]:
        """
        Verify a plugin's signature.
        
        Args:
            mmd_path: Path to the .mmd file
            require_signature: If True, missing signature = failure
        
        Returns:
            (is_valid, message, metadata)
        """
        mmd_path = Path(mmd_path)
        
        # Check cache
        cache_key = str(mmd_path.resolve())
        if cache_key in self._verification_cache:
            is_valid = self._verification_cache[cache_key]
            status = "✅ VERIFIED (cached)" if is_valid else "❌ FAILED (cached)"
            return is_valid, status, {}
        
        # Load metadata
        metadata = self.signer.load_metadata(mmd_path)
        
        if metadata is None:
            if require_signature:
                msg = "Plugin signature not found (not officially signed)"
                self._verification_cache[cache_key] = False
                return False, msg, {}
            else:
                msg = "No signature found (local/unsigned plugin)"
                return True, msg, {}  # Allow unsigned in dev mode
        
        # Verify
        is_valid, msg = self.signer.verify_plugin(mmd_path, metadata)
        self._verification_cache[cache_key] = is_valid
        
        return is_valid, msg, metadata
    
    def clear_cache(self):
        """Clear the verification cache."""
        self._verification_cache.clear()
