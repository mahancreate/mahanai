# MahanAI Plugin Security Guide

This guide explains MahanAI's security system for plugins, including threat detection, signature verification, and best practices for plugin developers.

## Overview

MahanAI v7.4+ includes a **multi-layer security system** designed to protect users from malicious or buggy plugins:

1. **Threat Detection** — Scans for dangerous patterns (rm -rf, command injection, etc.)
2. **Signature Verification** — Cryptographically verifies plugins are from trusted developers
3. **Audit Logging** — Records all plugin-related security events for transparency
4. **Foundation Verification** — Flags plugins not from the official maifoundation

## For Users: Understanding Plugin Security

### What Gets Blocked? 🚫

MahanAI automatically **blocks** plugins that try to:
- **Destroy data:** `rm -rf /`, `dd`, `mkfs`
- **Inject commands:** `curl | bash`, `eval()`
- **Escalate privileges:** `sudo`, `su`
- **Lock you out:** `chmod 000`
- **Tamper with system:** `crontab`, `/etc/passwd`, `iptables`

### What Triggers Warnings? ⚠️

These activities are flagged but can be approved:
- Network requests (`curl`, `wget`)
- File operations (`chmod`, etc.)
- User management (`useradd`)
- System service control (`systemctl`)
- Plugins from unknown sources (not maifoundation)

### Loading Plugins Safely

```bash
# Local plugin with security scan
/plugin-load ./my-plugin.mmd
# MahanAI will:
# 1. Check signature (if available)
# 2. Scan for threats
# 3. Show warnings
# 4. Ask for approval if needed

# From the store (automatically scanned!)
/store install user/plugin-name

# View security audit log
/plugin-audit 50  # Show last 50 events
```

## For Developers: Signing Your Plugins

### Install the Signer Tool

```bash
pip install mahanai
mahanai-signer --help
```

### Sign Your Plugin

```bash
# Sign your plugin
mahanai-signer sign my-plugin.mmd

# This creates: my-plugin.mmd.sig
# Contains metadata about your signature
```

### What the Signature Proves

✅ You (the developer) created this plugin
✅ The plugin hasn't been modified
✅ It's safe to use (from your perspective)

### Verify a Plugin

```bash
mahanai-signer verify my-plugin.mmd

# Output:
# ✅ Plugin signature verified!
#    Signed by: mahanai-official
#    Signed at: 2026-05-15T14:32:10.123456
#    Content hash: a1b2c3d4e5f6g7h8...
```

### Manage Your Signing Key

```bash
# Show key info
mahanai-signer key show

# Create a new key (auto-created on first use)
mahanai-signer key create

# Rotate to a new key (backs up the old one)
mahanai-signer key rotate
```

## Best Practices for Plugin Developers

### 1. Write Safe Plugins

✅ **DO:**
- Use Claude Code commands for AI-driven automation
- Use MahanAI commands for plugin lifecycle management
- Request explicit user confirmation before dangerous actions
- Document what your plugin does in the .mmd file

❌ **DON'T:**
- Use shell commands for things that can be done with Claude Code
- Try to bypass safety checks or modify permissions
- Make network requests without explaining why
- Distribute unsigned plugins (they'll get warnings)

### 2. Sign Your Plugins Before Distribution

```bash
# Before uploading to the store:
mahanai-signer sign my-awesome-plugin.mmd
git add my-awesome-plugin.mmd*
git commit -m "Sign plugin"
git push origin main
```

### 3. Use Descriptive Registry Information

```
plugin.name = "Image Processor"
plugin.codename = "mahfoundation.image-processor"
plugin.reg.store = "mai-foundation"
plugin.reg.name = "MahanAI Foundation"
plugin.version = "1.0.0"
```

This helps users understand:
- What your plugin does
- Who maintains it
- Where it comes from
- Its version

### 4. Handle Warnings Gracefully

If your plugin triggers legitimate warnings, document why:

```
# my-plugin.mmd
# ⚠️ NOTE: This plugin makes network requests to fetch data.
# It's safe—we only contact api.example.com for user-approved queries.

plugin.name = "Data Fetcher"
...
```

## Security Event Logging

All plugin security events are logged to: `~/.config/mahanai/plugin-audit.log`

### View Audit Log

```bash
/plugin-audit        # Last 20 events
/plugin-audit 100    # Last 100 events
```

### Log Entries Include

- **Timestamp** — When the event occurred
- **Event type** — plugin_loaded, security_blocked, etc.
- **Plugin name & version**
- **Details** — Why it was blocked, what triggered warnings, etc.

## FAQ

### Q: I'm getting warnings for a legitimate network request. What should I do?

A: Document why in your plugin file comments. Users will see your explanation when loading the plugin. They can approve and load it anyway.

### Q: My plugin signature keeps failing verification. Why?

A: The most common cause is plugin file modification after signing. If you edit your .mmd file, re-sign it:

```bash
mahanai-signer sign my-plugin.mmd --force
```

### Q: Do I have to sign my plugins?

A: No, but unsigned plugins will show a warning when loaded. Signing builds trust with users and is highly recommended for plugins distributed via the store.

### Q: What if I lose my signing key?

A: Your key is stored at `~/.config/mahanai/.mahanai-plugin-signer.key`. If you lose it and create a new one, old plugins will fail verification. You'll need to re-sign them with the new key.

### Q: Can I use someone else's signing key?

A: No, and you shouldn't try. Signatures are tied to a specific key pair. Using someone else's key (or creating a fake signature) will be detected and the plugin will be marked as untrusted.

## Reporting Security Issues

Found a bug in the security system? **Don't post it publicly.** Instead:

1. Email: security@mahancreate.dev (or open a private issue)
2. Describe the vulnerability clearly
3. Include steps to reproduce
4. Allow time for a fix before public disclosure

Security vulnerabilities in plugins should be reported directly to the plugin developer.

## For MahanAI Maintainers

### Auditing Plugins

The security system leaves a complete audit trail:

```bash
# Check what plugins loaded recently
/plugin-audit 50

# Look for: security_blocked, security_warning events
# These indicate users attempted to load suspicious plugins
```

### Reviewing Store Plugins

Before adding a plugin to an official category:
1. Clone and review the code
2. Run security scan: `/plugin-load <path>`
3. Check the audit log for any warnings
4. Ask the developer to sign their plugin

## Version History

- **v7.4.0** (May 2026) — Initial security system release
  - Threat detection engine
  - Signature verification
  - Audit logging
  - CLI signer tool
