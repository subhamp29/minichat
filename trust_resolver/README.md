# Trust Resolver Python Package

Python bindings for the Rust-based trust resolution engine, with a pure-Python fallback.

## Installation

### Option 1: Native Rust Extension (Recommended for Performance)

1. Install Rust toolchain: https://rustup.rs/
2. Install maturin: `pip install maturin`
3. Build and install:
   ```bash
   cd D:\ai\rust\crates\trust-resolver-py
   maturin develop -m .
   ```

Or use the provided build script:
```powershell
.\build.ps1
```

### Option 2: Pure-Python Fallback

If the native extension is not available, the package automatically falls back to a pure-Python implementation. No additional build steps required.

## Usage

```python
from trust_resolver import TrustConfig, TrustResolver, TrustPolicy, TrustEvent

# Configure trust rules
config = TrustConfig()
config.add_allowlisted("/tmp/worktrees/*")
config.add_denied("/tmp/malicious")

# Create resolver
resolver = TrustResolver(config)

# Resolve trust
decision = resolver.resolve(
    cwd="/tmp/worktrees/repo-a",
    worktree=None,
    screen_text="Do you trust the files in this folder?"
)

print(f"Policy: {decision.policy}")
print(f"Events: {decision.events}")
```

## Integration with MiniChat

The `MiniChat` Streamlit app has been updated to use the trust resolver for working directory validation. On startup, it checks if the current directory is trusted and displays an error if manual approval is required or the path is denied.
