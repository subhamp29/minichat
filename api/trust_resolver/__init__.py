"""
Trust resolver Python package.

This package provides trust resolution for file system paths, with support for
allowlist/denylist pattern matching, trust prompt detection, and event emission.

It will try to use the native Rust extension (`trust_resolver._native`) if available,
and fall back to a pure-Python implementation if not.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

try:
    from trust_resolver._native import (
        PyTrustAllowlistEntry as NativeAllowlistEntry,
        PyTrustConfig as NativeTrustConfig,
        PyTrustDecision as NativeTrustDecision,
        PyTrustEvent as NativeTrustEvent,
        PyTrustPolicy as NativeTrustPolicy,
        PyTrustResolution as NativeTrustResolution,
        PyTrustResolver as NativeTrustResolver,
        detect_manual_approval as native_detect_manual_approval,
        detect_trust_prompt as native_detect_trust_prompt,
    )

    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pure-Python fallback implementation
# ---------------------------------------------------------------------------

_TRUST_PROMPT_CUES = [
    "do you trust the files in this folder",
    "trust the files in this folder",
    "trust this folder",
    "allow and continue",
    "yes, proceed",
]

_MANUAL_APPROVAL_CUES = [
    "yes, i trust",
    "i trust this",
    "trusted manually",
    "approval granted",
]


class TrustPolicy:
    """Resolution method for trust decisions."""

    AUTO_TRUST = 0
    REQUIRE_APPROVAL = 1
    DENY = 2

    @classmethod
    def name(cls, value: int) -> str:
        return {0: "auto_trust", 1: "require_approval", 2: "deny"}.get(value, "unknown")


class TrustResolution:
    """How trust was resolved."""

    AUTO_ALLOWLISTED = 0
    MANUAL_APPROVAL = 1

    @classmethod
    def name(cls, value: int) -> str:
        return {0: "auto_allowlisted", 1: "manual_approval"}.get(value, "unknown")


@dataclass
class TrustEvent:
    """Event emitted during trust resolution."""

    event_type: str
    cwd: str
    repo: Optional[str] = None
    worktree: Optional[str] = None
    policy: Optional[int] = None
    resolution: Optional[int] = None
    reason: Optional[str] = None

    def to_json(self) -> str:
        import json

        return json.dumps(self.__dict__)


@dataclass
class TrustAllowlistEntry:
    """Entry in the trust allowlist."""

    pattern: str
    worktree_pattern: Optional[str] = None
    description: Optional[str] = None


class TrustDecision:
    """Result of a trust resolution."""

    def __init__(self, is_required: bool, policy: Optional[int], events: List[TrustEvent]):
        self._is_required = is_required
        self._policy = policy
        self._events = events

    @property
    def is_required(self) -> bool:
        return self._is_required

    @property
    def policy(self) -> Optional[int]:
        return self._policy

    @property
    def events(self) -> List[TrustEvent]:
        return list(self._events)


class TrustConfig:
    """Configuration for trust resolution."""

    def __init__(self):
        self.allowlisted: List[TrustAllowlistEntry] = []
        self.denied: List[str] = []
        self.emit_events: bool = True

    def add_allowlisted(self, pattern: str) -> None:
        self.allowlisted.append(TrustAllowlistEntry(pattern=pattern))

    def add_allowlisted_entry(self, entry: TrustAllowlistEntry) -> None:
        self.allowlisted.append(entry)

    def add_denied(self, path: str) -> None:
        self.denied.append(os.path.normpath(path))

    def is_allowlisted(
        self, cwd: str, worktree: Optional[str]
    ) -> Optional[TrustAllowlistEntry]:
        cwd = os.path.normpath(cwd)
        for entry in self.allowlisted:
            if self._pattern_matches(entry.pattern, cwd):
                if entry.worktree_pattern is not None:
                    if worktree is None:
                        continue
                    if not self._pattern_matches(entry.worktree_pattern, worktree):
                        continue
                return entry
        return None

    def _pattern_matches(self, pattern: str, path: str) -> bool:
        import fnmatch

        pattern = pattern.strip()
        path = path.strip()

        if pattern == path:
            return True

        pattern = os.path.normpath(pattern)
        path = os.path.normpath(path)

        if fnmatch.fnmatch(path, pattern):
            return True

        if pattern.endswith(os.sep):
            pattern = pattern.rstrip(os.sep)

        if path.startswith(pattern + os.sep):
            return True

        return False

    def to_json(self) -> str:
        import json

        return json.dumps(
            {
                "allowlisted": [
                    {
                        "pattern": e.pattern,
                        "worktree_pattern": e.worktree_pattern,
                        "description": e.description,
                    }
                    for e in self.allowlisted
                ],
                "denied": self.denied,
                "emit_events": self.emit_events,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TrustConfig":
        import json

        data = json.loads(json_str)
        config = cls()
        config.emit_events = data.get("emit_events", True)
        config.denied = data.get("denied", [])
        for entry in data.get("allowlisted", []):
            config.allowlisted.append(
                TrustAllowlistEntry(
                    pattern=entry["pattern"],
                    worktree_pattern=entry.get("worktree_pattern"),
                    description=entry.get("description"),
                )
            )
        return config


class TrustResolver:
    """Resolves trust decisions for file system paths."""

    def __init__(self, config: TrustConfig):
        self.config = config

    def resolve(self, cwd: str, worktree: Optional[str], screen_text: str) -> TrustDecision:
        if not _detect_trust_prompt(screen_text):
            return TrustDecision(is_required=False, policy=None, events=[])

        repo = _extract_repo_name(cwd)
        events: List[TrustEvent] = [
            TrustEvent(
                event_type="trust_required",
                cwd=cwd,
                repo=repo,
                worktree=worktree,
            )
        ]

        cwd_norm = os.path.normpath(cwd)
        for denied_path in self.config.denied:
            if cwd_norm == denied_path or cwd_norm.startswith(denied_path + os.sep):
                events.append(
                    TrustEvent(
                        event_type="trust_denied",
                        cwd=cwd,
                        reason=f"cwd matches denied trust root: {denied_path}",
                    )
                )
                return TrustDecision(
                    is_required=True,
                    policy=TrustPolicy.DENY,
                    events=events,
                )

        matched = self.config.is_allowlisted(cwd, worktree)
        if matched is not None:
            events.append(
                TrustEvent(
                    event_type="trust_resolved",
                    cwd=cwd,
                    policy=TrustPolicy.AUTO_TRUST,
                    resolution=TrustResolution.AUTO_ALLOWLISTED,
                )
            )
            return TrustDecision(
                is_required=True,
                policy=TrustPolicy.AUTO_TRUST,
                events=events,
            )

        if _detect_manual_approval(screen_text):
            events.append(
                TrustEvent(
                    event_type="trust_resolved",
                    cwd=cwd,
                    policy=TrustPolicy.REQUIRE_APPROVAL,
                    resolution=TrustResolution.MANUAL_APPROVAL,
                )
            )
            return TrustDecision(
                is_required=True,
                policy=TrustPolicy.REQUIRE_APPROVAL,
                events=events,
            )

        return TrustDecision(
            is_required=True,
            policy=TrustPolicy.REQUIRE_APPROVAL,
            events=events,
        )

    def trusts(self, cwd: str, worktree: Optional[str]) -> bool:
        cwd_norm = os.path.normpath(cwd)
        for denied_path in self.config.denied:
            if cwd_norm == denied_path or cwd_norm.startswith(denied_path + os.sep):
                return False
        return self.config.is_allowlisted(cwd, worktree) is not None


def _detect_trust_prompt(screen_text: str) -> bool:
    lowered = screen_text.lower()
    return any(cue in lowered for cue in _TRUST_PROMPT_CUES)


def _detect_manual_approval(screen_text: str) -> bool:
    lowered = screen_text.lower()
    return any(cue in lowered for cue in _MANUAL_APPROVAL_CUES)


def _extract_repo_name(cwd: str) -> Optional[str]:
    path = Path(cwd)
    if path.is_dir():
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            root = result.stdout.strip()
            if root:
                return Path(root).name
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
    return path.name


# ---------------------------------------------------------------------------
# Public API - native if available, otherwise pure-Python
# ---------------------------------------------------------------------------

if _NATIVE_AVAILABLE:

    class TrustResolverNative:
        """Trust resolver backed by the native Rust extension."""

        def __init__(self, config: TrustConfig):
            native_config = NativeTrustConfig()
            for entry in config.allowlisted:
                native_entry = NativeAllowlistEntry(entry.pattern)
                if entry.worktree_pattern:
                    native_entry.set_worktree_pattern(entry.worktree_pattern)
                if entry.description:
                    native_entry.set_description(entry.description)
                native_config.add_allowlisted_entry(native_entry)
            for denied in config.denied:
                native_config.add_denied(denied)
            self._resolver = NativeTrustResolver(native_config)

        def resolve(self, cwd: str, worktree: Optional[str], screen_text: str) -> TrustDecision:
            native_decision = self._resolver.resolve(cwd, worktree, screen_text)
            events = [
                TrustEvent(
                    event_type=ev.event_type(),
                    cwd=ev.cwd(),
                    repo=ev.repo(),
                    worktree=ev.worktree(),
                    policy=ev.policy().value if ev.policy() is not None else None,
                    resolution=ev.resolution().value if ev.resolution() is not None else None,
                    reason=ev.reason(),
                )
                for ev in native_decision.events()
            ]
            return TrustDecision(
                is_required=native_decision.is_required(),
                policy=native_decision.policy().value if native_decision.policy() is not None else None,
                events=events,
            )

        def trusts(self, cwd: str, worktree: Optional[str]) -> bool:
            return self._resolver.trusts(cwd, worktree)

    TrustResolverImpl = TrustResolverNative
    detect_trust_prompt_impl = staticmethod(native_detect_trust_prompt)
    detect_manual_approval_impl = staticmethod(native_detect_manual_approval)

else:

    class TrustResolverFallback:
        """Trust resolver using pure-Python implementation."""

        def __init__(self, config: TrustConfig):
            self._resolver = TrustResolver(config)

        def resolve(self, cwd: str, worktree: Optional[str], screen_text: str) -> TrustDecision:
            return self._resolver.resolve(cwd, worktree, screen_text)

        def trusts(self, cwd: str, worktree: Optional[str]) -> bool:
            return self._resolver.trusts(cwd, worktree)

    TrustResolverImpl = TrustResolverFallback
    detect_trust_prompt_impl = staticmethod(_detect_trust_prompt)
    detect_manual_approval_impl = staticmethod(_detect_manual_approval)


# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    "TrustPolicy",
    "TrustResolution",
    "TrustEvent",
    "TrustAllowlistEntry",
    "TrustConfig",
    "TrustDecision",
    "TrustResolverImpl",
    "detect_trust_prompt_impl",
    "detect_manual_approval_impl",
]
