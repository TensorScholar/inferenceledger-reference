from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .pilot_contract import PILOT_MANIFEST_VERSION, ArtifactRetention


@dataclass(frozen=True, order=True)
class EvidenceManifestEntry:
    logical_path: str
    role: str
    retention: ArtifactRetention
    sha256: str | None
    byte_size: int | None
    required: bool
    verification: str
    explicitly_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        logical = self.logical_path.strip().replace("\\", "/")
        if not logical or logical.startswith("/") or ".." in Path(logical).parts:
            raise ValueError("manifest logical_path must be a relative path without parent traversal")
        object.__setattr__(self, "logical_path", logical)
        if not self.role.strip():
            raise ValueError("manifest entry role must be non-empty")
        if self.retention == ArtifactRetention.EXPLICITLY_UNAVAILABLE:
            if self.sha256 is not None or self.byte_size is not None:
                raise ValueError("explicitly unavailable manifest entries must not claim bytes")
            if self.explicitly_unavailable_reason is None or not self.explicitly_unavailable_reason.strip():
                raise ValueError("explicitly unavailable manifest entry requires a reason")
            if self.verification != "UNAVAILABLE":
                raise ValueError("unavailable manifest entry verification must be UNAVAILABLE")
            return

        if self.sha256 is None or not _valid_sha256(self.sha256):
            raise ValueError("manifest byte-bearing/reference entry requires SHA-256")
        object.__setattr__(self, "sha256", self.sha256.lower())
        if self.byte_size is None or self.byte_size < 0:
            raise ValueError("manifest byte-bearing/reference entry requires non-negative byte_size")
        if self.explicitly_unavailable_reason is not None:
            raise ValueError("available manifest entries must not carry unavailable reason")
        if (
            self.retention == ArtifactRetention.COMMITTED_SANITIZED
            and self.verification != "LOCAL_BYTES"
        ):
            raise ValueError("committed/sanitized manifest entries require LOCAL_BYTES verification")
        if (
            self.retention == ArtifactRetention.PRIVATE_RAW_HASH_REFERENCED
            and self.verification != "HASH_REFERENCE_ONLY"
        ):
            raise ValueError("private raw manifest entries require HASH_REFERENCE_ONLY verification")


@dataclass(frozen=True)
class EvidenceManifest:
    version: str
    entries: tuple[EvidenceManifestEntry, ...]

    def __post_init__(self) -> None:
        if self.version != PILOT_MANIFEST_VERSION:
            raise ValueError(f"manifest version must be {PILOT_MANIFEST_VERSION}")
        ordered = tuple(sorted(self.entries, key=lambda item: (item.logical_path, item.role)))
        if ordered != self.entries:
            raise ValueError("manifest entries must be deterministically sorted")
        keys = [(entry.logical_path, entry.role) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("manifest contains duplicate logical_path/role entries")


@dataclass(frozen=True)
class ManifestVerification:
    valid: bool
    checked_local_count: int
    private_reference_count: int
    unavailable_count: int
    errors: tuple[str, ...]


def committed_manifest_entry(
    *,
    root: Path,
    path: Path,
    role: str,
    logical_path: str | None = None,
    required: bool = True,
) -> EvidenceManifestEntry:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError("manifest committed artifact must live under bundle root") from exc
    data = resolved_path.read_bytes()
    return EvidenceManifestEntry(
        logical_path=logical_path or relative,
        role=role,
        retention=ArtifactRetention.COMMITTED_SANITIZED,
        sha256=sha256(data).hexdigest(),
        byte_size=len(data),
        required=required,
        verification="LOCAL_BYTES",
    )


def private_reference_manifest_entry(
    *,
    logical_path: str,
    role: str,
    sha256_hex: str,
    byte_size: int,
    required: bool = True,
) -> EvidenceManifestEntry:
    return EvidenceManifestEntry(
        logical_path=logical_path,
        role=role,
        retention=ArtifactRetention.PRIVATE_RAW_HASH_REFERENCED,
        sha256=sha256_hex,
        byte_size=byte_size,
        required=required,
        verification="HASH_REFERENCE_ONLY",
    )


def unavailable_manifest_entry(
    *,
    logical_path: str,
    role: str,
    reason: str,
    required: bool = True,
) -> EvidenceManifestEntry:
    return EvidenceManifestEntry(
        logical_path=logical_path,
        role=role,
        retention=ArtifactRetention.EXPLICITLY_UNAVAILABLE,
        sha256=None,
        byte_size=None,
        required=required,
        verification="UNAVAILABLE",
        explicitly_unavailable_reason=reason,
    )


def build_manifest(entries: Iterable[EvidenceManifestEntry]) -> EvidenceManifest:
    ordered = tuple(sorted(entries, key=lambda item: (item.logical_path, item.role)))
    return EvidenceManifest(version=PILOT_MANIFEST_VERSION, entries=ordered)


def write_manifest(manifest: EvidenceManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": manifest.version,
        "entries": [asdict(entry) for entry in manifest.entries],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> EvidenceManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest must contain a JSON object")
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("manifest entries must be an array")
    entries: list[EvidenceManifestEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("manifest entry must be an object")
        entries.append(_entry_from_dict(raw_entry))
    return EvidenceManifest(version=str(raw.get("version")), entries=tuple(entries))


def verify_manifest(manifest: EvidenceManifest, *, root: Path) -> ManifestVerification:
    errors: list[str] = []
    checked_local = 0
    private_refs = 0
    unavailable = 0
    for entry in manifest.entries:
        if entry.retention == ArtifactRetention.PRIVATE_RAW_HASH_REFERENCED:
            private_refs += 1
            continue
        if entry.retention == ArtifactRetention.EXPLICITLY_UNAVAILABLE:
            unavailable += 1
            if entry.required:
                errors.append(
                    f"required artifact explicitly unavailable: {entry.logical_path}: "
                    f"{entry.explicitly_unavailable_reason}"
                )
            continue

        checked_local += 1
        path = root / entry.logical_path
        if not path.is_file():
            errors.append(f"missing required bundle artifact: {entry.logical_path}")
            continue
        data = path.read_bytes()
        if len(data) != entry.byte_size:
            errors.append(
                f"byte-size mismatch for {entry.logical_path}: expected={entry.byte_size}, actual={len(data)}"
            )
        digest = sha256(data).hexdigest()
        if digest != entry.sha256:
            errors.append(
                f"SHA-256 mismatch for {entry.logical_path}: expected={entry.sha256}, actual={digest}"
            )
    return ManifestVerification(
        valid=not errors,
        checked_local_count=checked_local,
        private_reference_count=private_refs,
        unavailable_count=unavailable,
        errors=tuple(errors),
    )


def _entry_from_dict(raw: dict[str, Any]) -> EvidenceManifestEntry:
    return EvidenceManifestEntry(
        logical_path=str(raw["logical_path"]),
        role=str(raw["role"]),
        retention=ArtifactRetention(str(raw["retention"])),
        sha256=_optional_text(raw.get("sha256")),
        byte_size=_optional_int(raw.get("byte_size")),
        required=_required_bool(raw.get("required")),
        verification=str(raw["verification"]),
        explicitly_unavailable_reason=_optional_text(raw.get("explicitly_unavailable_reason")),
    )


def _valid_sha256(value: str) -> bool:
    text = value.strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("manifest byte_size must be integer or null")
    return value


def _required_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("manifest required must be boolean")
    return value
