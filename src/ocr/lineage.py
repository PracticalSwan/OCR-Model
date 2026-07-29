"""Immutable split and transformation lineage for OCR-derived artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable


ALLOWED_SPLIT_ROLES = frozenset(
    {
        "train",
        "dev_select",
        "dev_calibration",
        "test_in_domain",
        "unseen_coru",
        "unseen_domain_test",
        "private_operational",
        "private_test",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LineageRecord:
    """Trace one source or derivative back to an immutable document family."""

    sample_id: str
    document_id: str
    document_family_id: str
    split_role: str
    source_dataset: str
    source_hash: str
    parent_sample_id: str | None = None
    transformation_chain: tuple[str, ...] = ()
    license_id: str = ""
    annotation_version: str = ""

    def __post_init__(self) -> None:
        required = {
            "sample_id": self.sample_id,
            "document_id": self.document_id,
            "document_family_id": self.document_family_id,
            "source_dataset": self.source_dataset,
            "license_id": self.license_id,
            "annotation_version": self.annotation_version,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"lineage fields must be nonempty: {', '.join(missing)}")
        if self.split_role not in ALLOWED_SPLIT_ROLES:
            raise ValueError(f"unsupported split role: {self.split_role}")
        normalized_hash = self.source_hash.casefold()
        if not _SHA256_RE.fullmatch(normalized_hash):
            raise ValueError("source_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_hash", normalized_hash)
        object.__setattr__(
            self,
            "transformation_chain",
            tuple(str(value) for value in self.transformation_chain),
        )
        if any(not value.strip() for value in self.transformation_chain):
            raise ValueError("transformation_chain entries must be nonempty")

    def derive(self, *, sample_id: str, transformation: str) -> "LineageRecord":
        """Return a derivative that inherits all split and source identities."""
        if not str(sample_id).strip():
            raise ValueError("derived sample_id must be nonempty")
        if sample_id == self.sample_id:
            raise ValueError("derived sample_id must differ from its parent")
        if not str(transformation).strip():
            raise ValueError("transformation must be nonempty")
        return replace(
            self,
            sample_id=str(sample_id),
            parent_sample_id=self.sample_id,
            transformation_chain=(*self.transformation_chain, str(transformation)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "document_id": self.document_id,
            "document_family_id": self.document_family_id,
            "split_role": self.split_role,
            "source_dataset": self.source_dataset,
            "source_hash": self.source_hash,
            "parent_sample_id": self.parent_sample_id or "",
            "transformation_chain": list(self.transformation_chain),
            "license_id": self.license_id,
            "annotation_version": self.annotation_version,
        }


def ensure_allowed_roles(
    records: Iterable[LineageRecord], allowed_roles: set[str] | frozenset[str]
) -> None:
    """Refuse records outside the caller's explicitly permitted split roles."""
    allowed = set(allowed_roles)
    unknown = allowed - ALLOWED_SPLIT_ROLES
    if unknown:
        raise ValueError(f"unsupported allowed roles: {sorted(unknown)}")
    for record in records:
        if record.split_role not in allowed:
            raise ValueError(
                f"role {record.split_role} is not allowed for sample {record.sample_id}"
            )


def validate_family_isolation(records: Iterable[LineageRecord]) -> None:
    """Ensure one document family never crosses model-governance roles."""
    roles_by_family: dict[str, set[str]] = {}
    for record in records:
        roles_by_family.setdefault(record.document_family_id, set()).add(record.split_role)
    violations = {
        family: sorted(roles)
        for family, roles in roles_by_family.items()
        if len(roles) > 1
    }
    if violations:
        sample = next(iter(sorted(violations.items())))
        raise ValueError(
            f"document family crosses split roles: {sample[0]} -> {sample[1]}"
        )
