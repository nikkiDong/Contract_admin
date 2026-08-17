"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Reference layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Requirement:
    """One row of References/Reference_Checklist.csv."""

    requirement_id: str
    tier: str
    requirement_name: str
    reference_source: str
    section: str
    applicability_rule: str
    review_expectation: str
    severity_guidance: str
    evidence_required: str
    challenge_reference_rule: str

    @property
    def reference_location(self) -> str:
        """Authoritative citation used in the submission."""
        return f"{self.reference_source} - {self.section}"


# ---------------------------------------------------------------------------
# Ingest layer
# ---------------------------------------------------------------------------

@dataclass
class Clause:
    """A heading-anchored block of text lifted out of one package document."""

    package_id: str
    file_name: str
    doc_type: str
    heading: str
    text: str
    page: int
    is_replacement: bool = False
    revises_heading: Optional[str] = None

    @property
    def location(self) -> str:
        """Human-checkable pointer back into the reviewed package."""
        return f"{self.file_name} > {self.heading} (page {self.page})"


@dataclass
class Package:
    """A single contract package: metadata, document index and parsed clauses."""

    package_id: str
    root: str
    project_title: str
    metadata: dict
    doc_index: list[dict] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)
    doc_files: list[str] = field(default_factory=list)

    # -- metadata accessors (normalised to bool / list) ---------------------

    def _yes(self, key: str) -> bool:
        return str(self.metadata.get(key, "")).strip().lower() in {"yes", "true", "y"}

    @property
    def federal_aid(self) -> bool:
        return self._yes("federal_aid")

    @property
    def baba_applicable(self) -> bool:
        return self._yes("buy_america_baba_applicable")

    @property
    def subcontracting_planned(self) -> bool:
        return self._yes("subcontracting_planned")

    @property
    def claim_event(self) -> bool:
        return self._yes("claim_event")

    @property
    def delay_event(self) -> bool:
        return self._yes("delay_event")

    @property
    def changed_work_event(self) -> bool:
        return self._yes("changed_work_event")

    @property
    def issued_addenda(self) -> list[str]:
        raw = self.metadata.get("issued_addenda") or []
        if isinstance(raw, str):
            raw = [p.strip() for p in raw.split(",") if p.strip()]
        return [a for a in raw if a and a.strip().lower() != "none"]

    @property
    def contract_value(self) -> Optional[int]:
        val = self.metadata.get("assumed_contract_value")
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    @property
    def has_fhwa_1273_attachment(self) -> bool:
        """FHWA-1273 counts as physically included only if the form is in Docs/."""
        return any("fhwa" in f.lower() and "1273" in f.lower() for f in self.doc_files)


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """One submission row: document_id x requirement_id."""

    document_id: str
    requirement_id: str
    applicability_decision: str
    applicability_reason: str
    predicted_label: str
    severity: str
    governing_document: str
    draft_location: str
    draft_evidence: str
    reference_id: str
    reference_location: str
    reference_evidence: str
    explanation: str
    confidence: float
    recommended_human_action: str

    # provenance (not part of the submission schema, kept for audit/debug)
    decided_by: str = "rule"
    rule_id: str = ""

    SUBMISSION_FIELDS = (
        "document_id",
        "requirement_id",
        "applicability_decision",
        "applicability_reason",
        "predicted_label",
        "severity",
        "governing_document",
        "draft_location",
        "draft_evidence",
        "reference_id",
        "reference_location",
        "reference_evidence",
        "explanation",
        "confidence",
        "recommended_human_action",
    )

    def to_submission_row(self) -> dict:
        row = asdict(self)
        row["confidence"] = f"{self.confidence:.2f}"
        return {k: row[k] for k in self.SUBMISSION_FIELDS}

    def to_audit_row(self) -> dict:
        row = self.to_submission_row()
        row["decided_by"] = self.decided_by
        row["rule_id"] = self.rule_id
        return row


@dataclass
class Verdict:
    """Output of a single requirement detector before it becomes a Finding."""

    label: str                      # FLAG | NO_FLAG
    explanation: str
    confidence: float
    rule_id: str
    evidence: str = ""
    uncertain: bool = False         # True -> escalate to the LLM adjudicator
