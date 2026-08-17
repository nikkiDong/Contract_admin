"""Orchestration: contract package -> one submission row per requirement.

Execution order matters and is fixed:

    ingest -> applicability gate -> precedence resolution
           -> invariant detector -> (LLM only if uncertain) -> emit

Applicability runs before retrieval so out-of-scope requirements cannot generate
findings. Precedence runs before detection so superseded base text is never
tested. The model runs last and only on residual uncertainty.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import applicability, detectors, precedence
from .extract import discover_packages, load_package
from .llm import Adjudicator, build_provider
from .models import Finding, Package
from .reference import ReferenceChecklist

NO_ACTION = "No action required; retain as reviewed with no finding."
CONFIRM = "Confirm superseding document controls before accepting; no separate action."
REVIEW = "Route to contract reviewer for confirmation; human decision required."
PRIORITY_REVIEW = "Escalate for priority contract review before award/execution."


def _recommended_action(label: str, severity: str, superseded: bool) -> str:
    if label == "FLAG":
        return PRIORITY_REVIEW if severity in {"Critical", "High"} else REVIEW
    return CONFIRM if superseded else NO_ACTION


def analyse_package(
    package: Package,
    checklist: ReferenceChecklist,
    adjudicator: Adjudicator | None = None,
) -> list[Finding]:
    """Produce one Finding per requirement for a single package."""
    findings: list[Finding] = []

    for requirement in checklist:
        req_id = requirement.requirement_id
        decision, reason = applicability.decide(package, req_id)

        # --- out of scope: no clause read, no flag possible -----------------
        if decision == applicability.DOES_NOT_APPLY:
            findings.append(
                Finding(
                    document_id=package.package_id,
                    requirement_id=req_id,
                    applicability_decision=decision,
                    applicability_reason=reason,
                    predicted_label="NO_FLAG",
                    severity="Info",
                    governing_document="Not applicable",
                    draft_location="",
                    draft_evidence="",
                    reference_id=req_id,
                    reference_location=requirement.reference_location,
                    reference_evidence=requirement.challenge_reference_rule,
                    explanation=(
                        f"{requirement.requirement_name} is outside the scope of this "
                        f"package. {reason} No deviation review is performed and no "
                        "finding is raised."
                    ),
                    confidence=0.97,
                    recommended_human_action=NO_ACTION,
                    decided_by="applicability",
                    rule_id=f"{req_id}.not_applicable",
                )
            )
            continue

        # --- in scope: resolve the governing clause -------------------------
        governing, superseded, precedence_note = precedence.resolve(
            package, req_id, checklist
        )
        verdict = detectors.run(req_id, governing, package)
        decided_by = "rule"

        if verdict.uncertain and adjudicator is not None and governing is not None:
            verdict = adjudicator.adjudicate(requirement, governing, package, verdict)
            decided_by = "llm" if verdict.rule_id.endswith("+llm") else "rule"

        severity = requirement.severity_guidance if verdict.label == "FLAG" else "Info"

        if governing is not None:
            governing_document = f"{governing.doc_type} ({governing.file_name})"
            draft_location = governing.location
            draft_evidence = verdict.evidence or governing.text
        else:
            governing_document = "Not located in package"
            draft_location = f"{package.package_id} package documents (no matching section)"
            draft_evidence = verdict.evidence

        explanation = verdict.explanation
        if superseded:
            explanation = f"{explanation} {precedence_note}"

        findings.append(
            Finding(
                document_id=package.package_id,
                requirement_id=req_id,
                applicability_decision=decision,
                applicability_reason=reason,
                predicted_label=verdict.label,
                severity=severity,
                governing_document=governing_document,
                draft_location=draft_location,
                draft_evidence=_tidy(draft_evidence),
                reference_id=req_id,
                reference_location=requirement.reference_location,
                reference_evidence=requirement.challenge_reference_rule,
                explanation=_tidy(explanation),
                confidence=round(verdict.confidence, 2),
                recommended_human_action=_recommended_action(
                    verdict.label, severity, bool(superseded)
                ),
                decided_by=decided_by,
                rule_id=verdict.rule_id,
            )
        )

    return findings


def _tidy(text: str) -> str:
    return " ".join((text or "").split())


def run_split(
    split_root: str | Path,
    checklist: ReferenceChecklist,
    llm_provider: str = "null",
    llm_model: str | None = None,
) -> tuple[list[Finding], list[Package], Adjudicator]:
    """Analyse every package under a split directory (Development/ or Validation/)."""
    adjudicator = Adjudicator(build_provider(llm_provider, llm_model))
    findings: list[Finding] = []
    packages: list[Package] = []

    for package_dir in discover_packages(split_root):
        package = load_package(package_dir, checklist)
        packages.append(package)
        findings.extend(analyse_package(package, checklist, adjudicator))

    return findings, packages, adjudicator


def write_submission(findings: list[Finding], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(Finding.SUBMISSION_FIELDS))
        writer.writeheader()
        for finding in findings:
            writer.writerow(finding.to_submission_row())
    return path


def write_audit(findings: list[Finding], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Finding.SUBMISSION_FIELDS) + ["decided_by", "rule_id"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            writer.writerow(finding.to_audit_row())
    return path
