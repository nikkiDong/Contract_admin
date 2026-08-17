"""PDF ingest: package documents -> heading-anchored `Clause` records.

Text extraction uses `pdftotext -layout` when available and falls back to
`pypdf`. Segmentation is heading-driven: a line is treated as a heading when it
normalises to a known checklist heading (or the Addendum "Revision to ..."
form). Everything between two headings is the clause body.

This is deliberately not a chunker. Clauses are the unit the checklist compares
against, so splitting on anything other than headings would break evidence
grounding.
"""

from __future__ import annotations

import json
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

from .models import Clause, Package
from .reference import ReferenceChecklist, REVISION_PREFIX, normalise

# Boilerplate stamped on every rendered page.
_NOISE = re.compile(
    r"CONTRACT CLAUSE RISK FLAGGING|FOR EVALUATION USE ONLY|"
    r"SAMPLE MATERIAL|SAMPLE ATTACHMENT|NOT AN EXECUTED CONTRACT|"
    r"^\s*Page \d+\s*$",
    re.IGNORECASE,
)
_REPLACEMENT_MARKER = re.compile(r"^\s*REPLACEMENT TEXT\s*:?\s*$", re.IGNORECASE)


class PdfTextError(RuntimeError):
    pass


# Backend selection. AWS Lambda has no poppler binary, so the deployed path uses
# pypdf. Set CRF_PDF_BACKEND=pypdf to exercise that path locally and confirm the
# two backends agree before deploying.
_BACKEND = os.environ.get("CRF_PDF_BACKEND", "auto").strip().lower()


def _pages_via_pdftotext(path: Path) -> list[str] | None:
    if not shutil.which("pdftotext"):
        return None
    proc = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.split("\f") if proc.returncode == 0 else None


def _pages_via_pypdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise PdfTextError(
            f"Cannot read {path.name}: install poppler (pdftotext) or `pip install pypdf`."
        ) from exc
    return [(page.extract_text() or "") for page in PdfReader(str(path)).pages]


def pdf_pages(path: Path) -> list[str]:
    """Return the text of each page, in order.

    `pdftotext -layout` is preferred locally for its column handling. `pypdf` is
    the pure-Python fallback and the backend used in Lambda.
    """
    if _BACKEND == "pypdf":
        return _pages_via_pypdf(path)
    if _BACKEND == "pdftotext":
        pages = _pages_via_pdftotext(path)
        if pages is None:
            raise PdfTextError("CRF_PDF_BACKEND=pdftotext but pdftotext is not on PATH.")
        return pages
    return _pages_via_pdftotext(path) or _pages_via_pypdf(path)


def _clean_lines(page_text: str) -> list[str]:
    out = []
    for line in page_text.splitlines():
        line = line.strip()
        if not line or _NOISE.search(line):
            continue
        out.append(line)
    return out


def parse_document(
    path: Path,
    package_id: str,
    doc_type: str,
    checklist: ReferenceChecklist,
) -> list[Clause]:
    """Split one PDF into heading-anchored clauses."""
    clauses: list[Clause] = []
    current: Clause | None = None

    for page_no, page_text in enumerate(pdf_pages(path), start=1):
        for line in _clean_lines(page_text):
            if _REPLACEMENT_MARKER.match(line):
                if current is not None:
                    current.is_replacement = True
                continue

            if checklist.is_known_heading(line):
                is_revision = bool(REVISION_PREFIX.match(line))
                current = Clause(
                    package_id=package_id,
                    file_name=path.name,
                    doc_type=doc_type,
                    heading=REVISION_PREFIX.sub("", line).strip(),
                    text="",
                    page=page_no,
                    is_replacement=is_revision,
                    revises_heading=(
                        REVISION_PREFIX.sub("", line).strip() if is_revision else None
                    ),
                )
                clauses.append(current)
                continue

            if current is not None:
                current.text = (current.text + " " + line).strip()

    for clause in clauses:
        clause.text = re.sub(r"\s+", " ", clause.text).strip()
        # Rejoin words split across a line break by the PDF layout.
        clause.text = re.sub(r"(\w)-\s(\w)", r"\1-\2", clause.text)
    return [c for c in clauses if c.text]


def _read_doc_index(package_root: Path) -> list[dict]:
    index_path = package_root / "Document_Index.csv"
    if not index_path.exists():
        return []
    with open(index_path, newline="", encoding="utf-8-sig") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _doc_type_for(file_name: str, doc_index: list[dict]) -> str:
    for row in doc_index:
        if row.get("File_Name", "").strip() == file_name:
            return row.get("Document_Type", "").strip()
    stem = Path(file_name).stem.replace("_", " ")
    return stem


def load_package(package_root: str | Path, checklist: ReferenceChecklist) -> Package:
    """Load metadata, document index and all clauses for one contract package."""
    root = Path(package_root)
    meta_path = root / "Project_Metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    doc_index = _read_doc_index(root)

    docs_dir = root / "Docs"
    pdf_paths = sorted(docs_dir.glob("*.pdf")) if docs_dir.exists() else []

    package = Package(
        package_id=str(metadata.get("package_id") or root.name).strip(),
        root=str(root),
        project_title=str(metadata.get("project_title") or root.name).strip(),
        metadata=metadata,
        doc_index=doc_index,
        doc_files=[p.name for p in pdf_paths],
    )

    for pdf in pdf_paths:
        doc_type = _doc_type_for(pdf.name, doc_index)
        package.clauses.extend(
            parse_document(pdf, package.package_id, doc_type, checklist)
        )
    return package


def discover_packages(split_root: str | Path) -> list[Path]:
    """Every immediate subdirectory holding a Project_Metadata.json."""
    root = Path(split_root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "Project_Metadata.json").exists()
    )


def clause_summary(package: Package, checklist: ReferenceChecklist) -> list[tuple]:
    """(requirement_id, file, heading) for every resolved clause. Debug helper."""
    rows = []
    for clause in package.clauses:
        rows.append(
            (
                checklist.resolve_heading(clause.heading) or "UNRESOLVED",
                clause.file_name,
                clause.heading,
            )
        )
    return rows
