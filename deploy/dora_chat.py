"""DORA chat service: serves the review-assistant web page and proxies chat
turns to DORA's system prompt (prompts/dora_agent_prompt_v2.xml).

Two interchangeable backends, selected by DORA_BACKEND (default "ces"):

  DORA_BACKEND=ces (default)
    Proxies to a hosted Gemini Enterprise (CES) deployment where DORA's
    persona and its six tools are configured on Google's side. This process
    only forwards user text to `sessions:runSession` and returns the agent's
    text — no tool loop runs locally. Auth is an OAuth access token minted
    from Application Default Credentials (equivalent to
    `gcloud auth print-access-token`, but usable from a server process).
    Configure via CES_PROJECT / CES_LOCATION / CES_APP / CES_DEPLOYMENT /
    CES_APP_VERSION / CES_API_VERSION (defaults match the deployment this was
    built against).

  DORA_BACKEND=gemini
    Runs the tool-use loop locally against the raw Gemini API. Six tools,
    matching the {@TOOL: ...} references in the prompt:

        lookup_reference_requirement   Tier 1 - checklist row (Challenge_Reference_Rule)
        get_project_metadata           Applicability gate - Project_Metadata.json booleans
        search_contract_package        Clause retrieval - base + addendum clauses for a heading
        resolve_governing_document     Precedence - the one clause that governs
        verify_evidence_verbatim       Evidence check - is this span really in that document
        specifications                 Tier 3 - best-effort live fetch of the external authority text

    The first five are deterministic lookups over the local challenge dataset
    (same data tool_server.py wraps). `specifications` is the one tool that
    reaches the open internet; it degrades to "nothing found" rather than
    fabricating text. Auth is a Gemini API key from GOOGLE_API_KEY. Kept as a
    fallback for when the hosted CES deployment is unreachable or misconfigured.

No credential is ever sent to, or embedded in, the browser either way.

Run locally (CES backend):
    pip install -r deploy/requirements-dora-chat.txt
    gcloud auth application-default login     # once, interactively
    python3 deploy/dora_chat.py                  # http://localhost:8081

Run locally (Gemini fallback):
    export DORA_BACKEND=gemini
    export GOOGLE_API_KEY=AQ...                # from Google AI Studio / Cloud Console
    python3 deploy/dora_chat.py

Deploy (Cloud Run):
    gcloud run deploy dora-chat --source . \
        --region us-central1 \
        --set-env-vars DORA_BACKEND=ces \
        --allow-unauthenticated
    # grant the Cloud Run service account access to the CES app/deployment
    # instead of relying on `gcloud auth application-default login`.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Lets this module run as `python3 deploy/dora_chat.py` from any cwd,
    # not just when the repo root happens to already be on sys.path.
    sys.path.insert(0, str(_REPO_ROOT))

from crf import precedence
from crf.extract import discover_packages, load_package
from crf.models import Clause, Package
from crf.reference import ReferenceChecklist

LOG = logging.getLogger("dora_chat")
logging.basicConfig(level=logging.INFO)

ROOT = _REPO_ROOT
DATA_ROOT = Path(os.environ.get("CRF_DATA_ROOT", ROOT / "Contract_Clause_Risk_Flagging")).resolve()
CHECKLIST_PATH = DATA_ROOT / "References" / "Reference_Checklist.csv"
SYSTEM_PROMPT_PATH = Path(
    os.environ.get("DORA_SYSTEM_PROMPT", ROOT / "prompts" / "dora_agent_prompt_v2.xml")
)
STATIC_DIR = ROOT / "website"
SPLITS = ("Development", "Validation")
MODEL_ID = os.environ.get("DORA_MODEL_ID", "gemini-3.6-flash")
MAX_TOOL_ITERATIONS = 12
MAX_SESSIONS = 200  # simple bound so an in-memory demo server can't grow unbounded

# DORA_BACKEND selects who runs the agent loop:
#   "ces"    (default) - proxy to a hosted Gemini Enterprise (CES) deployment,
#            where the DORA persona and the six tools are configured on
#            Google's side. This process just forwards text and returns text.
#   "gemini" - run the tool-use loop locally against the raw Gemini API, using
#            the TOOLS/_TOOL_IMPL defined above. This is the path that was
#            tested end-to-end before the CES deployment existed; kept as a
#            fallback in case the hosted deployment is unreachable or
#            misconfigured.
DORA_BACKEND = os.environ.get("DORA_BACKEND", "ces").strip().lower()

CES_API_VERSION = os.environ.get("CES_API_VERSION", "v1beta")
CES_PROJECT = os.environ.get("CES_PROJECT", "hackathon-2026-transport-2")
CES_LOCATION = os.environ.get("CES_LOCATION", "us")
CES_APP = os.environ.get("CES_APP", "a9c38b91-99b9-429d-a249-0154dba7969a")
CES_DEPLOYMENT = os.environ.get("CES_DEPLOYMENT", "b4649aef-1936-47f8-adb2-56ed11f68b55")
# Optional: the console's "test this deployment" snippet includes a pinned app
# version. Leave unset to let the deployment resolve its own current version.
CES_APP_VERSION = os.environ.get("CES_APP_VERSION", "4c6b5387-fae7-42ec-a17c-2da3dcd88d8f")

# User-uploaded packages: never mixed into the sample dataset directory, and
# excluded from git (see .gitignore) since they may be private documents.
UPLOAD_ROOT = Path(os.environ.get("DORA_UPLOAD_ROOT", ROOT / "uploads")).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
_PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ALLOWED_UPLOAD_NAMES = {"Project_Metadata.json", "Document_Index.csv"}
MAX_UPLOAD_FILES = 40
MAX_UPLOAD_FILE_BYTES = 25 * 1024 * 1024  # 25 MB/file — generous for a contract PDF

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

# --------------------------------------------------------------------------- #
# Dataset cache (same shape as deploy/tool_server.py, kept independent so the
# two tool servers can be deployed separately)
# --------------------------------------------------------------------------- #

_checklist: Optional[ReferenceChecklist] = None
_packages: dict[str, Package] = {}


def checklist() -> ReferenceChecklist:
    global _checklist
    if _checklist is None:
        _checklist = ReferenceChecklist.load(CHECKLIST_PATH)
    return _checklist


def _load_all() -> None:
    if _packages:
        return
    for split_root in [DATA_ROOT / split for split in SPLITS] + [UPLOAD_ROOT]:
        if not split_root.exists():
            continue
        for pkg_root in discover_packages(split_root):
            pkg = load_package(pkg_root, checklist())
            _packages[pkg.package_id.upper()] = pkg


def get_package(package_id: str) -> Package:
    _load_all()
    pkg = _packages.get(package_id.strip().upper())
    if pkg is None:
        raise KeyError(
            f"Unknown package_id {package_id!r}. Known: {', '.join(sorted(_packages))}"
        )
    return pkg


def clause_payload(c: Clause) -> dict:
    return {
        "file_name": c.file_name,
        "doc_type": c.doc_type,
        "heading": c.heading,
        "page": c.page,
        "location": c.location,
        "text": c.text,
        "is_replacement": c.is_replacement,
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #

def tool_lookup_reference_requirement(requirement_id: str) -> dict:
    req_id = (requirement_id or "").strip().upper()
    try:
        r = checklist().get(req_id)
    except KeyError:
        return {"error": f"Unknown requirement_id {requirement_id!r}. Known: {', '.join(checklist().ids)}"}
    return {
        "requirement_id": r.requirement_id,
        "tier": r.tier,
        "requirement_name": r.requirement_name,
        "reference_source": r.reference_source,
        "section": r.section,
        "applicability_rule": r.applicability_rule,
        "review_expectation": r.review_expectation,
        "severity_guidance": r.severity_guidance,
        "evidence_required": r.evidence_required,
        "challenge_reference_rule": r.challenge_reference_rule,
        "reference_location": r.reference_location,
    }


def tool_get_project_metadata(package_id: str) -> dict:
    try:
        pkg = get_package(package_id)
    except KeyError as exc:
        return {"error": str(exc)}
    return {
        "package_id": pkg.package_id,
        "project_title": pkg.project_title,
        "federal_aid": pkg.federal_aid,
        "buy_america_baba_applicable": pkg.baba_applicable,
        "subcontracting_planned": pkg.subcontracting_planned,
        "claim_event": pkg.claim_event,
        "delay_event": pkg.delay_event,
        "changed_work_event": pkg.changed_work_event,
        "issued_addenda": pkg.issued_addenda,
        "assumed_contract_value": pkg.contract_value,
        "document_files": pkg.doc_files,
    }


def tool_search_contract_package(package_id: str, requirement_name: str) -> dict:
    try:
        pkg = get_package(package_id)
    except KeyError as exc:
        return {"error": str(exc)}

    req_id = checklist().resolve_heading(requirement_name)
    if req_id is None:
        return {
            "package_id": pkg.package_id,
            "requirement_name": requirement_name,
            "resolved_requirement_id": None,
            "clause_count": 0,
            "clauses": [],
            "message": (
                "No requirement matched this heading. Use the exact Requirement_Name "
                "string from lookup_reference_requirement, not a paraphrase."
            ),
        }

    clauses = precedence.candidates(pkg, req_id, checklist())
    return {
        "package_id": pkg.package_id,
        "requirement_name": requirement_name,
        "resolved_requirement_id": req_id,
        "clause_count": len(clauses),
        "clauses": [clause_payload(c) for c in clauses],
        "message": (
            "Includes base documents and any Addenda ('Revision to <name>'). "
            "Resolve precedence with resolve_governing_document before judging any one clause."
        ),
    }


def _precedence_basis(governing: Optional[Clause], superseded: list[Clause]) -> str:
    if governing is None:
        return "not_located"
    if precedence.is_addendum(governing) and governing.is_replacement:
        return "addendum_replacement"
    return "deldot_105_6" if superseded else "single_occurrence"


def tool_resolve_governing_document(package_id: str, requirement_id: str) -> dict:
    try:
        pkg = get_package(package_id)
    except KeyError as exc:
        return {"error": str(exc)}

    req_id = (requirement_id or "").strip().upper()
    if req_id not in set(checklist().ids):
        return {"error": f"Unknown requirement_id {requirement_id!r}"}

    governing, superseded, note = precedence.resolve(pkg, req_id, checklist())
    return {
        "package_id": pkg.package_id,
        "requirement_id": req_id,
        "found": governing is not None,
        "governing": clause_payload(governing) if governing else None,
        "governing_document": (
            f"{governing.doc_type} ({governing.file_name})" if governing else "Not located in package"
        ),
        "superseded": [clause_payload(c) for c in superseded],
        "resolution_basis": _precedence_basis(governing, superseded),
        "resolution_note": note,
    }


def tool_verify_evidence_verbatim(package_id: str, file_name: str, span: str) -> dict:
    try:
        pkg = get_package(package_id)
    except KeyError as exc:
        return {"error": str(exc)}

    wanted_file = (file_name or "").strip().lower()
    in_file = [c for c in pkg.clauses if c.file_name.lower() == wanted_file]
    if not in_file:
        return {"error": f"{file_name!r} is not a document of {pkg.package_id}. Documents: {', '.join(pkg.doc_files)}"}

    needle = _norm(span)
    if not needle:
        return {
            "verbatim": False,
            "package_id": pkg.package_id,
            "file_name": file_name,
            "matched_heading": None,
            "matched_page": None,
            "message": "Empty span. An empty draft_evidence is only correct for a DOES_NOT_APPLY row.",
        }

    for c in in_file:
        if needle in _norm(c.text):
            return {
                "verbatim": True,
                "package_id": pkg.package_id,
                "file_name": c.file_name,
                "matched_heading": c.heading,
                "matched_page": c.page,
                "message": f"Verbatim span of {c.location}.",
            }

    return {
        "verbatim": False,
        "package_id": pkg.package_id,
        "file_name": file_name,
        "matched_heading": None,
        "matched_page": None,
        "message": (
            "This span does not occur in the cited document. It is not evidence. "
            "Retrieve the governing clause again and quote text that exists, or lower "
            "confidence and escalate the row for human review."
        ),
    }


# Reference_Source keyword -> external authority page, per the knowledge_source_rules
# in the DORA prompt. Best-effort: these are large government pages, so the fetch is
# whole-page text, not a section-precise API. Section-precision is left to Tier 1.
_SPEC_SOURCES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"fhwa-?\s*1273|federal-aid", re.I), "https://www.fhwa.dot.gov/construction/cqit/form1273.cfm"),
    (re.compile(r"contractor registration", re.I), "https://delcode.delaware.gov/title19/c036/index.html"),
    (re.compile(r"public works licens|business.*licens|subcontractor.*licens", re.I), "https://delcode.delaware.gov/title29/c069/index.html"),
    (re.compile(r"deldot", re.I), "https://engineeringsupport.deldot.gov/index.php/Standard_Specifications"),
]
_HTML_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_HTML_TAG = re.compile(r"<[^>]+>")


def _pick_spec_url(reference_source: str, section: str) -> Optional[str]:
    haystack = f"{reference_source} {section}"
    for pattern, url in _SPEC_SOURCES:
        if pattern.search(haystack):
            return url
    return None


def tool_specifications(reference_source: str, section: str) -> dict:
    url = _pick_spec_url(reference_source, section)
    if url is None:
        return {
            "section": section,
            "reference_source": reference_source,
            "url": None,
            "found": False,
            "excerpt": "",
            "message": "No mapped external source for this Reference_Source. Tier 3 is silent; rely on Tier 1.",
        }
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "DORA-contract-reviewer/1.0"})
        resp.raise_for_status()
    except Exception as exc:  # network unavailable, blocked, 4xx/5xx, timeout, ...
        return {
            "section": section,
            "reference_source": reference_source,
            "url": url,
            "found": False,
            "excerpt": "",
            "message": f"Live fetch failed ({exc.__class__.__name__}). Tier 3 returned nothing from this server; rely on Tier 1.",
        }

    stripped = _HTML_SCRIPT_STYLE.sub(" ", resp.text)
    text = re.sub(r"\s+", " ", html.unescape(_HTML_TAG.sub(" ", stripped))).strip()
    needle = re.sub(r"[^0-9A-Za-z.§ ]", "", section).strip()
    idx = text.find(needle) if needle else -1
    if idx >= 0:
        excerpt = text[max(0, idx - 300): idx + 1200]
    else:
        excerpt = text[:1500]

    return {
        "section": section,
        "reference_source": reference_source,
        "url": url,
        "found": True,
        "excerpt": excerpt,
        "message": (
            "Live page fetched. Whole-page text, not guaranteed section-precise. "
            "Interpretive support only — never a source for a number, deadline, or rate that Tier 1 already states."
        ),
    }


TOOLS: list[dict] = [
    {
        "name": "lookup_reference_requirement",
        "description": (
            "Tier 1. Return the full Reference_Checklist.csv row for one requirement ID: "
            "Requirement_Name, Reference_Source, Section, Applicability_Rule, Review_Expectation, "
            "Severity_Guidance, Evidence_Required, and the authoritative Challenge_Reference_Rule "
            "text. Call this first, before anything else, for every requirement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"requirement_id": {"type": "string", "description": "e.g. CC-04"}},
            "required": ["requirement_id"],
        },
    },
    {
        "name": "get_project_metadata",
        "description": (
            "Return the exact Project_Metadata.json booleans for a package (federal_aid, "
            "buy_america_baba_applicable, subcontracting_planned, claim_event, delay_event, "
            "changed_work_event, issued_addenda, assumed_contract_value, document_files). "
            "Call before deciding applicability; never infer these from clause text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"package_id": {"type": "string", "description": "e.g. DEV-HARBOR-CROSSING"}},
            "required": ["package_id"],
        },
    },
    {
        "name": "search_contract_package",
        "description": (
            "Retrieve every clause in the package (base documents AND addenda) whose heading "
            "resolves to the given Requirement_Name, or to 'Revision to <Requirement_Name>'. "
            "Must be called before resolve_governing_document — a retrieval that skips addenda "
            "produces a confidently wrong precedence decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "package_id": {"type": "string"},
                "requirement_name": {
                    "type": "string",
                    "description": "The exact Requirement_Name from the checklist row, e.g. 'Performance and payment bonds'.",
                },
            },
            "required": ["package_id", "requirement_name"],
        },
    },
    {
        "name": "resolve_governing_document",
        "description": (
            "Resolve which single clause governs a requirement in a package: Addendum "
            "supersession first (latest ordinal wins), else the DelDOT 105.6 precedence ladder. "
            "Returns the governing clause and everything it supersedes. Test only the governing clause."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "package_id": {"type": "string"},
                "requirement_id": {"type": "string"},
            },
            "required": ["package_id", "requirement_id"],
        },
    },
    {
        "name": "verify_evidence_verbatim",
        "description": (
            "Check whether a candidate draft_evidence span appears verbatim (whitespace/case "
            "normalised only) in the named document. Call on every non-empty draft_evidence "
            "before emitting a row. If verbatim is false, the quote is not evidence — retrieve "
            "the clause again and quote text that exists, or lower confidence and escalate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "package_id": {"type": "string"},
                "file_name": {"type": "string", "description": "e.g. Addendum_B.pdf"},
                "span": {"type": "string", "description": "The exact text proposed as draft_evidence."},
            },
            "required": ["package_id", "file_name", "span"],
        },
    },
    {
        "name": "specifications",
        "description": (
            "Tier 3. Best-effort live fetch of the external authoritative source page for a "
            "Section citation (DelDOT engineeringsupport.deldot.gov, fhwa.dot.gov FHWA-1273, or "
            "delcode.delaware.gov). Interpretive support only — it may clarify a Tier 1 term but "
            "never supplies or overrides a number, deadline, threshold, or rate that Tier 1 states. "
            "May report that no network-reachable text was found; that is a valid, expected result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reference_source": {"type": "string", "description": "The checklist Reference_Source column value."},
                "section": {"type": "string", "description": "The exact Section citation, e.g. 'DelDOT 103.5' or '19 Del. C. § 3604'."},
            },
            "required": ["reference_source", "section"],
        },
    },
]

_TOOL_IMPL = {
    "lookup_reference_requirement": lambda i: tool_lookup_reference_requirement(**i),
    "get_project_metadata": lambda i: tool_get_project_metadata(**i),
    "search_contract_package": lambda i: tool_search_contract_package(**i),
    "resolve_governing_document": lambda i: tool_resolve_governing_document(**i),
    "verify_evidence_verbatim": lambda i: tool_verify_evidence_verbatim(**i),
    "specifications": lambda i: tool_specifications(**i),
}


# --------------------------------------------------------------------------- #
# CES (hosted Gemini Enterprise deployment) client
#
# DORA's persona and tools are configured on Google's side for this
# deployment, so this process is a thin proxy: forward the user's text,
# return the agent's text. No local tool loop runs on this path.
# --------------------------------------------------------------------------- #

_ces_credentials = None
_ces_auth_attempted = False
_ces_auth_error: Optional[str] = None


def _ces_access_token() -> str:
    """Mint/refresh an OAuth access token via Application Default Credentials.

    Equivalent to `gcloud auth print-access-token`, but usable from a server
    process without shelling out to the CLI. Needs either
    `gcloud auth application-default login` run once in this environment, or
    a service account attached (e.g. Cloud Run's runtime service account).

    A missing-ADC lookup itself is slow (google-auth probes the GCE/Cloud Run
    metadata server, which can take several seconds to time out on a machine
    that isn't GCP). That failure is cached after the first attempt so a
    misconfigured server fails every subsequent call fast instead of re-paying
    that probe on every request; restart the process after fixing credentials.
    """
    global _ces_credentials, _ces_auth_attempted, _ces_auth_error
    if not _ces_auth_attempted:
        _ces_auth_attempted = True
        import google.auth
        import google.auth.exceptions

        try:
            _ces_credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except google.auth.exceptions.DefaultCredentialsError as exc:
            _ces_auth_error = str(exc)

    if _ces_credentials is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No Google Cloud Application Default Credentials found on this server. "
                "Run `gcloud auth application-default login` in the environment running "
                "dora_chat.py (or attach a service account in production), then restart "
                f"the service. Underlying error: {_ces_auth_error}"
            ),
        )

    if not _ces_credentials.valid:
        import google.auth.transport.requests

        _ces_credentials.refresh(google.auth.transport.requests.Request())
    return _ces_credentials.token


def _ces_session_resource(session_id: str) -> str:
    return f"projects/{CES_PROJECT}/locations/{CES_LOCATION}/apps/{CES_APP}/sessions/{session_id}"


def _ces_run_session(session_id: str, message: str) -> tuple[str, list[dict]]:
    token = _ces_access_token()
    session_resource = _ces_session_resource(session_id)

    config: dict[str, Any] = {
        "session": session_resource,
        "deployment": f"projects/{CES_PROJECT}/locations/{CES_LOCATION}/apps/{CES_APP}/deployments/{CES_DEPLOYMENT}",
    }
    if CES_APP_VERSION:
        config["app_version"] = (
            f"projects/{CES_PROJECT}/locations/{CES_LOCATION}/apps/{CES_APP}/versions/{CES_APP_VERSION}"
        )

    url = f"https://ces.googleapis.com/{CES_API_VERSION}/{session_resource}:runSession"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"config": config, "inputs": [{"text": message}]},
            timeout=90,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"CES request failed: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"CES returned {resp.status_code}: {resp.text[:2000]}")

    data = resp.json()
    texts: list[str] = []
    trace: list[dict] = []
    for out in data.get("outputs", []):
        if "text" in out:
            texts.append(out["text"])
        elif "toolCalls" in out:
            # Tool execution is configured to happen on Google's side for this
            # deployment; surfaced here only as an audit-trail entry, not
            # something this process needs to act on.
            trace.append({"tool": "ces:toolCalls", "input": None, "output": out["toolCalls"]})
        elif "citations" in out:
            trace.append({"tool": "ces:citations", "input": None, "output": out["citations"]})
        elif "endSession" in out:
            trace.append({"tool": "ces:endSession", "input": None, "output": out["endSession"]})

    if not texts:
        LOG.warning("CES runSession returned no text output: %s", json.dumps(data)[:2000])
        return (
            "DORA's hosted agent responded without a text output. This can happen if the "
            "deployment expects streaming (streamRunSession, not implemented here) or the turn "
            "didn't complete. Check the server logs for the raw response.",
            trace,
        )
    return "".join(texts), trace


# --------------------------------------------------------------------------- #
# Gemini client + tool declarations + chat loop (local fallback; see
# DORA_BACKEND above)
# --------------------------------------------------------------------------- #

_genai_client = None
_genai_tool = None  # built lazily too, since it needs the `types` module


def _client():
    global _genai_client
    if _genai_client is None:
        if not os.environ.get("GOOGLE_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="GOOGLE_API_KEY is not set on the server. Export it in the server "
                "environment (never in the browser) and restart the service.",
            )
        from google import genai  # lazy import: app can boot with no key configured

        _genai_client = genai.Client()
    return _genai_client


def _tool() -> Any:
    """The single genai.types.Tool wrapping all six DORA tool declarations."""
    global _genai_tool
    if _genai_tool is None:
        from google.genai import types

        _genai_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["input_schema"],
                )
                for t in TOOLS
            ]
        )
    return _genai_tool


# session_id -> genai.types.Content list. In-memory only: a demo-scale
# convenience, not a durable store. Restarting the server clears all sessions.
_sessions: dict[str, list[Any]] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    reply: str
    trace: list[dict]


app = FastAPI(title="DORA chat", version="1.0.0")


@app.get("/api/packages")
def list_packages() -> dict:
    _load_all()
    return {"packages": sorted(_packages)}


def _safe_upload_package_id(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        value = f"UPLOAD-{uuid.uuid4().hex[:8].upper()}"
    if not _PACKAGE_ID_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail="package_id may contain only letters, digits, '-' and '_' (max 64 characters).",
        )
    return value


@app.post("/api/packages/upload")
async def upload_package(
    package_id: str = Form(""),
    files: list[UploadFile] = File(...),
) -> dict:
    """Accept a user-supplied contract package and make it reviewable immediately.

    Expects, in any order, as one multipart upload: a file literally named
    Project_Metadata.json (required), an optional Document_Index.csv, and one or
    more *.pdf documents. Anything else is ignored. Classification is by exact
    filename / extension, not by any folder structure the browser may have sent,
    so a flat multi-file picker works with no client-side directory handling.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_UPLOAD_FILES}).")

    _load_all()  # ensure the sample dataset is loaded before we add to _packages

    pkg_id = _safe_upload_package_id(package_id)
    pkg_root = (UPLOAD_ROOT / pkg_id).resolve()
    if UPLOAD_ROOT not in pkg_root.parents:
        raise HTTPException(status_code=400, detail="Invalid package_id.")

    # A re-upload under the same id replaces the prior contents outright, so
    # stale documents from an earlier attempt can never linger and get judged.
    if pkg_root.exists():
        shutil.rmtree(pkg_root)
    docs_dir = pkg_root / "Docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    has_metadata = False
    saved_pdfs: list[str] = []
    try:
        for f in files:
            name = Path(f.filename or "").name  # drop any path component the browser sent
            ext = Path(name).suffix.lower()
            if name not in ALLOWED_UPLOAD_NAMES and ext != ".pdf":
                continue  # ignore .DS_Store and anything else unrecognised

            data = await f.read()
            if len(data) > MAX_UPLOAD_FILE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} exceeds the {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB per-file limit.",
                )

            if name == "Project_Metadata.json":
                (pkg_root / name).write_bytes(data)
                has_metadata = True
            elif name == "Document_Index.csv":
                (pkg_root / name).write_bytes(data)
            elif ext == ".pdf":
                (docs_dir / name).write_bytes(data)
                saved_pdfs.append(name)

        if not has_metadata:
            raise HTTPException(
                status_code=400,
                detail="Project_Metadata.json is required (exact file name) so applicability can be checked.",
            )
        if not saved_pdfs:
            raise HTTPException(status_code=400, detail="At least one PDF document is required.")

        try:
            pkg = load_package(pkg_root, checklist())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse the package: {exc}") from exc

    except HTTPException:
        shutil.rmtree(pkg_root, ignore_errors=True)
        raise

    _packages[pkg.package_id.upper()] = pkg
    LOG.info("uploaded package %s: %d PDF(s), %d clause(s)", pkg.package_id, len(pkg.doc_files), len(pkg.clauses))
    return {
        "package_id": pkg.package_id,
        "project_title": pkg.project_title,
        "documents": pkg.doc_files,
        "clauses_extracted": len(pkg.clauses),
    }


@app.get("/api/health")
def health() -> dict:
    base = {"ok": True, "backend": DORA_BACKEND, "system_prompt_chars": len(SYSTEM_PROMPT)}
    if DORA_BACKEND == "ces":
        adc_found = True
        try:
            _ces_access_token()
        except HTTPException:
            adc_found = False
        base.update(
            {
                "model": "ces:" + CES_DEPLOYMENT,
                "model_key_configured": adc_found,
                "ces_session_resource_prefix": f"projects/{CES_PROJECT}/locations/{CES_LOCATION}/apps/{CES_APP}",
            }
        )
    else:
        base.update({"model": MODEL_ID, "model_key_configured": bool(os.environ.get("GOOGLE_API_KEY"))})
    return base


@app.post("/api/session/reset")
def reset_session(payload: dict) -> dict:
    session_id = str(payload.get("session_id", "")).strip()
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if DORA_BACKEND == "ces":
        reply, trace = _ces_run_session(req.session_id, req.message)
        return ChatResponse(reply=reply, trace=trace)
    return _gemini_chat(req)


def _gemini_chat(req: ChatRequest) -> ChatResponse:
    from google.genai import types

    client = _client()
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[_tool()],
        max_output_tokens=8192,
    )

    if req.session_id not in _sessions and len(_sessions) >= MAX_SESSIONS:
        # Evict an arbitrary session rather than growing without bound.
        _sessions.pop(next(iter(_sessions)), None)
    history = _sessions.setdefault(req.session_id, [])
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=req.message)]))

    trace: list[dict[str, Any]] = []
    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            resp = client.models.generate_content(model=MODEL_ID, contents=history, config=cfg)
        except Exception as exc:
            LOG.exception("Gemini API call failed")
            raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc

        if not resp.candidates or resp.candidates[0].content is None:
            reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
            return ChatResponse(
                reply=f"DORA got no usable response from the model (finish_reason={reason}). Please retry.",
                trace=trace,
            )

        turn = resp.candidates[0].content
        history.append(turn)

        function_calls = [p.function_call for p in turn.parts if p.function_call]
        if not function_calls:
            final_text = "".join(p.text for p in turn.parts if p.text)
            return ChatResponse(reply=final_text, trace=trace)

        response_parts = []
        for call in function_calls:
            impl = _TOOL_IMPL.get(call.name)
            if impl is None:
                result: dict = {"error": f"Unknown tool {call.name!r}"}
            else:
                try:
                    result = impl(dict(call.args or {}))
                except Exception as exc:
                    result = {"error": f"{exc.__class__.__name__}: {exc}"}
            trace.append({"tool": call.name, "input": call.args, "output": result})
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id,
                        name=call.name,
                        response=json.loads(json.dumps(result, default=str)),
                    )
                )
            )
        history.append(types.Content(role="user", parts=response_parts))

    return ChatResponse(
        reply=(
            "DORA stopped after too many tool calls without a final answer. "
            "Please retry, or narrow the request to one document/requirement pair."
        ),
        trace=trace,
    )


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
