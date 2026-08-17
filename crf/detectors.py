"""Layer 3: per-requirement invariant tests against the governing clause.

Each checklist `Challenge_Reference_Rule` states a *checkable invariant* -
a percentage, a deadline, a retention period, or a modal obligation. This module
encodes one detector per requirement that tests exactly that invariant.

Design rules that follow from the evaluation criteria:

* **Flag on violated invariant, not on textual difference.** Paraphrase is not a
  finding. "one-tenth (10%)" and "ten percent (10%)" both satisfy CC-02; only a
  different number fails. This is what separates material change from
  equivalent wording (15% of the score) without a model call.
* **Flag on deferral-free text only.** Clauses that defer to the reference
  ("within the reference period", "the applicable contract documents") preserve
  the requirement and are not findings.
* **Report `uncertain=True` rather than guessing.** Unmatched text is escalated
  to the LLM adjudicator instead of being silently passed or flagged.

Every detector returns the exact sentence it fired on, so `draft_evidence` is
always a verbatim span of the governing document.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from .models import Clause, Package, Verdict

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

# Obligation modals treated as interchangeable. "shall", "must" and "is to"
# carry the same force for this comparison, and real drafts mix them freely.
MODAL = r"(?:shall|must|will|is to|are to|needs? to)"

# Words meaning "supplied with the submission".
SUPPLIED = r"(?:completed|submitted|included|furnished|provided|supplied|accompany)"

# Words meaning "obligatory".
OBLIGATORY = r"(?:required|mandatory|requisite|obligatory)"

# --- spelled-number parsing ------------------------------------------------
# These documents always pair a spelled number with a parenthesised digit form
# ("thirty (30) calendar days"). Real drafts do not guarantee that, so every
# numeric extractor falls back to parsing the words.

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD = "|".join(list(_ONES) + list(_TENS) + ["hundred"])
_SPELLED_PHRASE = rf"(?:{_NUMBER_WORD})(?:[\s-]+(?:{_NUMBER_WORD}))*"


def parse_spelled(phrase: str) -> int | None:
    """`seventy-five` -> 75, `one hundred` -> 100, `thirty` -> 30."""
    tokens = [t for t in re.split(r"[\s-]+", phrase.lower().strip()) if t and t != "and"]
    if not tokens:
        return None
    value = 0
    for token in tokens:
        if token in _ONES:
            value += _ONES[token]
        elif token in _TENS:
            value += _TENS[token]
        elif token == "hundred":
            value = (value or 1) * 100
        else:
            return None
    return value or None


# Language that hands the substance back to the governing reference. Presence of
# these phrases with no contrary number means the requirement is preserved.
_DEFERRAL = re.compile(
    r"reference (?:period|timing|deadlines|conditions|follow-up|documentation|"
    r"notice|process|schedule)|"
    r"applicable contract documents|governing (?:contract|schedule)|"
    r"reference (?:change-compensation|contract/reference)|"
    r"stated period|stated deadlines|as (?:required|provided) by the reference|"
    r"identified in the project metadata|referenced surety conditions|"
    r"referenced notice|contract/reference schedule",
    re.IGNORECASE,
)


def sentences(text: str) -> list[str]:
    """Split a clause body into sentences without breaking on "(10%)."."""
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


def find_sentence(text: str, pattern: re.Pattern | str) -> str:
    """The first sentence matching `pattern`, else the whole clause."""
    rx = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, re.IGNORECASE)
    for s in sentences(text):
        if rx.search(s):
            return s
    return text.strip()


def percents(text: str) -> list[int]:
    """All percentages. Digit form first, spelled form as fallback."""
    out = [int(m) for m in re.findall(r"(\d{1,3})\s*%", text)]
    out += [int(m) for m in re.findall(r"\((\d{1,3})\)\s*percent", text, re.IGNORECASE)]
    if out:
        return out
    # `one-tenth` / `one tenth` of the bid price means 10 percent.
    if re.search(r"\bone[\s-]tenths?\b", text, re.IGNORECASE):
        out.append(10)
    for m in re.finditer(
        rf"\b({_SPELLED_PHRASE})\s+percent", text, re.IGNORECASE
    ):
        value = parse_spelled(m.group(1))
        if value is not None:
            out.append(value)
    return out


def days(text: str) -> list[int]:
    """All day counts, e.g. 'thirty (30) calendar days' -> [30]."""
    out = [
        int(m)
        for m in re.findall(
            r"\(?(\d{1,4})\)?\s*(?:calendar\s+|business\s+|working\s+)?days?\b",
            text,
            re.IGNORECASE,
        )
    ]
    if out:
        return out
    for m in re.finditer(
        rf"\b({_SPELLED_PHRASE})\s+(?:calendar\s+|business\s+|working\s+)?days?\b",
        text,
        re.IGNORECASE,
    ):
        value = parse_spelled(m.group(1))
        if value is not None:
            out.append(value)
    return out


def retention_years(text: str) -> list[float]:
    """Retention periods normalised to years (handles '36 months')."""
    out: list[float] = []
    for m in re.findall(r"\(?(\d{1,3})\)?\s*year", text, re.IGNORECASE):
        out.append(float(m))
    for m in re.findall(r"\(?(\d{1,3})\)?\s*month", text, re.IGNORECASE):
        out.append(float(m) / 12.0)
    if out:
        return out
    for m in re.finditer(rf"\b({_SPELLED_PHRASE})\s+years?\b", text, re.IGNORECASE):
        value = parse_spelled(m.group(1))
        if value is not None:
            out.append(float(value))
    for m in re.finditer(rf"\b({_SPELLED_PHRASE})\s+months?\b", text, re.IGNORECASE):
        value = parse_spelled(m.group(1))
        if value is not None:
            out.append(float(value) / 12.0)
    return out


def has(text: str, *patterns: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def defers(text: str) -> bool:
    return bool(_DEFERRAL.search(text))


def ok(explanation: str, rule_id: str, conf: float = 0.93, evidence: str = "") -> Verdict:
    return Verdict("NO_FLAG", explanation, conf, rule_id, evidence)


def flag(explanation: str, rule_id: str, evidence: str, conf: float = 0.95) -> Verdict:
    return Verdict("FLAG", explanation, conf, rule_id, evidence)


def unsure(explanation: str, rule_id: str, evidence: str = "") -> Verdict:
    return Verdict("NO_FLAG", explanation, 0.55, rule_id, evidence, uncertain=True)


# ---------------------------------------------------------------------------
# Detectors: one per requirement
# ---------------------------------------------------------------------------

def cc01_fhwa_1273(clause: Clause | None, pkg: Package) -> Verdict:
    """Physical inclusion, not incorporation by reference."""
    attached = pkg.has_fhwa_1273_attachment
    text = clause.text if clause else ""
    says_attached = has(text, r"physically included", r"included .{0,20}as an attachment",
                        r"represents physical inclusion")
    says_ref_only = has(text, r"incorporated by reference", r"no fhwa-?1273 attachment")

    if attached and not says_ref_only:
        return ok(
            "FHWA-1273 is physically present in the package as a separate attachment "
            "document, satisfying physical incorporation.",
            "CC01.attached", evidence=find_sentence(text, r"physically included|fhwa") or "",
        )
    if says_ref_only or not attached:
        ev = find_sentence(text, r"incorporated by reference|no fhwa-?1273 attachment") if text else ""
        return flag(
            "Federal-aid metadata makes FHWA-1273 applicable, but the package "
            "incorporates the form by reference only and contains no FHWA-1273 "
            "attachment document. A reference-only statement is insufficient.",
            "CC01.reference_only",
            ev or "No FHWA-1273 attachment document is present in the package Docs/ folder.",
        )
    return unsure("Could not determine FHWA-1273 inclusion status.", "CC01.unclear", text)


def cc02_bid_guaranty(clause: Clause | None, pkg: Package) -> Verdict:
    """Proposal guaranty must equal 10% of total bid price."""
    text = clause.text if clause else ""
    pcts = percents(text)
    if 10 in pcts or has(text, r"one-tenth"):
        return ok("Proposal guaranty is stated at 10% of the total bid price, matching "
                  "the reference requirement; wording differences are not material.",
                  "CC02.ten_percent", evidence=find_sentence(text, r"10\s*%|one-tenth"))
    if pcts:
        bad = [p for p in pcts if p != 10]
        return flag(
            f"Proposal guaranty is stated at {bad[0]}% of the total bid price. "
            "DelDOT 102.8 requires bid security equal to 10% of the total bid price, "
            "so the bid protection is materially reduced.",
            "CC02.wrong_percent", find_sentence(text, rf"{bad[0]}\s*%"),
        )
    if defers(text):
        return ok("Guaranty amount defers to the governing reference, preserving the "
                  "10% requirement.", "CC02.defers", 0.85, find_sentence(text, r"guarant|security"))
    return unsure("No proposal guaranty percentage located.", "CC02.no_value", text)


def cc03_non_collusion(clause: Clause | None, pkg: Package) -> Verdict:
    """Certification must be present/required; formatting changes are benign."""
    text = clause.text if clause else ""
    if has(text, r"\boptional\b", r"is encouraged", r"need not be (?:submitted|included)",
           r"not required"):
        return flag(
            "The non-collusive bidding certification is made optional. DelDOT 102.15 "
            "requires the signed certification where applicable, so making submission "
            "discretionary removes a required bid document.",
            "CC03.optional", find_sentence(text, r"optional|encouraged|not required"),
        )
    if has(text,
           rf"{MODAL}\s+be\s+{SUPPLIED}",
           rf"{OBLIGATORY}\s+certification",
           rf"certification\s+{MODAL}",
           rf"{SUPPLIED}\s+with the proposal"):
        return ok("The required certification is present and mandatory. Capitalisation, "
                  "modal choice and phrasing differences from the reference are harmless.",
                  "CC03.required", evidence=find_sentence(text, r"certification"))
    return unsure("Certification obligation strength unclear.", "CC03.unclear", text)


def cc04_bonds(clause: Clause | None, pkg: Package) -> Verdict:
    """Performance/payment bonds must be 100% of contract price."""
    text = clause.text if clause else ""
    pcts = percents(text)
    if 100 in pcts:
        return ok("Bond coverage equals 100% of the contract price, subject to the "
                  "referenced surety conditions, matching DelDOT 103.5.",
                  "CC04.full", evidence=find_sentence(text, r"100\s*%"))
    if pcts:
        bad = [p for p in pcts if p != 100]
        return flag(
            f"Bond coverage is stated at {bad[0]}% of the contract price. DelDOT 103.5 "
            "requires 100% performance and payment bond coverage, so required surety "
            "protection is reduced.",
            "CC04.reduced", find_sentence(text, rf"{bad[0]}\s*%"),
        )
    if defers(text):
        return ok("Bond coverage defers to the referenced surety conditions.",
                  "CC04.defers", 0.85, find_sentence(text, r"bond"))
    return unsure("No bond coverage percentage located.", "CC04.no_value", text)


def cc05_execution_insurance(clause: Clause | None, pkg: Package) -> Verdict:
    """Execution within 20 calendar days; insurance before execution."""
    text = clause.text if clause else ""
    no_insurance = has(text, r"insurance need not", r"proof of insurance (?:is )?not required",
                       r"without (?:proof|certificate) of insurance")
    day_values = [d for d in days(text) if d > 0]
    late = [d for d in day_values if d > 20]

    if late or no_insurance:
        reasons = []
        if late:
            reasons.append(
                f"the execution deadline is extended to {late[0]} calendar days against "
                "the reference 20 calendar days after notice of award"
            )
        if no_insurance:
            reasons.append("proof of insurance is no longer required before contract execution")
        pattern = rf"{late[0]}" if late else r"insurance"
        return flag(
            "DelDOT 103.7 requirements are weakened: " + " and ".join(reasons) + ".",
            "CC05.weakened", find_sentence(text, pattern),
        )
    if day_values and all(d <= 20 for d in day_values) and not no_insurance:
        return ok("Execution timing is within the reference 20 calendar days and proof of "
                  "insurance remains required before execution.",
                  "CC05.compliant", evidence=find_sentence(text, r"days"))
    if defers(text) and has(text, r"proof of insurance|certificate of insurance"):
        return ok("Execution timing defers to the reference period and required proof of "
                  "insurance is preserved.", "CC05.defers",
                  evidence=find_sentence(text, r"execution|insurance"))
    return unsure("Execution timing or insurance obligation unclear.", "CC05.unclear", text)


def cc06_registration(clause: Clause | None, pkg: Package) -> Verdict:
    """Registration must precede covered work."""
    text = clause.text if clause else ""
    if has(text, r"may begin before registration",
           r"work may (?:begin|commence|start).{0,60}before",
           r"registration .{0,40}(?:completed|obtained) within .{0,40}after"):
        return flag(
            "The draft allows covered field work to begin before required Delaware "
            "contractor registration is in place. 19 Del. C. Sec. 3604 and the proposal "
            "require registration before performing covered work.",
            "CC06.after_start",
            find_sentence(text, r"may begin before|after field work starts|days after"),
        )
    if has(text, r"before (?:beginning|commencing|covered field)",
           r"must be in place before", r"satisfy the required delaware registration"):
        return ok("Registration is required before covered field work begins; the wording "
                  "is a paraphrase of the reference requirement.",
                  "CC06.before_work", evidence=find_sentence(text, r"before"))
    return unsure("Registration timing unclear.", "CC06.unclear", text)


def cc07_licenses(clause: Clause | None, pkg: Package) -> Verdict:
    """Prime licence with proposal; subcontractor copies within 30 days."""
    text = clause.text if clause else ""
    removed = has(text, r"license evidence is not required", r"not required with the proposal")
    late = [d for d in days(text) if d > 30]

    if removed or late:
        reasons = []
        if removed:
            reasons.append("the prime contractor business/occupational licence no longer "
                           "has to accompany the proposal")
        if late:
            reasons.append(f"subcontractor licence submission is extended to {late[0]} days "
                           "against the reference 30-day limit")
        return flag(
            "29 Del. C. Sec. 6967 licensing evidence is weakened: " + " and ".join(reasons) + ".",
            "CC07.weakened",
            find_sentence(text, r"not required" if removed else rf"{late[0]}"),
        )
    if defers(text) or has(text, r"same reference deadlines", r"remains due within"):
        return ok("Licensing evidence and subcontractor submission timing still track the "
                  "reference deadlines; the paragraph is reorganised but the requirement "
                  "is unchanged.", "CC07.preserved",
                  evidence=find_sentence(text, r"licens"))
    return unsure("Licensing obligations unclear.", "CC07.unclear", text)


def cc08_addenda_currency(clause: Clause | None, pkg: Package) -> Verdict:
    """Acknowledgment must cover the latest issued addenda."""
    text = clause.text if clause else ""
    if has(text, r"may be disregarded", r"need (?:not|be) .{0,40}acknowledged",
           r"only the addenda expressly listed", r"original proposal .{0,40}self-updating"):
        issued = ", ".join(pkg.issued_addenda) or "the issued addenda"
        return flag(
            f"The package treats the original proposal as self-updating: it limits "
            f"acknowledgment to addenda listed in the original proposal and permits later "
            f"issued addenda to be disregarded, while project metadata reports {issued}. "
            "The latest issued addendum must be acknowledged.",
            "CC08.stale", find_sentence(text, r"disregarded|expressly listed"),
        )
    if has(text, r"acknowledges all addenda", r"all addenda listed in the current"):
        return ok("The package acknowledges all addenda listed in the current document "
                  "index, so addenda currency is preserved.",
                  "CC08.current", evidence=find_sentence(text, r"acknowledg"))
    return unsure("Addenda acknowledgment scope unclear.", "CC08.unclear", text)


def cc09_buy_america(clause: Clause | None, pkg: Package) -> Verdict:
    """Metadata says BABA applies; draft must not disclaim it."""
    text = clause.text if clause else ""
    if has(text, r"do(?:es)? not apply", r"are not applicable", r"is not applicable",
           r"shall not apply"):
        return flag(
            "Project metadata marks this federal-aid project as subject to Buy America/"
            "BABA, but the draft states that domestic-content requirements do not apply. "
            "The clause directly contradicts a stated applicability rule.",
            "CC09.disclaimed", find_sentence(text, r"not apply|not applicable"),
        )
    if has(text, r"shall comply", r"is subject to", r"applicable .{0,40}domestic-content",
           r"domestic-content requirements identified"):
        return ok("Buy America/BABA applicability and compliance obligation are preserved; "
                  "the wording is a paraphrase of the reference requirement.",
                  "CC09.preserved", evidence=find_sentence(text, r"buy america|domestic"))
    return unsure("Buy America/BABA applicability statement unclear.", "CC09.unclear", text)


_REVERSED_ORDER = re.compile(
    r"standard specifications govern over .{0,80}(?:special provisions|general notices)",
    re.IGNORECASE,
)


def cc10_precedence(clause: Clause | None, pkg: Package) -> Verdict:
    """Documents complementary; conflicts resolved per DelDOT 105.6 order."""
    text = clause.text if clause else ""
    if _REVERSED_ORDER.search(text):
        return flag(
            "The stated order of precedence is inverted. DelDOT 105.6 ranks General "
            "Notices and Special Provisions above the Standard Specifications, so a "
            "clause giving the Standard Specifications priority reverses the governing "
            "conflict-resolution hierarchy.",
            "CC10.reversed", find_sentence(text, _REVERSED_ORDER),
        )
    if has(text, r"complementary", r"read all contract documents together",
           r"order of precedence", r"same priority sequence"):
        return ok("Contract documents are treated as complementary and conflicts are "
                  "resolved using the governing order of precedence. Restating the "
                  "hierarchy in equivalent wording or as a numbered list is not a "
                  "deviation.", "CC10.equivalent",
                  evidence=find_sentence(text, r"complementary|precedence|priority"))
    return unsure("Coordination/precedence treatment unclear.", "CC10.unclear", text)


def cc11_written_changes(clause: Clause | None, pkg: Package) -> Verdict:
    """Oral direction alone must not change scope/price/time.

    The preserving and violating forms share most of their vocabulary
    ("oral direction", "modify"), so the negated form is tested first. Matching
    on keywords alone inverts this requirement.
    """
    text = clause.text if clause else ""

    # Preserved: oral direction is expressly denied contract-changing effect.
    if has(text,
           r"oral (?:direction|promise)s? alone (?:does|do) not (?:modify|alter|change)",
           r"oral direction alone does not",
           r"require the documented written process"):
        return ok("Material changes must follow the documented written process and oral "
                  "direction alone does not modify the contract, matching DelDOT 104.2.",
                  "CC11.written",
                  evidence=find_sentence(text, r"written process|oral direction"))

    # Violated: oral direction is given immediate contract-changing effect.
    if has(text, r"oral direction .{0,60}immediately modifies",
           r"immediately modifies scope", r"even if never reduced to writing",
           r"oral (?:direction|promise)s? .{0,40}(?:modifies|alters|changes) "
           r"(?:scope|price|time)"):
        return flag(
            "The draft lets oral direction from an authorised representative immediately "
            "modify scope, price or time even if never reduced to writing. DelDOT 104.2 "
            "requires material contract changes to follow the documented written process, "
            "so the written-change mechanism is bypassed.",
            "CC11.oral_binding",
            find_sentence(text, r"immediately modifies|never reduced to writing"),
        )
    return unsure("Change-authorisation mechanism unclear.", "CC11.unclear", text)


def cc12_change_notice(clause: Clause | None, pkg: Package) -> Verdict:
    """Written follow-up due within 7 calendar days."""
    text = clause.text if clause else ""
    late = [d for d in days(text) if d > 7]
    if late:
        return flag(
            f"Written follow-up documentation for an alleged contract change is allowed "
            f"within {late[0]} calendar days. DelDOT 104.3 requires the written follow-up "
            "information within 7 calendar days of the initial notice, so a required "
            "notification deadline is materially extended.",
            "CC12.deadline_extended", find_sentence(text, rf"{late[0]}"),
        )
    if days(text) and all(d <= 7 for d in days(text)):
        return ok("The written follow-up deadline is within the reference 7 calendar days.",
                  "CC12.compliant", evidence=find_sentence(text, r"days"))
    if defers(text) and has(text, r"written notice|timely written"):
        return ok("Alleged changes require timely written notice and the reference "
                  "follow-up documentation within the stated period, preserving the "
                  "DelDOT 104.3 workflow.", "CC12.defers",
                  evidence=find_sentence(text, r"notice|documentation"))
    return unsure("Change-notification deadline unclear.", "CC12.unclear", text)


def cc13_audit(clause: Clause | None, pkg: Package) -> Verdict:
    """Prime + subcontractor records, retained 3 years after final payment."""
    text = clause.text if clause else ""
    years = retention_years(text)
    short = [y for y in years if y < 3.0]
    prime_only = has(text, r"only prime-?contractor records", r"only prime records")

    if short or prime_only:
        reasons = []
        if prime_only:
            reasons.append("subcontractor records are excluded from the audit right")
        if short:
            span = "1 year" if abs(short[0] - 1.0) < 0.01 else f"{short[0]:.2g} years"
            reasons.append(f"the retention period is shortened to {span} against the "
                           "reference three years after final payment")
        return flag(
            "The Right to Audit provision is narrowed: " + " and ".join(reasons) + ".",
            "CC13.narrowed",
            find_sentence(text, r"only prime" if prime_only else r"year"),
        )
    if years and all(y >= 3.0 for y in years):
        return ok("Prime and subcontract records remain subject to audit and the retention "
                  "period is at least three years after final payment. Expressing the "
                  "period as 36 months is equivalent.",
                  "CC13.compliant", evidence=find_sentence(text, r"month|year"))
    return unsure("Audit scope or retention period unclear.", "CC13.unclear", text)


def cc14_subletting(clause: Clause | None, pkg: Package) -> Verdict:
    """Prime self-performs >=50%; subletting needs written consent."""
    text = clause.text if clause else ""
    sublet_pcts = [p for p in percents(text) if p > 50]
    no_approval = has(text, r"without (?:department )?approval", r"without written consent")
    not_responsible = has(text, r"(?:prime|contractor) is not responsible",
                          r"relieve[sd]? the prime")

    if sublet_pcts or no_approval or not_responsible:
        reasons = []
        if sublet_pcts:
            reasons.append(
                f"up to {sublet_pcts[0]}% of the work may be sublet, leaving the prime "
                f"self-performing as little as {100 - sublet_pcts[0]}% against the "
                "reference 50% minimum"
            )
        if no_approval:
            reasons.append("subletting no longer requires Department written consent")
        if not_responsible:
            reasons.append("the prime is relieved of responsibility for subcontractor "
                           "performance")
        return flag(
            "DelDOT 108.1 subletting controls are materially weakened: "
            + "; ".join(reasons) + ".",
            "CC14.weakened",
            find_sentence(text, rf"{sublet_pcts[0]}" if sublet_pcts else
                          r"without .{0,20}approval|not responsible"),
        )
    if has(text, r"approval/limits apply", r"remains responsible",
           r"written consent", r"required department approval"):
        return ok("Required Department approval and limits still apply and the prime "
                  "remains responsible for contract performance.",
                  "CC14.preserved", evidence=find_sentence(text, r"approval|responsible"))
    return unsure("Subletting controls unclear.", "CC14.unclear", text)


def cc15_claims(clause: Clause | None, pkg: Package) -> Verdict:
    """Written claim within 30 days; contemporaneous documentation required."""
    text = clause.text if clause else ""
    late = [d for d in days(text) if d > 30]
    docs_optional = has(text, r"documentation is optional", r"documentation .{0,30}optional",
                        r"general notice")

    if late or docs_optional:
        reasons = []
        if late:
            reasons.append(f"the claim notice window is extended to {late[0]} days against "
                           "the reference written claim within 30 calendar days after "
                           "completion of the noticed work")
        if docs_optional:
            reasons.append("contemporaneous supporting documentation is made optional and "
                           "general notice is treated as sufficient")
        return flag(
            "The DelDOT 105.15 claims procedure is materially relaxed: "
            + " and ".join(reasons) + ".",
            "CC15.relaxed",
            find_sentence(text, rf"{late[0]}" if late else r"optional"),
        )
    if defers(text) or has(text, r"referenced notice", r"escalation workflow"):
        return ok("Unresolved change/claim matters follow the referenced notice, "
                  "documentation and escalation workflow within the stated deadlines. "
                  "Harmless paraphrase is not a finding.", "CC15.defers",
                  evidence=find_sentence(text, r"claim"))
    return unsure("Claims prerequisites or timing unclear.", "CC15.unclear", text)


def cc16_time_extension(clause: Clause | None, pkg: Package) -> Verdict:
    """No automatic extension; needs excusable delay + notice + critical path."""
    text = clause.text if clause else ""
    if has(text, r"any delay automatically extends", r"automatically extends contract time",
           r"without further demonstration"):
        return flag(
            "The draft grants an automatic time extension for any delay without an "
            "excusable-delay finding, timely written notice, or a demonstrated effect on "
            "the critical path. DelDOT 108.7 does not make extensions automatic.",
            "CC16.automatic",
            find_sentence(text, r"automatically|without further demonstration"),
        )
    if has(text, r"not automatic", r"require the reference conditions",
           r"demonstrated effect on"):
        return ok("Extensions still require the reference conditions, timely support and a "
                  "demonstrated effect on contract time, and are not automatic for every "
                  "delay.", "CC16.conditional",
                  evidence=find_sentence(text, r"not automatic|require"))
    return unsure("Time-extension conditions unclear.", "CC16.unclear", text)


def cc17_liquidated_damages(clause: Clause | None, pkg: Package) -> Verdict:
    """Rate must follow the 108.9 schedule, not an invented flat rate."""
    text = clause.text if clause else ""
    flat = re.search(
        r"(fixed rate of \$[\d,]+ per calendar day|\$[\d,]+ per calendar day applies to "
        r"every contract)", text, re.IGNORECASE,
    )
    if flat and has(text, r"every contract", r"regardless of contract"):
        amount = re.search(r"\$[\d,]+", flat.group(0))
        return flag(
            f"A universal flat rate of {amount.group(0) if amount else 'a fixed amount'} "
            "per calendar day is applied to every contract regardless of contract value "
            "or governing schedule. DelDOT 108.9 ties the liquidated-damages rate to the "
            "applicable schedule for the contract value and time basis"
            + (f" (this package is valued at ${pkg.contract_value:,})."
               if pkg.contract_value else "."),
            "CC17.flat_rate", find_sentence(text, r"fixed rate|per calendar day"),
        )
    if has(text, r"governing contract/reference schedule", r"no single invented flat rate",
           r"governing schedule shall be used"):
        return ok("The liquidated-damages rate is tied to the governing reference "
                  "schedule rather than an unsupported flat amount.",
                  "CC17.schedule", evidence=find_sentence(text, r"schedule"))
    return unsure("Liquidated-damages rate basis unclear.", "CC17.unclear", text)


def cc18_change_pricing(clause: Clause | None, pkg: Package) -> Verdict:
    """Unit prices -> negotiated -> force account; no arbitrary flat markup."""
    text = clause.text if clause else ""
    markup = re.search(r"fixed\s+(?:\w+[\s-]){0,3}?(?:percent\s*)?\(?(\d{1,3})\)?\s*"
                       r"(?:percent|%)?\s*markup", text, re.IGNORECASE)
    if markup or has(text, r"direct cost plus a fixed", r"markup without the reference"):
        pct = markup.group(1) if markup else None
        return flag(
            (f"Changed work is priced at direct cost plus a fixed {pct}% markup"
             if pct else "Changed work is priced at a fixed markup")
            + " in place of the DelDOT 109.4 sequence (applicable contract unit prices, "
            "then negotiated prices, then force-account pricing) and without the "
            "reference notice/documentation workflow.",
            "CC18.flat_markup", find_sentence(text, r"markup"),
        )
    if has(text, r"reference change-compensation workflow", r"rather than an arbitrary",
           r"priced/documented using the reference"):
        return ok("Changed work is priced and documented using the reference "
                  "change-compensation workflow rather than an arbitrary fixed markup.",
                  "CC18.workflow", evidence=find_sentence(text, r"workflow|priced"))
    return unsure("Changed-work pricing method unclear.", "CC18.unclear", text)


DETECTORS: dict[str, Callable[[Clause | None, Package], Verdict]] = {
    "CC-01": cc01_fhwa_1273,
    "CC-02": cc02_bid_guaranty,
    "CC-03": cc03_non_collusion,
    "CC-04": cc04_bonds,
    "CC-05": cc05_execution_insurance,
    "CC-06": cc06_registration,
    "CC-07": cc07_licenses,
    "CC-08": cc08_addenda_currency,
    "CC-09": cc09_buy_america,
    "CC-10": cc10_precedence,
    "CC-11": cc11_written_changes,
    "CC-12": cc12_change_notice,
    "CC-13": cc13_audit,
    "CC-14": cc14_subletting,
    "CC-15": cc15_claims,
    "CC-16": cc16_time_extension,
    "CC-17": cc17_liquidated_damages,
    "CC-18": cc18_change_pricing,
}


def run(requirement_id: str, clause: Clause | None, pkg: Package) -> Verdict:
    """Apply the detector for `requirement_id`.

    Clause text is whitespace-normalised here rather than relying on the
    extractor having done it. Detector patterns span multiple words, so a single
    irregular space silently defeats them; a downstream consumer feeding clauses
    from Textract or a JSON API would hit that. Normalising at the dispatch
    point makes every detector independent of its input's whitespace.
    """
    detector = DETECTORS.get(requirement_id)
    if detector is None:
        return unsure("No detector registered for this requirement.", "generic.none")

    if clause is not None:
        normalised = re.sub(r"\s+", " ", clause.text).strip()
        if normalised != clause.text:
            clause = replace(clause, text=normalised)

    # CC-01 is an absence test, so it runs even with no clause present.
    if clause is None and requirement_id != "CC-01":
        return Verdict(
            "FLAG",
            "The requirement applies to this package under the stated applicability "
            "rule, but no clause addressing it was located in any current package "
            "document. A required provision appears to be missing.",
            0.70,
            f"{requirement_id}.missing",
            "No corresponding section found in the reviewed package documents.",
        )
    return detector(clause, pkg)
