"""Layer 4: LLM adjudication for clauses the invariant tests cannot settle.

Scope discipline is the point of this module. The model is *not* asked to find
findings, choose a governing document, or decide applicability - those are
deterministic and already resolved upstream. It answers one bounded question:

    given this reference invariant and this single governing clause, is the
    clause materially deviant or an equivalent restatement?

Consequences of keeping the scope this narrow:

* Prompts are small and fixed-shape, so results are reproducible.
* The model cannot invent evidence: it receives one clause and its verdict is
  attached to that clause's citation.
* A provider outage degrades to the deterministic verdict instead of failing.

Providers: Bedrock Converse (boto3), Anthropic API (ANTHROPIC_API_KEY), or the
null provider, which keeps the rule verdict and runs with no network calls.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .models import Clause, Package, Requirement, Verdict

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = """\
You are a contract-review adjudicator for transportation construction contract \
packages. You decide one narrow question and nothing else.

You are given:
  - a reference requirement and its governing invariant (the scoring authority);
  - ONE clause from a contract package that has already been confirmed as the \
governing text for that requirement after order-of-precedence and Addendum \
resolution.

Decide whether the clause MATERIALLY deviates from the reference invariant.

Rules you must follow:
1. Paraphrase, reordering, capitalisation, synonyms, and equivalent numeric \
forms (for example "one-tenth (10%)" and "ten percent (10%)", or "36 months" \
and "three years") are NOT deviations. Answer NO_FLAG.
2. A clause that defers to the reference ("within the reference period", "as \
stated in the applicable contract documents") preserves the requirement. \
Answer NO_FLAG.
3. Flag only when the clause changes a required number, removes a required \
document or protection, reverses a required order, inverts an applicability \
statement, or replaces a required workflow.
4. Quote evidence verbatim from the clause you were given. Never invent text \
and never cite a document you were not shown.
5. You are producing decision support for a human reviewer, not a legal \
conclusion.

Reply with JSON only, no prose and no code fence:
{"label":"FLAG|NO_FLAG","evidence":"<verbatim span from the clause>",
 "explanation":"<one or two sentences>","confidence":<0.00-1.00>}"""

USER_TEMPLATE = """\
REFERENCE REQUIREMENT
  id: {req_id}
  name: {req_name}
  authority: {req_location}
  review expectation: {expectation}
  governing invariant: {invariant}

GOVERNING CLAUSE FROM THE REVIEWED PACKAGE
  package: {package_id}
  document: {doc_type} ({file_name}, page {page})
  heading: {heading}
  text: {text}

PROJECT CONTEXT
{context}

Does this clause materially deviate from the governing invariant?"""


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        ...


@dataclass
class NullProvider:
    """No-op provider: adjudication is skipped and rule verdicts stand."""

    name: str = "null"

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("null provider: no model configured")


@dataclass
class BedrockProvider:
    """Amazon Bedrock Converse API."""

    model_id: str = DEFAULT_BEDROCK_MODEL
    region: str = os.environ.get("AWS_REGION", "us-east-1")
    max_tokens: int = 600
    name: str = "bedrock"

    def __post_init__(self):
        import boto3  # imported lazily so the rule-only path needs no boto3

        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )
        return "".join(
            block.get("text", "") for block in resp["output"]["message"]["content"]
        )


@dataclass
class AnthropicProvider:
    """Anthropic Messages API via ANTHROPIC_API_KEY."""

    model_id: str = DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = 600
    name: str = "anthropic"

    def __post_init__(self):
        from anthropic import Anthropic  # lazy import

        self._client = Anthropic()

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def build_provider(kind: str, model_id: Optional[str] = None) -> Provider:
    """Construct a provider, degrading to NullProvider on any failure."""
    kind = (kind or "null").lower()
    try:
        if kind == "bedrock":
            return BedrockProvider(model_id=model_id or DEFAULT_BEDROCK_MODEL)
        if kind == "anthropic":
            return AnthropicProvider(model_id=model_id or DEFAULT_ANTHROPIC_MODEL)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[llm] provider '{kind}' unavailable ({exc}); falling back to rules only.")
    return NullProvider()


# ---------------------------------------------------------------------------
# Adjudicator
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class Adjudicator:
    """Escalation target for uncertain rule verdicts."""

    def __init__(self, provider: Provider):
        self.provider = provider
        self.calls = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return not isinstance(self.provider, NullProvider)

    def _context(self, pkg: Package) -> str:
        return "\n".join(
            [
                f"  federal_aid: {pkg.metadata.get('federal_aid')}",
                f"  buy_america_baba_applicable: "
                f"{pkg.metadata.get('buy_america_baba_applicable')}",
                f"  assumed_contract_value: {pkg.metadata.get('assumed_contract_value')}",
                f"  issued_addenda: {', '.join(pkg.issued_addenda) or 'none'}",
                f"  subcontracting_planned: {pkg.metadata.get('subcontracting_planned')}",
                f"  claim_event: {pkg.metadata.get('claim_event')}",
                f"  delay_event: {pkg.metadata.get('delay_event')}",
                f"  changed_work_event: {pkg.metadata.get('changed_work_event')}",
            ]
        )

    def adjudicate(
        self,
        requirement: Requirement,
        clause: Clause,
        pkg: Package,
        fallback: Verdict,
    ) -> Verdict:
        """Return an LLM verdict, or `fallback` if unavailable/unparseable."""
        if not self.enabled:
            return fallback

        user = USER_TEMPLATE.format(
            req_id=requirement.requirement_id,
            req_name=requirement.requirement_name,
            req_location=requirement.reference_location,
            expectation=requirement.review_expectation,
            invariant=requirement.challenge_reference_rule,
            package_id=pkg.package_id,
            doc_type=clause.doc_type,
            file_name=clause.file_name,
            page=clause.page,
            heading=clause.heading,
            text=clause.text,
            context=self._context(pkg),
        )

        try:
            self.calls += 1
            raw = self.provider.complete(SYSTEM_PROMPT, user)
            match = _JSON_BLOCK.search(raw)
            if not match:
                raise ValueError(f"no JSON object in response: {raw[:200]!r}")
            data = json.loads(match.group(0))

            label = str(data.get("label", "")).strip().upper()
            if label not in {"FLAG", "NO_FLAG"}:
                raise ValueError(f"invalid label {label!r}")

            evidence = str(data.get("evidence", "")).strip()
            # Reject evidence that is not actually in the clause.
            if evidence and _norm(evidence) not in _norm(clause.text):
                evidence = fallback.evidence or clause.text

            confidence = float(data.get("confidence", 0.7))
            confidence = min(max(confidence, 0.0), 1.0)

            return Verdict(
                label=label,
                explanation=str(data.get("explanation", "")).strip()
                or fallback.explanation,
                confidence=confidence,
                rule_id=f"{fallback.rule_id}+llm",
                evidence=evidence or clause.text,
                uncertain=False,
            )
        except Exception as exc:
            self.failures += 1
            print(f"[llm] adjudication failed for {requirement.requirement_id} "
                  f"/{pkg.package_id}: {exc}")
            return fallback

    def stats(self) -> dict:
        return {
            "provider": self.provider.name,
            "enabled": self.enabled,
            "calls": self.calls,
            "failures": self.failures,
        }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
