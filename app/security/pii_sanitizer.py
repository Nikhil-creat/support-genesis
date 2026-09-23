"""Dual-layer PII scrubber: deterministic regex pass + spaCy NER pass.

Regex handles high-confidence structured PII (SSNs, credit cards, API keys,
emails, phone numbers). spaCy NER catches unstructured PII (person names,
locations, orgs) that regex cannot reliably detect. Both passes run on every
inbound message before it touches the LLM, and on every tool payload before
it is persisted to the audit trail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

try:
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - spaCy is an optional heavy dep
    spacy = None
    Language = None


_REGEX_RULES: dict[str, re.Pattern] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "API_KEY": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b"),
    "AWS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

_NER_ENTITY_LABELS = {"PERSON", "GPE", "LOC", "ORG", "CARDINAL"}
_NER_LABEL_TO_TAG = {
    "PERSON": "PERSON_NAME",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "CARDINAL": "NUMBER",
}


@dataclass
class SanitizationResult:
    original: str
    sanitized: str
    redactions: list[dict[str, str]] = field(default_factory=list)

    @property
    def had_pii(self) -> bool:
        return len(self.redactions) > 0


@lru_cache(maxsize=1)
def _load_nlp() -> "Language | None":
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Model not downloaded in this environment; regex-only mode.
        return None


class PIISanitizer:
    """Scrubs PII from free text using regex first, then NER."""

    def __init__(self, enable_ner: bool = True, luhn_validate_cc: bool = True) -> None:
        self.enable_ner = enable_ner
        self.luhn_validate_cc = luhn_validate_cc
        self._nlp = _load_nlp() if enable_ner else None

    @staticmethod
    def _luhn_check(candidate: str) -> bool:
        digits = [int(c) for c in re.sub(r"\D", "", candidate)]
        if len(digits) < 13:
            return False
        checksum = 0
        parity = len(digits) % 2
        for i, digit in enumerate(digits):
            if i % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0

    def _regex_pass(self, text: str) -> tuple[str, list[dict[str, str]]]:
        redactions: list[dict[str, str]] = []
        result = text
        for tag, pattern in _REGEX_RULES.items():
            def _replace(match: re.Match, tag: str = tag) -> str:
                value = match.group(0)
                if tag == "CREDIT_CARD" and self.luhn_validate_cc and not self._luhn_check(value):
                    return value  # not a valid card number, leave untouched
                redactions.append({"tag": tag, "original_span": value})
                return f"[REDACTED_{tag}]"

            result = pattern.sub(_replace, result)
        return result, redactions

    def _ner_pass(self, text: str) -> tuple[str, list[dict[str, str]]]:
        if self._nlp is None:
            return text, []
        doc = self._nlp(text)
        redactions: list[dict[str, str]] = []
        result = text
        # Replace longest spans first to avoid offset corruption.
        entities = sorted(
            (ent for ent in doc.ents if ent.label_ in _NER_ENTITY_LABELS),
            key=lambda e: len(e.text),
            reverse=True,
        )
        for ent in entities:
            tag = _NER_LABEL_TO_TAG.get(ent.label_, "ENTITY")
            if ent.text in result:
                redactions.append({"tag": tag, "original_span": ent.text})
                result = result.replace(ent.text, f"[REDACTED_{tag}]")
        return result, redactions

    def sanitize(self, text: str) -> SanitizationResult:
        regex_clean, regex_redactions = self._regex_pass(text)
        final_clean, ner_redactions = (
            self._ner_pass(regex_clean) if self.enable_ner else (regex_clean, [])
        )
        return SanitizationResult(
            original=text,
            sanitized=final_clean,
            redactions=regex_redactions + ner_redactions,
        )

    def sanitize_payload(self, payload: dict) -> dict:
        """Recursively sanitize string values in a nested dict (for audit logs)."""
        clean: dict = {}
        for key, value in payload.items():
            if isinstance(value, str):
                clean[key] = self.sanitize(value).sanitized
            elif isinstance(value, dict):
                clean[key] = self.sanitize_payload(value)
            elif isinstance(value, list):
                clean[key] = [
                    self.sanitize(v).sanitized if isinstance(v, str) else v for v in value
                ]
            else:
                clean[key] = value
        return clean
