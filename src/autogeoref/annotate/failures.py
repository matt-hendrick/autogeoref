"""Annotation failure taxonomy and shared failure classification."""

from __future__ import annotations

import re


class AnnotateError(Exception):
    """Base class for all annotator errors."""


class ModelQualityError(AnnotateError):
    """Configured model fails the Sonnet-class-minimum quality gate."""


class AnnotationCallError(AnnotateError):
    """Base class for a failed annotation call."""


class EmptyResponseError(AnnotationCallError):
    """The model returned no output at all."""


class BudgetLimitError(AnnotationCallError):
    """The model/provider reported a budget, quota, or usage-limit condition."""


class TransientRateLimitError(AnnotationCallError):
    """The provider rate-limited the request."""


class AnnotationTimeoutError(AnnotationCallError):
    """The call exceeded its timeout."""


class MalformedResponseError(AnnotationCallError):
    """The model responded, but the output is not valid annotation JSON."""


class AnnotatorProcessError(AnnotationCallError):
    """The annotator subprocess failed without usable output."""


#: ``_S`` is the separator between the words of a limit message: prose says
#: "usage limit", a JSON error code says ``usage_limit_reached``, and both mean
#: the same thing. Matching only the spaced form lets the machine-readable spelling
#: through, and an unrecognised limit writes a per-page marker that reads like an
#: unreadable sheet, so a later pass spends into the same wall. A bare HTTP 429 is
#: deliberately NOT here: transient back-off pressure, not this terminal path.
_S = r"[ _-]"
_BUDGET_RE = re.compile(
    rf"usage{_S}limit|rate{_S}limit|session{_S}limit|credit{_S}balance|"
    rf"out of credits|quota|budget|"
    rf"spending{_S}(?:limit|cap)|limit{_S}(?:reached|exceeded)|"
    # subscription refusals name the limit's window ("hit your weekly limit")
    rf"(?:hourly|daily|weekly|monthly){_S}limit",
    re.IGNORECASE,
)


def _raise_budget_if_matched(text: str, scope: re.Pattern[str] | None = None) -> None:
    """Raise ``BudgetLimitError`` when provider output reports a limit."""
    lines = scope.findall(text) if scope is not None else [text]
    for chunk in lines:
        match = _BUDGET_RE.search(chunk)
        if match is None:
            continue
        line = chunk
        for candidate in chunk.splitlines():
            if match.group(0) in candidate:
                line = candidate
                break
        raise BudgetLimitError(f"budget/limit message from model: {line.strip()[:300]}")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    return re.sub(r"\s*```$", "", stripped).strip()
