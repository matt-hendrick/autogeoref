"""Cached-fixture qualification scoring for annotation backends.

Score a candidate backend against annotations already on disk before anyone
trusts it. No pipeline stage imports this: it is a harness library, and living
here rather than in the package is what says so. It spends nothing itself — the
backend it scores is injected, and every case is a cached fixture.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from autogeoref.annotate.failures import AnnotationCallError

if TYPE_CHECKING:
    from autogeoref.annotate.schema import Annotation, AnnotatorBackend

logger = logging.getLogger(__name__)
_NAME_JUNK_RE = re.compile(r"[^A-Z0-9 ]")


def normalize_street_name(name: str) -> str:
    """Normalize a street label for recall/precision matching."""
    return " ".join(_NAME_JUNK_RE.sub("", name.upper()).split())


@dataclass(frozen=True)
class SheetScore:
    image_path: Path
    recall: float
    precision: float
    expected: int
    predicted: int
    error: str | None = None


@dataclass(frozen=True)
class QualificationReport:
    scores: tuple[SheetScore, ...] = field(default_factory=tuple)

    @property
    def mean_recall(self) -> float:
        return sum(score.recall for score in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def mean_precision(self) -> float:
        return (
            sum(score.precision for score in self.scores) / len(self.scores) if self.scores else 0.0
        )

    @property
    def failures(self) -> tuple[SheetScore, ...]:
        return tuple(score for score in self.scores if score.error is not None)


def score_annotation(predicted: Annotation, expected: Annotation) -> tuple[float, float]:
    """Return street-name recall and precision for one cached reference sheet."""
    expected_names = {normalize_street_name(street.name) for street in expected.streets}
    predicted_names = {normalize_street_name(street.name) for street in predicted.streets}
    hits = len(expected_names & predicted_names)
    recall = hits / len(expected_names) if expected_names else 1.0
    precision = (
        hits / len(predicted_names) if predicted_names else (1.0 if not expected_names else 0.0)
    )
    return recall, precision


def qualify_backend(
    backend: AnnotatorBackend, cases: Iterable[tuple[Path, Annotation]]
) -> QualificationReport:
    """Score a backend against cached fixture annotations without making it authoritative."""
    scores: list[SheetScore] = []
    for image_path, expected in cases:
        started = time.monotonic()
        try:
            predicted = backend.annotate(image_path)
        except AnnotationCallError as exc:
            logger.warning("qualification sheet %s failed: %s", image_path.name, exc)
            scores.append(
                SheetScore(
                    image_path,
                    0.0,
                    0.0,
                    len(expected.streets),
                    0,
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        recall, precision = score_annotation(predicted, expected)
        logger.info(
            "qualification sheet %s: recall=%.2f precision=%.2f (%.1fs)",
            image_path.name,
            recall,
            precision,
            time.monotonic() - started,
        )
        scores.append(
            SheetScore(
                image_path,
                recall,
                precision,
                len(expected.streets),
                len(predicted.streets),
            )
        )
    return QualificationReport(tuple(scores))
