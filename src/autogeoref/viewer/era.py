"""Era vocabulary: the year-range buckets and the year -> chip-label rule.

Its own module because :mod:`.config` parses buckets and :mod:`.sources`
labels years with them; sharing a base keeps those two off each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EraBucket:
    """Year range -> era chip label."""

    first_year: int
    last_year: int
    label: str
    #: attribution HTML for this era's layers. Per-era on purpose: crediting
    #: volunteer georeferencers on auto-georeferenced eras (or vice versa)
    #: was a real attribution inaccuracy in the single global footer.
    credits_html: str | None = None


def era_label(year: int | None, buckets: Sequence[EraBucket]) -> str | None:
    """Bucket a volume year into its era chip label.

    A year no bucket covers labels itself; ``None`` stays ``None``.
    """
    if year is None:
        return None
    for b in buckets:
        if b.first_year <= year <= b.last_year:
            return b.label
    return str(year)
