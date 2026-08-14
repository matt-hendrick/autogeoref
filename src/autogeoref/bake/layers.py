"""Which sheets a bake is allowed to touch: the committed records, in paint order.

Every back-half stage starts here, and the answer is read from the result
records alone — nothing recomputes a placement to decide what to serve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..paths import iter_results
from ..slugs import page_sort_key, slug_for_page
from ..volume import is_committed

if TYPE_CHECKING:
    from ..paths import VolumePaths


def committed_layers(paths: VolumePaths, volume: str) -> list[tuple[str, str, dict[str, Any]]]:
    """``(page, slug, record)`` for every committed result, in page order."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for page, record, _path in iter_results(paths):
        if is_committed(record):
            out.append((page, record.get("layer") or slug_for_page(volume, page), record))
    out.sort(key=lambda item: page_sort_key(item[1]))
    return out
