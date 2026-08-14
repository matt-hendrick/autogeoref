"""THE page<->slug mapping: one function each way, shared by every consumer.

One shared parser prevents consumers from binding a page's GCPs to another
sheet. Slugs may be recorded or use ``<volume>_p<N>``. Page ids are digits with
an optional letter suffix (``12``, ``7a``), and the ``p`` token must start the
name or follow an underscore — ``map3`` is not page 3. Some books bind named
sheets with non-numeric ids, and ``_NAMED_PAGES`` admits those known ids
explicitly so map sheets cannot be silently skipped.

The regex REFUSES a trailing region index (``..._p10_1``), and that refusal is
a safety property. Those are ground-truth layers for a sheet a volunteer split
into crops before pinning: their GCP pixels live in the CROP's frame and the
export carries no offset back to the page, so admitting them would bind one
frame's control points to another frame's pixels.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Literal page ids for named (non-numeric) sheets. An explicit allow-list,
#: never a pattern: a general "letters allowed" rule would also admit the
#: crop-layer ids the module docstring refuses. PARSING only — which pages are
#: district-scale *overview* sheets is a per-volume declaration, not a property
#: of the id, so the class and the grammar cannot share a list.
_NAMED_PAGES = ("cbd1", "cbd2", "cbda", "cbdb")

_PAGE_RE = re.compile(
    r"(?:^|_)p(\d+[a-z]?|" + "|".join(_NAMED_PAGES) + r")$",
    re.IGNORECASE,
)

#: Sheet kinds that carry no map and so have no page id: the binding cover, the
#: title plate, the street index (``ind``, ``ind1``, ``ind2``), and one
#: publisher's note. Prep rejects every other unpageable filename so a missing
#: page cannot be mistaken for an intentional skip.
_NON_ADDRESSABLE_RE = re.compile(r"(?:^|_)p(covr|titl|ind\d*|note)$", re.IGNORECASE)


def page_from_slug(slug: str) -> str | None:
    """Page id from a layer slug or sheet-file stem, else ``None``.

    Accepts ``..._p12``, ``p12``, letter-suffixed ``..._p7a``, and the named
    named sheets such as ``..._pcbd1`` / ``..._pcbd2``.

    A NAMED id is canonicalized to lower case (it is matched case-insensitively,
    and one sheet must not become two page ids). A numeric id is returned exactly
    as it was written: case-folding those would silently re-key existing pages
    (``_p7A`` -> ``7a``), which is not this function's business.
    """
    m = _PAGE_RE.search(slug)
    if m is None:
        return None
    page = m.group(1)
    return page.lower() if page.lower() in _NAMED_PAGES else page


def non_addressable_kind(slug: str) -> str | None:
    """The KIND of a known map-less sheet (``covr``/``titl``/``ind``/``note``), else ``None``.

    ``None`` here does NOT mean "addressable" — it means "not a kind we know
    about". Callers pair this with :func:`page_from_slug`: a slug that answers
    ``None`` to BOTH is a file nothing in this repo recognizes, and prep refuses
    to skip it silently.
    """
    m = _NON_ADDRESSABLE_RE.search(slug)
    return m.group(1).lower() if m else None


def slug_for_page(volume: str, page: str) -> str:
    """Canonical layer slug for a page with no recorded export slug."""
    return f"{volume}_p{page}"


#: The page-id alphabet on its own, for validating an id that arrives WITHOUT
#: its ``p`` token (a review URL, a persisted sidecar field).
_PAGE_ID_RE = re.compile(r"^\d+[A-Za-z]?$")


def valid_review_page(page: str) -> bool:
    """True for a page id review may address: THE narrow page grammar.

    Digits with an optional letter suffix (``12``, ``7a``, ``13S``) or one of
    the literal named sheets (``cbd1``/``cbd2``, canonical lower case). An
    explicit allowlist, never a permissive alphabetic pattern: ``10_1``-style
    crop ids, path separators, and every other stranger stay rejected, for the
    reasons the module docstring pins. Review interpolates a page id into
    result and sidecar paths, so this is a path-safety boundary too.
    """
    return bool(_PAGE_ID_RE.match(page)) or page in _NAMED_PAGES


#: Candidate skeleton sheets: an UPPERCASE ``S`` after a numeric page id
#: (``13S``). Uppercase-only is deliberate: a lowercase letter is the
#: ``7a``-style continuation form, a distinct map area that must never be
#: treated as a duplicate. The form ALONE does not settle it — see
#: :func:`skeleton_pages`.
_SKELETON_PAGE_RE = re.compile(r"^(\d+)S$")


def skeleton_pages(pages: Collection[str]) -> frozenset[str]:
    """Which of a volume's page ids are skeleton twins.

    A skeleton is the uncolored outline duplicate of a numeric sheet, so it is
    one only when that twin is in the same volume: ``13S`` beside ``13``. The
    form alone is underdetermined — a volume can print the division letter as
    part of the sheet number (``20 S``), and then every map page carries the
    suffix and none duplicates anything. Resolved from the volume's whole page
    inventory, never from what placed. Takes page ids, not slugs. Warns when a
    volume answers BOTH ways — see the comment on the warning.
    """
    numeric = {page for page in pages if page.isdigit()}
    matched = {page: m for page in pages if (m := _SKELETON_PAGE_RE.match(page))}
    twinned = frozenset(page for page, m in matched.items() if m.group(1) in numeric)
    orphans = sorted(set(matched) - twinned)
    # a volume is one shape or the other; both at once means the inventory is
    # incomplete, and a real skeleton whose twin is missing demotes silently to
    # a regular sheet that then competes with the very page it duplicates
    if twinned and orphans:
        logger.warning(
            "skeleton twins: %s have a numeric twin here but %s do not — an "
            "incomplete page inventory demotes a real skeleton to a regular sheet",
            sorted(twinned),
            orphans,
        )
    return twinned


@dataclass(frozen=True)
class DuplicateCoverage:
    """A volume's pages whose map repeats ground its regular pages own.

    Both kinds in one value, resolved once per volume: the DECLARED overview
    pages (``VolumeConfig.overview_pages``) and the twins :func:`skeleton_pages`
    finds. They travel together through every bake stage so no two of them can
    classify one sheet differently. The default is the empty class.
    """

    overview_pages: frozenset[str] = frozenset()
    skeletons: frozenset[str] = frozenset()

    @classmethod
    def resolve(
        cls, pages: Collection[str], overview_pages: Collection[str] = ()
    ) -> DuplicateCoverage:
        """Build one from a volume's whole page inventory and its declaration."""
        return cls(frozenset(overview_pages), skeleton_pages(pages))


#: The empty class, for a volume with neither kind. A module-level singleton
#: because a call cannot be a default argument.
NO_DUPLICATE_COVERAGE = DuplicateCoverage()


def duplicate_coverage_page(page: str, duplicates: DuplicateCoverage) -> bool:
    """True when a page's map repeats ground the volume's regular pages own.

    The bake keeps these out of the regular sheets' overlap competition (a
    whole-sheet overlap makes the bisector splitter halve both sheets along a
    mid-sheet diagonal) and paints them underneath, as fallback coverage.
    """
    return page in duplicates.overview_pages or page in duplicates.skeletons


def duplicate_coverage_slug(slug: str, duplicates: DuplicateCoverage) -> bool:
    """:func:`duplicate_coverage_page` lifted to slugs; False when unpageable."""
    page = page_from_slug(slug)
    return page is not None and duplicate_coverage_page(page, duplicates)


def overview_page(page: str, duplicates: DuplicateCoverage) -> bool:
    """True for a DECLARED overview page — the district-scale duplicate-coverage kind.

    The two duplicate-coverage kinds differ in reach. A skeleton twin coincides
    with its numeric twin, so its whole mask is ground its own fit constrained.
    An overview page spans a district, most of it far from any of its GCPs, so
    its mask is clipped to what its inliers earned and its paint is kept out of
    the detail mosaic rather than presented as coverage. Declared per volume,
    never derived from the id: overview segments can carry plain numeric ids.
    """
    return page in duplicates.overview_pages


def overview_slug(slug: str, duplicates: DuplicateCoverage) -> bool:
    """:func:`overview_page` lifted to slugs; False when unpageable."""
    page = page_from_slug(slug)
    return page is not None and overview_page(page, duplicates)


def page_sort_key(slug: str) -> tuple[int, str]:
    """Natural page order for slugs; non-numeric tails sort last, stably."""
    page = page_from_slug(slug)
    if page is not None and page.isdigit():
        return (int(page), slug)
    return (1 << 30, slug)


def mosaic_paint_order(slugs: list[str], duplicates: DuplicateCoverage) -> list[str]:
    """Paint order: duplicate-coverage sheets first, then regular page order.

    Later parts paint over earlier ones, and page order alone puts
    duplicate-coverage sheets last (non-numeric ids sort after every number),
    i.e. on top of the very pages they duplicate. They belong at the bottom:
    fallback coverage that shows only where no regular sheet does.
    """
    ordered = sorted(slugs, key=page_sort_key)
    ordered.sort(key=lambda slug: 0 if duplicate_coverage_slug(slug, duplicates) else 1)
    return ordered
