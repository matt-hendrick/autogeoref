"""Read street-index pages into coarse per-sheet placement priors.

Tiled reads are cached and malformed or ambiguous entries abstain rather than
becoming placement evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from .addresses import BLOCK_SIZE, AddressNumeral, RenumberingTable, match_address, modern_numeral
from .affine import TO_3857
from .annotate.cli_call import classify, run_cli
from .annotate.failures import AnnotationCallError, MalformedResponseError
from .annotate.providers import canonical_model, ensure_model_allowed
from .annotate.schema import _json_object_from_cli_text
from .margins import PriorWindow, _mercator_scale

if TYPE_CHECKING:
    from .names import Aliases

logger = logging.getLogger(__name__)

#: Prior radius in EPSG:3857 planar meters.
HIT_RADIUS_M = 600.0

#: Per-tile extraction prompt.
INDEX_TILE_PROMPT = (
    "The image at IMGPATH is one horizontal tile of a street index page from a Sanborn "
    "fire insurance atlas: an alphabetical table mapping street name + house-number range "
    'to a sheet number (e.g. "Cullom Av 600-719 ... 114"). Output ONLY a JSON object '
    '(no prose, no code fences): {"entries": [{"street": "<name as printed>", '
    '"from": <int>, "to": <int>, "sheet": <int>}]}. Read EVERY legible row in this image '
    "tile (it is a tile of a larger page; adjacent tiles overlap slightly). Skip rows "
    "where any field is illegible; precision matters more than coverage."
)

#: Explicit prompt version, part of every read's cache fingerprint. Bump it
#: when the extraction semantics change without a text change to
#: :data:`INDEX_TILE_PROMPT` (the prompt digest already covers text edits).
INDEX_PROMPT_VERSION = "tile-v1"


def _cache_namespace(image_path: Path, model: str, n_tiles: int, overlap_frac: float) -> str:
    """Fingerprint every semantic input of a tiled index read.

    The digest covers the source-image BYTES (replacing the image in place
    invalidates the cache), the tiling geometry, the canonical model
    identity (bare and provider-qualified spellings of one model share a
    namespace), and the prompt version + digest. Cache reuse is therefore
    exactly "every semantic input unchanged".
    """
    key = json.dumps(
        {
            "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "n_tiles": n_tiles,
            "overlap_frac": repr(float(overlap_frac)),
            "model": canonical_model(model),
            "prompt_version": INDEX_PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(INDEX_TILE_PROMPT.encode()).hexdigest(),
        },
        sort_keys=True,
    )
    return f"{image_path.stem}.{hashlib.sha256(key.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class IndexEntry:
    """One street-index row: street + house-number range -> sheet.

    ``sheet`` keeps the pipeline's string page ids (``"92"``), matching
    ``sheets/manifest.json`` and the results records.
    """

    street: str
    from_number: int
    to_number: int
    sheet: str


def parse_index_entries(raw: dict[str, Any]) -> list[IndexEntry]:
    """Parse a raw index-read JSON object into entries, tolerantly.

    Accepts the ``{"entries": [{"street", "from", "to", "sheet"}, ...]}``
    shape the validated prompt produces. Malformed entries (missing keys,
    non-integer ranges, empty street, non-int/str sheet) are dropped with a
    debug log — never fatal; the index channel is additive evidence.
    """
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list):
        logger.debug("index read has no 'entries' list (got %s)", type(entries_raw).__name__)
        return []
    out: list[IndexEntry] = []
    for item in entries_raw:
        try:
            street_raw = item["street"]
            sheet_raw = item["sheet"]
            if not isinstance(street_raw, str) or isinstance(sheet_raw, bool):
                raise TypeError("bad street/sheet type")
            if isinstance(sheet_raw, int):
                sheet = str(sheet_raw)
            elif isinstance(sheet_raw, str):
                sheet = sheet_raw.strip()
            else:
                raise TypeError("sheet must be int or str")
            entry = IndexEntry(
                street=street_raw.strip(),
                from_number=int(item["from"]),
                to_number=int(item["to"]),
                sheet=sheet,
            )
        except (KeyError, TypeError, ValueError):
            logger.debug("dropping malformed index entry %r", item)
            continue
        if not entry.street or not entry.sheet:
            logger.debug("dropping malformed index entry %r", item)
            continue
        out.append(entry)
    return out


def _entry_key(entry: IndexEntry) -> tuple[str, int, int]:
    """Dedupe key: whitespace-normalized uppercase street + the range."""
    street = re.sub(r"\s+", " ", entry.street.strip()).upper()
    return (street, entry.from_number, entry.to_number)


def dedupe_entries(entries: Iterable[IndexEntry]) -> list[IndexEntry]:
    """Dedupe the union of tile reads on (normalized street, from, to).

    Identical duplicates (the tile overlap re-reading a boundary row)
    collapse to the first occurrence. If the SAME key maps to DIFFERENT
    sheets, ALL entries under that key are dropped with a warning — at
    least one of the reads is wrong, and a misread must not become a prior.
    """
    first: dict[tuple[str, int, int], IndexEntry] = {}
    order: list[tuple[str, int, int]] = []
    conflicts: set[tuple[str, int, int]] = set()
    for entry in entries:
        key = _entry_key(entry)
        prior = first.get(key)
        if prior is None:
            first[key] = entry
            order.append(key)
        elif prior.sheet != entry.sheet:
            conflicts.add(key)
    for key in conflicts:
        logger.warning(
            "index entry %s %d-%d maps to conflicting sheets across tiles; dropping all",
            *key,
        )
    return [first[key] for key in order if key not in conflicts]


def _require_entries_list(raw: dict[str, Any]) -> None:
    """Reject a payload that is not an index read at all.

    The distinction is between a tile the model read and found nothing in
    (``{"entries": []}``, a legitimate result) and a response that never
    answered the prompt — a provider refusal being the case that matters, since
    it is valid JSON. Entry-level junk stays tolerated; that is
    :func:`parse_index_entries`'s job.
    """
    if not isinstance(raw.get("entries"), list):
        raise MalformedResponseError(f"index read has no 'entries' list: {str(raw)[:200]}")


def _read_tile_cli(
    tile_path: Path,
    model: str,
    timeout_s: float,
    executable: str | None = None,
) -> dict[str, Any]:
    """Read one index tile through the annotator's model choke point.

    All conduct (Sonnet-class gate, the per-provider argv shape, stdin detached, the distinct
    failure taxonomy) lives in :mod:`autogeoref.annotate.cli_call` — this module never builds
    its own model invocation, and gets provider routing for free: ``model`` may carry a
    CLI ``provider:`` prefix.

    The schema check sits INSIDE the classified block: a refusal decodes cleanly, and
    judging it outside would cache it as an empty index that never re-spends and never errors.
    """
    outcome = run_cli(
        tile_path,
        INDEX_TILE_PROMPT,
        model=model,
        timeout_s=timeout_s,
        executable=executable,
    )
    try:
        raw = _json_object_from_cli_text(outcome.text)
        _require_entries_list(raw)
    except AnnotationCallError as exc:
        classify(outcome, exc)
    return raw


def read_index(
    image_path: Path,
    model: str = "claude-sonnet-5",
    annotate_fn: Callable[[Path], dict[str, Any]] | None = None,
    n_tiles: int = 5,
    overlap_frac: float = 0.08,
    cache_dir: Path | None = None,
    timeout_s: float = 600.0,
) -> list[IndexEntry]:
    """Read a street-index page by TILING it into horizontal bands.

    Measured constraint: single-call full-page reads do not complete, while bounded tile reads
    work cleanly — so the page is cropped into ``n_tiles`` full-width horizontal bands with
    ``overlap_frac`` vertical overlap. Index rows are horizontal text lines, so full-width bands
    preserve whole rows and the overlap prevents losing one cut at a boundary; the duplicates
    collapse in :func:`dedupe_entries`. Bands and their parsed reads live in a fingerprinted
    namespace under ``cache_dir``, whose fingerprint covers the image digest, tiling, model and
    prompt, so an unchanged repeat spends nothing.
    """
    ensure_model_allowed(model)
    if cache_dir is None:
        raise ValueError("cache_dir is required: tile crops and per-tile reads are cached there")
    if n_tiles < 1:
        raise ValueError(f"n_tiles must be >= 1, got {n_tiles}")
    if overlap_frac < 0:
        raise ValueError(f"overlap_frac must be >= 0, got {overlap_frac}")
    stem = image_path.stem
    tile_dir = cache_dir / _cache_namespace(image_path, model, n_tiles, overlap_frac)
    tile_dir.mkdir(parents=True, exist_ok=True)

    def _tile_jpg(i: int) -> Path:
        return tile_dir / f"tile{i}of{n_tiles}.jpg"

    def _tile_json(i: int) -> Path:
        return tile_dir / f"tile{i}of{n_tiles}.json"

    need_crops = [
        i for i in range(1, n_tiles + 1) if not _tile_json(i).exists() and not _tile_jpg(i).exists()
    ]
    if need_crops:
        with Image.open(image_path) as img:
            band_h = img.height / n_tiles
            pad = band_h * overlap_frac
            for i in need_crops:
                top = max(0, math.floor((i - 1) * band_h - pad))
                bottom = min(img.height, math.ceil(i * band_h + pad))
                img.crop((0, top, img.width, bottom)).convert("RGB").save(_tile_jpg(i))

    def _default_annotate(tile_path: Path) -> dict[str, Any]:
        return _read_tile_cli(tile_path, model=model, timeout_s=timeout_s)

    read_tile = annotate_fn if annotate_fn is not None else _default_annotate

    entries: list[IndexEntry] = []
    for i in range(1, n_tiles + 1):
        json_path = _tile_json(i)
        if json_path.exists():
            raw = json.loads(json_path.read_text())
            if not isinstance(raw, dict):
                raise ValueError(f"corrupt tile cache (not a JSON object): {json_path}")
        else:
            raw = read_tile(_tile_jpg(i))
            # before the write, so a non-answer cannot become a permanent empty index
            _require_entries_list(raw)
            json_path.write_text(json.dumps(raw, indent=2))
        tile_entries = parse_index_entries(raw)
        logger.debug("index tile %d/%d of %s: %d entries", i, n_tiles, stem, len(tile_entries))
        entries.extend(tile_entries)
    deduped = dedupe_entries(entries)
    logger.info(
        "index %s: %d entries across %d tiles (%d after dedupe)",
        stem,
        len(entries),
        n_tiles,
        len(deduped),
    )
    return deduped


def _centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest(
    points: Sequence[tuple[float, float]], reference: tuple[float, float]
) -> tuple[float, float]:
    return min(points, key=lambda p: _dist(p, reference))


def index_priors(
    entries: Iterable[IndexEntry],
    centerline_features: Sequence[Mapping[str, Any]],
    aliases: Aliases | None,
    renumbering: RenumberingTable | None = None,
    hit_radius_m: float = HIT_RADIUS_M,
    address_block_size: int = BLOCK_SIZE,
) -> dict[str, PriorWindow]:
    """Convert index entries into per-page placement prior windows.

    Range-unmatched entries abstain. ``renumbering=None`` is for modern printed numbers; a
    renumbered edition must supply a conversion table. Per entry, the range midpoint is
    converted and matched against the centerline address ranges, abstaining on either failure.
    Per page the entries combine, each contributing candidate points, and an entry whose range
    repeats along its street is disambiguated toward the centroid of the single-hit entries. If
    the resolved points' spread exceeds ``hit_radius_m`` the page ABSTAINS — inconsistent index
    rows must not average into a fake window.
    """
    per_page: dict[str, list[list[tuple[float, float]]]] = {}
    for entry in entries:
        mid = (entry.from_number + entry.to_number) // 2
        numeral = modern_numeral(
            entry.street, mid, renumbering, aliases, block_size=address_block_size
        )
        if numeral is None:
            logger.debug(
                "index entry %s %d-%d: no modern numeral (renumbering unknown); abstain",
                entry.street,
                entry.from_number,
                entry.to_number,
            )
            continue
        hits = match_address(
            AddressNumeral(value=numeral, bbox=(0.0, 0.0, 1.0, 1.0), street_hint=entry.street),
            entry.street,
            centerline_features,
            aliases,
        )
        if not hits:
            logger.debug(
                "index entry %s %d-%d: no centerline range match; abstain",
                entry.street,
                entry.from_number,
                entry.to_number,
            )
            continue
        points = [(float(x), float(y)) for x, y in (TO_3857.transform(*h.point_4326) for h in hits)]
        per_page.setdefault(entry.sheet, []).append(points)

    windows: dict[str, PriorWindow] = {}
    for page, candidate_lists in sorted(per_page.items()):
        single = [pts[0] for pts in candidate_lists if len(pts) == 1]
        if single:
            reference = _centroid(single)
        else:
            reference = _centroid([p for pts in candidate_lists for p in pts])
        resolved = list(single)
        resolved.extend(_nearest(pts, reference) for pts in candidate_lists if len(pts) > 1)
        spread = max(
            (_dist(p, q) for i, p in enumerate(resolved) for q in resolved[i + 1 :]),
            default=0.0,
        )
        if spread > hit_radius_m:
            logger.warning(
                "index page %s: resolved entry points spread %.0f m > %.0f m "
                "(mutually inconsistent rows); abstaining",
                page,
                spread,
                hit_radius_m,
            )
            continue
        center = _centroid(resolved)
        # Convert the planar radius to PriorWindow ground meters.
        windows[page] = PriorWindow(
            center_3857=center, radius_m=hit_radius_m * _mercator_scale(center[1])
        )
    logger.info("index priors: %d pages with windows (of %d named)", len(windows), len(per_page))
    return windows


def fuse_windows(a: PriorWindow, b: PriorWindow) -> PriorWindow | None:  # noqa: DS002
    """Smallest circle covering the intersection lens of two prior windows.

    This is the tightening step that lets a wide index window (600 m) shrink an N-S-only margin
    window (~200 m radius) toward the junction verifier's <= +-100 m contract: the fused window
    can be far smaller than either input.

    Geometry (radii converted to 3857 units at the mean latitude; place disk A's center at 0 and
    disk B's at ``d`` on the line between them):

    - ``d >= ra + rb``: disjoint (a tangency's zero-area lens is useless as
      a window) — the priors CONTRADICT each other; return ``None`` and let
      the caller log the contradiction.
    - ``d <= |ra - rb|``: one disk contains the other; the lens IS the
      smaller disk — return the smaller window unchanged.
    - Otherwise the lens is bounded by two arcs meeting at the corner
      points ``(x0, +-h)``, where ``x0 = (d^2 + ra^2 - rb^2) / 2d`` is the
      radical-line offset and ``h = sqrt(ra^2 - x0^2)`` the half-chord.
      For any covering circle centered on the axis, distance along each
      bounding arc is monotone in the arc angle, so the farthest lens
      points are always among {the two corners, the two axial extremes
      ``(ra, 0)`` and ``(d - rb, 0)``}. The chord-diameter circle
      (center ``(x0, 0)``, radius ``h``) passes through both corners and
      covers the axial extremes exactly when ``0 <= x0 <= d`` (algebra:
      ``h >= ra - x0  <=>  x0 >= 0`` since ``h^2 = (ra-x0)(ra+x0)``, and
      symmetrically ``h >= x0 - (d - rb)  <=>  x0 <= d``), so it is then
      the smallest cover. When ``x0 < 0`` (i.e. ``rb^2 > d^2 + ra^2``) the
      binding points are the corners plus A's axial extreme — all of which
      lie ON circle A, so the smallest cover degenerates to disk A itself
      (and ``ra < rb`` there, so it is the smaller disk); symmetrically
      ``x0 > d`` returns disk B. The axial-diameter circle is never
      optimal in partial overlap: requiring it to cover the corners
      reduces to ``(ra+rb)^2 - d^2 <= (ra+rb-d)^2  <=>  d <= 0``.
    """
    ax, ay = a.center_3857
    bx, by = b.center_3857
    # radius_m is ground meters (PriorWindow contract); the center distance
    # is 3857 units — convert radii to 3857 units at the mean latitude so
    # the circle geometry is done in one frame.
    scale = _mercator_scale((ay + by) / 2.0)
    ra = a.radius_m / scale
    rb = b.radius_m / scale
    d = math.hypot(bx - ax, by - ay)
    if d >= ra + rb:
        return None  # disjoint: contradictory priors (callers log)
    if d <= abs(ra - rb):
        return a if a.radius_m <= b.radius_m else b  # containment: the smaller disk IS the lens
    x0 = (d * d + ra * ra - rb * rb) / (2.0 * d)
    if x0 <= 0.0:
        return a  # disk A (the smaller) is itself the minimal cover of the lens
    if x0 >= d:
        return b  # symmetric: disk B
    h = math.sqrt(max(ra * ra - x0 * x0, 0.0))
    ux, uy = (bx - ax) / d, (by - ay) / d
    return PriorWindow(
        center_3857=(ax + ux * x0, ay + uy * x0),
        radius_m=h * scale,
    )


__all__ = [
    "HIT_RADIUS_M",
    "INDEX_PROMPT_VERSION",
    "INDEX_TILE_PROMPT",
    "IndexEntry",
    "dedupe_entries",
    "fuse_windows",
    "index_priors",
    "parse_index_entries",
    "read_index",
]
