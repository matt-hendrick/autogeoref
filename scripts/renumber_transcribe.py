"""Pure parsing core for dual-read renumbering-table transcription.

Rows are accepted only when independent reads agree. Format-specific behavior
belongs in selectable classifiers, not the matcher pipeline.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

#: Accepted numeral bounds for the source table.
MAX_NEW = 13999
MAX_OLD = 9999

_NUM = re.compile(r"^(\d{1,5})([SWsw])?$")
_TO = re.compile(r"^(to|t0)([\s,.]+(to|t0))?$")
_QUOTEISH = re.compile(r"[\s\"'`“”‘’«»,.′″~-]+")  # noqa: RUF001 — ditto marks as OCR renders them
_ALPHA2 = re.compile(r"[A-Za-z]{2,}")
# Bare "No./Nos." can start a street name and must remain a header.
_COLHEAD = re.compile(r"^(new|old|0ld|o1d|numbers)([\s.]+(new|old|0ld|o1d|numbers))*$")


@dataclass(frozen=True)
class Token:
    """One OCR token in page-image pixel coordinates (y = row center)."""

    x0: float
    x1: float
    y: float
    text: str


@dataclass(frozen=True)
class RowClass:
    """A classified table row.

    ``kind`` is one of ``pair`` (NEW OLD numerals), ``lone`` (a single
    numeral — half a pair, always queued), ``multi`` (3+ numerals — a strip
    boundary artifact, queued), ``to`` (the printed run-compression word),
    ``ditto`` (repeated-value marks; skipped, never guessed), ``label``
    (Odd/Even Nos. column headings), ``header`` (street names and inline
    annotations like ``cor``/``rear``), or ``junk``.
    """

    kind: str
    new: int | None = None
    old: int | None = None
    new_suffix: str = ""
    old_suffix: str = ""
    parity: str = ""
    cont: bool = False
    text: str = ""


@dataclass(frozen=True)
class MergedRow:
    """One row after aligning the two independent reads by y position.

    ``status``: ``agreed`` (both reads produced the identical pair — the only
    status that feeds accepted output), ``disagree`` (both read a pair or one
    did and they differ — manual/tiebreak queue), ``only_a``/``only_b`` (a
    pair seen by a single read — queue), ``context`` (labels, headers, to/
    ditto rows — used for chain segmentation, never as data).
    """

    y: float
    status: str
    a: RowClass | None
    b: RowClass | None
    #: set for ``concat`` rows: the pair-side reading both engines' digit
    #: strings support
    resolved: RowClass | None = None


def detect_rules(
    dark: npt.NDArray[np.bool_], min_frac: float = 0.2, group_gap: int = 6
) -> list[int]:
    """X centers of the printed vertical rules.

    A rule is a pixel column whose longest vertical dark run exceeds
    ``min_frac`` of the page height; adjacent hits are grouped into one center.
    """
    h = dark.shape[0]
    d = dark.astype(np.int32)
    c = np.cumsum(d, axis=0)
    reset = np.where(d == 0, c, 0)
    run = c - np.maximum.accumulate(reset, axis=0)
    longest = run.max(axis=0)
    xs = np.flatnonzero(longest > min_frac * h)
    if xs.size == 0:
        return []
    centers: list[int] = []
    start = prev = int(xs[0])
    for x in xs[1:].tolist():
        if x - prev > group_gap:
            centers.append((start + prev) // 2)
            start = x
        prev = x
    centers.append((start + prev) // 2)
    return centers


def strip_bounds(rules: list[int]) -> list[tuple[int, int]]:
    """Column-strip (x0, x1) bounds from detected rule centers.

    Interior gaps much wider than the median rule spacing get evenly infilled
    (weak/broken rules), and one leading boundary is prepended when the first
    rule sits a full column-width into the page (the leftmost rule is often
    the table frame's inner line, not the page edge).
    """
    if len(rules) < 3:
        return []
    gaps = np.diff(np.asarray(rules))
    median = float(np.median(gaps))
    if median <= 0:
        return []
    xs: list[int] = []
    if rules[0] > 0.6 * median:
        xs.append(max(int(rules[0] - median), 0))
    for i, r in enumerate(rules[:-1]):
        xs.append(r)
        k = round((rules[i + 1] - r) / median)
        xs.extend(round(r + (rules[i + 1] - r) * j / k) for j in range(1, k))
    xs.append(rules[-1])
    return [(a, b) for a, b in itertools.pairwise(xs) if b - a > 0.4 * median]


def cluster_rows(tokens: list[Token], gap: float = 12.0) -> list[list[Token]]:
    """Group tokens into printed rows by y proximity, each row sorted by x."""
    rows: list[list[Token]] = []
    for tok in sorted(tokens, key=lambda t: t.y):
        if rows and tok.y - rows[-1][-1].y <= gap:
            rows[-1].append(tok)
        else:
            rows.append([tok])
    return [sorted(r, key=lambda t: t.x0) for r in rows]


_STREET_HEADER_1911 = re.compile(
    r"^[NSEW]?[A-Z'. ]{2,40}\b(ST|STREET|AVE?|AVENUE|BLVD|BOULEVARD|CT|COURT|PL"
    r"|PLACE|DR|DRIVE|ROW|SQ)\.?\s*(continued|cont'?d?)?\s*$",
    re.IGNORECASE,
)


def classify_row_leading(texts: list[str]) -> RowClass:
    """1911 Loop-guide row classifier: the number pair LEADS the row.

    The typewritten register prints ``NEW OLD [building name] [entrance code]`` per row, so only
    numerals BEFORE the first word count — a building name's stray digits must never form a
    pair. Rows without a leading pair fall through to :func:`classify_row`, except that a header
    verdict survives only when the text looks like a street heading (name + suffix word at the
    END, optional 'continued'); building names ('Elk Hotel', 'Rand McNally', 'La Salle St.
    Station') become junk so they can neither break the chain nor hijack its street attribution.
    """
    raw = " ".join(t.strip() for t in texts if t.strip())
    cleaned = [c for c in (p.strip(".,;:|()!") for p in raw.split()) if c]
    nums: list[tuple[int, str]] = []
    for c in cleaned:
        m = _NUM.match(c)
        if m is None:
            break
        nums.append((int(m.group(1)), (m.group(2) or "").upper()))
    # a third leading numeral is the ENTR column's letter misread as a digit
    # ('417 293 8' — the entrance code 's'); the pair is the first two
    nums = nums[:2]
    if len(nums) == 2:
        (new, new_suffix), (old, old_suffix) = nums
        if 1 <= new <= MAX_NEW and 1 <= old <= MAX_OLD:
            return RowClass(
                kind="pair",
                new=new,
                old=old,
                new_suffix=new_suffix,
                old_suffix=old_suffix,
                text=raw,
            )
    rc = classify_row(texts)
    if rc.kind in ("header", "label") and _STREET_HEADER_1911.match(raw):
        # 'S. CLARK ST. continued' must stay a street header even though the
        # bare-'continued' rule would call it a label
        return RowClass(kind="header", text=raw)
    if rc.kind == "pair":
        # digits existed but not leading (building-name digits): not data
        return RowClass(kind="junk", text=raw)
    if rc.kind == "header":
        return RowClass(kind="junk", text=raw)
    return rc


def classify_row(texts: list[str]) -> RowClass:
    """Classify one printed row from its left-to-right token texts.

    Digits are matched exactly — no 0/O style repair, which would correlate
    the two reads' errors and defeat the agreement rule.
    """
    raw = " ".join(t.strip() for t in texts if t.strip())
    lower = raw.lower()
    if raw and not _QUOTEISH.sub("", raw):
        return RowClass(kind="ditto", text=raw)
    if _TO.match(lower):
        return RowClass(kind="to", text=raw)
    cleaned = [c for c in (p.strip(".,;:|()!") for p in raw.split()) if c]
    nums: list[tuple[int, str]] = []
    words: list[str] = []
    i = 0
    while i < len(cleaned):
        m = _NUM.match(cleaned[i])
        if m:
            suffix = (m.group(2) or "").upper()
            if not suffix and i + 1 < len(cleaned) and cleaned[i + 1] in ("S", "W", "s", "w"):
                suffix = cleaned[i + 1].upper()
                i += 1
            nums.append((int(m.group(1)), suffix))
        else:
            words.append(cleaned[i])
        i += 1
    alpha = [w for w in words if _ALPHA2.search(w)]
    collapsed = re.sub(r"[^a-z0-9]", "", lower)
    if (
        not nums
        and ("odd" in collapsed or "even" in collapsed)
        and ("no" in collapsed or "cont" in collapsed)
    ):
        parity = "odd" if "odd" in collapsed else "even"
        return RowClass(kind="label", parity=parity, cont="cont" in collapsed, text=raw)
    if alpha and _COLHEAD.match(" ".join(a.lower() for a in alpha)):
        return RowClass(kind="label", text=raw)
    if "cont" in lower and not nums:
        return RowClass(kind="label", cont=True, text=raw)
    if alpha:
        return RowClass(kind="header", text=raw)
    if len(nums) == 2:
        (new, new_suffix), (old, old_suffix) = nums
        if 1 <= new <= MAX_NEW and 1 <= old <= MAX_OLD:
            return RowClass(
                kind="pair",
                new=new,
                old=old,
                new_suffix=new_suffix,
                old_suffix=old_suffix,
                text=raw,
            )
        return RowClass(kind="junk", text=raw)
    if len(nums) == 1:
        return RowClass(kind="lone", new=nums[0][0], new_suffix=nums[0][1], text=raw)
    if len(nums) >= 3:
        return RowClass(kind="multi", text=raw)
    return RowClass(kind="junk", text=raw)


def _pairs_equal(a: RowClass, b: RowClass) -> bool:
    return (
        a.new == b.new
        and a.old == b.old
        and a.new_suffix == b.new_suffix
        and a.old_suffix == b.old_suffix
    )


def _merge_one(y: float, a: RowClass | None, b: RowClass | None) -> MergedRow:
    if a is not None and b is not None:
        if a.kind == "pair" and b.kind == "pair":
            return MergedRow(y, "agreed" if _pairs_equal(a, b) else "disagree", a, b)
        if a.kind == "pair" or b.kind == "pair":
            # concat agreement: typewritten kerning makes one engine glue the
            # pair into a single token ('411291' vs '411 291'). When the
            # non-pair read's digit string equals the pair's concatenation,
            # both engines read the same digits — only the segmentation is
            # single-read, and chain monotonicity validates that downstream.
            pair = a if a.kind == "pair" else b
            other = b if a.kind == "pair" else a
            if re.sub(r"\D", "", other.text) == f"{pair.new}{pair.old}":
                return MergedRow(y, "concat", a, b, resolved=pair)
            return MergedRow(y, "disagree", a, b)
        return MergedRow(y, "context", a, b)
    if a is not None:
        return MergedRow(y, "only_a" if a.kind == "pair" else "context", a, None)
    assert b is not None
    return MergedRow(y, "only_b" if b.kind == "pair" else "context", None, b)


def merge_reads(
    rows_a: list[tuple[float, RowClass]],
    rows_b: list[tuple[float, RowClass]],
    y_tol: float = 14.0,
) -> list[MergedRow]:
    """Align two independent reads of one strip by row y position.

    Both inputs must be y-sorted. Rows within ``y_tol`` pixels pair up
    (half the ~27 px row pitch); everything else surfaces as single-read.
    """
    out: list[MergedRow] = []
    i = j = 0
    while i < len(rows_a) or j < len(rows_b):
        if j >= len(rows_b) or (i < len(rows_a) and rows_a[i][0] < rows_b[j][0] - y_tol):
            out.append(_merge_one(rows_a[i][0], rows_a[i][1], None))
            i += 1
        elif i >= len(rows_a) or rows_b[j][0] < rows_a[i][0] - y_tol:
            out.append(_merge_one(rows_b[j][0], None, rows_b[j][1]))
            j += 1
        else:
            out.append(_merge_one((rows_a[i][0] + rows_b[j][0]) / 2, rows_a[i][1], rows_b[j][1]))
            i += 1
            j += 1
    return out


def rows_from_tokens(
    tokens: list[Token],
    gap: float = 12.0,
    classifier: Callable[[list[str]], RowClass] = classify_row,
) -> list[tuple[float, RowClass]]:
    """Cluster tokens into rows and classify each: the per-strip read result."""
    out: list[tuple[float, RowClass]] = []
    for row in cluster_rows(tokens, gap=gap):
        y = sum(t.y for t in row) / len(row)
        out.append((y, classifier([t.text for t in row])))
    return out
