"""Catalog and layer discovery: LOC titles, community areas, served archives.

The one shapely importer in the package — config-only consumers must not
import this module at module level (see :mod:`.config`).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shapely.geometry import box, shape

logger = logging.getLogger(__name__)

#: Physical-format sentences LOC leads a description with, in any order:
#: date ("1901." / "Apr 1933."), sheet count, binding, dimensions. What
#: survives their removal is the map's catalogued subject.
_SUBJECT_BOILERPLATE = (
    # month tokens as LOC writes them: "Apr 1933.", "Sept. 1916.", "July 1922."
    # Enumerated, not prefix-widened: a widening like `[a-z]*` also swallows a
    # subject's own leading place name ("Maywood 1901.", "Junction 1905.").
    re.compile(
        r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?"
        r"|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\.?\s+)?\d{4}\.\s*"
    ),
    re.compile(r"\d+ sheet\(s\)\.\s*"),
    re.compile(r"Bound\.\s*"),
    re.compile(r"\d+ X \d+ cm\.\s*"),
)
_OTHER_PLACES_RE = re.compile(r"Other places as they appear on original:\s*(?P<place>.+?)\s*$")


def _catalog_subject(desc: str) -> str | None:
    """The catalogued subject of an unnumbered (special) map: the LOC
    description minus its physical-format boilerplate. The "Other places as
    they appear on original:" clause is a cross-reference, not the name — prose
    before it is the map's own subject and wins; the referenced place stands in
    only when nothing else survives."""
    remainder = desc.strip()
    while True:
        for pattern in _SUBJECT_BOILERPLATE:
            m = pattern.match(remainder)
            if m:
                remainder = remainder[m.end() :]
                break
        else:
            break
    m = _OTHER_PLACES_RE.search(remainder)
    if m:
        remainder = remainder[: m.start()].strip().rstrip(".") or m.group("place")
    return remainder.strip().rstrip(".").strip() or None


def _catalog_year(item: Mapping[str, Any], desc: str, vol_num: str | None) -> tuple[str, str]:
    """Resolve one catalog item's year: ``(date, year_source)``.

    The date string may still be unusable — the caller checks
    ``date[:4].isdigit()`` and drops the item. ``year_source`` is one of the
    three values documented on :func:`loc_titles`.
    """
    date = item.get("date") or ""
    source = "date"
    desc_year = re.match(r"(\d{4})\.", desc)
    desc_month_year = re.match(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\.", desc
    )
    created = (item.get("item") or {}).get("created_published") or ""
    created_year = re.search(r"\b(\d{4})\b", created)
    if (
        desc_year
        and not vol_num
        and date[:4].isdigit()
        and desc_year.group(1) != date[:4]
        and created_year
        and desc_year.group(1) == created_year.group(1)
    ):
        # _188 is catalogued as 1963-11 despite independently matching 1927
        # description and created_published fields. Keep that map year, but never
        # present it as a trusted structured date to the address-era tool.
        date = desc_year.group(1)
        source = "description-conflict"
    elif not date[:4].isdigit():
        if desc_year:
            date = desc_year.group(1)
            source = "description"
        elif desc_month_year and created_year and desc_month_year.group(1) == created_year.group(1):
            date = desc_month_year.group(1)
            source = "description"
    return date, source


def loc_titles(catalog_path: Path, city_name: str) -> dict[str, dict[str, Any]]:
    """Title, year, volume number and subject per item from a LOC catalog dump.

    Keeps the original parsing rules, including the SPECIALS quirk: a subject-map item has no
    "Vol." prefix and an empty date field, so the year leads the description and the subject is
    what remains once the format boilerplate is stripped (func:`_catalog_subject`). **Each entry
    says WHERE its year came from** (``year_source``): ``"date"`` is LOC's structured field,
    ``"description"`` is scraped from the blurb of an item catalogued with a null date, and
    ``"description-conflict"`` is a scraped year contradicting it. ``era.py`` proposes
    ``addresses_modern`` from this, arming the one channel that may REFUTE.
    """
    meta: dict[str, dict[str, Any]] = {}
    for item in json.loads(catalog_path.read_text(encoding="utf-8")):
        ident = (item.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
        desc = (item.get("description") or [""])[0]
        vol_match = re.match(r"Vol\. ([\w.]+)", desc)
        vol_num = vol_match.group(1).rstrip(".,") if vol_match else None
        date, source = _catalog_year(item, desc, vol_num)
        subject = _catalog_subject(desc) if not vol_num else None
        if ident and date[:4].isdigit():
            meta[ident] = {
                "title": f"{city_name} | {date[:4]} | {subject or ('Vol. ' + (vol_num or '?'))}",
                "year": int(date[:4]),
                "year_source": source,
                "volume_number": vol_num,
                "subject": subject,
            }
    return meta


class AreaIndex:
    """Community-area polygons -> the top-N names covering a bounds box."""

    def __init__(self, geojson_path: Path, name_property: str = "community") -> None:
        self._areas: list[tuple[str, Any]] = []
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
        for feature in data.get("features") or []:
            name = (feature["properties"].get(name_property) or "").title()
            if name and feature.get("geometry"):
                self._areas.append((name, shape(feature["geometry"])))

    def names(self, bounds: Sequence[float], top: int = 3) -> list[str]:
        bb = box(*bounds)
        hits = []
        for name, geom in self._areas:
            inter = bb.intersection(geom)
            if not inter.is_empty:
                hits.append((inter.area, name))
        hits.sort(reverse=True)
        return [n for _, n in hits[:top]]


def _relpath(target: Path, start_dir: Path) -> str:
    return Path(os.path.relpath(target, start_dir)).as_posix()


def classify_pmtiles(pmtiles_dir: Path | None) -> dict[str, Path]:
    """Every per-volume ``*.pmtiles`` archive in a directory, keyed by identifier.

    Zero-byte files are in-progress bakes: skipped.

    ``<volume>-overview.pmtiles`` is skipped: nothing serves one, and calling it
    a volume would mint a phantom whose id no metadata matches."""
    volume_files: dict[str, Path] = {}
    if pmtiles_dir is None or not pmtiles_dir.is_dir():
        return volume_files
    for path in sorted(pmtiles_dir.glob("*.pmtiles")):
        if path.stat().st_size == 0:
            continue
        tail = path.stem.rsplit("-", 1)[-1] if "-" in path.stem else None
        if tail == "overview":
            # said out loud: a volume whose own identifier ends in `-overview`
            # would otherwise leave the map with no diagnostic anywhere
            logger.info("%s: overview companion, served by nothing; skipped", path.name)
        else:
            volume_files[path.stem] = path
    return volume_files
