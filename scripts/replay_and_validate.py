"""Replay cached annotation reads through the matcher, and validate what they win.

Both subcommands are ZERO model spend and read-only on the source work tree.
``replay`` feeds every cached read of a volume's flagged pages back through the
production ``match_sheet``; a page that then clears the full constrained gates is a
placement the pipeline could hold but does not. ``validate`` builds an evidence
dossier per candidate page — margin adjacency against the neighbours a sheet prints
in its own margins, corroboration with committed neighbours, the addresses vote, the
drawn-junction verdict, the implied scale and rotation — and renders a ghost
composite, which is the decisive check and needs human eyes.

The dossier exists because these pages are admitted by a WIDER gate than the one
that has been qualified, so each win needs evidence the matcher did not itself
produce. ``--margins`` is ``{"<volume>:<page>": {"N": "43", ...}}``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# private, but the addresses channel's READERS rather than a producer; if they move,
# update these imports
from autogeoref.address_channel import (
    AddressVoteConfig,
    _numerals_in_source_frame,
    _sidecar_numerals,
    _street_segments,
    addr_tol_numbers,
    address_vote,
)
from autogeoref.addresses import EMPTY_RENUMBERING, RenumberingTable
from autogeoref.affine import TO_3857, TO_4326, apply_affine
from autogeoref.bounds_bootstrap import persisted_bounds
from autogeoref.config.load import load_city_config
from autogeoref.corroborate import corroborations
from autogeoref.era import era_from_config
from autogeoref.geometry import clip_features_4326
from autogeoref.paths import VolumePaths
from autogeoref.review.render import render_ghost_composite
from autogeoref.run_inputs import NoBoundsSourceError, build_index, resolve_bounds
from autogeoref.seam import sheet_fit_from_result
from autogeoref.sheet_inputs import sheet_input_from
from autogeoref.volume import (
    constraints_for_page,
    constraints_from_constants,
    match_sheet,
)
from autogeoref.vouchers import committed_vouch_nodes

ROOT = Path(__file__).resolve().parents[1]

#: Compass bearing (degrees, 0 = east, 90 = north) each margin direction should point.
_BEARINGS = {"E": 0.0, "N": 90.0, "W": 180.0, "S": -90.0}
#: A margin neighbour this far off its expected bearing is a disagreement. Generous
#: because sheet centres are close together and a neighbour's own extent skews the
#: bearing; a misplacement shows up as 90-180 degrees, not 45.
_BEARING_TOL_DEG = 45.0
#: Producers whose cached reads this replays, in the order a page would have used them.
_PRODUCERS = ("annotation", "escalated", "v2")


def _constraints(volume: Path, config: Any) -> Any | None:
    """A volume's scale/rotation window, from its persisted constants or its config."""
    constants = volume / "volume-constants.json"
    if constants.exists():
        raw = json.loads(constants.read_text())
        return constraints_from_constants(float(raw["scale_m_per_px"]), float(raw["rotation_deg"]))
    if config.scale_m_per_px is not None and config.rotation_deg is not None:
        return constraints_from_constants(config.scale_m_per_px, config.rotation_deg)
    return None


class VolumeContext:
    """Everything a volume needs for a replay or a dossier, built once."""

    def __init__(self, city: Any, volume: Path, fixtures: Path, features: list[dict[str, Any]]):
        self.volume = volume
        self.city = city
        self.config = city.volume(volume.name)
        self.constraints = _constraints(volume, self.config)
        try:
            self.bounds = resolve_bounds(city, self.config, fixtures / "viewer-manifest.json")
        except NoBoundsSourceError:
            # replay the box the finished run actually used, not one derived here
            persisted = persisted_bounds(VolumePaths(root=volume))
            if persisted is None:
                raise
            # ANNOUNCED: bounds is a hard membership filter, and this box is not
            # the pin-derived one earlier replays used — neither contains the
            # other. A human commits a placement from this dossier.
            print(f"{volume.name}: bounds from the persisted bootstrap")
            self.bounds = persisted
        self.clipped = clip_features_4326(features, self.bounds)
        self.index = build_index(city, self.config, self.bounds, features)
        self.manifest = json.loads((volume / "sheets" / "manifest.json").read_text())
        self.era = era_from_config(
            self.config.addresses_modern,
            volume=volume.name,
            city_renumbered=city.renumbering_table_path is not None,
        )
        table_path = self.config.renumbering_table_path or city.renumbering_table_path
        self.renumbering = (
            RenumberingTable.from_json(table_path)
            if self.era == "renumbered" and table_path
            else EMPTY_RENUMBERING
        )
        self._segments: dict[str, Any] | None = None

    @property
    def segments(self) -> dict[str, Any]:
        if self._segments is None:
            self._segments = _street_segments(
                self.clipped,
                self.index.aliases,
                self.city.centerline_name_property,
                self.city.centerline_type_property,
            )
        return self._segments

    def result(self, page: str) -> dict[str, Any] | None:
        path = self.volume / "results" / f"p{page}.json"
        if not path.exists():
            return None
        record: dict[str, Any] = json.loads(path.read_text())
        return record

    def info(self, page: str) -> dict[str, Any] | None:
        entry: dict[str, Any] | None = self.manifest.get(f"p{page}")
        return entry

    def centre_3857(self, page: str, record: dict[str, Any] | None = None) -> Any:
        """Placement centre of a page, from its record or a supplied replay."""
        info = self.info(page)
        record = record or self.result(page)
        if info is None or record is None:
            return None
        fit = sheet_fit_from_result(page, record)
        if fit is None:
            return None
        w, h = float(info["full_size"][0]), float(info["full_size"][1])
        pts = [apply_affine(fit.coef, x, y) for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
        return (sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4)


def _cached_reads(annotations: Path, page: str) -> list[tuple[str, Path]]:
    """``(producer, path)`` for every SUCCESSFUL cached read of one page.

    ``p<N>.annotation.active.json`` is a pointer, not a reading, and failure markers
    are skipped — deleting a marker is the deliberate gesture that re-spends.

    The primary read has TWO spellings on disk, the current
    ``p<N>.annotation.<key>.json`` and the legacy bare ``p<N>.json``. Both must be
    found, or the primary-vs-second attribution reports "no primary cache" for every
    page in an older volume and the comparison is vacuous.
    """
    out: list[tuple[str, Path]] = []
    legacy = annotations / f"p{page}.json"
    if legacy.exists():
        out.append(("annotation", legacy))
    for producer in _PRODUCERS:
        for path in sorted(annotations.glob(f"p{page}.{producer}.*.json")):
            key = path.name[len(f"p{page}.{producer}.") : -len(".json")]
            if key == "active" or key.endswith(".failed"):
                continue
            out.append((producer, path))
    return out


def replay(ctx: VolumeContext, only_flagged: bool = True) -> list[dict[str, Any]]:
    """Re-run the matcher over every cached read of the volume's flagged pages."""
    rows: list[dict[str, Any]] = []
    for path in sorted((ctx.volume / "results").glob("p*.json")):
        page = path.name[1 : -len(".json")]
        record = json.loads(path.read_text())
        status = str(record.get("status", ""))
        if only_flagged and not status.startswith("REJECTED"):
            continue
        info = ctx.info(page)
        if info is None or ctx.constraints is None:
            continue
        for producer, cache in _cached_reads(ctx.volume / "annotations", page):
            try:
                out = match_sheet(
                    sheet_input_from(page, json.loads(cache.read_text()), info),
                    ctx.index,
                    constraints_for_page(page, ctx.constraints, ctx.config.page_scale_multiples),
                    ctx.index.aliases,
                )
            except Exception as exc:  # a replay failure is a measurement gap, not a crash
                rows.append({"page": page, "producer": producer, "error": repr(exc)[:120]})
                continue
            rows.append(
                {
                    "volume": ctx.volume.name,
                    "page": page,
                    "producer": producer,
                    "cache": cache.name,
                    "current_status": status,
                    "replay_status": out.get("status"),
                    "strict": str(out.get("status", "")).startswith("OK"),
                    "primary_n_candidates": record.get("n_candidates"),
                    "replay_n_candidates": out.get("n_candidates"),
                    "replay_n_inliers": out.get("n_inliers"),
                }
            )
    return rows


def dossier(
    ctx: VolumeContext, page: str, record: dict[str, Any], margins: dict[str, str], out_dir: Path
) -> dict[str, Any]:
    """Independent evidence for one candidate placement, plus its ghost composite."""
    info = ctx.info(page)
    fit = sheet_fit_from_result(page, record)
    entry: dict[str, Any] = {"volume": ctx.volume.name, "page": page}
    if info is None or fit is None:
        entry["error"] = "no manifest entry or no fit"
        return entry
    current = ctx.result(page) or {}
    entry["current_status"] = current.get("status")
    entry["replay_status"] = record.get("status")
    entry["n_inliers"] = record.get("n_inliers")

    centre = ctx.centre_3857(page, record)
    entry["centre_4326"] = [round(c, 6) for c in TO_4326.transform(*centre)]
    existing = ctx.centre_3857(page)
    if existing is not None:
        entry["shift_from_current_m"] = round(
            math.hypot(centre[0] - existing[0], centre[1] - existing[1]), 1
        )

    linear = np.array([[fit.coef[0][1], fit.coef[0][2]], [fit.coef[1][1], fit.coef[1][2]]])
    entry["implied_scale_m_per_px"] = round(float(np.sqrt(abs(np.linalg.det(linear)))), 5)
    entry["implied_rotation_deg"] = round(
        float(np.degrees(math.atan2(linear[1, 0], linear[0, 0]))), 3
    )

    # margin adjacency: the sheet's own printed neighbour index vs measured bearings
    checks: list[dict[str, Any]] = []
    for direction, neighbour in (margins or {}).items():
        other = ctx.centre_3857(str(neighbour))
        if other is None:
            checks.append({"dir": direction, "page": neighbour, "verdict": "unplaced"})
            continue
        bearing = math.degrees(math.atan2(other[1] - centre[1], other[0] - centre[0]))
        err = abs((bearing - _BEARINGS[direction] + 180) % 360 - 180)
        checks.append(
            {
                "dir": direction,
                "page": neighbour,
                "bearing_err_deg": round(err, 1),
                "distance_m": round(math.hypot(other[0] - centre[0], other[1] - centre[1])),
                "verdict": "agrees" if err <= _BEARING_TOL_DEG else "DISAGREES",
            }
        )
    checked = [c for c in checks if "bearing_err_deg" in c]
    entry["margin_adjacency"] = {
        "checked": len(checked),
        "agree": sum(1 for c in checked if c["verdict"] == "agrees"),
        "detail": checks,
    }

    # corroboration with committed neighbours (shared centerline nodes)
    hits = corroborations(fit, committed_vouch_nodes(VolumePaths(root=ctx.volume)))
    best: dict[Any, float] = {}
    for key, _pg, dist in hits:
        best[key] = min(dist, best.get(key, dist))
    distances = sorted(best.values())
    entry["corroboration"] = {
        "shared_nodes": len(best),
        "within_8m": sum(1 for d in distances if d <= 8),
        "best_m": [round(d, 1) for d in distances[:6]],
    }

    # addresses at THIS placement, from readings already on disk
    per_model = {
        model: _numerals_in_source_frame(numerals, info)
        for model, numerals in _sidecar_numerals(ctx.volume / "annotations", page).items()
    }
    vote, detail = address_vote(
        per_model,
        fit.coef,
        1.0 / float(info["scale"]),
        ctx.clipped,
        ctx.index.aliases,
        ctx.era,
        ctx.renumbering if ctx.era == "renumbered" else None,
        segments_by_street=ctx.segments,
        config=AddressVoteConfig(
            addr_tol=addr_tol_numbers(ctx.city.address_block_size),
            name_property=ctx.city.centerline_name_property,
            type_property=ctx.city.centerline_type_property,
            address_block_size=ctx.city.address_block_size,
        ),
    )
    entry["addresses"] = {
        "vote": vote,
        "votable": detail.get("votable"),
        "in_block": detail.get("in_block"),
        "models": detail.get("models"),
    }

    # drawn-junction constellation
    from autogeoref.junction_snap import (
        JunctionSnapError,
        extract_junctions,
        extraction_in_source_frame,
        verify_placement,
        world_from_centerlines,
    )

    try:
        minx, miny = TO_3857.transform(ctx.bounds[0], ctx.bounds[1])
        maxx, maxy = TO_3857.transform(ctx.bounds[2], ctx.bounds[3])
        world = world_from_centerlines(ctx.clipped, bounds_3857=(minx, miny, maxx, maxy))
        extraction = extraction_in_source_frame(
            extract_junctions(ctx.volume / "sheets" / str(info["file"])),
            int(info.get("rotation_applied", 0)),
        )
        verdict = verify_placement(
            extraction, fit.coef, world, small_to_full=1.0 / float(info["scale"])
        )
        entry["junction"] = {
            "supports": verdict.supports,
            "n_junctions": extraction.n_junctions,
            "best_offset_m": round(verdict.best_offset_m, 1),
        }
    except JunctionSnapError as exc:
        # expected on this population: a label-rich, corridor-poor sheet extracts none
        entry["junction"] = {"error": str(exc)[:100]}

    # ghost composite of THIS placement, rendered from a scratch copy so the source
    # work tree is never written
    scratch = Path(tempfile.mkdtemp(prefix=f"ghost-{ctx.volume.name}-{page}-"))
    try:
        (scratch / "results").mkdir(parents=True)
        (scratch / "sheets").mkdir(parents=True)
        (scratch / "results" / f"p{page}.json").write_text(json.dumps(record, indent=2))
        (scratch / "sheets" / "manifest.json").write_text(json.dumps(ctx.manifest))
        shutil.copy2(
            ctx.volume / "sheets" / str(info["file"]), scratch / "sheets" / str(info["file"])
        )
        render_ghost_composite(
            VolumePaths(root=scratch),
            ctx.volume.name,
            page,
            Path(ctx.city.centerlines_path),
            out_dir,
        )
        entry["ghost"] = str(out_dir / ctx.volume.name / f"p{page}_qa.jpg")
    except Exception as exc:
        entry["ghost"] = f"render failed: {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return entry


def _best_strict_replay(ctx: VolumeContext, page: str) -> dict[str, Any] | None:
    """The first cached read of a page whose replay clears the gates."""
    info = ctx.info(page)
    if info is None or ctx.constraints is None:
        return None
    for _producer, cache in _cached_reads(ctx.volume / "annotations", page):
        out = match_sheet(
            sheet_input_from(page, json.loads(cache.read_text()), info),
            ctx.index,
            constraints_for_page(page, ctx.constraints, ctx.config.page_scale_multiples),
            ctx.index.aliases,
        )
        if str(out.get("status", "")).startswith("OK"):
            return out
    return None


_MARGINS_HELP = (
    '{"<volume>:<page>": {"N": "43", "W": "42", ...}} — the neighbour page numbers '
    "printed in the sheet's own margins, read off the scan (they cannot be derived). "
    'Beware division suffixes: in a multi-division city a printed "70w" is the WEST '
    "division's page 70, not this volume's p70, and comparing the two fails falsely."
)

_EPILOG = """examples:
  uv run python scripts/replay_and_validate.py replay \\
      --volume <vid> --city configs/chicago/chicago.toml
  uv run python scripts/replay_and_validate.py validate \\
      --pages <vid>:1,11,52 --city configs/chicago/chicago.toml \\
      --out /tmp/validate --margins margins.json
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=("replay", "validate"))
    parser.add_argument("--city", type=Path, default=ROOT / "configs/chicago/chicago.toml")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--volume", action="append", help="volume id (repeatable); replay mode")
    parser.add_argument("--pages", action="append", help="<volume>:<p1>,<p2>; validate mode")
    parser.add_argument("--margins", type=Path, help=_MARGINS_HELP)
    parser.add_argument("--out", type=Path, default=Path("replay-validate-out"))
    args = parser.parse_args()

    city = load_city_config(args.city)
    features = json.loads(Path(city.centerlines_path).read_text())["features"]
    args.out.mkdir(parents=True, exist_ok=True)
    margins = json.loads(args.margins.read_text()) if args.margins else {}

    if args.mode == "replay":
        volumes = args.volume or [p.name for p in sorted(args.work.glob("sanborn*"))]
        rows: list[dict[str, Any]] = []
        for name in volumes:
            ctx = VolumeContext(city, args.work / name, args.fixtures, features)
            rows.extend(replay(ctx))
        strict = [r for r in rows if r.get("strict")]
        pages = sorted({(r["volume"], r["page"]) for r in strict})
        # THE CONFOUND, and why this split is printed rather than left to be noticed:
        # a replay runs against TODAY's index. On a volume whose alias file changed
        # since its last run, a cached read places pages the original run could not —
        # that is alias/index drift, which a spend-free re-run already collects. Only
        # a page whose PRIMARY read still fails, while a second read succeeds, is
        # evidence that a re-read was needed.
        by_page: dict[tuple[str, str], set[str]] = {}
        for row in strict:
            by_page.setdefault((row["volume"], row["page"]), set()).add(row["producer"])
        drift = sorted(k for k, v in by_page.items() if "annotation" in v)
        second = sorted(k for k, v in by_page.items() if "annotation" not in v)
        (args.out / "replay.json").write_text(json.dumps(rows, indent=2))
        print(f"replayed {len(rows)} cached reads over {len(volumes)} volume(s)")
        print(f"reads whose replay clears the gates: {len(strict)}")
        print(f"DISTINCT still-flagged pages a cached read would place: {len(pages)}")
        print("  alias/index drift — the PRIMARY read places it under today's index,")
        print(f"  so a spend-free re-run already collects it: {len(drift)}")
        for volume, page in drift:
            print(f"     {volume} p{page}")
        print(f"  genuine second-read wins — the primary still fails: {len(second)}")
        for volume, page in second:
            print(f"     {volume} p{page}")
        print(f"wrote {args.out / 'replay.json'}")
        return

    entries: list[dict[str, Any]] = []
    for spec in args.pages or []:
        name, _, pagelist = spec.partition(":")
        ctx = VolumeContext(city, args.work / name, args.fixtures, features)
        for page in pagelist.split(","):
            record = _best_strict_replay(ctx, page)
            if record is None:
                entries.append({"volume": name, "page": page, "error": "no cached read places it"})
                continue
            entries.append(dossier(ctx, page, record, margins.get(f"{name}:{page}", {}), args.out))
    (args.out / "dossier.json").write_text(json.dumps(entries, indent=2, default=str))
    for entry in entries:
        adjacency = entry.get("margin_adjacency") or {}
        print(
            f"{entry['volume']} p{entry['page']}: "
            f"margins {adjacency.get('agree')}/{adjacency.get('checked')}  "
            f"corrob<=8m {(entry.get('corroboration') or {}).get('within_8m')}  "
            f"ghost {entry.get('ghost')}"
        )
    print(f"wrote {args.out / 'dossier.json'} — now LOOK at every ghost composite")


if __name__ == "__main__":
    main()
