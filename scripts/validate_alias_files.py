"""Validate every alias file on disk against its volume's index.

The structural rules live in ``autogeoref.alias.validate`` — one implementation,
shared with the automated sweep, which must abort a volume rather than write a
table that fails them. This adds the two things a corpus regression needs and
the library has no business holding: DISCOVERY of every ``aliases-*.json`` in
the city's alias directory, so a new table is validated the day it lands with
no edit here; and the NORMALIZER-CONTRACT cases below — reads a file MUST catch
and reads it must NOT catch — which pin the end-to-end behaviour the
structural rules only approximate.
Each file is read through the real loader and checked against an ALIAS-FREE
bounded index: the ground the aliases must land on. A volume with no declared
and no persisted bounds is reported and skipped, not failed — a fresh checkout
has neither. Zero model calls, zero network; needs a populated ``work/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from autogeoref.alias.validate import (
    alias_files,
    is_inert,
    redirects,
    validate_table,
    volume_of,
)
from autogeoref.bounds_bootstrap import persisted_bounds
from autogeoref.centerlines import CenterlineIndex
from autogeoref.config.load import load_city_config
from autogeoref.names import load_aliases, normalize
from autogeoref.paths import VolumePaths
from autogeoref.run_inputs import NoBoundsSourceError, resolve_bounds
from autogeoref.viewer.layout import city_manifest

#: A COUNTDOWN, not a feature. Every line is a real violation already baked
#: into a committed run, listed by exact message with the repair it waits on.
#: Nothing by-design belongs here: an entry the rules should not have objected
#: to is a rule to fix. Anything else, and any CHANGE to one of these, fails;
#: an excuse for a violation that no longer occurs is stale and fails too.
PRE_EXISTING: dict[tuple[str, str], str] = {
    ("sanborn01790_006.5", "'ELIZABETH': KEY SHADOWS AN IN-BOUNDS STREET (twin)"): (
        "SUSPECTED DEFECT, not a pattern — owner-gated fixture correction"
    ),
    (
        "sanborn01790_006.5",
        "centerline 'ELIZABETH' RE-KEYED to 'CLYDE', joining ['CLYDE']",
    ): "SUSPECTED DEFECT, not a pattern — owner-gated fixture correction",
}

# (volume, annotated read, expected key after aliasing) — expected None means
# the alias table must not change the read's outcome at all (court-guarded,
# deliberately unaliased, or a direct index match the table must not touch).
CONTRACT: list[tuple[str, str, str | None]] = [
    ("sanborn01790_037", "N. 40TH AV.", "PULASKI"),
    ("sanborn01790_037", "S. 40TH AV.", "PULASKI"),
    ("sanborn01790_037", "N. FORTIETH", "PULASKI"),
    ("sanborn01790_037", "N. 41 ST. AV.", "KARLOV"),
    ("sanborn01790_037", "N. 41 ST. CT.", None),  # court guard: '41' never aliased
    ("sanborn01790_037", "N. 42ND CT.", "TRIPP"),
    ("sanborn01790_037", "N. 42ND. CT.", "TRIPP"),
    ("sanborn01790_037", "S. 43RD CT.", "KOLIN"),
    ("sanborn01790_037", "S. 40TH CT.", None),  # Komensky out of bounds
    ("sanborn01790_037", "N. 43RD CT.", None),  # Lowell out of bounds
    ("sanborn01790_037", "N. 51ST AV.", "LECLAIRE"),
    ("sanborn01790_037", "S. 51ST AV.", "LEAMINGTON"),
    ("sanborn01790_037", "N. 51ST CT.", "LEAMINGTON"),
    ("sanborn01790_037", "S. 51ST CT.", None),  # S range undocumented
    ("sanborn01790_037", "N. 52D AV.", "LARAMIE"),
    # re-pointed: the landing control put every independent read on these
    # streets and none on the targets they replace
    ("sanborn01790_037", "PARK AV.", "MAYPOLE"),
    ("sanborn01790_037", "N. PARK AV.", "PARKSIDE"),
    ("sanborn01790_037", "N. PARK", "PARKSIDE"),
    ("sanborn01790_037", "S. PARK AV.", None),  # no documented range
    ("sanborn01790_037", "PARK PL.", None),  # court guard
    ("sanborn01790_037", "SOUTH BOULEVARD", "CORCORAN"),
    ("sanborn01790_037", "PHILADELPHIA PL.", "CARROLL"),
    ("sanborn01790_037", "W. INDIANA", "FERDINAND"),
    ("sanborn01790_025", "W. 22ND ST.", "CERMAK"),
    ("sanborn01790_025", "W. 20TH ST.", "CULLERTON"),
    ("sanborn01790_025", "W. 12TH ST.", "ROOSEVELT"),
    ("sanborn01790_025", "W. 22ND PL.", None),  # numbered-PL twin matches directly
    ("sanborn01790_025", "JOHN PL.", "22ND PL"),
    ("sanborn01790_025", "O'NEIL", "23RD"),
    ("sanborn01790_025", "(JOHNSON)", None),  # normalizes to ''
    ("sanborn01790_022", "N. ROBEY", "DAMEN"),
    ("sanborn01790_022", "McHENRY", "THROOP"),
    ("sanborn01790_022", "MC HENRY", "THROOP"),
    ("sanborn01790_022", "ALICE PL.", "CONCORD"),
    ("sanborn01790_019", "THE LAKE SHORE DRIVE", "LAKE SHORE"),
    ("sanborn01790_019", "HAMILTON CT.", "GENEVA"),
    ("sanborn01790_019", "HIGH", "JANSSEN"),  # the Clybourn-corridor High St
    ("sanborn01790_019", "MARCY", "MARCEY"),
    ("sanborn01790_019", "MARCY St.", "MARCEY"),
    ("sanborn01790_019", "NURSERY", None),  # erased from the modern grid
    ("sanborn01790_019", "GROVE PL.", None),  # court guard + Shakespeare too far
    ("sanborn01790_019", "WASHINGTON PL.", "DELAWARE"),
    ("sanborn01790_039", "W. 12TH ST.", "ROOSEVELT"),
    ("sanborn01790_039", "W. TWELFTH", "ROOSEVELT"),
    ("sanborn01790_039", "W. 20TH ST.", "CULLERTON"),
    ("sanborn01790_039", "W. 22ND ST.", "CERMAK"),
    # court guard: the W-qualified numbered keys must not reach the PL twins
    # (neither PL is in this volume's index, so both stay unmatched — the
    # assertion is that the table does not map them onto 22ND/12TH either)
    ("sanborn01790_039", "W. 22ND PL.", None),
    ("sanborn01790_039", "W. 12TH PL.", None),  # 14TH also refuted on-sheet
    ("sanborn01790_039", "S. CRAWFORD AV. (S. 40TH AV.)", "PULASKI"),
    ("sanborn01790_039", "S. 40TH AV.", "PULASKI"),
    ("sanborn01790_039", "COLORADO AV.", "FIFTH"),
    ("sanborn01790_039", "S. MANSFIELD AV.", "MONITOR"),
    # the neighbouring town's numbered grid stays unmatched: right mapping, no geometry
    ("sanborn01790_039", "S. 46TH AV.", None),
    ("sanborn01790_039", "S. 48TH AV.", None),
    ("sanborn01790_039", "S. 51ST AV.", None),
    ("sanborn01790_039", "S. 51ST CT.", None),
    ("sanborn01790_039", "S. 55TH AV.", None),
    ("sanborn01790_039", "W. 29TH ST.", None),  # one-block error vs 28TH
    ("sanborn01790_039", "W. 15TH PL.", None),  # one-block error vs 16TH
    ("sanborn01790_083", "W. CENTER", "ARMITAGE"),
    ("sanborn01790_083", "W. GARFIELD AV.", "DICKENS"),
    # the suffix loop reaches the key through two suffixes and a direction
    ("sanborn01790_083", "W. GARFIELD AV. BLVD.", "DICKENS"),
    ("sanborn01790_083", "N. MARCY", "MARCEY"),
    ("sanborn01790_083", "N. HIGH", "JANSSEN"),
    ("sanborn01790_083", "N. HAMMOND", "ORLEANS"),
    ("sanborn01790_083", "W. CUSTER", "SHAKESPEARE"),
    ("sanborn01790_083", "N. LASALLE", "LA SALLE"),
    ("sanborn01790_083", "W. HUBER PL.", "MEDILL"),  # full-string place key
    # held: refuted on this volume's own sheets and carried by its neighbour
    ("sanborn01790_083", "N. SMITH AV.", None),
    ("sanborn01790_083", "W. COURTLAND", None),  # held: one-sheet locality
    ("sanborn01790_084", "W. REES", "EVERGREEN"),
    ("sanborn01790_084", "W. SIGEL", "EVERGREEN"),
    ("sanborn01790_084", "W. SIEGEL", None),  # Martin's rival reading, left held
    ("sanborn01790_084", "W. VEDDER", "SCOTT"),
    # both printed directions reach the one documented value
    ("sanborn01790_084", "N. VEDDER AV.", "SCOTT"),
    ("sanborn01790_084", "N. SMITH AV.", "FREMONT"),
    ("sanborn01790_084", "W. CARL", "BURTON"),
    ("sanborn01790_084", "N. CHATHAM CT.", "HOWE"),  # full-string court key
    ("sanborn01790_084", "W. BEETHOVEN PL.", "SCOTT"),
    ("sanborn01790_084", "W. HEIN PL.", "GOETHE"),
    ("sanborn01790_084", "N. ROBERTS", None),  # held: one-sheet locality
    ("sanborn01790_017", "5TH AV.", "WELLS"),
    ("sanborn01790_017", "CUSTOM HOUSE C'T", "FEDERAL"),
    ("sanborn01790_017", "PECK PL.", "8TH"),
    ("sanborn01790_017", "LASALLE", "LA SALLE"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--city", type=Path, default=Path("configs/chicago/chicago.toml"))
    ap.add_argument(
        "--viewer-manifest",
        type=Path,
        default=None,
        help="default: viewer/<city-slug>/manifest.json",
    )
    args = ap.parse_args()

    logging.getLogger().setLevel(logging.ERROR)

    city = load_city_config(args.city)
    features = json.loads(city.centerlines_path.read_text())["features"]
    declared = args.viewer_manifest or city_manifest(city.name)
    viewer_manifest = declared if declared.is_file() else None

    files = alias_files(city.aliases_dir)
    if not files:
        # Curating no aliases is a real state and passes: a city whose reads
        # already resolve needs no table, and one whose table the normalizer
        # made redundant should be able to delete it and stay green. A
        # directory that is not THERE is a different thing — a typo in
        # aliases_dir drops every table a city has, silently, and this is the
        # one place that would notice.
        if not city.aliases_dir.is_dir():
            print(f"aliases_dir does not exist: {city.aliases_dir}")
            return 1
        print(f"no alias files under {city.aliases_dir} — nothing to validate")
        return 0

    failures = 0
    tables: dict[str, dict[str, str]] = {}
    seen_problems: dict[str, list[str]] = {}
    skipped: list[str] = []
    for path in files:
        volume = volume_of(path)
        vol = city.volume(volume)
        paths = VolumePaths(root=args.work / volume)
        bounds: tuple[float, float, float, float] | None
        try:
            bounds = resolve_bounds(city, vol, viewer_manifest)
        except NoBoundsSourceError:
            bounds = persisted_bounds(paths)
        aliases = load_aliases(path)
        tables[volume] = aliases
        if not aliases:
            print(f"{volume}: alias file has no entries ({path})")
            failures += 1
            continue
        if bounds is None:
            # Not checked is not the same as clean, and it is certainly not
            # "these excuses are stale": a checkout without this volume's
            # bounds cannot judge its table either way.
            print(f"{volume}: {len(aliases)} entries, NO BOUNDS on this checkout — skipped")
            skipped.append(volume)
            continue
        # alias-free index: the ground the aliases must land on
        index = CenterlineIndex(
            features,
            aliases={},
            bounds_4326=bounds,
            name_property=city.centerline_name_property,
            type_property=city.centerline_type_property,
        )
        problems = validate_table(
            aliases,
            index,
            city.centerline_name_property,
            city.centerline_type_property,
        )
        inert = sum(1 for key in aliases if is_inert(key, aliases))
        redirected = redirects(aliases, index)
        seen_problems[volume] = problems
        fresh = [p for p in problems if (volume, p) not in PRE_EXISTING]
        excused = [p for p in problems if (volume, p) in PRE_EXISTING]
        suffix = f" — {len(fresh)} FAILURE(S)" if fresh else ""
        if excused:
            suffix += f" [{len(excused)} pre-existing, excused]"
        if redirected:
            suffix += f" [{len(redirected)} redirect(s), see below]"
        if inert:
            # Not a failure: an entry that cannot change an outcome cannot
            # break one either. It is dead weight, and only a reader can decide
            # whether the table is still worth carrying.
            suffix += f" [{inert} inert]"
        print(f"{volume}: {len(aliases)} entries{suffix}")
        for line in redirected:
            print(f"  note: {line}")
        for problem in fresh:
            print(f"  {problem}")
        for problem in excused:
            print(f"  pre-existing: {problem} — {PRE_EXISTING[volume, problem]}")
        failures += len(fresh)

    # Both recorded lists are keyed by volume, and a volume belongs to exactly
    # one city. Without this guard a second city reports every other city's row
    # as a failure of its own.
    for volume, read, expected in CONTRACT:
        if volume not in city.volumes:
            continue
        if volume not in tables:
            print(f"CONTRACT {volume} {read!r}: no alias file for this volume")
            failures += 1
            continue
        got = normalize(read, tables[volume])
        if expected is None:
            bare = normalize(read)
            if got != bare:
                print(f"CONTRACT {volume} {read!r}: table changed {bare!r} -> {got!r}")
                failures += 1
        elif got != expected:
            print(f"CONTRACT {volume} {read!r}: expected {expected!r}, got {got!r}")
            failures += 1

    stale = sorted(
        f"{volume} {message}"
        for (volume, message) in PRE_EXISTING
        if volume in city.volumes
        and volume not in skipped
        and (volume not in tables or message not in seen_problems.get(volume, ()))
    )
    if skipped:
        print(
            f"\n{len(skipped)} volume(s) not checked (no bounds on this checkout): "
            f"{', '.join(skipped)}. Point --work at a populated tree to check them."
        )
    if stale:
        # An excuse for a violation that no longer occurs is an excuse that
        # would silently cover a DIFFERENT one later.
        print(f"\n{len(stale)} stale PRE_EXISTING entr(y/ies) — delete them:")
        for line in stale:
            print(f"  {line}")
        failures += len(stale)
    print(f"\n{'FAIL' if failures else 'OK'}: {failures} failure(s) over {len(files)} file(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
