"""Does ``fixtures/<vid>/`` actually hold volume ``<vid>``?

Every "we beat the origin baseline" claim reads ``fixtures/<vid>/results/`` on
the assumption that the directory name identifies the volume. Nothing checks
it: the fixture-integrity test verifies the tree has not DRIFTED, a different
question from whether it was filed under the right id at all.

The check is exact and free: ``sheets/p<N>_small.jpg`` is a prepped scan of one
physical sheet, so hashing it against the ``work/`` counterpart settles it with
no interpretation. Where the small scans are missing, annotation street-name
sets are compared instead — weaker, but two volumes share almost no street set.
Read-only. Zero model spend, zero network.

    uv run python scripts/audit_fixture_volume_identity.py --work work
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: below this share of matching pages a tree is reported as MISFILED
MATCH_FLOOR = 0.5


def _sheet_hashes(root: Path) -> dict[str, str]:
    d = root / "sheets"
    if not d.is_dir():
        return {}
    return {
        f.name.removesuffix("_small.jpg").removeprefix("p"): hashlib.md5(f.read_bytes()).hexdigest()
        for f in sorted(d.glob("p*_small.jpg"))
    }


def _street_sets(root: Path) -> dict[str, frozenset[str]]:
    d = root / "annotations"
    if not d.is_dir():
        return {}
    out: dict[str, frozenset[str]] = {}
    for f in sorted(d.glob("p*.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        names = frozenset(s.get("name", "") for s in rec.get("streets") or [])
        if names:
            out[f.stem.removeprefix("p")] = names
    return out


def _compare(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, int]:
    shared = sorted(set(a) & set(b))
    return sum(1 for p in shared if a[p] == b[p]), len(shared)


def audit(work: Path, fixtures: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fixture_vols = sorted(
        d.name for d in fixtures.iterdir() if d.is_dir() and (d / "results").is_dir()
    )
    for vid in fixture_vols:
        froot, wroot = fixtures / vid, work / vid
        row: dict[str, Any] = {"volume": vid, "work_tree": wroot.is_dir()}
        if not wroot.is_dir():
            row["verdict"] = "no work counterpart — not checkable here"
            rows.append(row)
            continue

        fh, wh = _sheet_hashes(froot), _sheet_hashes(wroot)
        same, shared = _compare(fh, wh)
        method = "sheet image md5"
        if not shared:
            fs, ws = _street_sets(froot), _street_sets(wroot)
            same, shared = _compare(fs, ws)
            method = "annotation street sets"
        row.update(
            {
                "method": method,
                "pages_shared": shared,
                "pages_identical": same,
                "match_rate": round(same / shared, 3) if shared else None,
            }
        )
        if not shared:
            row["verdict"] = "no comparable pages"
        elif same / shared >= MATCH_FLOOR:
            row["verdict"] = "OK"
        else:
            row["verdict"] = "MISFILED"
            # name the volume it actually holds, if one of the others matches
            for other in sorted(d.name for d in work.iterdir() if d.is_dir()):
                if other == vid:
                    continue
                oh = _sheet_hashes(work / other)
                s2, n2 = _compare(fh or _street_sets(froot), oh or _street_sets(work / other))
                if n2 and s2 / n2 >= MATCH_FLOOR:
                    row["actually_holds"] = other
                    row["actually_holds_match"] = f"{s2}/{n2}"
                    break
        rows.append(row)
    return {
        "match_floor": MATCH_FLOOR,
        "volumes": rows,
        "misfiled": [r["volume"] for r in rows if r.get("verdict") == "MISFILED"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = audit(args.work, args.fixtures)
    for r in report["volumes"]:
        rate = f"{r['match_rate']:.0%}" if r.get("match_rate") is not None else "  - "
        extra = (
            f"  -> actually holds {r['actually_holds']} ({r['actually_holds_match']})"
            if r.get("actually_holds")
            else ""
        )
        print(
            f"{r['volume']:22s} {r['verdict']:12s} {rate:>5s} "
            f"({r.get('pages_identical', '-')}/{r.get('pages_shared', '-')} pages, "
            f"{r.get('method', '-')}){extra}"
        )
    print(f"\nmisfiled: {report['misfiled'] or 'none'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
