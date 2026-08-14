"""Golden-testbed expansion, phase 2: annotate + replay + score vs human GT.

SPENDS BUDGET: one model call per uncached small in the `annotate` step.

Follow-on to scripts/fetch_new_goldens.py, over the volunteer-pinned subsets it
froze. Per volume: `annotate` takes one v2 read per small (cached, so a re-run
spends nothing); `replay` runs the full production pipeline
with bounds from the GT layer extents; `score` reports the acceptance funnel,
median grid-RMSE against the human GCPs, and the within-tolerance committed
count — the ref-volume validation pattern.

Annotation lanes for different volumes may run in PARALLEL, since they write
disjoint per-volume directories. Never run two lanes on the SAME volume: the
per-page cache check is not atomic across processes.
"""

import json
import statistics
import sys
from pathlib import Path

from autogeoref.annotate.failures import AnnotationCallError, BudgetLimitError
from autogeoref.annotate.invocation import ClaudeCLIBackend
from autogeoref.bounds import load_ground_truth, volume_bounds
from autogeoref.centerlines import CenterlineIndex
from autogeoref.paths import VolumePaths
from autogeoref.report import build_report, load_results_dir
from autogeoref.score_pass import score_volume
from autogeoref.stages.corroborate import stage_corroborate
from autogeoref.stages.match import stage_match
from autogeoref.stages.report import stage_report
from autogeoref.stages.rescue import stage_rescue, stage_revoke_shared_street_rescues
from autogeoref.stages.seam import stage_seam

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "goldens"
VOLUMES = ["sanborn01790_041", "sanborn01790_089", "sanborn01790_130"]
MODEL = "claude-sonnet-5"

#: Cached one-off Overpass rail extracts per volume (rail rescue anchors);
#: volumes without a covering extract simply run without the channel.
RAIL_EXTRACTS = {
    "sanborn01790_041": ROOT / "work" / "rail-vol22-overpass.json",
}


def slug_prefix_of(gt_path: Path) -> str:
    layers = json.loads(gt_path.read_text())
    slug = str(layers[0]["slug"])
    return slug[: slug.rfind("_p") + 2]


def annotate_volume(vid: str) -> bool:
    """Annotate every small; returns False when halted by a budget limit."""
    vol = OUT / vid
    ann_dir = vol / "annotations"
    ann_dir.mkdir(exist_ok=True)
    backend = ClaudeCLIBackend(model=MODEL)
    smalls = sorted((vol / "sheets").glob("p*_small.jpg"))
    for img in smalls:
        page = img.name.removeprefix("p").removesuffix("_small.jpg")
        out = ann_dir / f"p{page}.json"
        failed = ann_dir / f"p{page}.failed.json"
        if out.exists() or failed.exists():
            continue
        try:
            raw = backend.annotate_extended(img).raw
        except BudgetLimitError as exc:
            print(f"{vid}: BUDGET LIMIT — halting annotation ({exc})", flush=True)
            return False
        except AnnotationCallError as exc:
            print(f"{vid} p{page}: {type(exc).__name__}: {exc}", flush=True)
            failed.write_text(json.dumps({"error": str(exc)}))
            continue
        out.write_text(json.dumps(raw, indent=2))
        print(f"{vid} p{page}: {len(raw.get('streets') or [])} streets", flush=True)
    return True


def replay_volume(vid: str) -> None:
    vol = OUT / vid
    paths = VolumePaths(root=vol)
    gt_path = vol / f"api-layers-{vid}.json"
    # a CANDIDATE volume has no config stanza yet, so its search box comes from
    # the pinned extents. That is this tool's bootstrap and nothing else's: the
    # pipeline never takes bounds from pins (run_inputs.resolve_bounds)
    bounds = volume_bounds(load_ground_truth(gt_path, slug_prefix=slug_prefix_of(gt_path)))
    from autogeoref.names import load_aliases

    aliases_path = ROOT / "configs" / "chicago" / "aliases" / f"aliases-{vid}.json"
    if not aliases_path.exists():
        # bootstrap: geometrically inferred aliases (scripts/infer_aliases.py)
        aliases_path = vol / "aliases-inferred.json"
    aliases = load_aliases(aliases_path if aliases_path.exists() else None)
    index = CenterlineIndex.from_geojson(
        ROOT / "fixtures" / "reference" / "street_center_lines.geojson",
        aliases=aliases,
        bounds_4326=bounds,
    )
    rail_index = None
    rail_path = RAIL_EXTRACTS.get(vid)
    if rail_path is not None and rail_path.exists():
        from autogeoref.rail import RailIndex

        rail_index = RailIndex.from_json(rail_path)
    from autogeoref.config.model import VolumeConfig

    vcfg = VolumeConfig(identifier=vid)  # constants derived two-pass
    stage_match(paths, index, vcfg)
    stage_revoke_shared_street_rescues(paths, aliases)
    _rescued, _provisional = stage_rescue(paths, index, vcfg, rail_index=rail_index)
    stage_seam(paths)
    stage_corroborate(paths)
    # the keystone evidence stages: advisory junction verdicts, then the
    # >=2-independent-verifiers adjudication (all three volumes post-1909)
    from autogeoref.geometry import clip_features_4326
    from autogeoref.verified_accept import stage_verified_accept
    from autogeoref.verify import stage_junction_verify

    features = json.loads(
        (ROOT / "fixtures" / "reference" / "street_center_lines.geojson").read_text()
    )["features"]
    stage_junction_verify(paths, features, bounds)
    stage_verified_accept(
        paths, clip_features_4326(features, bounds), aliases, address_era="modern"
    )
    stage_report(paths, vid)

    # placement is finished and GT-free; NOW grade it. The scorer parses pages
    # with page_from_slug, which DROPS a volunteer's region crops — a crop's
    # pixels are not the page's, and pairing them fabricates a clean-looking fit
    scores = score_volume(paths, vid, [gt_path.parent])["pages"]
    results = load_results_dir(paths.results)
    report = build_report(vid, results, scores={p: e["rmse_vs_human_m"] for p, e in scores.items()})
    rmses = sorted(e["rmse_vs_human_m"] for e in scores.values())
    committed = sum(1 for v in rmses if v <= 15.0)
    print(
        f"\n{vid}: {report.n_sheets} sheets | strict {report.strict_accepted} | "
        f"rescued {report.rescued} | corroborated {report.corroborated} | "
        f"flagged {report.flagged}"
    )
    if rmses:
        print(
            f"{vid}: grid-RMSE vs human GCPs over {len(rmses)} scored accepts: "
            f"median {statistics.median(rmses):.2f} m, p90 "
            f"{rmses[max(0, int(0.9 * len(rmses)) - 1)]:.2f} m; committed (<=15 m): "
            f"{committed}/{len(rmses)}"
        )
    over = [(p, e["rmse_vs_human_m"]) for p, e in scores.items() if e["rmse_vs_human_m"] > 15.0]
    if over:
        print(f"{vid}: accepted-but-over-gate pages (a demotion pass would act on these): {over}")
    summary = {
        "funnel": {
            "n_sheets": report.n_sheets,
            "strict": report.strict_accepted,
            "rescued": report.rescued,
            "corroborated": report.corroborated,
            "verified": report.verified,
            "flagged": report.flagged,
        },
        "rmse_median_m": statistics.median(rmses) if rmses else None,
        "rmse_scored": len(rmses),
        "committed_15m": committed,
    }
    (OUT / f"{vid}_validation.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    volumes = sys.argv[2:] or VOLUMES
    for vid in volumes:
        if not (OUT / vid / "sheets" / "manifest.json").exists():
            print(f"{vid}: not fetched yet, skipping")
            continue
        if mode in ("annotate", "all") and not annotate_volume(vid):
            break
        if mode in ("replay", "all"):
            replay_volume(vid)


if __name__ == "__main__":
    main()
