"""One writer at a time, and a sidecar is untrusted input until it is validated.

Apply rewrites results and masks, so it refuses to interleave with a run, a prep
or a bake that owns the same tree — and so does save. A persisted sidecar naming
another volume, another page, or a path outside the tree is skipped and touches
nothing. Two writers racing a barrier serialize, and neither edit is stranded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.paths import VolumePaths
from autogeoref.review.app import review_queue
from autogeoref.review.apply import apply_reviews
from autogeoref.review.sidecars import (
    result_sha256 as sha_of,
)
from autogeoref.review.sidecars import (
    sidecar_path,
)
from autogeoref.volume import (
    STATUS_REVIEWER_VERIFIED,
)
from review_support import (
    PENTAGON_PX,
    composed_affine,
    look,
    make_app,
    make_sidecar_dict,
    make_volume,
    ops_translate,
    save_ui_sidecar,
    sidecar_with_mask,
)


def test_apply_is_refused_while_another_operation_owns_the_volume(tmp_path: Path) -> None:
    """`review --apply` rewrites results and masks, so it must not interleave
    with a run/prep/bake that owns the same tree — and must work once released."""
    from autogeoref.paths import VolumeBusyError, volume_lock

    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    with volume_lock(paths, operation="run --warp-only"):
        with pytest.raises(VolumeBusyError, match="is busy"):
            apply_reviews(paths, "volX", do_warp=False)
        # nothing was materialized under the refusal
        r = json.loads((paths.results / "p4.json").read_text())
        assert r.get("status") != STATUS_REVIEWER_VERIFIED
    summary = apply_reviews(paths, "volX", do_warp=False)  # released: same apply lands
    assert summary["applied"] == ["4"]


def test_cli_apply_says_busy_and_fails_instead_of_materializing(tmp_path: Path) -> None:
    from autogeoref.cli.entry import main
    from autogeoref.paths import volume_lock

    root = Path(__file__).resolve().parent.parent
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        "[city]\n"
        'name = "X"\n'
        f'aliases_dir = "{root / "configs" / "chicago" / "aliases"}"\n'
        f'centerlines = "{root / "fixtures" / "reference" / "street_center_lines.geojson"}"\n'
    )
    work = tmp_path / "work"
    paths = VolumePaths(root=work / "volX")
    make_volume(work)  # make_volume builds under <root>/volX
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    with volume_lock(paths, operation="run"):
        rc = main(["review", "--city", str(cfg), "--work", str(work), "--apply", "--no-warp"])
    assert rc == 1
    r = json.loads((paths.results / "p4.json").read_text())
    assert r.get("status") != STATUS_REVIEWER_VERIFIED


def test_apply_skips_forged_traversal_sidecar_and_touches_nothing_outside(tmp_path: Path) -> None:
    """A persisted sidecar is untrusted input: a forged page id must not be
    interpolated into any path — no sentinel outside the volume changes."""
    paths = make_volume(tmp_path)
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("untouched")
    forged = make_sidecar_dict(page="../../sentinel", verdict="accept", mask_px=None)
    p = paths.root / "review" / "p4.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(forged))
    before = {f: f.read_text() for f in paths.results.glob("*.json")}
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["applied"] == []
    assert any("invalid sidecar" in w for w in summary["warnings"])
    assert sentinel.read_text() == "untouched"
    assert not (tmp_path / "results").exists()  # nothing materialized above the volume
    assert {f: f.read_text() for f in paths.results.glob("*.json")} == before


def test_apply_skips_wrong_volume_sidecar(tmp_path: Path) -> None:
    """A sidecar carried over from another volume's review dir must not
    materialize that volume's placement onto this volume's same-numbered page."""
    paths = make_volume(tmp_path)
    d = make_sidecar_dict(
        volume="volY",
        page="4",
        verdict="adjusted",
        ops=ops_translate(-400.0, 0.0),
        mask_px=None,
        base_result_sha256=sha_of(paths.results / "p4.json"),
    )
    p = sidecar_path(paths, "4")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(d))
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["applied"] == []
    assert any("belongs to volume 'volY'" in w for w in summary["warnings"])
    assert json.loads((paths.results / "p4.json").read_text()).get("status") != (
        STATUS_REVIEWER_VERIFIED
    )
    # the UI queue surfaces it instead of silently hiding the file
    q = review_queue(paths, "volX")
    assert next(e for e in q if e["page"] == "4")["verdict"] == "invalid-sidecar"


def test_apply_skips_sidecar_whose_filename_disagrees_with_its_page(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    d = make_sidecar_dict(
        page="2",
        verdict="adjusted",
        ops=ops_translate(-400.0, 0.0),
        mask_px=None,
        base_result_sha256=sha_of(paths.results / "p2.json"),
    )
    p = paths.root / "review" / "p4.json"  # claims p2, lives at p4
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(d))
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["applied"] == []
    assert any("mismatched file" in w for w in summary["warnings"])
    assert json.loads((paths.results / "p2.json").read_text())["status"] == "OK"


def test_save_is_refused_while_another_operation_owns_the_volume(tmp_path: Path) -> None:
    """The sidecar write is part of the volume transaction: while a run owns
    the tree a save answers 409 (retryable), and the same edit lands after."""
    from autogeoref.paths import volume_lock

    app, paths = make_app(tmp_path)
    payload = look(app, "volX", "4")
    ops = ops_translate(-400.0, 0.0)
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "adjusted",
        "ops": ops,
        "affine": composed_affine(payload, ops),
        "mask_px": None,
    }
    with volume_lock(paths, operation="run --warp-only"):
        code, resp = app.save("volX", "4", body)
        assert code == 409 and "busy" in resp["error"]
        assert not sidecar_path(paths, "4").exists()  # refused whole, not half-saved
    code, resp = app.save("volX", "4", body)  # released: the same edit lands
    assert code == 200 and sidecar_path(paths, "4").exists()


def test_barrier_save_vs_apply_strands_no_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force a save into the middle of an apply: the save is refused while the
    apply holds the volume (never persisted against a result mid-rewrite), and
    a post-apply reload saves cleanly — no reviewer edit is stranded."""
    import threading

    import autogeoref.review.apply as review_apply

    app, paths = make_app(tmp_path)
    payload = look(app, "volX", "4")
    ops = ops_translate(-400.0, 0.0)
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "adjusted",
        "ops": ops,
        "affine": composed_affine(payload, ops),
        "mask_px": None,
    }
    assert app.save("volX", "4", body)[0] == 200

    entered, release = threading.Event(), threading.Event()
    real_write = review_apply.write_result

    def paused_write(path: Path, record: dict[str, Any]) -> Path:
        entered.set()
        assert release.wait(10)
        return real_write(path, record)

    monkeypatch.setattr(review_apply, "write_result", paused_write)
    summaries: list[dict[str, Any]] = []
    t = threading.Thread(
        target=lambda: summaries.append(apply_reviews(paths, "volX", do_warp=False))
    )
    t.start()
    try:
        assert entered.wait(10)  # apply is mid-rewrite, holding the volume
        code, resp = app.save("volX", "4", body)
        assert code == 409 and "review --apply" in resp["error"]
    finally:
        release.set()
        t.join(10)
    assert summaries and summaries[0]["applied"] == ["4"]
    # the refused edit is not stranded: reload the sheet and the verdict lands
    payload2 = look(app, "volX", "4")
    body2 = {
        "base_result_sha256": payload2["base_result_sha256"],
        "verdict": "accept",
        "ops": [],
        "affine": payload2["affine"],
        "mask_px": None,
    }
    assert app.save("volX", "4", body2)[0] == 200
    assert json.loads(sidecar_path(paths, "4").read_text())["verdict"] == "accept"


def test_barrier_two_mask_upserts_serialize_and_both_slugs_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two mask-writing applies over one volume: the second is refused while
    the first holds the lock (the masks.geojson read-modify-write can never
    interleave), and a retry finds both slugs in the collection, once each."""
    import threading

    import autogeoref.review.apply as review_apply
    from autogeoref.paths import VolumeBusyError

    paths = make_volume(tmp_path)
    paths.regions.mkdir()
    (paths.regions / "volX_p2.jpg").write_bytes(b"\xff\xd8fake")
    (paths.regions / "volX_p4.jpg").write_bytes(b"\xff\xd8fake")
    save_ui_sidecar(paths, "2", verdict="adjusted", ops=[], mask_px=PENTAGON_PX)
    sidecar_with_mask(paths, "4")

    entered, release = threading.Event(), threading.Event()

    def slow_dryrun(*_a: Any, **_k: Any) -> tuple[bool, str]:
        entered.set()
        assert release.wait(10)
        return True, ""

    monkeypatch.setattr(review_apply, "dryrun_against_region", slow_dryrun)
    summaries: list[dict[str, Any]] = []
    t = threading.Thread(
        target=lambda: summaries.append(apply_reviews(paths, "volX", do_warp=False))
    )
    t.start()
    try:
        assert entered.wait(10)  # first apply is mid-mask-validation, holding the volume
        with pytest.raises(VolumeBusyError, match="is busy"):
            apply_reviews(paths, "volX", do_warp=False)
    finally:
        release.set()
        t.join(10)
    assert summaries and sorted(summaries[0]["masks_written"]) == ["volX_p2", "volX_p4"]
    # the refused apply retries clean: everything already applied, nothing lost
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["already_applied"] == ["2", "4"] and summary["applied"] == []
    coll = json.loads((paths.masks / "masks.geojson").read_text())
    slugs = [f["properties"]["slug"] for f in coll["features"]]
    assert slugs == ["volX_p2", "volX_p4"]  # both survive, each exactly once
