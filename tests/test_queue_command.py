"""The command a leg spawns: the one a human would type, with the roots the queue owns.

``--run-arg`` may tune a run and never re-point it. Every queue-owned shape —
the city, the work tree, the viewer manifest, the leg's own mode — is refused
before anything is logged or spawned, and the trusted arguments are appended
after the extras so argparse's last-flag-wins keeps them. Ground truth is not
reachable from a queued leg at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autogeoref.queue import command as qcommand
from autogeoref.queue import run as qrun
from autogeoref.queue import store as qstore
from autogeoref.queue.command import DrainContext
from autogeoref.viewer import publish as viewer_publish
from queue_support import CITY, _ctx, _spy, _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def test_serve_runs_the_back_half_only_never_the_placement_stages(tmp_path: Path) -> None:
    """`--warp` would re-run the whole funnel before tiling; the serve leg is --warp-only."""
    entry = qstore.QueueEntry(volume="vol_a", track="serve")
    cmd = qcommand._command("serve", entry, _ctx(tmp_path))
    assert "--warp-only" in cmd and "--warp" not in cmd


#: One sequence per queue-owned override shape: separate-token, `=`-joined, and
#: an abbreviated prefix the child's argparse would still resolve.
_OWNED_OVERRIDES = [
    ["--city", "elsewhere.toml"],
    ["--city=elsewhere.toml"],
    ["--work", "/tmp/other-work"],
    ["--work=/tmp/other-work"],
    ["--viewer-manifest", "other/manifest.json"],
    ["--viewer-manifest=other/manifest.json"],
    ["--viewer=other/manifest.json"],
    ["--warp"],
    ["--warp-only"],
    ["--warp-only=1"],
    ["--dry-run"],
    ["--dry"],
    ["--no-escalate", "--dry-run"],
]


@pytest.mark.parametrize("smuggled", _OWNED_OVERRIDES, ids=lambda s: " ".join(s))
@pytest.mark.parametrize("leg", ["place", "serve"])
def test_the_command_refuses_queue_owned_identity_and_mode_overrides(
    tmp_path: Path, leg: str, smuggled: list[str]
) -> None:
    """`--run-arg` may tune a run, never re-point its roots or change what the leg is.

    Otherwise the queue holds state and locks for one work tree while the child acts
    on another, or reports successful placement for a --dry-run that did no work.
    """
    entry = qstore.QueueEntry(volume="vol_a", track=leg)
    with pytest.raises(qstore.QueueError, match="cannot override"):
        qcommand._command(leg, entry, _ctx(tmp_path, extra=tuple(smuggled)))


@pytest.mark.parametrize("smuggled", _OWNED_OVERRIDES, ids=lambda s: " ".join(s))
def test_an_override_fails_the_drain_before_any_log_or_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smuggled: list[str]
) -> None:
    """The rejection is a controlled QueueError raised before anything happens: no
    child spawns, no per-volume log is created, and the entry stays queued."""
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a")
    spawned = _spy(monkeypatch)

    with pytest.raises(qstore.QueueError, match="cannot override"):
        qrun.run_queue(DrainContext(work=tmp_path, city=CITY, extra=smuggled), track="place")

    assert spawned == [], "no child may launch"
    assert not qstore.log_path(tmp_path, "vol_a", "place").exists(), "no log may be created"
    assert [e.status for e in qstore.load_queue(tmp_path)] == ["queued"]


def test_the_drain_spawns_a_binary_that_actually_exists(tmp_path: Path) -> None:
    """The bare name `autogeoref` is NOT enough, and mocks hide it."""
    binary = qcommand.autogeoref_bin()
    if binary != "autogeoref":
        assert Path(binary).exists() and Path(binary).is_absolute()
    cmd = qcommand._command("place", qstore.QueueEntry(volume="v", track="place"), _ctx(tmp_path))
    assert cmd[0] == binary


def test_the_queued_command_is_the_command_a_human_would_type(tmp_path: Path) -> None:
    entry = qstore.QueueEntry(volume="vol_a", track="place")
    assert qcommand._command("place", entry, _ctx(tmp_path)) == [
        qcommand.autogeoref_bin(),
        "run",
        "vol_a",
        "--city",
        str(CITY),
        "--work",
        str(tmp_path),
        "--viewer-manifest",
        "viewer/chicago-ill/manifest.json",
    ]
    assert qcommand._command("serve", entry, _ctx(tmp_path))[-1] == "--warp-only"


def test_allowed_extras_pass_through_but_never_outrank_the_configured_roots(
    tmp_path: Path,
) -> None:
    """Behavior flags survive, and the queue-owned arguments come AFTER them so the
    configured city/work/manifest and the leg's mode win argparse's last-flag-wins
    even against an override shape the rejection does not know yet."""
    entry = qstore.QueueEntry(volume="vol_a", track="place")
    extras = ("--no-escalate", "--annotate-jobs", "10")

    cmd = qcommand._command("place", entry, _ctx(tmp_path, extra=extras))
    assert cmd[3:6] == list(extras)
    tail = cmd[6:]
    assert tail == [
        "--city",
        str(CITY),
        "--work",
        str(tmp_path),
        "--viewer-manifest",
        "viewer/chicago-ill/manifest.json",
    ], "trusted roots follow the extras, with the configured values"
    assert "--warp-only" not in cmd and "--warp" not in cmd and "--dry-run" not in cmd

    serve = qcommand._command("serve", entry, _ctx(tmp_path, extra=extras))
    assert serve.index("--no-escalate") < serve.index("--city")
    assert serve[-1] == "--warp-only", "the serve leg's mode is appended last"


def test_no_queued_leg_can_hand_ground_truth_to_a_run(tmp_path: Path) -> None:
    """A queued place run must be the run a human would type — and neither has pins.

    The queue used to append the per-volume export to every place leg, which made a
    volume's search box come from hand-placed pins. The fix was to delete the input
    rather than teach the queue about the rule: a rule only one code path knows is
    not a rule, and this asserts the flag is unreachable from here at all.
    """
    gt_dir = tmp_path / "ground-truth"
    gt_dir.mkdir()
    (gt_dir / "api-layers-vol_a.json").write_text("[]")
    entry = qstore.QueueEntry(volume="vol_a", track="place")

    for leg in ("place", "serve"):
        assert "--ground-truth" not in qcommand._command(leg, entry, _ctx(tmp_path))
    with pytest.raises(TypeError):
        _ctx(tmp_path, ground_truth=gt_dir)


def test_the_queue_passes_a_viewer_manifest_to_bounds_from_volumes(tmp_path: Path) -> None:
    cmd = qcommand._command(
        "place", qstore.QueueEntry(volume="vol_a", track="place"), _ctx(tmp_path)
    )
    assert cmd[-2:] == ["--viewer-manifest", "viewer/chicago-ill/manifest.json"]


def test_the_queue_uses_the_publication_manifest_for_child_runs(tmp_path: Path) -> None:
    publication = viewer_publish.PublicationConfig(
        work=tmp_path,
        city_toml=CITY,
        manifest=tmp_path / "custom-viewer" / "manifest.json",
    )
    cmd = qcommand._command(
        "place",
        qstore.QueueEntry(volume="vol_a", track="place"),
        _ctx(tmp_path, publication=publication),
    )
    assert cmd[-2:] == ["--viewer-manifest", str(publication.manifest)]


def test_a_queue_run_passes_the_publication_manifest_to_its_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", then_serve=False)
    publication = viewer_publish.PublicationConfig(
        work=tmp_path,
        city_toml=CITY,
        manifest=tmp_path / "custom-viewer" / "manifest.json",
    )
    spawned = _spy(monkeypatch)

    qrun.run_queue(DrainContext(work=tmp_path, city=CITY, publication=publication), track="place")

    assert spawned[0][-2:] == ["--viewer-manifest", str(publication.manifest)]


def test_default_serve_lanes_is_conservative() -> None:
    """Derived from CPU, but capped and floored — never raw cpu_count (GDAL is already
    multithreaded, baking is memory-heavy, the box is shared)."""
    lanes = qrun.default_serve_lanes()
    assert 1 <= lanes <= 4
