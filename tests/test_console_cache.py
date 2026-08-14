"""The board cache: reuse an unchanged tree, and show every write on the next poll.

The page polls every few seconds and most polls see a byte-identical tree, which
must not be re-parsed. A cached payload is returned only while the fingerprint
proves nothing changed — including a drain that died without writing anything,
and an edit to the catalog the city TOML names. The one-shot paths build no
cache at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autogeoref import drain_lock
from autogeoref.console import cli as console_cli
from autogeoref.console import payload as console_payload
from autogeoref.queue import store as qstore
from console_support import _city_toml, _images, _results, _status, _tree

if TYPE_CHECKING:
    import pytest


def _bump(path: Path) -> None:
    """Advance a path's mtime past filesystem timestamp granularity."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def _areas_geojson(name: str) -> str:
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "properties": {"community": name},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-87.70, 41.85],
                                [-87.60, 41.85],
                                [-87.60, 41.95],
                                [-87.70, 41.95],
                                [-87.70, 41.85],
                            ]
                        ],
                    },
                }
            ],
        }
    )


def _fingerprint(roots: dict[str, Path]) -> tuple[Any, ...]:
    return console_payload.board_fingerprint(
        work=roots["work"],
        fixtures=roots["fixtures"],
        tiles=roots["tiles"],
        ground_truth=roots["ground_truth"],
    )


def test_the_board_cache_reuses_an_unchanged_tree_and_rebuilds_on_a_write(
    tmp_path: Path,
) -> None:
    """The page polls every few seconds and most polls see a byte-identical tree;
    those must not re-parse every result record. But a cached payload is only ever
    returned while the fingerprint proves the tree unchanged — every write shows
    on the NEXT poll, never after a timeout."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    builds = 0

    def build() -> dict[str, Any]:
        nonlocal builds
        builds += 1
        return console_payload.board(work=roots["work"], rows=_status(roots))

    cache = console_payload.BoardCache(build, lambda: _fingerprint(roots))
    first = cache()
    assert cache() is first and builds == 1  # unchanged tree: no rebuild

    qstore.add(roots["work"], "vol_a", "place")  # creates queue/queue.json
    second = cache()
    assert builds == 2
    assert [e["volume"] for e in second["entries"]] == ["vol_a"]

    _results(roots["work"], "vol_a", accepted=1, flagged=0)  # creates results/
    _bump(roots["work"] / "vol_a" / "results")  # granularity-proof
    third = cache()
    assert builds == 3
    assert third["entries"][0]["progress"]["results"] == 1
    assert cache() is third and builds == 3


def test_the_fingerprint_sees_a_drain_die_without_any_file_changing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain killed -9 releases its flock but rewrites nothing, so no mtime
    moves; the fingerprint carries `live_drain`'s answer so the board still stops
    saying "running" on the next poll."""
    roots = _tree(tmp_path)
    monkeypatch.setattr(drain_lock, "live_drain", lambda *_: 4242)
    running = _fingerprint(roots)
    assert _fingerprint(roots) == running  # deterministic while nothing changes
    monkeypatch.setattr(drain_lock, "live_drain", lambda *_: None)
    assert _fingerprint(roots) != running


def test_the_community_area_index_is_reparsed_only_when_its_file_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "areas.geojson"
    path.write_text(_areas_geojson("TESTVILLE"))
    first = console_payload._area_index(path)
    assert console_payload._area_index(path) is first
    path.write_text(_areas_geojson("OTHERTOWN"))
    _bump(path)
    fresh = console_payload._area_index(path)
    assert fresh is not first
    assert fresh.names((-87.70, 41.85, -87.60, 41.95)) == ["Othertown"]


def test_the_served_board_is_the_cached_one_and_still_reflects_the_next_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`queue --serve` is the ONE surface that wraps its builder in the cache, and
    the wired fingerprint must cover what the wired builder reads — a queue write
    lands on the very next poll."""
    from autogeoref.cli.entry import main

    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    captured: dict[str, Any] = {}

    def fake_serve(work: Path, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(console_cli, "serve", fake_serve)
    assert (
        main(
            [
                "queue",
                "--serve",
                "--work",
                str(roots["work"]),
                "--fixtures",
                str(roots["fixtures"]),
                "--tiles",
                str(roots["tiles"]),
            ]
        )
        == 0
    )
    build_board = captured["build_board"]
    assert isinstance(build_board, console_payload.BoardCache)
    first = build_board()
    assert build_board() is first
    qstore.add(roots["work"], "vol_a", "place")
    assert [e["volume"] for e in build_board()["entries"]] == ["vol_a"]


def test_the_served_board_sees_an_edit_to_the_config_derived_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wired fingerprint must watch the catalog the wired builder actually
    reads — the config fallback included, not just the flag. Otherwise a year
    edit stays invisible until an unrelated tree write happens to bump the key."""
    from autogeoref.cli.entry import main

    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    cfg = _city_toml(tmp_path, loc_catalog="cat.json")
    catalog = tmp_path / "cat.json"

    def entry(year: int) -> str:
        return json.dumps([{"id": "http://www.loc.gov/item/vol_a/", "date": str(year)}])

    catalog.write_text(entry(1896))
    captured: dict[str, Any] = {}
    monkeypatch.setattr(console_cli, "serve", lambda _work, **kwargs: captured.update(kwargs))
    assert (
        main(
            [
                "queue",
                "--serve",
                "--city",
                str(cfg),
                "--work",
                str(roots["work"]),
                "--fixtures",
                str(roots["fixtures"]),
                "--tiles",
                str(roots["tiles"]),
            ]
        )
        == 0
    )
    build_board = captured["build_board"]
    assert build_board()["context"]["vol_a"]["year"] == 1896

    catalog.write_text(entry(1912))
    _bump(catalog)  # granularity-proof
    assert build_board()["context"]["vol_a"]["year"] == 1912


def test_one_shot_status_and_candidates_paths_never_touch_the_board_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`make status` and `queue --candidates` are one-shot derivations; they must
    re-read the tree every invocation and never construct the cache at all."""
    from autogeoref.cli.entry import main

    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)

    def tripwire(*args: object, **kwargs: object) -> None:
        raise AssertionError("one-shot paths must not construct a BoardCache")

    monkeypatch.setattr(console_cli, "BoardCache", tripwire)
    argv = [
        "--work",
        str(roots["work"]),
        "--fixtures",
        str(roots["fixtures"]),
        "--tiles",
        str(roots["tiles"]),
    ]
    assert main(["queue", "--candidates", *argv]) == 0
    out = capsys.readouterr().out
    assert "vol_a" in out and "vol_b" not in out

    _images(roots["work"], "vol_b", 4)
    assert main(["queue", "--candidates", *argv]) == 0
    assert "vol_b" in capsys.readouterr().out  # re-derived, not replayed

    assert main(["status", *argv]) == 0
    assert "vol_b" in capsys.readouterr().out
