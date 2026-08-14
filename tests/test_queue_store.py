"""The queue file: membership rules, and surviving whatever is written into it.

Two queues, and a volume may be live on only one of them. The file outlives the
code that wrote it, so an unknown field is kept, an entry on a queue this
version does not have is dropped loudly, and a row whose status or field types
make no sense is quarantined — visible, terminal, and no bar to a clean re-add.
Concurrent writers all land.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from autogeoref.queue import render as qrender
from autogeoref.queue import run as qrun
from autogeoref.queue import store as qstore
from autogeoref.queue.command import DrainContext
from queue_support import CITY, _by, _spy, _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def test_add_and_render(tmp_path: Path) -> None:
    _volume(tmp_path, "vol_a")
    entry = qstore.add(tmp_path, "vol_a")
    assert entry.status == "queued"
    assert entry.track == "place", "the default enqueue is the place queue"
    assert entry.then_serve is True, "and end to end by default: it will promote to serve"
    assert [e.volume for e in qstore.load_queue(tmp_path)] == ["vol_a"]
    board = qrender.render_text(tmp_path)
    assert "PLACE" in board and "SERVE" in board and "vol_a" in board


def test_a_place_enqueue_needs_no_results_yet(tmp_path: Path) -> None:
    """Its own place run writes them; the promotion to serve happens after."""
    _volume(tmp_path, "vol_a")  # no results/
    e = qstore.add(tmp_path, "vol_a", "place")
    assert e.track == "place" and e.then_serve is True


def test_serve_enqueue_needs_no_assertion_a_human_cannot_be_made_to_mean(tmp_path: Path) -> None:
    """A serve enqueue takes no sign-off — the removed `--reviewed` gate stays removed."""
    _volume(tmp_path, "vol_a", results=True)
    assert qstore.add(tmp_path, "vol_a", "serve").track == "serve"


def test_serve_enqueue_refuses_a_volume_with_nothing_placed(tmp_path: Path) -> None:
    """The precondition that IS real: --warp-only consumes records it cannot produce."""
    _volume(tmp_path, "vol_a")  # no results/
    with pytest.raises(qstore.QueueError, match="nothing to serve"):
        qstore.add(tmp_path, "vol_a", "serve")


def test_a_serve_entry_never_carries_then_serve(tmp_path: Path) -> None:
    """then_serve is meaningless on a serve entry (nothing left to promote to)."""
    _volume(tmp_path, "vol_a", results=True)
    assert qstore.add(tmp_path, "vol_a", "serve", then_serve=True).then_serve is False


def test_add_refuses_an_unknown_volume(tmp_path: Path) -> None:
    with pytest.raises(qstore.QueueError, match="no work tree"):
        qstore.add(tmp_path, "nope", "place")


def test_no_duplicate_live_entries(tmp_path: Path) -> None:
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", "place")
    with pytest.raises(qstore.QueueError, match="already on the place queue"):
        qstore.add(tmp_path, "vol_a", "place")


def test_a_volume_cannot_be_live_on_two_queues(tmp_path: Path) -> None:
    """A serve run against a volume still being placed would bake a moving funnel."""
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "place", then_serve=False)
    with pytest.raises(qstore.QueueError, match="on the place queue"):
        qstore.add(tmp_path, "vol_a", "serve")


def test_remove_without_a_track_removes_both(tmp_path: Path) -> None:
    _volume(tmp_path, "vol_a", results=True)
    place = qstore.add(tmp_path, "vol_a", "place", then_serve=False)
    place.status = "needs-review"  # terminal, so a serve entry can coexist
    qstore.save_queue(tmp_path, [place])
    qstore.add(tmp_path, "vol_a", "serve")
    assert qstore.remove(tmp_path, "vol_a") == 2
    assert qstore.load_queue(tmp_path) == []


def test_concurrent_writers_do_not_lose_an_entry(tmp_path: Path) -> None:
    """queue_write_lock: many threads adding at once all land (no lost update)."""
    for i in range(12):
        _volume(tmp_path, f"vol_{i}")
    barrier = threading.Barrier(12)

    def _add(i: int) -> None:
        barrier.wait()
        qstore.add(tmp_path, f"vol_{i}", "place", then_serve=False)

    threads = [threading.Thread(target=_add, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(qstore.load_queue(tmp_path)) == 12, "every concurrent add survived"


def test_the_queue_file_survives_a_field_this_version_does_not_know(tmp_path: Path) -> None:
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", "place")
    raw = json.loads(qstore.queue_path(tmp_path).read_text())
    raw["entries"][0]["some_future_field"] = 42
    qstore.queue_path(tmp_path).write_text(json.dumps(raw))
    entries = qstore.load_queue(tmp_path)  # must not raise TypeError
    assert [e.volume for e in entries] == ["vol_a"] and entries[0].status == "queued"


def test_an_entry_on_a_queue_this_version_does_not_know_is_dropped_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A removed track (`all`) must not become an invisible, un-drainable zombie.

    Kept, it matches neither queue — drained by nobody, invisible everywhere — while
    still blocking a re-add with an error naming a queue that is gone. So it is dropped
    with a warning that says to re-add it. Validation, not migration.
    """
    _volume(tmp_path, "vol_legacy")
    _volume(tmp_path, "vol_ok")
    qstore.queue_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    qstore.queue_path(tmp_path).write_text(
        json.dumps(
            {
                "entries": [
                    {"volume": "vol_legacy", "track": "all", "status": "queued"},
                    {"volume": "vol_ok", "track": "place", "status": "queued"},
                ]
            }
        )
    )
    import logging

    with caplog.at_level(logging.WARNING):
        entries = qstore.load_queue(tmp_path)
    assert [e.volume for e in entries] == ["vol_ok"], "the zombie is gone; the good one stays"
    assert "vol_legacy" in caplog.text and "unknown queue" in caplog.text
    # and the volume can now be re-added cleanly (no phantom-queue conflict)
    assert qstore.add(tmp_path, "vol_legacy").track == "place"


def test_an_old_queue_file_still_loads(tmp_path: Path) -> None:
    """The queue file OUTLIVES the code: an entry written before `then_serve` must still
    load and take its default."""
    _volume(tmp_path, "vol_a")
    qstore.queue_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    qstore.queue_path(tmp_path).write_text(
        json.dumps({"entries": [{"volume": "vol_a", "track": "place", "status": "queued"}]})
    )
    entries = qstore.load_queue(tmp_path)
    assert entries[0].then_serve is True  # the default
    # a field a LATER version removed (`leg`, `review`) is dropped, not fatal
    qstore.queue_path(tmp_path).write_text(
        json.dumps({"entries": [{"volume": "vol_a", "track": "place", "leg": "place"}]})
    )
    assert qstore.load_queue(tmp_path)[0].volume == "vol_a"


def _write_queue_file(work: Path, entries: list[dict[str, object]]) -> None:
    qstore.queue_path(work).parent.mkdir(parents=True, exist_ok=True)
    qstore.queue_path(work).write_text(json.dumps({"entries": entries}))


def test_an_unknown_status_is_quarantined_visible_and_nonblocking(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A status a newer version wrote would be a zombie kept as-is: drains select
    only `queued`, and a nonterminal stranger blocks a re-add. Quarantine keeps it
    VISIBLE (a failed row whose note names the original status) and TERMINAL, so
    the operator sees it and a clean re-add just works."""
    import logging

    _volume(tmp_path, "vol_bad")
    _volume(tmp_path, "vol_ok")
    _write_queue_file(
        tmp_path,
        [
            {"volume": "vol_bad", "track": "place", "status": "paused"},
            {"volume": "vol_ok", "track": "place", "status": "queued", "then_serve": False},
        ],
    )
    with caplog.at_level(logging.WARNING):
        entries = qstore.load_queue(tmp_path)
    assert [e.volume for e in entries] == ["vol_bad", "vol_ok"], "nothing was dropped"
    bad = entries[0]
    assert bad.status == "failed" and bad.terminal
    assert "quarantined" in (bad.note or "") and "'paused'" in (bad.note or "")
    assert "quarantining vol_bad" in caplog.text
    # visible on the board, not just in the loader's memory
    board = qrender.render_text(tmp_path)
    assert "vol_bad" in board and "FAILED" in board and "quarantined" in board
    # nonblocking: the volume can be re-added over its quarantined row
    assert qstore.add(tmp_path, "vol_bad", then_serve=False).status == "queued"


def test_valid_rows_still_drain_alongside_a_quarantined_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _volume(tmp_path, "vol_bad")
    _volume(tmp_path, "vol_ok")
    _write_queue_file(
        tmp_path,
        [
            {"volume": "vol_bad", "track": "place", "status": "paused", "added": 1.0},
            {
                "volume": "vol_ok",
                "track": "place",
                "status": "queued",
                "added": 2.0,
                "then_serve": False,
            },
        ],
    )
    spawned = _spy(monkeypatch)
    drained = qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")
    assert [e.volume for e in drained] == ["vol_ok"], "the quarantined row runs nothing"
    assert len(spawned) == 1
    statuses = _by(qstore.load_queue(tmp_path))
    assert statuses[("vol_ok", "place")] == "needs-review"
    assert statuses[("vol_bad", "place")] == "failed", "still visible after the drain's writes"


def test_malformed_fields_are_quarantined_with_inert_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Wrong-typed fields would crash or corrupt whatever touches them (persist keys
    on `added`; the views subtract timestamps). The entry survives as a terminal
    failed row whose bad fields are replaced with inert values, and the note names
    every offender."""
    import logging

    _volume(tmp_path, "vol_bad")
    _write_queue_file(
        tmp_path,
        [
            {
                "volume": "vol_bad",
                "track": "serve",
                "status": "queued",
                "added": "yesterday",
                "exit_code": "boom",
                "then_serve": "yes",
            }
        ],
    )
    with caplog.at_level(logging.WARNING):
        (entry,) = qstore.load_queue(tmp_path)
    assert entry.status == "failed" and entry.terminal
    assert entry.added == 0.0 and entry.exit_code is None and entry.then_serve is False
    assert "added" in (entry.note or "") and "exit_code" in (entry.note or "")
    assert "then_serve" in (entry.note or "")
    # deterministic: a second load produces the same key, so persist's merge holds
    (again,) = qstore.load_queue(tmp_path)
    assert (again.volume, again.track, again.added) == (entry.volume, entry.track, entry.added)
    qrender.render_text(tmp_path)  # the views survive the sanitized row


def test_a_malformed_status_type_is_quarantined_not_crashed_on(tmp_path: Path) -> None:
    _volume(tmp_path, "vol_bad")
    _write_queue_file(tmp_path, [{"volume": "vol_bad", "track": "place", "status": 7}])
    (entry,) = qstore.load_queue(tmp_path)
    assert entry.status == "failed" and "status" in (entry.note or "")


def test_a_quarantined_row_survives_other_writers(tmp_path: Path) -> None:
    """add/remove/persist rewrite the file through load_queue; the quarantined row
    must come out the other side terminal and annotated, not resurrected raw."""
    _volume(tmp_path, "vol_bad")
    _volume(tmp_path, "vol_ok")
    _write_queue_file(tmp_path, [{"volume": "vol_bad", "track": "place", "status": "paused"}])
    qstore.add(tmp_path, "vol_ok")  # a write: load + save under the queue lock
    raw = json.loads(qstore.queue_path(tmp_path).read_text())["entries"]
    persisted = {e["volume"]: e for e in raw}
    assert persisted["vol_bad"]["status"] == "failed"
    assert "quarantined" in persisted["vol_bad"]["note"]
    # and removing it works like any other row
    assert qstore.remove(tmp_path, "vol_bad") == 1


def test_a_non_object_entry_is_dropped_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An entry that is not even an object has no identity to quarantine under."""
    import logging

    _volume(tmp_path, "vol_ok")
    _write_queue_file(tmp_path, ["garbage", {"volume": "vol_ok", "track": "place"}])  # type: ignore[list-item]
    with caplog.at_level(logging.WARNING):
        entries = qstore.load_queue(tmp_path)
    assert [e.volume for e in entries] == ["vol_ok"]
    assert "non-object" in caplog.text
