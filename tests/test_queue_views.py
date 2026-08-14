"""The read side: per-volume progress, the board, and the log tail.

Progress is derived from the filesystem and never from the queue file, so a
runner killed mid-stage cannot leave a state file lying about what it did. The
board has a section per queue. The log tail is a backwards block scan, and it
has to agree with a whole-file read at every block boundary without cutting a
multibyte character.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogeoref import logfiles
from autogeoref.queue import progress as qprogress
from autogeoref.queue import render as qrender
from autogeoref.queue import store as qstore
from queue_support import _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def test_progress_is_read_off_the_filesystem_not_the_queue_file(tmp_path: Path) -> None:
    """A runner killed at 3am must not be able to leave a state file lying."""
    vol = _volume(tmp_path, "vol_a", results=True)
    (vol / "annotations").mkdir()
    (vol / "annotations" / "p1.json").write_text("{}")
    (vol / "annotations" / "p1.v2.claude-sonnet-5.json").write_text("{}")
    (vol / "annotations" / "p2.failed.json").write_text("{}")  # a call that did not land
    (vol / "markers").mkdir()
    (vol / "markers" / "match.marker.json").write_text(
        json.dumps({"stage": "match", "status": "ok", "finished": 1_000_000.0})
    )
    prog = qprogress.volume_progress(tmp_path, "vol_a")
    assert prog.pages == 2  # the orientation sentinel is not a page
    assert prog.reads == 1  # p1 (two sidecars, one page); p2's failed read excluded
    assert prog.failed_markers == 1  # p2's marker: what a clear-and-retry would re-spend
    assert prog.results == 2
    assert prog.accepted == 1 and prog.flagged == 1
    assert prog.stage == "match" and prog.stage_status == "ok"


def test_progress_surfaces_a_failed_stage(tmp_path: Path) -> None:
    vol = _volume(tmp_path, "vol_a")
    (vol / "markers").mkdir()
    (vol / "markers" / "prep.marker.json").write_text(
        json.dumps(
            {
                "stage": "prep",
                "status": "failed",
                "finished": 2_000_000.0,
                "error": "UnrecognizedSheetError: p_weird.jpg\nTraceback...",
            }
        )
    )
    prog = qprogress.volume_progress(tmp_path, "vol_a")
    assert prog.stage == "prep" and prog.stage_status == "failed"
    assert prog.error is not None and "UnrecognizedSheetError" in prog.error


def test_a_fetch_row_renders_on_its_own_board_section(tmp_path: Path) -> None:
    """A queue nobody can see is a queue nobody drains."""
    qstore.add(tmp_path, "vol_new", "fetch")
    board = qrender.render_text(tmp_path)
    assert "FETCH" in board and "vol_new" in board


def _log(tmp_path: Path, text: str) -> Path:
    lp = qstore.log_path(tmp_path, "vol_a", "place")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_bytes(text.encode("utf-8"))
    return lp


def test_tail_matches_a_whole_file_read_at_every_block_boundary(tmp_path: Path) -> None:
    """The backwards block scan must agree with the naive read-everything tail on
    files smaller than, equal to, and much larger than the block — including a
    final line with no newline — for any requested line count."""
    body = "\n".join(f"line {i} {'x' * (i % 37)}" for i in range(500)) + "\ntrailing partial"
    lp = _log(tmp_path, body)
    reference = body.splitlines()
    size = lp.stat().st_size
    for block in (7, 64, size, size + 1, 8192):
        for lines in (1, 3, 200, 499, 501, 10_000):
            got = logfiles._tail_lines(lp, lines, block=block)
            assert got == reference[-lines:], (block, lines)


def test_tail_never_cuts_a_multibyte_character_in_a_returned_line(tmp_path: Path) -> None:
    """Block boundaries land mid-UTF-8-sequence for almost every block size here
    (2-, 3- and 4-byte characters); every returned line must still decode intact,
    because the only cut line is the one the slice drops."""
    body = "\n".join(f"π→𐍈 café {i} {'é' * (i % 11)}" for i in range(120))
    lp = _log(tmp_path, body)
    reference = body.splitlines()
    for block in range(3, 40):
        got = logfiles._tail_lines(lp, 50, block=block)
        assert got == reference[-50:], block
        assert "�" not in "".join(got)


def test_tail_log_reads_the_current_end_of_the_log_on_every_call(tmp_path: Path) -> None:
    """The console's /api/log serves this live per GET (never through the board
    cache), so an append must show up on the very next call."""
    lp = _log(tmp_path, "first\n")
    assert logfiles.tail_log(tmp_path, "vol_a", "place") == "first"
    with lp.open("a") as fh:
        fh.write("second\n")
    assert logfiles.tail_log(tmp_path, "vol_a", "place") == "first\nsecond"
    assert logfiles.tail_log(tmp_path, "vol_a", "serve").startswith("(no log at ")


def test_tail_stops_scanning_a_pathological_single_line_log(tmp_path: Path) -> None:
    """One enormous line has no newlines to satisfy the scan, so the byte cap is
    what keeps the tail from swallowing the file whole."""
    lp = _log(tmp_path, "x" * (logfiles._TAIL_SCAN_CAP + 50_000))
    (got,) = logfiles._tail_lines(lp, 200)
    assert len(got) <= logfiles._TAIL_SCAN_CAP + 8192  # the cap, not the file size


def test_rotate_log_moves_an_oversized_log_aside_and_keeps_one_generation(
    tmp_path: Path,
) -> None:
    lp = tmp_path / "drain.both.log"
    assert logfiles.rotate_log(lp) is False  # missing file: nothing to do

    lp.write_text("small\n")
    assert logfiles.rotate_log(lp, cap=1024) is False
    assert lp.read_text() == "small\n"  # under the cap: untouched

    lp.write_text("old drain " * 200)
    assert logfiles.rotate_log(lp, cap=1024) is True
    rotated = tmp_path / "drain.both.log.1"
    assert not lp.exists() and rotated.read_text().startswith("old drain ")

    lp.write_text("newer drain " * 200)
    assert logfiles.rotate_log(lp, cap=1024) is True
    assert rotated.read_text().startswith("newer drain ")  # one generation kept
