"""Log tailing and rotation: bounded reads of the queue's per-leg and drain logs.

Imports :mod:`.queue.store` for the shared ``log_path`` definition but is
otherwise queue-agnostic.
"""

from __future__ import annotations

import os
from pathlib import Path

from .queue.store import log_path


def tail_log(work: Path, volume: str, track: str, lines: int = 200) -> str:
    lp = log_path(work, volume, track)
    if not lp.is_file():
        return f"(no log at {lp})"
    return "\n".join(_tail_lines(lp, lines))


#: Stop a backwards tail scan after this many bytes even if the requested line
#: count was never reached — a log that is one enormous line must not pull the
#: whole file back into memory, which is what the tail exists to avoid.
_TAIL_SCAN_CAP = 1 << 20


def _tail_lines(lp: Path, lines: int, block: int = 8192) -> list[str]:
    """The last ``lines`` lines of ``lp``, read backwards in blocks from the end.

    The scan stops once the buffer holds MORE newlines than lines requested:
    only the buffer's first line can then be cut short by a block boundary
    (possibly mid-UTF-8-sequence), and that line is always outside the
    ``[-lines:]`` slice. A scan that reaches the start of the file holds the
    whole file, so nothing is cut. A scan that hits ``_TAIL_SCAN_CAP`` first
    returns what it has; only its oldest line can be truncated.
    """
    buf = b""
    with lp.open("rb") as fh:
        pos = fh.seek(0, os.SEEK_END)
        while pos > 0 and buf.count(b"\n") <= lines and len(buf) < _TAIL_SCAN_CAP:
            step = min(block, pos)
            pos -= step
            fh.seek(pos)
            buf = fh.read(step) + buf
    return buf.decode("utf-8", errors="replace").splitlines()[-lines:]


#: Rotate a drain log this size or larger at the START of the next drain. The
#: drain logs are the only append-mode logs (`console.start_drain`); per-leg
#: logs are rewritten per run (`_run_leg`) and bound themselves.
DRAIN_LOG_ROTATE_BYTES = 8 * 1024 * 1024


def rotate_log(lp: Path, cap: int | None = None) -> bool:
    """Move an oversized ``lp`` aside to ``<name>.1`` so appends start fresh.

    One rotated generation is kept — a new rotation replaces the previous
    ``.1`` — so the pair stays bounded near two caps while the tail of the
    displaced log remains readable on disk. Returns True when it rotated.
    """
    if cap is None:
        cap = DRAIN_LOG_ROTATE_BYTES
    try:
        size = lp.stat().st_size
    except OSError:
        return False
    if size < cap:
        return False
    lp.replace(lp.with_name(lp.name + ".1"))
    return True
