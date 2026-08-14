"""The browser harness's own teardown and its unstartable-browser path.

`_stop` deletes a profile directory the moment it returns, so anything the
browser left running would race that delete — and it runs in a `finally`, where
raising would replace the caller's result with its own. Both are asserted here
against a stand-in process, so they hold on a machine with no browser at all.

The launch tests use a stand-in that writes to stderr and exits, which is what
a browser installed under a confinement rule it cannot satisfy does.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# pytest exports no public handle on the exception its own `skip` raises.
from _pytest.outcomes import Skipped

import browser as harness
from browser import _drained, _launched, _stop, _wait_for_devtools

#: Refuses to start and says why, in chromium's own shape: several lines of
#: noise, the cause among them, and no DevTools endpoint.
DEAF_BROWSER = """#!/bin/sh
echo "cannot start document portal: dial unix /run/user/1000/bus" >&2
echo "ERROR:process_singleton_posix.cc: Failed to create socket directory." >&2
exit 1
"""


@pytest.fixture(autouse=True)
def _forget_the_unstartable_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """The memo is module state, and a worker runs more than one file.

    Restored rather than cleared: a stub left in it would skip every real
    browser test that ran after this one in the same process, and a real one
    wiped here would make the file after this one detect the fault again.
    """
    monkeypatch.setattr(harness, "_unstartable", None)


def _stub_browser(tmp_path: Path) -> Path:
    stub = tmp_path / "chromium"
    stub.write_text(DEAF_BROWSER)
    stub.chmod(0o755)
    return stub


#: Alive, holding stderr open, and slower to announce than the wait allows —
#: a launch starved by the workers beside it rather than a broken browser.
SLOW_BROWSER = """
import sys, time
print("still waking up", file=sys.stderr, flush=True)
time.sleep(30)
"""

#: Ignores SIGTERM, like a browser too busy to answer one, and forks a child
#: that outlives it. Prints the child's pid so the test can watch it.
DEAF_PARENT = """
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(child.pid, flush=True)
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").is_dir()


def test_stop_takes_the_children_of_a_process_that_ignores_the_ask() -> None:
    """The failure this guards is a leaked writer, not a leaked process: a child
    still flushing into the profile is what made the delete fail."""
    proc = subprocess.Popen(
        [sys.executable, "-c", DEAF_PARENT],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    child = int(proc.stdout.readline().strip())
    time.sleep(0.2)
    assert _alive(child)

    _stop(proc, timeout_s=2.0)

    assert proc.returncode == -9, "the group kill never landed"
    assert _drained(proc.pid, timeout_s=5.0), "something outlived the group kill"
    assert not _alive(child)


def test_stop_is_silent_on_a_process_that_is_already_gone() -> None:
    """It runs in a `finally`; anything it raises replaces the caller's result."""
    proc = subprocess.Popen([sys.executable, "-c", ""], start_new_session=True)
    proc.wait()

    _stop(proc, timeout_s=1.0)


def test_stop_leaves_the_runner_that_called_it_alone() -> None:
    """A group signal aimed one pid wrong takes the test session with it."""
    group = os.getpgid(0)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )

    _stop(proc, timeout_s=2.0)

    assert proc.returncode is not None
    assert os.getpgid(0) == group


def test_a_browser_that_will_not_start_skips_rather_than_failing_every_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed browser answers PATH and still cannot be driven; that is a
    missing browser, not 59 failures with no stated cause."""
    monkeypatch.delenv("CI", raising=False)
    stub = _stub_browser(tmp_path)

    with pytest.raises(Skipped) as skipped, _launched(str(stub), block_dns=False):
        pass  # the launch never yields

    assert "will not start" in str(skipped.value)
    assert "socket directory" in str(skipped.value), "the browser's own cause is dropped"


def test_the_first_refusal_to_start_spares_the_rest_of_the_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every later test skips on the memo rather than launching again — and the
    memo names the executable, since PATH may have offered several."""
    monkeypatch.delenv("CI", raising=False)
    stub = _stub_browser(tmp_path)
    with pytest.raises(Skipped), _launched(str(stub), block_dns=False):
        pass  # the launch never yields

    with pytest.raises(Skipped) as skipped:
        harness.browser()

    assert str(stub) in str(skipped.value)


def test_a_browser_that_will_not_start_stays_loud_on_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner ships a browser, so a skip there retires the tier unseen.

    Written the long way round on purpose. `pytest.raises(AssertionError)` does
    not catch a skip — it is a `BaseException` — so writing it that way would
    make THIS test skip if the guard it exists to protect were ever removed,
    and a guard that disappears the way the fault does guards nothing.
    """
    monkeypatch.setenv("CI", "1")
    stub = _stub_browser(tmp_path)

    complaint = ""
    try:
        with _launched(str(stub), block_dns=False):
            pass  # the launch never yields
    except Skipped as skipped:
        pytest.fail(f"skipped on CI instead of failing: {skipped}")
    except AssertionError as failed:
        complaint = str(failed)
    else:
        pytest.fail("an unstartable browser neither failed nor skipped")

    assert "socket directory" in complaint


def test_a_slow_browser_stays_a_failure_and_is_never_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eight workers deep a launch can be starved, and one starved launch must
    not retire the tier: what it says about the next launch is nothing."""
    monkeypatch.delenv("CI", raising=False)
    proc = subprocess.Popen(
        [sys.executable, "-c", SLOW_BROWSER],
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        with pytest.raises(AssertionError) as failed:
            _wait_for_devtools(proc, timeout_s=0.5)
    finally:
        _stop(proc, timeout_s=5.0)

    assert "still running" in str(failed.value)
    assert harness._unstartable is None, "a slow launch was remembered as a broken one"
