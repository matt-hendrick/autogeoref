"""The shared retry policy: spacing, budget passthrough, and stop hooks."""

import pytest

from autogeoref.annotate import invocation
from autogeoref.annotate.failures import BudgetLimitError, MalformedResponseError
from autogeoref.annotate.invocation import annotate_with_retry, retry_delay_s

BASE = 15.0


@pytest.fixture(autouse=True)
def real_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo conftest's zeroing: these tests are ABOUT the delay."""
    monkeypatch.setattr(invocation, "RETRY_BASE_S", BASE)


def test_retries_are_spaced_not_back_to_back() -> None:
    """Back-to-back retries all land inside one provider blip.

    Measured on a live drain: "Selected model is at capacity" killed single
    pages on four volumes in a row, each stopping the whole batch, while pages
    minutes either side of the failure read fine. Both attempts had fired
    within the same moment.
    """
    slept: list[float] = []
    calls: list[int] = []

    def call() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise MalformedResponseError("at capacity")
        return "ok"

    result, error = annotate_with_retry(call, 2, "p1", sleep=slept.append)

    assert (result, error) == ("ok", "")
    assert len(slept) == 1, "the failed attempt must be followed by a wait"
    assert slept[0] >= BASE


def test_no_wait_after_the_final_attempt() -> None:
    """Sleeping after the last attempt delays the failure and buys nothing."""
    slept: list[float] = []

    def call() -> str:
        raise MalformedResponseError("still at capacity")

    result, error = annotate_with_retry(call, 2, "p1", sleep=slept.append)

    assert result is None
    assert "still at capacity" in error
    assert len(slept) == 1


def test_budget_limit_is_never_retried_or_slept_on() -> None:
    slept: list[float] = []

    def call() -> str:
        raise BudgetLimitError("usage limit reached")

    with pytest.raises(BudgetLimitError):
        annotate_with_retry(call, 3, "p1", sleep=slept.append)
    assert slept == []


def test_a_cancelled_stage_does_not_sleep_before_giving_up() -> None:
    slept: list[float] = []

    def call() -> str:
        raise AssertionError("must not be called")

    result, _ = annotate_with_retry(call, 2, "p1", cancelled=lambda: True, sleep=slept.append)
    assert result is None
    assert slept == []


def test_a_sleeping_worker_rechecks_cancellation_before_spending() -> None:
    """Another worker's budget limit must land during the wait, not after it.

    The gap used to be microseconds; it is now tens of seconds, which is long
    enough for a sibling worker to hit the wall while this one sleeps.
    """
    stopped = False
    calls: list[int] = []

    def call() -> str:
        calls.append(1)
        raise MalformedResponseError("at capacity")

    def stop_during_the_wait(_delay: float) -> None:
        nonlocal stopped
        stopped = True

    result, _ = annotate_with_retry(
        call, 3, "p1", cancelled=lambda: stopped, sleep=stop_during_the_wait
    )
    assert result is None
    assert len(calls) == 1, "woke up and spent into a stopped stage"


def test_delay_grows_and_is_capped() -> None:
    """Jitter keeps a six-worker fan-out from retrying in lockstep."""
    assert retry_delay_s(0, rand=lambda: 0.0) == BASE
    assert retry_delay_s(1, rand=lambda: 0.0) == BASE * 2
    assert retry_delay_s(0, rand=lambda: 1.0) == BASE * 2
    # a raised `attempts` lengthens gaps; it must not stall the drain
    assert retry_delay_s(20, rand=lambda: 0.0) == invocation.RETRY_CAP_S
