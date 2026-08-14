"""The call estimate on a candidate card: what a run would cost, net of cached reads.

Cached reads replay free, so the low end is the unread primaries and nothing
else; the ceiling adds the escalation ladder over the gated pool and the one
retry every read is allowed. The gated fraction is a city measurement, so a city
that declares its own must move the ceiling.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from autogeoref.budget import estimate_spend
from autogeoref.console import backlog as console_backlog
from console_support import _city, _images, _status, _tree


def test_call_estimate_is_net_of_cached_reads_and_is_a_range(tmp_path: Path) -> None:
    """Cached reads replay free: a re-run of a partly-read volume costs far less than
    its sheet count. The low end is the unread primaries and nothing else."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 100)
    ann = roots["work"] / "vol_a" / "annotations"
    ann.mkdir()
    for i in range(1, 41):
        (ann / f"p{i}.json").write_text("{}")

    c = console_backlog.candidates(_status(roots), work=roots["work"], city=_city(declared={}))[0]
    assert c.calls is not None
    assert c.cached_reads == 40
    assert c.calls.low == 60  # 100 sheets - 40 already paid for
    # the ceiling contains BOTH other spenders (budget.estimate_spend): the gated pool (41
    # pages) through a 2-tier ladder, and the retry — every read is one attempt plus one
    # possible retry, and a retry is a second billable call. Escalation is the only other
    # spender: the addresses channel stopped buying reads when its producer was cut so a
    # declared "addresses" channel adds NOTHING to this ceiling.
    assert c.calls.ceiling == 2 * (60 + 41 * 2)
    assert str(c.calls) == "60-284"


def test_a_fully_read_volume_estimates_no_new_primary_reads(tmp_path: Path) -> None:
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 10)
    ann = roots["work"] / "vol_a" / "annotations"
    ann.mkdir()
    for i in range(1, 11):
        (ann / f"p{i}.json").write_text("{}")

    c = console_backlog.candidates(_status(roots), work=roots["work"])[0]
    assert c.calls is not None and c.calls.low == 0


def test_estimate_never_goes_negative_when_more_pages_were_read_than_addressed(
    tmp_path: Path,
) -> None:
    """Annotations can outnumber addressable sheets (front matter read once, an image
    later pruned). The floor is zero, not a negative "we owe you calls"."""
    est = estimate_spend(sheets=5, cached=9, escalation_tiers=2)
    assert est.low == 0 and est.ceiling >= 0 and est.cached == 5


def test_escalation_attempts_default_tracks_escalate_max_attempts() -> None:
    """Escalation bills up to `escalate.MAX_ATTEMPTS` per tier regardless of the
    annotate batch's retry setting, so the estimator's default must track it."""
    import inspect

    from autogeoref import escalate
    from autogeoref.budget import ESCALATION_ATTEMPTS

    assert ESCALATION_ATTEMPTS == escalate.MAX_ATTEMPTS
    sig = inspect.signature(estimate_spend)
    assert sig.parameters["escalation_attempts"].default == escalate.MAX_ATTEMPTS
    # the escalation term multiplies by ITS attempts, not the annotate batch's
    est = estimate_spend(sheets=10, cached=10, unread=0, escalation_tiers=1, attempts=1)
    assert est.ceiling == escalate.MAX_ATTEMPTS * est.gated


def test_the_call_estimate_reads_the_city_gated_fraction(tmp_path: Path) -> None:
    """The gated-pool fraction is a CITY measurement (Chicago's corpus: 0.41),
    so a city that declares its own `gated_fraction` must move the estimate
    ceiling — it cannot silently inherit another city's number."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 100)
    rows = _status(roots)
    # the fraction only ever scales the ESCALATION arm, so the volume must name a
    # ladder for this to measure anything (the helper's VolumeConfig inherits none)
    base = _city(declared={"vol_a": True})
    laddered = replace(
        base.volumes["vol_a"], escalation_models=("claude-sonnet-5", "claude-opus-4-8")
    )
    base = replace(base, volumes={"vol_a": laddered})
    half = replace(base, gated_fraction=0.5)

    default_est = console_backlog.candidates(rows, work=roots["work"], city=base)[0].calls
    half_est = console_backlog.candidates(rows, work=roots["work"], city=half)[0].calls
    assert default_est is not None and half_est is not None
    assert default_est.ceiling == 2 * (100 + 41 * 2)
    assert half_est.ceiling == 2 * (100 + 50 * 2)
