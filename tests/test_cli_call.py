"""Running one model CLI, and judging what came back.

The seam under test: `run_cli` reports what a process produced and judges
nothing; `classify` judges a failed read of that report. Splitting them is what
lets a caller own its schema without leaving the failure taxonomy — the whole
point, since a provider refusal is usually valid JSON and only a schema check
rejects it.

No network and no real subprocess: every test injects a runner.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from autogeoref.annotate import cli_call
from autogeoref.annotate.cli_call import CliOutcome, classify, run_cli
from autogeoref.annotate.failures import (
    AnnotationCallError,
    AnnotatorProcessError,
    BudgetLimitError,
    MalformedResponseError,
)
from autogeoref.annotate.invocation import CliBackend
from autogeoref.annotate.providers import CLAUDE_CLI, CODEX_CLI
from autogeoref.annotate.schema import _json_object_from_cli_text
from autogeoref.street_index import _read_tile_cli


@pytest.fixture
def image(tmp_path: Path) -> Path:
    img = tmp_path / "p26_small.jpg"
    img.write_bytes(b"x")
    return img


def codex_runner(last_message: str | None) -> Any:
    """A fake ``codex exec``: writes its answer where the real one does."""

    def runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        if last_message is not None and "--output-last-message" in argv:
            Path(argv[argv.index("--output-last-message") + 1]).write_text(last_message)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return runner


def _outcome(
    text: str, *, stdout: str = "", stderr: str = "", returncode: int = 0, provider: Any = CODEX_CLI
) -> CliOutcome:
    return CliOutcome(
        text=text,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        provider=provider,
        binary="codex",
        image_name="p26_small.jpg",
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        # a structured refusal is read whole, whatever the provider's log scope
        (_outcome('{"code": "usage_limit_reached"}'), BudgetLimitError),
        # ...but prose only reports a limit through the scope
        (_outcome("we are budget-gated per the agent instructions"), MalformedResponseError),
        (_outcome("x", stderr="ERROR: usage limit reached"), BudgetLimitError),
        (_outcome("x", stderr="quota exhausted"), MalformedResponseError),
        (_outcome("y", stderr="quota exhausted", provider=CLAUDE_CLI), BudgetLimitError),
        # a limit outranks a bad exit code; a bad exit code outranks a bad read
        (_outcome('{"code": "usage_limit_reached"}', returncode=1), BudgetLimitError),
        (_outcome("garbage", returncode=1), AnnotatorProcessError),
        (_outcome("garbage"), MalformedResponseError),
    ],
)
def test_classify_judges_an_outcome_directly(outcome: CliOutcome, expected: type) -> None:
    """The classifier is a function over a value, so its whole table is testable.

    As an `except` block inside the transport it could only be reached by
    driving a fake subprocess, which is why the arms below went unexamined long
    enough for a decoded refusal to slip past two of them.
    """
    with pytest.raises(expected):
        classify(outcome, MalformedResponseError("the read failed"))


def test_run_cli_judges_nothing(image: Path) -> None:
    """It reports; it does not interpret. A refusal comes back as data."""
    refusal = '{"type": "error", "code": "usage_limit_reached"}'
    outcome = run_cli(
        image, "prompt IMGPATH", model="codex:gpt-5.6-terra", runner=codex_runner(refusal)
    )
    assert outcome.text.strip() == refusal
    assert outcome.returncode == 0
    assert outcome.provider is CODEX_CLI
    assert outcome.image_name == image.name


def test_classify_never_downgrades_a_failure_that_is_already_terminal() -> None:
    """A caller wrapping every step can hand in a check that knows it is terminal.

    The evidence here — a payload with no budget vocabulary, a non-zero exit —
    would otherwise re-label it a retryable process error, which is the exact
    terminal-becomes-retryable inversion this seam exists to prevent.
    """
    already = BudgetLimitError("my schema knows this is a hard stop")
    with pytest.raises(BudgetLimitError) as info:
        classify(_outcome('{"ok": true}', returncode=1), already)
    assert info.value is already


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda img: CliBackend("codex:gpt-5.6-terra").annotate_extended(img), id="v2"),
        pytest.param(lambda img: _read_tile_cli(img, "codex:gpt-5.6-terra", 60.0), id="index-tile"),
    ],
)
def test_the_production_readers_wrap_their_schema_check(
    image: Path, monkeypatch: pytest.MonkeyPatch, call: Any
) -> None:
    """Both take no runner, so nothing else can reach their `try` blocks.

    Each owns a schema and decodes a refusal cleanly, so only the wrapping makes
    the limit terminal. Un-wrap either and it reports a retryable malformed
    read: doomed retries, then a marker that reads like an unreadable sheet.
    """
    refusal = '{"type": "error", "code": "usage_limit_reached"}'
    monkeypatch.setattr(cli_call, "_default_runner", codex_runner(refusal))
    with pytest.raises(BudgetLimitError):
        call(image)


def test_a_schema_check_inside_the_classified_block_is_what_catches_a_refusal(
    image: Path,
) -> None:
    """Why the seam is `run_cli` + `classify` and not "decode, then validate".

    A provider refusal is often itself valid JSON, so it decodes and only a
    schema check rejects it. Wrap that check and hand the failure to `classify`
    and it is terminal; run it on a value the transport already returned and it
    is a retryable malformed read — two doomed calls per page, then a marker
    that reads like an unreadable sheet.
    """
    limit = '{"type": "error", "code": "usage_limit_reached"}'

    def check(raw: dict[str, Any]) -> None:
        raise MalformedResponseError(f"not my schema: {raw}")

    def read_the_way_a_schema_owner_should() -> None:
        outcome = run_cli(
            image, "prompt IMGPATH", model="codex:gpt-5.6-terra", runner=codex_runner(limit)
        )
        try:
            check(_json_object_from_cli_text(outcome.text))
        except AnnotationCallError as exc:
            classify(outcome, exc)

    with pytest.raises(BudgetLimitError):
        read_the_way_a_schema_owner_should()

    # judged outside, the same response is merely malformed — the shape of the bug
    outcome = run_cli(
        image, "prompt IMGPATH", model="codex:gpt-5.6-terra", runner=codex_runner(limit)
    )
    with pytest.raises(MalformedResponseError):
        check(_json_object_from_cli_text(outcome.text))
