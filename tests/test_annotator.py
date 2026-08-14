"""Tests for the vision annotator interface and backends (all model calls mocked)."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

from autogeoref.annotate.api_call import AnthropicBackend
from autogeoref.annotate.failures import (
    AnnotationCallError,
    AnnotationTimeoutError,
    AnnotatorProcessError,
    BudgetLimitError,
    EmptyResponseError,
    MalformedResponseError,
    ModelQualityError,
    TransientRateLimitError,
)
from autogeoref.annotate.invocation import ClaudeCLIBackend
from autogeoref.annotate.providers import (
    DEFAULT_TIMEOUT,
    SMALL_MODEL_PATTERNS,
    ensure_model_allowed,
)
from autogeoref.annotate.schema import (
    EXTENDED_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE,
    Annotation,
    StreetLabel,
)
from conftest import load_script

qualify_backend = load_script("experiments/qualification.py").qualify_backend

SAMPLE = {
    "streets": [
        {"name": "W. MADISON", "bbox": [705, 213, 935, 253], "orientation": "horizontal"},
        {"name": "S. WESTERN AV.", "bbox": [168, 1075, 235, 1560], "orientation": "vertical"},
    ],
    "page_number_seen": "1 (699)",
}


# ----------------------------------------------------------------------
# Data model: JSON round-trip against the frozen fixture schema
# ----------------------------------------------------------------------


def _fixture_annotation_files(fixtures_dir: Path) -> list[Path]:
    # `p<N>.json` ONLY — the plain v1 reads whose schema this round-trip pins. The same
    # directory also holds the OTHER producers' caches, which `p*.json` happily matches and
    # which are a different schema entirely: the addresses channel's consensus sidecars
    # (`p<N>.v2.<model>.json`, raw v2 dicts carrying address_numerals) and the escalation
    # ladder's tier caches (`p<N>.escalated.<model>.json`). A model name is dotted, so "no dot
    # in the stem" separates them exactly and keeps doing so as producers are added.
    def is_v1(p: Path) -> bool:
        return "." not in p.stem

    files = sorted(
        p for p in (fixtures_dir / "sanborn01790_024" / "annotations").glob("p*.json") if is_v1(p)
    )
    files += sorted(
        p for p in (fixtures_dir / "ref-volume" / "annotations").glob("p*.json") if is_v1(p)
    )
    if not files:
        pytest.skip("no annotation fixtures present")
    # A representative spread, not the whole tree.
    return files[:: max(1, len(files) // 12)]


def test_round_trip_matches_fixture_files(fixtures_dir: Path) -> None:
    files = _fixture_annotation_files(fixtures_dir)
    assert len(files) >= 5
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        annotation = Annotation.from_dict(data)
        assert annotation.to_dict() == data, path
        assert json.loads(annotation.to_json()) == data, path
        assert Annotation.from_json(path.read_text(encoding="utf-8")) == annotation


def test_street_label_fields() -> None:
    annotation = Annotation.from_dict(SAMPLE)
    assert annotation.page_number_seen == "1 (699)"
    assert annotation.streets[0] == StreetLabel(
        name="W. MADISON", bbox=(705, 213, 935, 253), orientation="horizontal"
    )
    assert annotation.streets[1].orientation == "vertical"


def test_prompt_is_the_production_prompt() -> None:
    assert PROMPT_TEMPLATE.startswith("Look at the Sanborn fire insurance map sheet image at ")
    assert "IMGPATH" in PROMPT_TEMPLATE
    assert '"streets"' in PROMPT_TEMPLATE
    assert '"page_number_seen"' in PROMPT_TEMPLATE
    assert '"SEE VOLUME" notes' in PROMPT_TEMPLATE
    assert PROMPT_TEMPLATE.endswith("building names, and railroad names.")


# ----------------------------------------------------------------------
# Quality gate
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-4-5",
        "gpt-4o-mini",
        "gemini-2.0-flash",
        "llama-3.1-8b",
        "Claude-HAIKU",
        "gemma-2b-it",
        "qwen2-vl-2b-instruct",
        "phi-3-vision-4b",
    ],
)
def test_quality_gate_refuses_small_models(model: str) -> None:
    with pytest.raises(ModelQualityError):
        ensure_model_allowed(model)
    with pytest.raises(ModelQualityError):
        AnthropicBackend(model)
    with pytest.raises(ModelQualityError):
        ClaudeCLIBackend(model)


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-4-8"])
def test_quality_gate_allows_sonnet_class(model: str) -> None:
    ensure_model_allowed(model)
    # an injected client: building a real one now demands a credential, which
    # is a separate contract (tests/test_api_backends.py) from this gate
    assert AnthropicBackend(model, client=FakeAnthropicClient(None)).model == model  # type: ignore[arg-type]
    assert ClaudeCLIBackend(model).model == model


def test_quality_gate_is_user_extensible() -> None:
    blocked = (*SMALL_MODEL_PATTERNS, "sonnet")
    with pytest.raises(ModelQualityError):
        AnthropicBackend("claude-sonnet-5", blocked_patterns=blocked)


# ----------------------------------------------------------------------
# Failure classification
# ----------------------------------------------------------------------


def test_failure_kinds_are_mutually_distinguishable() -> None:
    kinds = (
        EmptyResponseError,
        BudgetLimitError,
        TransientRateLimitError,
        AnnotationTimeoutError,
        MalformedResponseError,
        AnnotatorProcessError,
    )
    for kind in kinds:
        assert issubclass(kind, AnnotationCallError)
        for other in kinds:
            if other is not kind:
                assert not issubclass(kind, other)


# ----------------------------------------------------------------------
# AnthropicBackend (mocked client — no real API calls)
# ----------------------------------------------------------------------


class FakeBlock:
    def __init__(self, text: str, block_type: str = "text") -> None:
        self.text = text
        self.type = block_type


class FakeMessage:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [FakeBlock(text)]
        self.stop_reason = stop_reason


class FakeMessagesAPI:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeAnthropicClient:
    def __init__(self, outcome: Any) -> None:
        self.messages = FakeMessagesAPI(outcome)


@pytest.fixture
def sheet_image(tmp_path: Path) -> Path:
    path = tmp_path / "p1_small.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fakejpegdata")
    return path


def _api_error_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def test_anthropic_backend_success(sheet_image: Path) -> None:
    fake = FakeAnthropicClient(FakeMessage(json.dumps(SAMPLE)))
    backend = AnthropicBackend(client=fake)  # type: ignore[arg-type]
    annotation = backend.annotate(sheet_image)
    assert annotation.to_dict() == SAMPLE

    (call,) = fake.messages.calls
    assert call["model"] == backend.model
    assert call["timeout"] == DEFAULT_TIMEOUT >= 600
    content = call["messages"][0]["content"]
    image_block, text_block = content
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/jpeg"
    decoded = base64.standard_b64decode(image_block["source"]["data"])
    assert decoded == sheet_image.read_bytes()
    # the v2 extended template is the production default (promoted on measured evidence);
    # the frozen v1 template stays selectable for reference-parity runs
    assert text_block["text"] == EXTENDED_PROMPT_TEMPLATE.replace("IMGPATH", sheet_image.name)
    assert "IMGPATH" not in text_block["text"]

    fake_v1 = FakeAnthropicClient(FakeMessage(json.dumps(SAMPLE)))
    v1 = AnthropicBackend(client=fake_v1, prompt_template=PROMPT_TEMPLATE)  # type: ignore[arg-type]
    v1.annotate(sheet_image)
    (v1_call,) = fake_v1.messages.calls
    assert v1_call["messages"][0]["content"][1]["text"] == PROMPT_TEMPLATE.replace(
        "IMGPATH", sheet_image.name
    )


def test_anthropic_backend_timeout_classified(sheet_image: Path) -> None:
    fake = FakeAnthropicClient(anthropic.APITimeoutError(request=_api_error_request()))
    backend = AnthropicBackend(client=fake)  # type: ignore[arg-type]
    with pytest.raises(AnnotationTimeoutError):
        backend.annotate(sheet_image)


def test_anthropic_backend_rate_limit_classified_as_transient(sheet_image: Path) -> None:
    """A 429 is transient back-off pressure, NOT a terminal budget condition —
    a batch driver halting on BudgetLimitError must not halt on rate limits."""
    request = _api_error_request()
    error = anthropic.RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )
    backend = AnthropicBackend(client=FakeAnthropicClient(error))  # type: ignore[arg-type]
    with pytest.raises(TransientRateLimitError) as info:
        backend.annotate(sheet_image)
    assert not isinstance(info.value, BudgetLimitError)


def test_anthropic_backend_credit_error_classified_as_budget(sheet_image: Path) -> None:
    request = _api_error_request()
    error = anthropic.BadRequestError(
        "Your credit balance is too low to access the API.",
        response=httpx.Response(400, request=request),
        body=None,
    )
    backend = AnthropicBackend(client=FakeAnthropicClient(error))  # type: ignore[arg-type]
    with pytest.raises(BudgetLimitError):
        backend.annotate(sheet_image)


def test_anthropic_backend_empty_response_classified(sheet_image: Path) -> None:
    backend = AnthropicBackend(client=FakeAnthropicClient(FakeMessage("")))  # type: ignore[arg-type]
    with pytest.raises(EmptyResponseError):
        backend.annotate(sheet_image)


def test_anthropic_backend_malformed_response_classified(sheet_image: Path) -> None:
    fake = FakeAnthropicClient(FakeMessage("I could not find any streets, sorry!"))
    backend = AnthropicBackend(client=fake)  # type: ignore[arg-type]
    with pytest.raises(MalformedResponseError):
        backend.annotate(sheet_image)


def test_anthropic_backend_timeout_is_configurable() -> None:
    backend = AnthropicBackend(timeout=900.0, client=FakeAnthropicClient(None))  # type: ignore[arg-type]
    assert backend.timeout == 900.0


# ----------------------------------------------------------------------
# ClaudeCLIBackend (mocked subprocess runner — no real processes)
# ----------------------------------------------------------------------


def make_runner(
    stdout: str, *, stderr: str = "", returncode: int = 0, raises: Exception | None = None
) -> Any:
    calls: list[tuple[list[str], float]] = []

    def runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append((argv, timeout))
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_cli_backend_success_and_argv(tmp_path: Path) -> None:
    image = tmp_path / "p7_small.jpg"
    image.write_bytes(b"x")
    runner = make_runner(json.dumps(SAMPLE))
    backend = ClaudeCLIBackend(runner=runner)
    annotation = backend.annotate(image)
    assert annotation.to_dict() == SAMPLE

    ((argv, timeout),) = runner.calls
    assert timeout == DEFAULT_TIMEOUT
    assert argv[0] == "claude"
    assert argv[1] == "-p"
    # v2 extended template is the default; v1 stays selectable
    assert argv[2] == EXTENDED_PROMPT_TEMPLATE.replace("IMGPATH", str(image))
    v1 = ClaudeCLIBackend(runner=make_runner(json.dumps(SAMPLE)), prompt_template=PROMPT_TEMPLATE)
    assert v1.build_argv(image)[2] == PROMPT_TEMPLATE.replace("IMGPATH", str(image))
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == backend.model
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "json"


def test_cli_backend_unwraps_json_envelope(tmp_path: Path) -> None:
    image = tmp_path / "p8_small.jpg"
    image.write_bytes(b"x")
    envelope = json.dumps({"type": "result", "result": json.dumps(SAMPLE)})
    backend = ClaudeCLIBackend(runner=make_runner(envelope))
    assert backend.annotate(image).to_dict() == SAMPLE


def test_cli_backend_timeout_classified(tmp_path: Path) -> None:
    image = tmp_path / "p9_small.jpg"
    image.write_bytes(b"x")
    err = subprocess.TimeoutExpired(cmd=["claude"], timeout=600)
    backend = ClaudeCLIBackend(runner=make_runner("", raises=err))
    with pytest.raises(AnnotationTimeoutError):
        backend.annotate(image)


def test_cli_backend_budget_message_classified(tmp_path: Path) -> None:
    image = tmp_path / "p10_small.jpg"
    image.write_bytes(b"x")
    backend = ClaudeCLIBackend(
        runner=make_runner("Claude usage limit reached. Try again later.", returncode=1)
    )
    with pytest.raises(BudgetLimitError):
        backend.annotate(image)


#: Verbatim shape of the claude CLI's subscription-limit refusal: a RESULT
#: envelope whose inner result is prose naming the limit's WINDOW, exit 1
#: (`` §4).
CLAUDE_WEEKLY_LIMIT_STDOUT = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": 429,
        "duration_ms": 452,
        "num_turns": 1,
        "result": "You've hit your weekly limit · resets Jul 24, 6am (America/Chicago)",
        "stop_reason": "stop_sequence",
    }
)


def test_cli_backend_weekly_limit_refusal_is_terminal(tmp_path: Path) -> None:
    """Terminal for the stage — must NOT classify as a process error.

    A process error is retried and then written as a permanent
    `.failed.json` marker: doomed spend now, silently blocked retries later.
    """
    image = tmp_path / "p13_small.jpg"
    image.write_bytes(b"x")
    backend = ClaudeCLIBackend(runner=make_runner(CLAUDE_WEEKLY_LIMIT_STDOUT, returncode=1))
    with pytest.raises(BudgetLimitError) as exc:
        backend.annotate(image)
    assert "weekly limit" in str(exc.value)


def test_cli_backend_empty_output_classified(tmp_path: Path) -> None:
    image = tmp_path / "p11_small.jpg"
    image.write_bytes(b"x")
    backend = ClaudeCLIBackend(runner=make_runner(""))
    with pytest.raises(EmptyResponseError):
        backend.annotate(image)


def test_cli_backend_crash_is_a_process_error_not_an_empty_response(tmp_path: Path) -> None:
    """A nonzero exit with no usable output is infrastructure failure — it
    must not masquerade as 'the model returned an empty response'."""
    image = tmp_path / "p12_small.jpg"
    image.write_bytes(b"x")
    with pytest.raises(AnnotatorProcessError) as info:
        ClaudeCLIBackend(runner=make_runner("", returncode=127)).annotate(image)
    assert "127" in str(info.value)
    assert not isinstance(info.value, EmptyResponseError)
    # nonzero exit with a stderr traceback: still a process error, not "malformed"
    with pytest.raises(AnnotatorProcessError):
        ClaudeCLIBackend(
            runner=make_runner("", stderr="Traceback (most recent call last): ...", returncode=1)
        ).annotate(image)
    # but a nonzero exit whose output is a BUDGET message keeps that class
    with pytest.raises(BudgetLimitError):
        ClaudeCLIBackend(
            runner=make_runner("", stderr="Claude usage limit reached.", returncode=1)
        ).annotate(image)
    # and a ZERO exit with valid JSON is unaffected
    assert ClaudeCLIBackend(runner=make_runner(json.dumps(SAMPLE))).annotate(image)


# ----------------------------------------------------------------------
# Model-qualification harness
# ----------------------------------------------------------------------


class FixtureBackend:
    """Fake backend that answers with the cached fixture annotation."""

    def __init__(self, answers: dict[Path, Annotation]) -> None:
        self.answers = answers

    def annotate(self, image_path: Path) -> Annotation:
        return self.answers[image_path]


class DegradedBackend:
    """Fake backend that drops half the streets and hallucinates one."""

    def __init__(self, answers: dict[Path, Annotation]) -> None:
        self.answers = answers

    def annotate(self, image_path: Path) -> Annotation:
        truth = self.answers[image_path]
        kept = truth.streets[: len(truth.streets) // 2]
        bogus = StreetLabel(name="IMAGINARY BLVD", bbox=(0, 0, 1, 1), orientation="horizontal")
        return Annotation(streets=(*kept, bogus), page_number_seen=truth.page_number_seen)


class FailingBackend:
    def annotate(self, image_path: Path) -> Annotation:
        raise BudgetLimitError(f"usage limit reached while annotating {image_path.name}")


def _qualification_cases(fixtures_dir: Path) -> list[tuple[Path, Annotation]]:
    files = sorted((fixtures_dir / "sanborn01790_024" / "annotations").glob("p*.json"))[:5]
    if not files:
        pytest.skip("no annotation fixtures present")
    cases = []
    for path in files:
        annotation = Annotation.from_json(path.read_text(encoding="utf-8"))
        if annotation.streets:
            cases.append((path.with_suffix(".jpg"), annotation))
    return cases


def test_qualification_perfect_backend_scores_one(fixtures_dir: Path) -> None:
    cases = _qualification_cases(fixtures_dir)
    backend = FixtureBackend(dict(cases))
    report = qualify_backend(backend, cases)
    assert len(report.scores) == len(cases)
    assert report.mean_recall == pytest.approx(1.0)
    assert report.mean_precision == pytest.approx(1.0)
    assert report.failures == ()
    for score in report.scores:
        assert score.expected == score.predicted > 0
        assert score.error is None


def test_qualification_degraded_backend_scores_lower(fixtures_dir: Path) -> None:
    cases = _qualification_cases(fixtures_dir)
    report = qualify_backend(DegradedBackend(dict(cases)), cases)
    assert report.mean_recall < 0.8
    assert report.mean_precision < 1.0
    assert report.mean_recall > 0.0  # it still finds some streets


def test_qualification_records_failures(fixtures_dir: Path) -> None:
    cases = _qualification_cases(fixtures_dir)
    report = qualify_backend(FailingBackend(), cases)
    assert len(report.failures) == len(cases)
    assert report.mean_recall == 0.0
    assert all(s.error is not None and "BudgetLimitError" in s.error for s in report.scores)


def test_a_crashed_cli_records_the_reason_not_just_its_banner(tmp_path: Path) -> None:
    """These CLIs open with a fixed banner and print the reason last, so a
    head-only excerpt keeps boilerplate and drops the diagnosis — and a failure
    marker that cannot say why costs a re-read to find out."""
    image = tmp_path / "p35_small.jpg"
    image.write_bytes(b"x")
    banner = "OpenAI Codex v0.144.1\n" + "workdir: /repo\nmodel: m\n" * 40
    reason = "error: upstream connect error, transport failure"
    with pytest.raises(AnnotatorProcessError) as info:
        ClaudeCLIBackend(
            runner=make_runner("", stderr=f"{banner}\n{reason}", returncode=1)
        ).annotate(image)
    assert reason in str(info.value)
    assert "OpenAI Codex" in str(info.value)


@pytest.mark.parametrize(
    "text",
    [
        "Claude usage limit reached. Your limit resets at 3am.",
        "Your credit balance is too low to run this request.",
        "You've hit your weekly limit · resets Jul 24, 6am",
        # the same limits spelled machine-readably. These DECODE, so the schema
        # check is what rejects them; run it outside the classifier and they
        # come back malformed, i.e. retryable
        '{"type": "error", "code": "usage_limit_reached"}',
        '{"type": "error", "code": "insufficient_quota"}',
        'ERROR: {"type": "error", "code": "usage_limit_reached"}',
    ],
)
@pytest.mark.parametrize("returncode", [0, 1])
def test_every_spelling_of_a_refusal_is_a_budget_stop(
    tmp_path: Path, text: str, returncode: int
) -> None:
    """An unrecognised limit is the worse failure: it writes a per-page marker
    that reads like an unreadable sheet, so the next pass clears it and spends
    again into the same wall.

    Both exit codes: a refusal that decodes used to reach the schema check past
    the point BOTH the budget arm and the non-zero-exit arm run, so a CLI
    exiting 1 on a limit was reported as a malformed read too.
    """
    image = tmp_path / "p3_small.jpg"
    image.write_bytes(b"x")
    with pytest.raises(BudgetLimitError):
        ClaudeCLIBackend(runner=make_runner(text, returncode=returncode)).annotate(image)
