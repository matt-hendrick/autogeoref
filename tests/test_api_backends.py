"""The direct-API providers: OpenAI Responses, Anthropic Messages, and routing.

No network. Every test injects a client, exactly as the CLI backend tests inject
a runner. What is under test is that a direct API is a MODEL REFERENCE and
nothing else: the prompt, the parsers, the Sonnet-class gate and the failure
taxonomy stay shared with the CLI transports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from autogeoref.annotate.api_call import AnthropicBackend, OllamaBackend, OpenAIBackend
from autogeoref.annotate.failures import (
    AnnotateError,
    AnnotationCallError,
    AnnotationTimeoutError,
    AnnotatorProcessError,
    BudgetLimitError,
    EmptyResponseError,
    MalformedResponseError,
    ModelQualityError,
    TransientRateLimitError,
)
from autogeoref.annotate.invocation import _NATIVE_BACKENDS, CliBackend, backend_for_model
from autogeoref.annotate.providers import (
    NATIVE_PROVIDERS,
    VARIANT_PROVIDERS,
    canonical_model,
    parse_model_ref,
    resolve_provider,
)
from autogeoref.annotate.schema import EXTENDED_PROMPT_TEMPLATE

SAMPLE: dict[str, Any] = {
    "streets": [{"name": "W. ADAMS", "bbox": [1, 2, 3, 4], "orientation": "horizontal"}],
    "page_number_seen": "26",
}
MODEL = "gpt-5.6-terra"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Placeholder keys: building a backend now demands one, and no test calls out.

    Set rather than inherited, so a developer's real key is never the reason a
    test passes — and never travels to a provider either.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-one")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-one")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    img = tmp_path / "p26_small.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpegdata")
    return img


class FakeResponse:
    """The subset of an OpenAI ``Response`` the backend reads."""

    def __init__(
        self,
        output_text: str = "",
        *,
        status: str = "completed",
        incomplete_reason: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.output_text = output_text
        self.status = status
        self.incomplete_details = (
            None if incomplete_reason is None else type("D", (), {"reason": incomplete_reason})()
        )
        self.error = None if error_message is None else type("E", (), {"message": error_message})()


class FakeMessage:
    """The subset of an Anthropic ``Message`` the backend reads."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [type("B", (), {"text": text, "type": "text"})()]
        self.stop_reason = stop_reason


class FakeResponsesAPI:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeOpenAIClient:
    def __init__(self, outcome: Any) -> None:
        self.responses = FakeResponsesAPI(outcome)


def _request(status: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        json={"error": body},
    )


@pytest.fixture
def sheet(tmp_path: Path) -> Path:
    path = tmp_path / "p1_small.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fakejpegdata")
    return path


def _anthropic_client(outcome: Any) -> Any:
    """An Anthropic client stub whose ``messages.create`` replays ``outcome``."""

    class Messages:
        def create(self, **_kwargs: Any) -> Any:
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    return type("Client", (), {"messages": Messages()})()


# ----------------------------------------------------------------------
# Routing: a direct API is a provider prefix, nothing more
# ----------------------------------------------------------------------


def test_every_native_provider_has_a_backend() -> None:
    """The two lists are written apart and must not drift: a name accepted by
    the config loader with no backend behind it fails at the first read."""
    assert set(_NATIVE_BACKENDS) == NATIVE_PROVIDERS


def test_backend_for_model_routes_the_direct_apis() -> None:
    assert isinstance(backend_for_model("anthropic-api:claude-sonnet-5"), AnthropicBackend)
    assert isinstance(backend_for_model(f"openai-api:{MODEL}"), OpenAIBackend)
    assert isinstance(backend_for_model("ollama:gemma4:12b"), OllamaBackend)
    assert isinstance(backend_for_model("claude-sonnet-5"), CliBackend)


def test_a_direct_api_read_is_its_own_cache_identity() -> None:
    """The CLI wraps the model in an agent harness, so the same name over the
    API is a different read. Sharing a cache key would replay one as the other."""
    assert canonical_model("anthropic-api:claude-sonnet-5") == "anthropic-api:claude-sonnet-5"
    assert canonical_model("claude-sonnet-5") == "claude-sonnet-5"


def test_a_bare_openai_name_names_both_spellings_that_work() -> None:
    with pytest.raises(AnnotateError) as info:
        parse_model_ref(MODEL)
    assert f"codex:{MODEL}" in str(info.value)
    assert f"openai-api:{MODEL}" in str(info.value)


def test_the_quality_gate_covers_the_direct_apis() -> None:
    with pytest.raises(ModelQualityError):
        parse_model_ref("openai-api:gpt-5.4-mini")
    with pytest.raises(ModelQualityError):
        parse_model_ref("anthropic-api:claude-haiku-4-5")


def test_a_native_provider_is_not_reported_as_an_unknown_one() -> None:
    """`resolve_provider` is the CLI lookup; a native name is a caller error
    with a backend behind it, not a typo."""
    with pytest.raises(AnnotateError, match="reached over HTTP"):
        resolve_provider("openai-api")
    with pytest.raises(AnnotateError, match="unknown CLI provider"):
        resolve_provider("openia-api")


def test_a_variant_is_refused_where_it_would_be_silently_dropped() -> None:
    assert {"codex", "opencode", "openai-api"} == VARIANT_PROVIDERS
    reader = backend_for_model(f"openai-api:{MODEL}", variant="high")
    assert isinstance(reader, OpenAIBackend)
    assert reader.variant == "high"
    for reference in ("anthropic-api:claude-sonnet-5", "ollama:gemma4:12b", "claude-sonnet-5"):
        with pytest.raises(AnnotateError, match=r"no reasoning variant|only by the Codex"):
            backend_for_model(reference, variant="high")


def test_a_missing_sdk_names_the_extra_that_installs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    with pytest.raises(AnnotateError, match=r"autogeoref\[openai\]"):
        OpenAIBackend(MODEL)
    with pytest.raises(AnnotateError, match=r"autogeoref\[annotate\]"):
        AnthropicBackend("claude-sonnet-5")
    # an injected client needs no SDK lookup at all
    assert OpenAIBackend(MODEL, client=FakeOpenAIClient(None)).model == MODEL  # type: ignore[arg-type]


def test_a_missing_key_is_a_build_error_not_a_worker_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both SDKs are built when the backend is, so a stage that builds its
    reader up front reports this before it spends. Left to the first read,
    OpenAI raises its own error class and Anthropic a bare ``TypeError``, and
    neither is in the taxonomy a worker catches."""
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(AnnotateError, match=r"OPENAI_API_KEY|could not be built"):
        OpenAIBackend(MODEL)
    with pytest.raises(AnnotateError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend("claude-sonnet-5")

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    assert AnthropicBackend("claude-sonnet-5").model == "claude-sonnet-5"


def test_the_run_preamble_names_an_uncallable_direct_api(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The CLI providers get a "not on PATH" line before prep; the direct APIs
    need the same warning, or a missing key surfaces after prep has worked."""
    import logging

    from autogeoref.runpolicy import RunPolicy

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    policy = RunPolicy(
        warp=False,
        escalation_models=(),
        run_escalation=False,
        run_junction=False,
        run_verified=False,
        allowed_channels=frozenset(),
    )
    with caplog.at_level(logging.WARNING):
        policy.warn_unavailable_model_clis(f"openai-api:{MODEL}")
    assert "OPENAI_API_KEY is not set" in caplog.text
    assert MODEL in caplog.text

    caplog.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with caplog.at_level(logging.WARNING):
        policy.warn_unavailable_model_clis(f"openai-api:{MODEL}")
        policy.warn_unavailable_model_clis("ollama:gemma4:12b")
    assert caplog.text == "", "a satisfied provider, and Ollama, warn about nothing"


# ----------------------------------------------------------------------
# OpenAI Responses request shape
# ----------------------------------------------------------------------


def test_openai_sends_the_sheet_as_a_base64_data_url(image: Path) -> None:
    fake = FakeOpenAIClient(FakeResponse(json.dumps(SAMPLE)))
    backend = OpenAIBackend(MODEL, client=fake)  # type: ignore[arg-type]
    assert backend.annotate(image).to_dict() == SAMPLE

    (call,) = fake.responses.calls
    assert call["model"] == MODEL
    assert call["timeout"] == 600.0
    assert call["max_output_tokens"] == backend.max_tokens
    assert "reasoning" not in call, "no variant configured means no effort override"
    image_part, text_part = call["input"][0]["content"]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "high", "street labels are small; auto may downsample them"
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")
    assert text_part["type"] == "input_text"
    assert text_part["text"] == EXTENDED_PROMPT_TEMPLATE.replace("IMGPATH", image.name)
    assert "IMGPATH" not in text_part["text"]


def test_openai_variant_sets_the_reasoning_effort(image: Path) -> None:
    fake = FakeOpenAIClient(FakeResponse(json.dumps(SAMPLE)))
    OpenAIBackend(MODEL, client=fake, variant="high").annotate(image)  # type: ignore[arg-type]
    assert fake.responses.calls[0]["reasoning"] == {"effort": "high"}


def test_openai_refuses_an_image_type_it_cannot_declare(tmp_path: Path) -> None:
    bad = tmp_path / "p1_small.tif"
    bad.write_bytes(b"x")
    with pytest.raises(AnnotateError, match="unsupported image type"):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(None)).annotate(bad)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# OpenAI failure classification
# ----------------------------------------------------------------------


def test_openai_timeout_classified(image: Path) -> None:
    error = openai.APITimeoutError(request=_request(408, {}).request)
    with pytest.raises(AnnotationTimeoutError):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(error)).annotate(image)  # type: ignore[arg-type]


def test_an_ordinary_429_is_transient_not_terminal(image: Path) -> None:
    """OpenAI says "rate limit" in the prose of both a back-off and an exhausted
    balance, so the two are told apart by error CODE. Getting this backwards
    halts a whole batch on ordinary pressure."""
    body = {"message": "Rate limit reached for gpt-5.6", "code": "rate_limit_exceeded"}
    error = openai.RateLimitError("rate limited", response=_request(429, body), body=body)
    with pytest.raises(TransientRateLimitError) as info:
        OpenAIBackend(MODEL, client=FakeOpenAIClient(error)).annotate(image)  # type: ignore[arg-type]
    assert type(info.value) is TransientRateLimitError, "a sibling class, not a subclass"


@pytest.mark.parametrize(
    "body",
    [
        {"message": "You exceeded your current quota", "code": "insufficient_quota"},
        # the marker has always been on `type`; `code` is the newer field and
        # is not always populated. Reading only one classifies a spent balance
        # as back-off pressure and every page then walks into the same wall.
        {"message": "You exceeded your current quota", "type": "insufficient_quota"},
    ],
    ids=["code", "type"],
)
def test_an_exhausted_quota_is_terminal_even_though_it_is_a_429(
    image: Path, body: dict[str, Any]
) -> None:
    error = openai.RateLimitError("quota", response=_request(429, body), body=body)
    with pytest.raises(BudgetLimitError):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(error)).annotate(image)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("server", "OpenAI returned 503"),
        ("auth", "OpenAI returned 401"),
        ("connection", "OpenAI call for"),
    ],
)
def test_a_provider_failure_is_one_page_s_problem_not_the_batch_s(
    image: Path, error: str, expected: str
) -> None:
    """A 5xx, a bad key or a dropped connection must land in the taxonomy.

    Outside it, `annotate_with_retry` does not retry and the stage writes no
    failure marker: the exception escapes the pool and kills the run, so a
    single 503 on page 40 also costs every page after it. The CLI transport
    turns the same failures into a per-page `AnnotatorProcessError`.
    """
    body = {"message": "boom"}
    raised: Exception = {
        "server": openai.InternalServerError("503", response=_request(503, body), body=body),
        "auth": openai.AuthenticationError("401", response=_request(401, body), body=body),
        "connection": openai.APIConnectionError(request=_request(500, body).request),
    }[error]
    with pytest.raises(AnnotatorProcessError, match=expected) as info:
        OpenAIBackend(MODEL, client=FakeOpenAIClient(raised)).annotate(image)  # type: ignore[arg-type]
    assert isinstance(info.value, AnnotationCallError), "so a retry and a marker both happen"


def test_an_anthropic_provider_failure_lands_in_the_taxonomy_too(sheet: Path) -> None:
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.InternalServerError(
        "overloaded", response=httpx.Response(529, request=request), body=None
    )
    with pytest.raises(AnnotatorProcessError, match="Anthropic returned 529"):
        AnthropicBackend(client=_anthropic_client(error)).annotate(sheet)


def test_a_billing_status_error_is_a_budget_limit_by_code_or_by_message(image: Path) -> None:
    """Both routes are live: the SDK builds its message out of the response body,
    but a provider that adds a code we know says so without any prose scan."""
    coded = {"message": "no prose match here", "code": "billing_not_active"}
    error = openai.BadRequestError("400", response=_request(400, coded), body=coded)
    with pytest.raises(BudgetLimitError):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(error)).annotate(image)  # type: ignore[arg-type]

    body = {"message": "Your credit balance is too low", "code": "unrecognised"}
    spelled = openai.BadRequestError(
        f"Error code: 400 - {{'error': {body}}}", response=_request(400, body), body=body
    )
    with pytest.raises(BudgetLimitError):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(spelled)).annotate(image)  # type: ignore[arg-type]


def test_a_truncated_reply_names_the_ceiling_only_when_it_cost_the_read(image: Path) -> None:
    """Reasoning tokens are billed against `max_output_tokens`, so a dense sheet
    can stop mid-JSON — and the marker should name the cap rather than read as
    an unreadable sheet. A reply cut off AFTER its JSON closed lost nothing,
    though, so hitting the ceiling is not itself a failure."""
    cut = FakeResponse('{"streets": [', status="incomplete", incomplete_reason="max_output_tokens")
    with pytest.raises(MalformedResponseError, match="max_output_tokens=32000"):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(cut)).annotate(image)  # type: ignore[arg-type]

    whole = FakeResponse(
        json.dumps(SAMPLE) + "\nand that is ev",
        status="incomplete",
        incomplete_reason="max_output_tokens",
    )
    backend = OpenAIBackend(MODEL, client=FakeOpenAIClient(whole))  # type: ignore[arg-type]
    assert backend.annotate(image).to_dict() == SAMPLE

    filtered = FakeResponse("", status="incomplete", incomplete_reason="content_filter")
    with pytest.raises(MalformedResponseError, match="content_filter") as info:
        OpenAIBackend(MODEL, client=FakeOpenAIClient(filtered)).annotate(image)  # type: ignore[arg-type]
    assert "max_output_tokens" not in str(info.value), "do not blame a cap that did not fire"


def test_the_anthropic_ceiling_behaves_the_same_way(sheet: Path) -> None:
    cut = FakeMessage('{"streets": [', stop_reason="max_tokens")
    with pytest.raises(MalformedResponseError, match="max_tokens=16000"):
        AnthropicBackend(client=_anthropic_client(cut)).annotate(sheet)

    whole = FakeMessage(json.dumps(SAMPLE) + "\nand that is ev", stop_reason="max_tokens")
    assert AnthropicBackend(client=_anthropic_client(whole)).annotate(sheet).to_dict() == SAMPLE


def test_a_failed_response_reports_the_provider_reason(image: Path) -> None:
    response = FakeResponse(status="failed", error_message="model overloaded")
    with pytest.raises(AnnotatorProcessError, match="model overloaded"):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(response)).annotate(image)  # type: ignore[arg-type]


def test_a_response_with_no_reply_says_which_state_it_was_in(image: Path) -> None:
    """`in_progress`, `queued` and `cancelled` carry no text. Read as an empty
    reply they look like an unreadable sheet; named, they say what happened."""
    for status in ("in_progress", "queued", "cancelled"):
        with pytest.raises(AnnotatorProcessError, match=status):
            OpenAIBackend(MODEL, client=FakeOpenAIClient(FakeResponse(status=status))).annotate(  # type: ignore[arg-type]
                image
            )


def test_openai_empty_and_malformed_replies_stay_in_the_shared_taxonomy(image: Path) -> None:
    with pytest.raises(EmptyResponseError):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(FakeResponse(""))).annotate(image)  # type: ignore[arg-type]
    prose = FakeResponse("I could not find any streets, sorry!")
    with pytest.raises(MalformedResponseError):
        OpenAIBackend(MODEL, client=FakeOpenAIClient(prose)).annotate(image)  # type: ignore[arg-type]
