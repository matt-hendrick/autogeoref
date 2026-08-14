"""Provider layer: model-reference routing and the Codex CLI backend.

No network and no real subprocess: every test injects a runner, exactly as the
``claude`` backend tests do. The contract under test is that adding a provider
changes the ARGV SHAPE and the OUTPUT CHANNEL and nothing else — the prompt,
the parsers, the Sonnet-class gate and the failure taxonomy stay shared.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from autogeoref.annotate.api_call import OllamaBackend
from autogeoref.annotate.cli_call import run_cli
from autogeoref.annotate.failures import (
    AnnotateError,
    AnnotationTimeoutError,
    AnnotatorProcessError,
    BudgetLimitError,
    EmptyResponseError,
    MalformedResponseError,
    ModelQualityError,
)
from autogeoref.annotate.invocation import (
    ClaudeCLIBackend,
    CliBackend,
    CodexCLIBackend,
    OpenCodeCLIBackend,
    backend_for_model,
)
from autogeoref.annotate.providers import (
    CLAUDE_CLI,
    CODEX_CLI,
    DEFAULT_MODEL,
    OPENCODE_CLI,
    canonical_model,
    model_cache_key,
    model_from_cache_key,
    parse_model_ref,
    prior_variant_cache_key,
)
from autogeoref.annotate.schema import EXTENDED_PROMPT_TEMPLATE

SAMPLE: dict[str, Any] = {
    "streets": [{"name": "W. ADAMS", "bbox": [1, 2, 3, 4], "orientation": "horizontal"}],
    "page_number_seen": "26",
}


@pytest.fixture
def image(tmp_path: Path) -> Path:
    img = tmp_path / "p26_small.jpg"
    img.write_bytes(b"x")
    return img


def codex_runner(
    last_message: str | None,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    raises: Exception | None = None,
) -> Any:
    """A fake ``codex exec``: writes its answer where the real one does."""
    calls: list[tuple[list[str], float]] = []

    def runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append((argv, timeout))
        if raises is not None:
            raise raises
        if last_message is not None and "--output-last-message" in argv:
            Path(argv[argv.index("--output-last-message") + 1]).write_text(last_message)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


# ----------------------------------------------------------------------
# Model-reference routing
# ----------------------------------------------------------------------


def test_bare_name_still_means_anthropic() -> None:
    """The compatibility contract: every existing TOML and golden uses one."""
    assert parse_model_ref("claude-sonnet-5") == parse_model_ref("anthropic:claude-sonnet-5")
    assert parse_model_ref("claude-sonnet-5").provider == "anthropic"
    # an UNRECOGNIZED bare name must not raise or reroute — it falls back
    assert parse_model_ref("some-future-model").provider == "anthropic"


def test_a_bare_foreign_model_name_is_refused_not_guessed() -> None:
    """Guessing is the only option that can be silently WRONG, so we don't.

    `"gpt-5.6-terra"` bare could mean "route to codex" (a silent provider
    switch behind the config's back) or "an Anthropic model with a funny name"
    (a confusing claude failure five stages in). Refusing, and naming the
    spelling that works, is neither.
    """
    with pytest.raises(AnnotateError, match=re.escape("codex:gpt-5.6-terra")):
        parse_model_ref("gpt-5.6-terra")


def test_a_backend_that_chose_its_provider_takes_its_own_bare_names() -> None:
    assert parse_model_ref("gpt-5.6-terra", default_provider="codex").provider == "codex"
    assert CodexCLIBackend(model="gpt-5.6-terra").model == "gpt-5.6-terra"
    assert OpenCodeCLIBackend(model="openai/gpt-5.6-terra").model == "openai/gpt-5.6-terra"


def test_explicit_prefix_resolves() -> None:
    ref = parse_model_ref("codex:gpt-5.5")
    assert (ref.provider, ref.model) == ("codex", "gpt-5.5")


def test_explicit_opencode_prefix_preserves_its_model_namespace() -> None:
    ref = parse_model_ref("opencode:openai/gpt-5.6-terra")
    assert (ref.provider, ref.model) == ("opencode", "openai/gpt-5.6-terra")


def test_a_bare_opencode_model_name_is_refused_not_guessed() -> None:
    with pytest.raises(AnnotateError, match=re.escape("opencode:openai/gpt-5.6-terra")):
        parse_model_ref("openai/gpt-5.6-terra")


# ----------------------------------------------------------------------
# Model IDENTITY — the addresses channel's two-voice floor depends on it
# ----------------------------------------------------------------------


def test_two_spellings_of_one_model_are_one_identity() -> None:
    """One model must not satisfy the consensus requirement by itself.

    Sidecars are named `p<N>.v2.<model>.json` and the model is read back OUT of
    that filename to count DISTINCT models. If "claude-sonnet-5" and
    "anthropic:claude-sonnet-5" were two names, one model could satisfy the
    ">=2 distinct models" consensus floor by agreeing with ITSELF — in the only
    channel permitted to REFUTE a placement.
    """
    assert canonical_model("anthropic:claude-sonnet-5") == canonical_model("claude-sonnet-5")


def test_anthropic_canonical_form_is_the_bare_name_that_is_already_on_disk() -> None:
    """Canonicalizing must not orphan the existing annotation cache.

    Every sidecar already written is `p<N>.v2.claude-sonnet-5.json`. A canonical
    form of "anthropic:claude-sonnet-5" would rename them all and silently
    re-spend model calls that were already paid for.
    """
    assert canonical_model("claude-sonnet-5") == "claude-sonnet-5"
    assert canonical_model("codex:gpt-5.6-terra") == "codex:gpt-5.6-terra"


def test_variant_cache_key_is_reversible_and_distinct() -> None:
    high = model_cache_key("codex:gpt-5.6-terra", "high")
    default = model_cache_key("codex:gpt-5.6-terra")
    assert high != default
    assert model_from_cache_key(high) == "codex:gpt-5.6-terra"
    assert model_from_cache_key(default) == default
    assert model_from_cache_key(prior_variant_cache_key("codex:gpt-5.6-terra", "high") or "") == (
        "codex:gpt-5.6-terra"
    )


# the end-to-end proof that the two spellings never become two VOICES lives with
# the channel it protects: tests/test_verified_accept.py
# ::test_two_spellings_of_one_model_are_still_one_voice


# ----------------------------------------------------------------------
# Provider precedence — ONE rule, at every entry point
# ----------------------------------------------------------------------


def test_a_model_prefix_beats_an_explicit_provider_everywhere(image: Path) -> None:
    """Honoring `provider=` over the prefix would run a codex model on claude.

    The transport and the backend disagreed about this (adversarial review: the
    choke point produced `['claude', '-p', ..., '--model', 'gpt-5.6-terra']`).
    Both entry points are checked, because agreement is the property at risk.
    """
    captured: list[list[str]] = []

    def runner(argv: list[str], _t: float) -> subprocess.CompletedProcess[str]:
        captured.append(argv)
        if "--output-last-message" in argv:
            Path(argv[argv.index("--output-last-message") + 1]).write_text(json.dumps(SAMPLE))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(SAMPLE), stderr="")

    run_cli(
        image, "prompt IMGPATH", model="codex:gpt-5.6-terra", provider="anthropic", runner=runner
    )
    assert captured[0][0] == "codex"

    captured.clear()
    CliBackend("codex:gpt-5.6-terra", provider=CLAUDE_CLI, runner=runner).annotate(image)
    assert captured[0][0] == "codex"


def test_a_mistyped_provider_fails_loudly(tmp_path: Path) -> None:
    """A colon PROMISES a provider on the left, and we hold it to that.

    The quiet alternative is the dangerous one: `codx:gpt-5.6-terra` becoming
    an "Anthropic model" of that literal name, shelling out to the claude CLI
    with a garbage --model. Fail at configure time, not at spend time.
    """
    with pytest.raises(AnnotateError, match="unknown provider"):
        parse_model_ref("codx:gpt-5.6-terra")
    with pytest.raises(AnnotateError):
        backend_for_model("nope:some-model")


def test_first_colon_splits_so_a_model_may_keep_its_own_tag() -> None:
    """A provider's tag syntax survives: `ollama:qwen2.5-vl:32b` -> `qwen2.5-vl:32b`."""
    ref = parse_model_ref("codex:gpt-5.6:preview")
    assert (ref.provider, ref.model) == ("codex", "gpt-5.6:preview")


def test_small_model_gate_survives_a_provider_prefix() -> None:
    """A small model wearing a provider name is still a small model."""
    with pytest.raises(ModelQualityError):
        parse_model_ref("codex:gpt-5.4-mini")
    with pytest.raises(ModelQualityError):
        CodexCLIBackend(model="gpt-5.4-mini")


def test_the_quality_gate_outranks_the_spelling_advice() -> None:
    """`gpt-4o-mini` is refused for being SMALL, not for missing a prefix.

    Both complaints are true of it. Only one of them is the reason that
    matters, and reporting the wrong one invites the caller to "fix" a
    hallucinating small model by adding `codex:` to its name.
    """
    with pytest.raises(ModelQualityError):
        parse_model_ref("gpt-4o-mini")


def test_backend_for_model_routes_to_the_right_cli() -> None:
    assert backend_for_model(DEFAULT_MODEL).provider is CLAUDE_CLI  # type: ignore[attr-defined]
    assert backend_for_model("codex:gpt-5.6-terra").provider is CODEX_CLI  # type: ignore[attr-defined]
    assert backend_for_model("opencode:openai/gpt-5.6-terra").provider is OPENCODE_CLI  # type: ignore[attr-defined]
    assert isinstance(backend_for_model("ollama:gemma4:12b"), OllamaBackend)


def test_explicit_ollama_prefix_preserves_its_tag() -> None:
    ref = parse_model_ref("ollama:gemma4:12b")
    assert (ref.provider, ref.model) == ("ollama", "gemma4:12b")


# ----------------------------------------------------------------------
# Ollama HTTP transport
# ----------------------------------------------------------------------


def test_ollama_posts_base64_image_and_parses_content(image: Path) -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []

    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append((url, payload, timeout))
        return {
            "model": "gemma4:12b",
            "message": {"role": "assistant", "content": json.dumps(SAMPLE)},
        }

    backend = OllamaBackend("gemma4:12b", endpoint="http://example.test/", transport=transport)
    assert backend.annotate(image).to_dict() == SAMPLE
    url, payload, timeout = calls[0]
    assert url == "http://example.test/api/chat"
    assert timeout == 600.0
    assert payload["model"] == "gemma4:12b"
    assert payload["stream"] is False
    assert payload["messages"][0]["content"] == EXTENDED_PROMPT_TEMPLATE.replace(
        "IMGPATH", image.name
    )
    assert payload["messages"][0]["images"]
    assert json.loads(backend.last_raw_response)["model"] == "gemma4:12b"


def test_ollama_provider_error_is_a_call_failure(image: Path) -> None:
    def transport(_url: str, _payload: dict[str, Any], _timeout: float) -> dict[str, Any]:
        return {"error": "model requires more system memory"}

    with pytest.raises(AnnotatorProcessError, match="more system memory"):
        OllamaBackend("gemma4:12b", transport=transport).annotate(image)


def test_ollama_missing_or_bad_content_is_malformed(image: Path) -> None:
    def transport(_url: str, _payload: dict[str, Any], _timeout: float) -> dict[str, Any]:
        return {"message": {"content": "not an annotation"}}

    backend = OllamaBackend("gemma4:12b", transport=transport)
    with pytest.raises(MalformedResponseError):
        backend.annotate(image)
    assert json.loads(backend.last_raw_response) == {"message": {"content": "not an annotation"}}


def test_ollama_timeout_and_unreachable_endpoint_are_call_failures(
    image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timed_out(*_args: object, **_kwargs: object) -> None:
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("autogeoref.annotate.api_call.httpx.post", timed_out)
    with pytest.raises(AnnotationTimeoutError):
        OllamaBackend("gemma4:12b").annotate(image)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("autogeoref.annotate.api_call.httpx.post", unavailable)
    with pytest.raises(AnnotatorProcessError, match="connection refused"):
        OllamaBackend("gemma4:12b").annotate(image)


# ----------------------------------------------------------------------
# Codex argv shape
# ----------------------------------------------------------------------


def test_codex_argv_shape(image: Path) -> None:
    backend = CodexCLIBackend(runner=codex_runner(json.dumps(SAMPLE)))
    out = image.parent / "last.txt"
    argv = backend.build_argv(image, out)

    assert argv[0] == "codex"
    assert argv[1] == "exec"
    # the image is ATTACHED, which is why no file-read tool and no dir grant
    assert argv[argv.index("--image") + 1] == str(image)
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--model") + 1] == backend.model
    assert argv[argv.index("--output-last-message") + 1] == str(out)
    assert "--skip-git-repo-check" in argv
    # the frozen v2 prompt, verbatim, as the trailing positional
    assert argv[-1] == EXTENDED_PROMPT_TEMPLATE.replace("IMGPATH", str(image))


def test_codex_variant_sets_reasoning_effort(image: Path) -> None:
    backend = CodexCLIBackend(variant="high")
    argv = backend.build_argv(image, image.parent / "last.txt")

    assert argv[argv.index("--config") + 1] == 'model_reasoning_effort="high"'
    assert argv[-1] == EXTENDED_PROMPT_TEMPLATE.replace("IMGPATH", str(image))


# ----------------------------------------------------------------------
# OpenCode argv shape and transport
# ----------------------------------------------------------------------


def test_opencode_argv_shape(image: Path) -> None:
    backend = OpenCodeCLIBackend("openai/gpt-5.6-terra", variant="high")
    workdir = image.parent / "opencode-workdir"
    argv = backend.build_argv(image, workdir=workdir)

    assert argv[:2] == ["opencode", "run"]
    assert "--pure" in argv
    assert argv[argv.index("--agent") + 1] == "autogeoref-annotation"
    assert argv[argv.index("--dir") + 1] == str(workdir)
    assert argv[argv.index("--model") + 1] == "openai/gpt-5.6-terra"
    assert argv[argv.index("--file") + 1] == str(image)
    assert argv[argv.index("--variant") + 1] == "high"
    assert argv[-2] == "--"
    assert "--output-last-message" not in argv
    assert argv[-1] == EXTENDED_PROMPT_TEMPLATE.replace("IMGPATH", str(image))


def test_an_unspecified_opencode_runner_is_the_isolated_one() -> None:
    """OpenCode's default spawn strips ambient agent instructions; an injected one is left alone.

    The isolation wrapper is chosen by identity against the module-level
    default, so a test session that replaces that default must not silently
    route the backend past it.
    """
    from autogeoref.annotate import cli_call

    def injected(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "", "")

    assert OpenCodeCLIBackend("openai/gpt-5.6-terra")._runner is cli_call._opencode_runner
    assert OpenCodeCLIBackend("openai/gpt-5.6-terra", runner=injected)._runner is injected
    assert CodexCLIBackend("gpt-5.6-terra")._runner is cli_call._default_runner


def test_opencode_success_reads_stdout(image: Path) -> None:
    calls: list[tuple[list[str], float]] = []
    config: dict[str, Any] = {}

    def runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append((argv, timeout))
        workdir = Path(argv[argv.index("--dir") + 1])
        config.update(json.loads((workdir / "opencode.json").read_text()))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(SAMPLE), stderr="")

    backend = OpenCodeCLIBackend("openai/gpt-5.6-terra", runner=runner)
    assert backend.annotate(image).to_dict() == SAMPLE
    assert calls[0][0][0] == "opencode"
    assert config["agent"]["autogeoref-annotation"]["permission"] == {"*": "deny"}
    assert config["attachment"]["image"]["auto_resize"] is False
    assert config["provider"]["openai"]["options"] == {
        "timeout": 600000,
        "chunkTimeout": 600000,
    }


def test_opencode_timeout_and_process_failure_are_classified(image: Path) -> None:
    def timed_out(_argv: list[str], _timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["opencode"], timeout=600)

    with pytest.raises(AnnotationTimeoutError):
        OpenCodeCLIBackend("openai/gpt-5.6-terra", runner=timed_out).annotate(image)

    def crashed(argv: list[str], _timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="opencode failed")

    with pytest.raises(AnnotatorProcessError):
        OpenCodeCLIBackend("openai/gpt-5.6-terra", runner=crashed).annotate(image)


def test_codex_success_reads_the_answer_file(image: Path) -> None:
    runner = codex_runner(json.dumps(SAMPLE))
    backend = CodexCLIBackend(runner=runner)
    assert backend.annotate(image).to_dict() == SAMPLE
    ((argv, _timeout),) = runner.calls
    assert "--output-last-message" in argv


def test_codex_answer_file_beats_noisy_stdout(image: Path) -> None:
    """THE reason the file is the answer channel.

    Codex streams reasoning and event chatter to stdout (measured: 28 KB of it
    around a 1.5 KB answer). If stdout won, the first ``{`` in a reasoning
    trace would be parsed as the annotation.
    """
    noise = 'thinking... {"streets": [{"name": "WRONG ST", "bbox": [9,9,9,9], '
    noise += '"orientation": "vertical"}], "page_number_seen": "99"}\nmore chatter'
    backend = CodexCLIBackend(runner=codex_runner(json.dumps(SAMPLE), stdout=noise))
    got = backend.annotate(image)
    assert got.to_dict() == SAMPLE
    assert [s.name for s in got.streets] == ["W. ADAMS"]


def test_codex_never_parses_stdout_as_the_annotation(image: Path) -> None:
    """No answer file means NO ANSWER — stdout is not a fallback answer channel.

    Falling back to stdout lets a JSON object
    inside a REASONING TRACE — a draft the model then discarded — be parsed as
    the sheet's annotation and cached, where every downstream channel would
    trust it. A fabricated read is worse than a failed one, so an empty answer
    file classifies as a failure even when stdout is full of plausible JSON.
    """
    draft = 'let me draft: {"streets": [{"name": "HALLUCINATED ST", "bbox": [9,9,9,9], '
    draft += '"orientation": "vertical"}], "page_number_seen": "99"} ... on reflection, no.'
    backend = CodexCLIBackend(runner=codex_runner(None, stdout=draft))
    with pytest.raises(EmptyResponseError):
        backend.annotate(image)


def test_codex_budget_limit_in_the_answer_file_is_still_terminal(image: Path) -> None:
    """A CLI may refuse through the same channel it ANSWERS on.

    If that refusal classified as MALFORMED it would be RETRIED (budget errors
    are the one class `annotate_with_retry` refuses to retry), burning two
    doomed calls per page and leaving permanent `.failed.json` markers.
    """
    backend = CodexCLIBackend(
        runner=codex_runner("ERROR: You've hit your usage limit. Try again later.")
    )
    with pytest.raises(BudgetLimitError):
        backend.annotate(image)


def test_codex_extended_channels_survive(image: Path) -> None:
    payload = dict(
        SAMPLE,
        address_numerals=[{"value": 1459, "bbox": [1, 2, 3, 4], "street": "W. ADAMS"}],
        margin_numbers=[{"side": "top", "text": "26"}],
    )
    backend = CodexCLIBackend(runner=codex_runner(json.dumps(payload)))
    ext = backend.annotate_extended(image)
    assert [n.value for n in ext.address_numerals] == [1459]
    assert [m.text for m in ext.margin_readings] == ["26"]


# ----------------------------------------------------------------------
# The five failure classes, through the codex transport
# ----------------------------------------------------------------------


def test_codex_empty_response_classified(image: Path) -> None:
    backend = CodexCLIBackend(runner=codex_runner(""))
    with pytest.raises(EmptyResponseError):
        backend.annotate(image)


def test_codex_malformed_response_classified(image: Path) -> None:
    backend = CodexCLIBackend(runner=codex_runner("I could not read that sheet, sorry."))
    with pytest.raises(MalformedResponseError):
        backend.annotate(image)


#: Placeholders of the original length, so the banner keeps its shape. They are
#: not real: paste no fresh capture over them.
REDACTED_WORKDIR = "/home/agent/autogeoref"  # local-path-ok — resolves nowhere
REDACTED_SESSION_ID = "00000000-0000-0000-0000-000000000000"

#: A usage-limit response has two properties, and they are the whole point of
#: this fixture: codex EXITS 0 while refusing, and it prints ~250 characters of
#: banner BEFORE the reason.
CODEX_LIMIT_STDERR = f"""Reading additional input from stdin...
OpenAI Codex v0.144.1
--------
workdir: {REDACTED_WORKDIR}
model: gpt-5.5
provider: openai
approval: never
sandbox: read-only
reasoning effort: none
reasoning summaries: none
session id: {REDACTED_SESSION_ID}
--------
user
Look at the Sanborn fire insurance map sheet
ERROR: You've hit your usage limit. Upgrade to Plus to continue using Codex \
(https://chatgpt.com/explore/plus), or try again at Aug 10th, 2026 6:05 PM.
"""


def test_codex_budget_message_classified(image: Path) -> None:
    """Terminal for the stage — must NOT look like an empty response.

    Note the exit code: codex reports a usage limit and STILL EXITS 0, so the
    classification cannot lean on returncode. Verified against the real CLI.
    """
    backend = CodexCLIBackend(runner=codex_runner("", stderr=CODEX_LIMIT_STDERR, returncode=0))
    with pytest.raises(BudgetLimitError) as exc:
        backend.annotate(image)
    # the operator must see the REASON, not the banner that precedes it
    assert "usage limit" in str(exc.value)
    assert "workdir" not in str(exc.value)


@pytest.mark.parametrize(
    "line",
    [
        'ERROR: {"type": "error", "code": "usage_limit_reached"}',
        'ERROR: {"type": "error", "code": "insufficient_quota"}',
        "ERROR: you've hit your weekly-limit, resets Aug 3",
    ],
)
def test_machine_readable_limit_codes_survive_the_error_line_scope(image: Path, line: str) -> None:
    """The codex scope reads only ``ERROR:`` lines, so a limit spelled as a JSON
    error code has to be recognised there too -- the whole-text scan that would
    otherwise catch it is exactly what this provider gives up."""
    backend = CodexCLIBackend(runner=codex_runner(None, stderr=line))
    with pytest.raises(BudgetLimitError):
        backend.annotate(image)


def test_a_structured_refusal_in_the_answer_file_is_read_whole(image: Path) -> None:
    """A DECODED payload is the provider's own structured output, so it is read
    whole -- the ``ERROR:`` scope would otherwise hide a refusal spelled as a
    JSON error code, which carries no such line."""
    backend = CodexCLIBackend(
        runner=codex_runner('{"type": "error", "code": "usage_limit_reached"}')
    )
    with pytest.raises(BudgetLimitError):
        backend.annotate(image)


@pytest.mark.parametrize(
    "answer",
    [
        # verbatim from the incident: codex ingests this repo and echoes it back
        "placement, status or GCP on a committed record. Cached, budget-gated, replayable;",
        "I could not read that sheet. The instructions say model calls are budget-gated.",
    ],
)
def test_repo_prose_echoed_into_the_answer_file_is_not_a_budget_limit(
    image: Path, answer: str
) -> None:
    """The reason PROSE stays under the reporting scope.

    `codex exec` ingests whatever agent instructions it finds in the working
    directory and echoes them into its own output -- not only its logs. Prose
    about budgets is dense with the vocabulary `_BUDGET_RE` hunts, so reading an
    answer whole reported OUR OWN instructions as the provider refusing us, and
    a BudgetLimitError is terminal: one malformed read would halt a healthy
    volume, permanently, quoting our own documentation as the reason.
    """
    backend = CodexCLIBackend(runner=codex_runner(answer))
    with pytest.raises(MalformedResponseError):  # retryable — NOT terminal
        backend.annotate(image)


def test_repo_prose_echoed_by_the_cli_is_not_a_budget_limit(image: Path) -> None:
    """A process failure must not be confused with a valid response.

    `codex exec` can ingest and echo repository instructions. A whole-text
    scan therefore read ordinary project budget vocabulary as the provider
    refusing us. A BudgetLimitError is terminal, so that halted a healthy stage
    on its first malformed read.

    A malformed response that merely quotes the word "budget" must stay
    malformed (i.e. retryable).
    """
    echoed = (
        "user\nModel calls are budget-gated and require approval.\n"
        "Estimate the call count before you start.\n"
        "I could not read that sheet.\n"
    )
    backend = CodexCLIBackend(runner=codex_runner("I could not read that sheet.", stderr=echoed))
    with pytest.raises(MalformedResponseError):  # retryable — NOT terminal
        backend.annotate(image)


def test_codex_timeout_classified(image: Path) -> None:
    err = subprocess.TimeoutExpired(cmd=["codex"], timeout=600)
    backend = CodexCLIBackend(runner=codex_runner(None, raises=err))
    with pytest.raises(AnnotationTimeoutError):
        backend.annotate(image)


def test_codex_process_crash_is_not_an_empty_response(image: Path) -> None:
    """Infrastructure failure != "the model returned nothing"."""
    backend = CodexCLIBackend(
        runner=codex_runner(None, stderr="codex: command not found", returncode=127)
    )
    with pytest.raises(AnnotatorProcessError):
        backend.annotate(image)


def test_codex_usable_output_survives_a_nonzero_exit(image: Path) -> None:
    """Real model output is never discarded on a false-positive failure signal."""
    backend = CodexCLIBackend(runner=codex_runner(json.dumps(SAMPLE), returncode=1))
    assert backend.annotate(image).to_dict() == SAMPLE


# ----------------------------------------------------------------------
# The claude path must not have moved
# ----------------------------------------------------------------------


def test_claude_backend_still_uses_its_own_argv_shape(image: Path) -> None:
    def runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(SAMPLE), stderr="")

    backend = ClaudeCLIBackend(runner=runner)
    argv = backend.build_argv(image)
    assert argv[0] == "claude"
    assert argv[1] == "-p"
    assert "--output-format" in argv and "--output-last-message" not in argv
    assert backend.annotate(image).to_dict() == SAMPLE
