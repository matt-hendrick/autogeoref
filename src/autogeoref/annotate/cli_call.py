"""Running one model CLI, and judging what came back.

Two steps, deliberately separate. :func:`run_cli` spawns the process and reports
everything that came out of it; :func:`classify` turns a failed read of that
output into the right member of the failure taxonomy.

They are separate because the judgement needs the WHOLE outcome — answer text,
both log streams, the exit code, the provider's reporting scope — while reading
a reply belongs to whoever knows its schema. A caller wraps its own parsing and
hands the failure here, so owning a schema never means leaving the taxonomy.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from .failures import (
    AnnotateError,
    AnnotationCallError,
    AnnotationTimeoutError,
    AnnotatorProcessError,
    BudgetLimitError,
    _raise_budget_if_matched,
)
from .providers import (
    CODEX_CLI,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    OPENCODE_CLI,
    CliProvider,
    _choose_provider,
    parse_model_ref,
)
from .schema import _json_object_from_cli_text

Runner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class CliOutcome:
    """One finished CLI invocation, as the failure taxonomy needs to see it.

    ``text`` is the ANSWER channel — the output file for a provider that writes
    one, else stdout falling back to stderr. ``stdout``/``stderr`` are the log
    channels, which for some providers carry a transcript rather than a reply.
    """

    text: str
    stdout: str
    stderr: str
    returncode: int
    provider: CliProvider
    binary: str
    image_name: str


def _default_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a model executable with stdin detached from the invoking process."""
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _opencode_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run an isolated OpenCode annotation without ambient agent instructions."""
    env = os.environ | {
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
    }
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
        env=env,
    )


def _runner_for(provider: CliProvider, runner: Runner | None) -> Runner:
    """Pick the spawn: OpenCode's isolated one by default, an injected one as given.

    Resolved here rather than in a signature default so that replacing the
    module-level default — which is what the test suite's tripwire does —
    actually replaces the runner a caller gets.
    """
    chosen = _default_runner if runner is None else runner
    return _opencode_runner if provider is OPENCODE_CLI and chosen is _default_runner else chosen


def _failure_excerpt(detail: str, head: int = 200, tail: int = 400) -> str:
    """Keep both ends of a crashed CLI's output.

    Taking the first N characters loses the diagnosis: these tools open with a
    fixed banner (version, workdir, model, session id) and print what actually
    went wrong last, so a head-only excerpt records boilerplate and drops the
    reason. Failure markers exist to answer "why", and a re-read costs a call.
    """
    if len(detail) <= head + tail:
        return detail
    return f"{detail[:head]}\n[...{len(detail) - head - tail} chars...]\n{detail[-tail:]}"


def _with_provider_variant(
    argv: list[str], provider: CliProvider, variant: str | None
) -> list[str]:
    """Insert a provider-owned reasoning effort before the final prompt argument."""
    if variant is None:
        return argv
    if provider is CODEX_CLI:
        # Codex exposes reasoning effort through its TOML-compatible config override.
        return [*argv[:-1], "--config", f"model_reasoning_effort={json.dumps(variant)}", argv[-1]]
    if provider is OPENCODE_CLI:
        if len(argv) < 2 or argv[-2] != "--":
            raise AnnotateError("OpenCode argv has no prompt delimiter")
        return [*argv[:-2], "--variant", variant, *argv[-2:]]
    raise AnnotateError("model variants are supported only by the Codex and OpenCode backends")


def _holds_a_json_object(text: str) -> bool:
    """Whether the answer channel carries a JSON object at all."""
    try:
        _json_object_from_cli_text(text)
    except AnnotationCallError:
        return False
    return True


def classify(outcome: CliOutcome, exc: AnnotationCallError) -> NoReturn:
    """Re-raise a failed read as the failure it actually was.

    Wrap EVERY step of reading ``outcome.text`` — decode and schema check both —
    and hand what they raise to this. A refusal is often valid JSON, so it
    decodes and only a schema check rejects it; judged outside, that reads as a
    retryable malformed response rather than the terminal limit it is.

    Only ever upgrades: a failure that already knows it is terminal is re-raised
    untouched, since no evidence here outranks it.
    """
    if isinstance(exc, BudgetLimitError):
        raise exc
    # A payload that DECODED is the provider's own structured output, so read it
    # whole. Prose is not: a CLI that ingests this repo echoes its budget-dense
    # docs back, and scanning that reported OUR guide as the provider refusing
    # us. Prose stays under the reporting scope.
    if _holds_a_json_object(outcome.text):
        _raise_budget_if_matched(outcome.text)
    _raise_budget_if_matched(
        f"{outcome.text}\n{outcome.stdout}\n{outcome.stderr}", scope=outcome.provider.budget_scope
    )
    if outcome.returncode != 0:
        detail = outcome.text.strip() or outcome.stderr.strip() or outcome.stdout.strip()
        raise AnnotatorProcessError(
            f"{outcome.binary} exited {outcome.returncode} for {outcome.image_name}: "
            f"{_failure_excerpt(detail) or '(no output)'}"
        ) from exc
    raise exc


def run_cli(
    image_path: Path,
    prompt_template: str,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT,
    executable: str | None = None,
    runner: Runner | None = None,
    provider: str | CliProvider | None = None,
    variant: str | None = None,
) -> CliOutcome:
    """Spawn one configured model CLI and collect everything it produced.

    Reads nothing: the only failure it raises is a timeout, which no reply can
    describe. Pass the result to :func:`classify` when a read of it fails.
    """
    cli = _choose_provider(model, provider)
    ref = parse_model_ref(model, default_provider=cli.name)
    binary = executable or cli.default_executable
    with contextlib.ExitStack() as stack:
        out_path: Path | None = None
        workdir: Path | None = None
        if cli.writes_output_file:
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory(prefix="autogeoref-cli-"))
            out_path = Path(tmpdir) / "last-message.txt"
        elif cli.configure_workdir is not None:
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory(prefix="autogeoref-cli-"))
            workdir = Path(tmpdir)
            cli.configure_workdir(workdir)
        argv = _with_provider_variant(
            cli.build_argv(image_path, prompt_template, ref.model, binary, out_path, workdir),
            cli,
            variant,
        )
        try:
            proc = _runner_for(cli, runner)(argv, timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise AnnotationTimeoutError(
                f"annotation of {image_path.name} timed out after {timeout_s}s"
            ) from exc
        if out_path is not None:
            text = (
                out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""
            )
        else:
            text = proc.stdout
            if not text.strip():
                text = proc.stderr
        return CliOutcome(
            text=text,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            provider=cli,
            binary=binary,
            image_name=image_path.name,
        )
    raise AssertionError("unreachable")  # pragma: no cover
