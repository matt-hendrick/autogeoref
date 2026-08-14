"""Choosing an annotation backend, and driving the model CLIs.

Two transports meet here. The executables are spawned by :mod:`.cli_call`, which
:class:`CliBackend` drives; the direct-HTTP providers live in :mod:`.api_call`.
Every pipeline stage that reads a sheet builds its reader through
:func:`backend_for_model`, so a provider is a config choice rather than a stage
capability.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .api_call import AnthropicBackend, OllamaBackend, OpenAIBackend
from .cli_call import Runner, _runner_for, _with_provider_variant, classify, run_cli
from .failures import AnnotateError, AnnotationCallError, BudgetLimitError
from .providers import (
    CLAUDE_CLI,
    CODEX_CLI,
    DEFAULT_CODEX_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    OPENCODE_CLI,
    SMALL_MODEL_PATTERNS,
    VARIANT_PROVIDERS,
    CliProvider,
    _choose_provider,
    ensure_model_allowed,
    parse_model_ref,
    resolve_provider,
)
from .schema import (
    Annotation,
    AnnotatorBackend,
    ExtendedAnnotation,
    _json_object_from_cli_text,
    _resolved_prompt_template,
    extended_from_raw,
)

logger = logging.getLogger(__name__)

#: First retry gap, in seconds. Sized against provider saturation, which
#: outlasts a tight loop: the whole allowance is two attempts by default.
RETRY_BASE_S = 15.0
#: Ceiling on one gap, so raising ``attempts`` cannot compound into a stall.
RETRY_CAP_S = 120.0


class CliBackend:
    """Annotate sheets through a configured model CLI."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        executable: str | None = None,
        runner: Runner | None = None,
        blocked_patterns: Sequence[str] = SMALL_MODEL_PATTERNS,
        prompt_template: str | None = None,
        provider: str | CliProvider | None = None,
        variant: str | None = None,
    ) -> None:
        self.provider = _choose_provider(model, provider)
        if variant is not None and self.provider not in (CODEX_CLI, OPENCODE_CLI):
            raise AnnotateError(
                "model variants are supported only by the Codex and OpenCode backends"
            )
        ref = parse_model_ref(model, default_provider=self.provider.name)
        ensure_model_allowed(ref.model, blocked_patterns)
        self.model = ref.model
        self.timeout = timeout
        self.executable = executable or self.provider.default_executable
        self._runner = _runner_for(self.provider, runner)
        self.prompt_template = _resolved_prompt_template(prompt_template)
        self.variant = variant

    def build_argv(
        self, image_path: Path, out_path: Path | None = None, workdir: Path | None = None
    ) -> list[str]:
        argv = self.provider.build_argv(
            image_path, self.prompt_template, self.model, self.executable, out_path, workdir
        )
        return _with_provider_variant(argv, self.provider, self.variant)

    def annotate(self, image_path: Path) -> Annotation:
        return self.annotate_extended(image_path).annotation

    def annotate_extended(self, image_path: Path) -> ExtendedAnnotation:
        outcome = run_cli(
            image_path,
            self.prompt_template,
            model=self.model,
            timeout_s=self.timeout,
            executable=self.executable,
            runner=self._runner,
            provider=self.provider,
            variant=self.variant,
        )
        try:
            return extended_from_raw(_json_object_from_cli_text(outcome.text))
        except AnnotationCallError as exc:
            classify(outcome, exc)


class ClaudeCLIBackend(CliBackend):
    """The reference ``claude -p`` annotation backend."""

    def __init__(self, model: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        kwargs.setdefault("provider", CLAUDE_CLI)
        super().__init__(model, **kwargs)


class CodexCLIBackend(CliBackend):
    """The opt-in ``codex exec`` annotation backend."""

    def __init__(self, model: str = DEFAULT_CODEX_MODEL, **kwargs: Any) -> None:
        kwargs.setdefault("provider", CODEX_CLI)
        super().__init__(model, **kwargs)


class OpenCodeCLIBackend(CliBackend):
    """The opt-in ``opencode run`` annotation backend."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        kwargs.setdefault("provider", OPENCODE_CLI)
        super().__init__(model, **kwargs)


#: The backends reached over HTTP. Keyed by the same provider names
#: `providers.NATIVE_PROVIDERS` lists; a test holds the two together.
_NATIVE_BACKENDS: dict[str, Callable[..., AnnotatorBackend]] = {
    "ollama": OllamaBackend,
    "anthropic-api": AnthropicBackend,
    "openai-api": OpenAIBackend,
}


def backend_for_model(model: str, *, variant: str | None = None, **kwargs: Any) -> AnnotatorBackend:
    """Create the configured backend for a provider-qualified model reference.

    Every stage builds its reader here, so a provider is a config choice rather
    than a stage capability.
    """
    ref = parse_model_ref(model)
    if variant is not None and ref.provider not in VARIANT_PROVIDERS:
        allowed = ", ".join(sorted(VARIANT_PROVIDERS))
        raise AnnotateError(
            f"provider {ref.provider!r} takes no reasoning variant (only {allowed} do)"
        )
    native = _NATIVE_BACKENDS.get(ref.provider)
    if native is not None:
        if variant is not None:
            kwargs["variant"] = variant
        return native(ref.model, **kwargs)
    return CliBackend(ref.model, provider=resolve_provider(ref.provider), variant=variant, **kwargs)


def retry_delay_s(attempt: int, rand: Callable[[], float] = random.random) -> float:
    """Seconds to wait after a failed attempt, exponential with jitter.

    Jitter spreads concurrent workers so a fleet does not retry in lockstep
    and re-hit the same saturated provider together. Capped, so raising
    ``attempts`` lengthens the gaps without stalling the drain.
    """
    return float(min(RETRY_BASE_S * (2**attempt), RETRY_CAP_S) * (1.0 + rand()))


def annotate_with_retry[AnnotationT](
    call: Callable[[], AnnotationT],
    attempts: int,
    label: str,
    cancelled: Callable[[], bool] | None = None,
    admit_attempt: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[AnnotationT | None, str]:
    """Retry transient failures while propagating budget limits.

    Attempts are spaced by :func:`retry_delay_s`, because back-to-back retries
    all land inside one provider blip and burn the page's whole allowance.

    ``cancelled`` is a stage-local stop hook, checked before every attempt so a
    terminal response in another worker cannot start a doomed retry.
    ``admit_attempt`` reserves an attempt against a stage's stop signal; a
    reserved attempt counts as in-flight for its overspend contract.
    """
    last_error = ""
    for attempt in range(attempts):
        if cancelled is not None and cancelled():
            return None, last_error
        if admit_attempt is not None and not admit_attempt():
            return None, last_error
        try:
            return call(), ""
        except BudgetLimitError:
            raise
        except AnnotationCallError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s attempt %d failed: %s", label, attempt + 1, exc)
        if attempt + 1 < attempts:
            delay = retry_delay_s(attempt)
            logger.info("%s: waiting %.1fs before retry", label, delay)
            sleep(delay)
    return None, last_error
