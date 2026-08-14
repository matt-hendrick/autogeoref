"""The annotation backends that speak HTTP: the only place a model SDK is called.

One per provider transport. Each turns one prepared sheet into an
:class:`~.schema.ExtendedAnnotation`, or into the member of the failure taxonomy
that says why it could not. The provider SDKs are optional extras, imported on
first use and reported by the extra that installs them.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx

from .failures import (
    _BUDGET_RE,
    AnnotateError,
    AnnotationCallError,
    AnnotationTimeoutError,
    AnnotatorProcessError,
    BudgetLimitError,
    MalformedResponseError,
    TransientRateLimitError,
)
from .providers import DEFAULT_MODEL, DEFAULT_TIMEOUT, SMALL_MODEL_PATTERNS, ensure_model_allowed
from .schema import (
    Annotation,
    ExtendedAnnotation,
    _parse_extended_classified,
    _resolved_prompt_template,
)

if TYPE_CHECKING:
    import anthropic
    import openai

_MediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
_MEDIA_TYPES: dict[str, _MediaType] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

OllamaTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]

#: OpenAI spells an exhausted balance as a 429 too, and that one is terminal
#: rather than back-off pressure. Told apart by error code, because the prose of
#: an ordinary rate limit also says "rate limit" and would match a text scan.
_OPENAI_TERMINAL_CODES = frozenset(
    {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active"}
)


def _require_sdk(package: str, extra: str) -> None:
    """Fail with the extra that installs a provider SDK, not with ``ImportError``.

    Raised when the backend is BUILT, not when it first reads, so a stage that
    builds its reader up front reports this before it spends anything.
    """
    if importlib.util.find_spec(package) is None:
        raise AnnotateError(
            f"the {package!r} package is not installed, so this provider cannot be called: "
            f"install it with `uv sync --extra {extra}` or `pip install 'autogeoref[{extra}]'`"
        )


def missing_credential(provider: str) -> str | None:
    """The env var a direct-API provider needs and does not have, or ``None``.

    A build-time question with no SDK object behind it, so a run can ask it in
    its preamble the way it asks whether a CLI is on ``PATH``.
    """
    variables = {
        "anthropic-api": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "openai-api": ("OPENAI_API_KEY",),
    }.get(provider, ())
    return None if not variables or any(os.environ.get(v) for v in variables) else variables[0]


def _parsed_or_truncated(text: str, cut_off: str | None, image_path: Path) -> ExtendedAnnotation:
    """Read a reply, blaming an output ceiling only when the reply is unusable.

    A reply that hit its ceiling after the JSON closed still parses, so the
    ceiling is reported where it explains a failure and nowhere else. It stays
    retryable: raise the ceiling for a sheet that keeps hitting it, because a
    retry at the same ceiling buys the same truncation.
    """
    try:
        return _parse_extended_classified(text)
    except AnnotationCallError as exc:
        if cut_off is None or isinstance(exc, BudgetLimitError):
            raise
        raise MalformedResponseError(
            f"the reply for {image_path.name} was cut off at {cut_off} and does not parse: {exc}"
        ) from exc


def _ollama_transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST one non-streaming Ollama chat request and preserve provider errors."""
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise AnnotationTimeoutError(f"Ollama request timed out after {timeout}s") from exc
    except httpx.HTTPError as exc:
        detail = exc.response.text[:300] if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        raise AnnotatorProcessError(f"Ollama request failed: {detail}") from exc
    try:
        decoded = response.json()
    except ValueError as exc:
        raise MalformedResponseError(
            f"Ollama returned non-JSON HTTP response: {response.text[:300]}"
        ) from exc
    if not isinstance(decoded, dict):
        raise MalformedResponseError("Ollama returned a non-object response")
    return decoded


class AnthropicBackend:
    """Annotate sheets through the Anthropic Messages API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
        blocked_patterns: Sequence[str] = SMALL_MODEL_PATTERNS,
        prompt_template: str | None = None,
    ) -> None:
        ensure_model_allowed(model, blocked_patterns)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client = client if client is not None else self._build_client()
        self.prompt_template = _resolved_prompt_template(prompt_template)

    def _build_client(self) -> anthropic.Anthropic:
        """Build the SDK client now, so a missing package or key is a build error.

        The client itself takes no credential check, so this asks for one: it
        would otherwise surface as a bare ``TypeError`` from inside a worker.
        """
        _require_sdk("anthropic", "annotate")
        import anthropic as anthropic_mod

        missing = missing_credential("anthropic-api")
        if missing is not None:
            raise AnnotateError(f"the Anthropic API needs a credential; set {missing}")
        return anthropic_mod.Anthropic(timeout=self.timeout)

    def annotate(self, image_path: Path) -> Annotation:
        return self.annotate_extended(image_path).annotation

    def annotate_extended(self, image_path: Path) -> ExtendedAnnotation:
        text, truncated = self._response_text(image_path)
        return _parsed_or_truncated(text, truncated, image_path)

    def _response_text(self, image_path: Path) -> tuple[str, str | None]:
        import anthropic as anthropic_mod

        media_type = _MEDIA_TYPES.get(image_path.suffix.lower())
        if media_type is None:
            raise AnnotateError(f"unsupported image type: {image_path.suffix!r}")
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": self.prompt_template.replace("IMGPATH", image_path.name),
                            },
                        ],
                    }
                ],
            )
        except anthropic_mod.APITimeoutError as exc:
            raise AnnotationTimeoutError(
                f"annotation of {image_path.name} timed out after {self.timeout}s"
            ) from exc
        except anthropic_mod.RateLimitError as exc:
            raise TransientRateLimitError(f"provider rate limit: {exc}") from exc
        except anthropic_mod.APIStatusError as exc:
            if _BUDGET_RE.search(str(exc)):
                raise BudgetLimitError(f"provider budget/limit error: {exc}") from exc
            raise AnnotatorProcessError(
                f"Anthropic returned {exc.status_code} for {image_path.name}: {exc}"
            ) from exc
        except anthropic_mod.APIError as exc:
            # a connection or protocol failure: one page's problem, like a
            # crashed CLI, not a reason to abandon the pages after it
            raise AnnotatorProcessError(
                f"Anthropic call for {image_path.name} failed: {exc}"
            ) from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        cut_off = f"max_tokens={self.max_tokens}" if response.stop_reason == "max_tokens" else None
        return text, cut_off


class OpenAIBackend:
    """Annotate sheets through the OpenAI Responses API.

    ``variant`` is the reasoning effort, the same knob the Codex CLI exposes.
    ``image_detail`` defaults to ``high`` because street labels are small and
    ``auto`` may downsample them away.
    """

    def __init__(
        self,
        model: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        # Reasoning tokens are billed against this ceiling as well as the reply,
        # so it sits above the Anthropic default rather than beside it.
        max_tokens: int = 32000,
        client: openai.OpenAI | None = None,
        blocked_patterns: Sequence[str] = SMALL_MODEL_PATTERNS,
        prompt_template: str | None = None,
        variant: str | None = None,
        image_detail: str = "high",
    ) -> None:
        ensure_model_allowed(model, blocked_patterns)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.prompt_template = _resolved_prompt_template(prompt_template)
        self.variant = variant
        self.image_detail = image_detail
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> openai.OpenAI:
        """Build the SDK client now, so a missing package or key is a build error."""
        _require_sdk("openai", "openai")
        import openai as openai_mod

        try:
            return openai_mod.OpenAI(timeout=self.timeout)
        except openai_mod.OpenAIError as exc:
            raise AnnotateError(f"the OpenAI client could not be built: {exc}") from exc

    def annotate(self, image_path: Path) -> Annotation:
        return self.annotate_extended(image_path).annotation

    def annotate_extended(self, image_path: Path) -> ExtendedAnnotation:
        text, cut_off = self._response_text(image_path)
        return _parsed_or_truncated(text, cut_off, image_path)

    def _request(self, image_path: Path) -> dict[str, Any]:
        media_type = _MEDIA_TYPES.get(image_path.suffix.lower())
        if media_type is None:
            raise AnnotateError(f"unsupported image type: {image_path.suffix!r}")
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        request: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_tokens,
            "timeout": self.timeout,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:{media_type};base64,{image_data}",
                            "detail": self.image_detail,
                        },
                        {
                            "type": "input_text",
                            "text": self.prompt_template.replace("IMGPATH", image_path.name),
                        },
                    ],
                }
            ],
        }
        if self.variant is not None:
            request["reasoning"] = {"effort": self.variant}
        return request

    def _response_text(self, image_path: Path) -> tuple[str, str | None]:
        import openai as openai_mod

        request = self._request(image_path)
        try:
            response = self._client.responses.create(**request)
        except openai_mod.APITimeoutError as exc:
            raise AnnotationTimeoutError(
                f"annotation of {image_path.name} timed out after {self.timeout}s"
            ) from exc
        except openai_mod.RateLimitError as exc:
            # An exhausted balance is a 429 as well, and unlike ordinary
            # pressure it is terminal. The marker may arrive as either field.
            if _OPENAI_TERMINAL_CODES & {exc.code, exc.type}:
                raise BudgetLimitError(f"provider budget/limit error: {exc}") from exc
            raise TransientRateLimitError(f"provider rate limit: {exc}") from exc
        except openai_mod.APIStatusError as exc:
            if _OPENAI_TERMINAL_CODES & {exc.code, exc.type} or _BUDGET_RE.search(str(exc)):
                raise BudgetLimitError(f"provider budget/limit error: {exc}") from exc
            raise AnnotatorProcessError(
                f"OpenAI returned {exc.status_code} for {image_path.name}: {exc}"
            ) from exc
        except openai_mod.OpenAIError as exc:
            # a connection or protocol failure: one page's problem, like a
            # crashed CLI, not a reason to abandon the pages after it
            raise AnnotatorProcessError(f"OpenAI call for {image_path.name} failed: {exc}") from exc
        return self._answer(response, image_path)

    def _answer(
        self, response: openai.types.responses.Response, image_path: Path
    ) -> tuple[str, str | None]:
        """The reply text and the ceiling that cut it, or why there is no reply.

        Only ``completed`` and ``incomplete`` carry one. Anything else — a
        failure, a background response still running, a cancellation — has no
        text to read and says which state it was in rather than reporting the
        empty string it would otherwise return.
        """
        if response.status == "failed":
            detail = response.error.message if response.error is not None else "no reason given"
            raise AnnotatorProcessError(f"OpenAI failed {image_path.name}: {detail}")
        if response.status not in ("completed", "incomplete"):
            raise AnnotatorProcessError(
                f"OpenAI returned no reply for {image_path.name}: status {response.status!r}"
            )
        reason = (
            response.incomplete_details.reason if response.incomplete_details is not None else None
        )
        ceiling = f"max_output_tokens={self.max_tokens}"
        return response.output_text, ceiling if reason == "max_output_tokens" else reason


class OllamaBackend:
    """Annotate sheets through Ollama's local ``/api/chat`` HTTP endpoint."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = "http://localhost:11434",
        timeout: float = DEFAULT_TIMEOUT,
        transport: OllamaTransport = _ollama_transport,
        blocked_patterns: Sequence[str] = SMALL_MODEL_PATTERNS,
        prompt_template: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        ensure_model_allowed(model, blocked_patterns)
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self.prompt_template = _resolved_prompt_template(prompt_template)
        self.options = options
        self.last_raw_response = ""

    def annotate(self, image_path: Path) -> Annotation:
        return self.annotate_extended(image_path).annotation

    def annotate_extended(self, image_path: Path) -> ExtendedAnnotation:
        if image_path.suffix.lower() not in _MEDIA_TYPES:
            raise AnnotateError(f"unsupported image type: {image_path.suffix!r}")
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt_template.replace("IMGPATH", image_path.name),
                    "images": [base64.standard_b64encode(image_path.read_bytes()).decode("ascii")],
                }
            ],
        }
        if self.options is not None:
            payload["options"] = self.options
        response = self._transport(f"{self.endpoint}/api/chat", payload, self.timeout)
        self.last_raw_response = json.dumps(response, sort_keys=True)
        error = response.get("error")
        if isinstance(error, str) and error:
            raise AnnotatorProcessError(f"Ollama provider error: {error}")
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise MalformedResponseError("Ollama response has no message.content string")
        return _parse_extended_classified(content)
