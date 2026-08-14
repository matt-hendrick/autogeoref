"""Provider and model-routing policy for annotation calls."""

from __future__ import annotations

import json
import re
from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from .failures import AnnotateError, ModelQualityError

# The production pipeline is tuned on claude-sonnet-5.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_CODEX_MODEL = "gpt-5.6-terra"
DEFAULT_TIMEOUT = 600.0
DEFAULT_PROVIDER = "anthropic"

SMALL_MODEL_PATTERNS: tuple[str, ...] = (
    "haiku",
    "mini",
    "flash",
    "nano",
    "lite",
    "tiny",
    "instant",
    "1b",
    "2b",
    "3b",
    "4b",
    "5b",
    "6b",
    "7b",
    "8b",
    "9b",
)


def ensure_model_allowed(
    model: str, blocked_patterns: Sequence[str] = SMALL_MODEL_PATTERNS
) -> None:
    """Enforce the Sonnet-class-minimum quality gate at configure time."""
    lowered = model.lower()
    for pattern in blocked_patterns:
        if re.search(rf"(^|[^a-z0-9]){re.escape(pattern.lower())}([^a-z0-9]|$)", lowered):
            raise ModelQualityError(
                f"model {model!r} matches small-model pattern {pattern!r}; the annotator "
                "requires a Sonnet-class model minimum (small models hallucinate street names)"
            )


def _build_cli_argv(
    image_path: Path,
    prompt_template: str,
    model: str,
    executable: str,
    _out_path: Path | None,
    _workdir: Path | None,
) -> list[str]:
    prompt = prompt_template.replace("IMGPATH", str(image_path))
    return [
        executable,
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
        "--add-dir",
        str(image_path.parent),
    ]


def _build_codex_argv(
    image_path: Path,
    prompt_template: str,
    model: str,
    executable: str,
    out_path: Path | None,
    _workdir: Path | None,
) -> list[str]:
    prompt = prompt_template.replace("IMGPATH", str(image_path))
    argv = [
        executable,
        "exec",
        "--model",
        model,
        "--image",
        str(image_path),
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
    ]
    if out_path is not None:
        argv += ["--output-last-message", str(out_path)]
    argv.append(prompt)
    return argv


def _build_opencode_argv(
    image_path: Path,
    prompt_template: str,
    model: str,
    executable: str,
    _out_path: Path | None,
    workdir: Path | None,
) -> list[str]:
    """Attach the sheet through OpenCode's tool-free isolated agent."""
    prompt = prompt_template.replace("IMGPATH", str(image_path))
    argv = [
        executable,
        "run",
        "--pure",
        "--agent",
        "autogeoref-annotation",
    ]
    if workdir is not None:
        argv += ["--dir", str(workdir)]
    argv += [
        "--model",
        model,
        "--file",
        str(image_path),
        "--",
        prompt,
    ]
    return argv


def _configure_opencode_workdir(workdir: Path) -> None:
    """Install a primary agent that can reply but cannot invoke any tool."""
    config = {
        "agent": {
            "autogeoref-annotation": {
                "description": "Return a structured annotation for an attached map image.",
                "mode": "primary",
                "permission": {"*": "deny"},
            }
        },
        "share": "disabled",
        "snapshot": False,
        # Annotation inputs are already prepared to this boundary. Reject rather
        # than silently resizing one provider's image differently from Claude's.
        "attachment": {
            "image": {
                "auto_resize": False,
                "max_width": 2000,
                "max_height": 2000,
                "max_base64_bytes": 5242880,
            }
        },
        # Match the shared backend's 600 second wall-clock boundary.
        "provider": {"openai": {"options": {"timeout": 600000, "chunkTimeout": 600000}}},
    }
    (workdir / "opencode.json").write_text(json.dumps(config), encoding="utf-8")


@dataclass(frozen=True)
class CliProvider:
    """A provider's argv shape, output channel, and model-name policy."""

    name: str
    default_executable: str
    build_argv: Callable[[Path, str, str, str, Path | None, Path | None], list[str]]
    writes_output_file: bool = False
    configure_workdir: Callable[[Path], None] | None = None
    model_prefixes: tuple[str, ...] = ()
    budget_scope: re.Pattern[str] | None = None


CLAUDE_CLI = CliProvider(
    name="anthropic",
    default_executable="claude",
    build_argv=_build_cli_argv,
    model_prefixes=("claude", "opus", "sonnet", "haiku", "fable"),
)
CODEX_CLI = CliProvider(
    name="codex",
    default_executable="codex",
    build_argv=_build_codex_argv,
    writes_output_file=True,
    model_prefixes=("gpt", "o3", "o4", "codex"),
    budget_scope=re.compile(r"(?im)^[^\S\n]*ERROR:.*$"),
)
OPENCODE_CLI = CliProvider(
    name="opencode",
    default_executable="opencode",
    build_argv=_build_opencode_argv,
    configure_workdir=_configure_opencode_workdir,
    # OpenCode model identifiers retain the provider namespace it needs, e.g.
    # ``openai/gpt-5.6-terra``.
    model_prefixes=("openai/",),
)
CLI_PROVIDERS: dict[str, CliProvider] = {
    provider.name: provider for provider in (CLAUDE_CLI, CODEX_CLI, OPENCODE_CLI)
}
#: Providers reached over HTTP instead of by spawning an executable, mapped to
#: the bare model-name prefixes that would be ambiguous without their provider.
#: ``anthropic-api`` is deliberately prefix-free: its model names are the ones a
#: bare reference already means, so only the explicit spelling selects it.
NATIVE_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "ollama": (),
    "anthropic-api": (),
    "openai-api": ("gpt", "o3", "o4"),
}
NATIVE_PROVIDERS = frozenset(NATIVE_MODEL_PREFIXES)
KNOWN_PROVIDERS = frozenset(CLI_PROVIDERS) | NATIVE_PROVIDERS
#: Providers that take a reasoning-effort variant. The others reject one rather
#: than accept a setting they would silently drop.
VARIANT_PROVIDERS = frozenset({"codex", "opencode", "openai-api"})


def _provider_prefixes() -> dict[str, tuple[str, ...]]:
    """Every provider's ambiguous bare-name prefixes, CLI and native alike."""
    return {p.name: p.model_prefixes for p in CLI_PROVIDERS.values()} | NATIVE_MODEL_PREFIXES


@dataclass(frozen=True)
class ModelRef:
    """A model name plus the provider that runs it."""

    provider: str
    model: str

    @property
    def key(self) -> str:
        return self.model if self.provider == DEFAULT_PROVIDER else f"{self.provider}:{self.model}"

    def __str__(self) -> str:
        return self.key


def parse_model_ref(ref: str, default_provider: str = DEFAULT_PROVIDER) -> ModelRef:
    """Resolve a config model reference while retaining bare-Anthropic compatibility."""
    provider, separator, rest = ref.partition(":")
    if separator:
        if provider not in KNOWN_PROVIDERS or not rest:
            known = ", ".join(sorted(KNOWN_PROVIDERS))
            raise AnnotateError(
                f"model reference {ref!r} names an unknown provider {provider!r} "
                f"(known: {known}). A bare model name means {DEFAULT_PROVIDER}."
            )
        resolved = ModelRef(provider=provider, model=rest)
    else:
        resolved = ModelRef(provider=default_provider, model=ref)
    ensure_model_allowed(resolved.model)
    if not separator:
        lowered = ref.lower()
        # One prefix can belong to several providers (`gpt` to both the Codex
        # CLI and the OpenAI API), so name every spelling that would work. A
        # caller that already chose one of them is not guessing and keeps it.
        candidates = [
            name
            for name, prefixes in _provider_prefixes().items()
            if any(lowered.startswith(prefix) for prefix in prefixes)
        ]
        if candidates and default_provider not in candidates:
            named = " or ".join(candidates)
            spellings = " or ".join(f"{name}:{ref}" for name in candidates)
            raise AnnotateError(
                f"bare model name {ref!r} looks like a {named} model, but a bare "
                f"name means {default_provider}. Write it as {spellings}."
            )
    return resolved


def canonical_model(ref: str, default_provider: str = DEFAULT_PROVIDER) -> str:
    """Collapse a model reference to its cache and consensus identity."""
    return parse_model_ref(ref, default_provider).key


def model_cache_key(model: str, variant: str | None = None, prompt: str | None = None) -> str:
    """Return a filesystem-safe, reversible cache identity for model, effort and prompt.

    The prompt belongs in the key because a cache hit is only free if it
    answers the same question: two prompts at one model would otherwise replay
    each other's reads. Omitting it preserves every existing cache path.
    """
    canonical = canonical_model(model)
    if variant is None and prompt is None:
        # Preserve every pre-variant cache path exactly.
        return canonical
    parts = [canonical, variant] if prompt is None else [canonical, variant, prompt]
    payload = json.dumps(parts, separators=(",", ":")).encode()
    return "v2-" + urlsafe_b64encode(payload).decode().rstrip("=")


def prior_variant_cache_key(model: str, variant: str | None) -> str | None:
    """Return the short-lived v1 variant key written before v2 cache encoding."""
    if variant is None:
        return None
    return f"{quote(canonical_model(model), safe='.-_')}.{quote(variant, safe='.-_')}"


def model_from_cache_key(cache_key: str) -> str:
    """Recover a canonical model identity from a model-cache filename component."""
    if cache_key.startswith("v2-"):
        encoded = cache_key.removeprefix("v2-")
        try:
            raw = urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            model, variant, *_prompt = json.loads(raw)
        except (BinasciiError, ValueError, TypeError):
            return cache_key
        # variant may be null in a prompt-keyed identity; that is still reversible
        return model if isinstance(model, str) and isinstance(variant, str | None) else cache_key
    # v1 escaped provider-qualified model IDs and appended ``.<variant>``.
    # This fallback only recognizes escaped provider separators, so an old
    # unvariant model name ending in a dot component remains unchanged.
    model, separator, _variant = cache_key.rpartition(".")
    return unquote(model) if separator and "%3A" in model else cache_key


def resolve_provider(name: str) -> CliProvider:
    """Look up a registered CLI provider by name.

    A native provider is a caller error here, not an unknown name: it has a
    backend, just not one that spawns anything.
    """
    try:
        return CLI_PROVIDERS[name]
    except KeyError:
        if name in NATIVE_PROVIDERS:
            raise AnnotateError(
                f"provider {name!r} is reached over HTTP and spawns no executable; "
                "build it with annotate.invocation.backend_for_model"
            ) from None
        known = ", ".join(sorted(KNOWN_PROVIDERS))
        raise AnnotateError(f"unknown CLI provider {name!r} (known: {known})") from None


def _choose_provider(model: str, provider: str | CliProvider | None) -> CliProvider:
    """Apply the shared provider-precedence rule."""
    if ":" in model:
        return resolve_provider(parse_model_ref(model).provider)
    if provider is None:
        return resolve_provider(parse_model_ref(model).provider)
    return resolve_provider(provider) if isinstance(provider, str) else provider
