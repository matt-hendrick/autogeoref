"""Resolved run behavior: spend, evidence channels, and back-half mode."""

from __future__ import annotations

import importlib.util
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .config.model import CityConfig, VolumeConfig, era_undeclared

logger = logging.getLogger(__name__)


class RunPolicyArgs(Protocol):
    """The narrow argument surface that determines run behavior."""

    city: Path
    volume: str
    warp: bool
    warp_only: bool
    escalate: bool
    no_escalate: bool
    escalate_model: Sequence[str] | None
    no_verify: bool
    verify_junctions: bool
    verified_accept: bool


@dataclass(frozen=True)
class RunPolicy:
    """All run-stage policy resolved before any pipeline work begins."""

    warp: bool
    escalation_models: tuple[str, ...]
    run_escalation: bool
    run_junction: bool
    run_verified: bool
    allowed_channels: frozenset[str]
    escalation_variants: tuple[str | None, ...] = field(default_factory=tuple)

    @staticmethod
    def is_warp_only(args: RunPolicyArgs) -> bool:
        """Validate the config-free serve-only mode before loading city config."""
        if not args.warp_only:
            return False
        if args.warp:
            raise SystemExit("--warp-only contradicts --warp")
        return True

    @classmethod
    def resolve(cls, args: RunPolicyArgs, city: CityConfig, vol: VolumeConfig) -> RunPolicy:
        """Resolve all placement policy, preserving CLI validation order and text."""
        override_models = tuple(args.escalate_model or ())
        escalation_models = override_models or vol.escalation_ladder()
        escalation_variants = (
            (None,) * len(override_models)
            if override_models
            else tuple(variant for _, variant in vol.escalation_tiers())
        )
        if (args.escalate or args.escalate_model) and not escalation_models:
            raise SystemExit(
                "--escalate: no escalation ladder configured — set escalation_models "
                "in the city TOML or pass --escalate-model"
            )
        if args.no_escalate and (args.escalate or args.escalate_model):
            raise SystemExit("--no-escalate contradicts --escalate / --escalate-model")

        channel_flags = (args.verify_junctions, args.verified_accept)
        if args.no_verify and any(channel_flags):
            raise SystemExit("--no-verify contradicts --verify-junctions / --verified-accept")
        channels = () if args.no_verify else vol.evidence_channels
        run_junction = bool(args.verify_junctions or args.verified_accept or "junction" in channels)
        run_verified = bool(args.verified_accept or channels)
        allowed_channels = frozenset(channels) | (
            {"junction"} if args.verify_junctions or args.verified_accept else set()
        )

        # The gate is the CHANNEL being allowed to vote, not any stage: the
        # addresses channel buys nothing, but it still reads sidecars already
        # on disk, and reading a pre-renumbering volume's numerals against
        # today's grid is how the only channel that may REFUTE vetoes correct
        # sheets. A config declaration is the whole trigger.
        if "addresses" in allowed_channels and era_undeclared(city, vol):
            raise SystemExit(
                f"{args.volume}: the addresses channel is ON and this volume declares no "
                f"address era, but {city.name} RENUMBERED its houses (the city config ships "
                f"a renumbering table). Undeclared means MODERN, and on a volume printed "
                f"before the renumbering that reads its numerals against today's grid and "
                f"REFUTES correct sheets — the addresses channel is the only one that can "
                f"veto.\n"
                f"  Set `addresses_modern` in [volumes.{args.volume}] of {args.city}:\n"
                f"    true  = the printed numbers ARE today's numbers (post-renumbering)\n"
                f"    false = the volume predates it; numbers convert through the table\n"
                f"            (check WHICH book: Chicago's Loop renumbered separately, in 1911)\n"
                f"  Or run without the channel: --no-verify, or evidence_channels = [] on "
                f"the volume."
            )
        return cls(
            warp=args.warp,
            escalation_models=escalation_models,
            escalation_variants=escalation_variants,
            run_escalation=bool(escalation_models) and not args.no_escalate,
            run_junction=run_junction,
            run_verified=run_verified,
            allowed_channels=allowed_channels,
        )

    def warn_unavailable_model_clis(self, annotation_model: str | None) -> None:
        """Warn once per model this run could reach and cannot call.

        A CLI provider needs its executable on PATH; a direct-API one needs its
        SDK installed and its key in the environment. Either way the failure
        otherwise surfaces partway in, after prep has already worked. Warn rather
        than refuse: a run whose reads are all cached needs neither, and a
        missing escalation provider is deliberately survivable. ``None`` means
        the run buys no annotation read. Ollama serves itself and needs nothing.
        """
        from .annotate.failures import AnnotateError
        from .annotate.providers import CLI_PROVIDERS, parse_model_ref

        wanted = ([annotation_model] if annotation_model else []) + (
            list(self.escalation_models) if self.run_escalation else []
        )
        missing: dict[str, list[str]] = {}
        for ref in dict.fromkeys(wanted):
            try:
                name = parse_model_ref(ref).provider
            except AnnotateError:
                # An unresolvable reference is the model layer's error to
                # report, with its own message, when a stage builds its backend.
                continue
            provider = CLI_PROVIDERS.get(name)
            # The default executable is what resolves here, since no run path
            # passes an override to `annotate.invocation`.
            if provider is not None:
                binary = provider.default_executable
                if shutil.which(binary) is None:
                    missing.setdefault(f"{binary!r} is not on PATH", []).append(ref)
                continue
            for want in self._unmet_api_requirements(name):
                missing.setdefault(want, []).append(ref)
        for want, refs in sorted(missing.items()):
            logger.warning(
                "%s, so %s cannot be read — see README Setup. Cached reads still replay.",
                want,
                ", ".join(refs),
            )

    @staticmethod
    def _unmet_api_requirements(provider: str) -> list[str]:
        """What a direct-API provider still needs before it can be called."""
        from .annotate.api_call import missing_credential

        packages = {"anthropic-api": ("anthropic", "annotate"), "openai-api": ("openai", "openai")}
        unmet = []
        if provider in packages:
            package, extra = packages[provider]
            if importlib.util.find_spec(package) is None:
                unmet.append(f"the {package!r} package is not installed (autogeoref[{extra}])")
        key = missing_credential(provider)
        if key is not None:
            unmet.append(f"{key} is not set")
        return unmet
