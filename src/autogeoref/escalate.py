"""Model-escalation stage: re-annotate hard sheets, cheap tiers first.

A stronger model reads the same streets more precisely, and one corrected
label is the difference between a poisoned RANSAC and a clean fit. The stage
is config-driven — no ladder, no stage — and evidence-gated: only REJECTED
pages with at least :data:`MIN_JUNCTIONS_TO_ESCALATE` drawn junctions are
re-read, because no model can read streets that are not drawn.

Escalated annotations cache separately, one file per tier; the v1 cache is
never overwritten, and a per-tier failure marker is the retry ledger. A
reported budget or usage limit is TERMINAL for the whole stage.

Acceptance still runs the full constrained gates: escalation adds evidence,
never weakens a gate. `docs/OPERATIONS.md` § Model Escalation is the contract.
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .annotate.failures import BudgetLimitError
from .annotate.invocation import annotate_with_retry, backend_for_model
from .annotate.providers import canonical_model, model_cache_key, prior_variant_cache_key
from .annotate.schema import Annotation, AnnotatorBackend
from .paths import VolumePaths, atomic_write_text, iter_results, small_sheet_entry, write_result
from .sheet_inputs import sheet_input_from
from .volume import (
    REJECTED_PREFIX,
    REVOKED_PREFIX,
    STATUS_OK,
    VolumeConstraints,
    constraints_for_page,
    match_sheet,
)

if TYPE_CHECKING:
    from .centerlines import CenterlineIndex

logger = logging.getLogger(__name__)

MIN_JUNCTIONS_TO_ESCALATE = 4
# One retry per tier for transient/malformed responses.
# budget.ESCALATION_ATTEMPTS mirrors this value.
MAX_ATTEMPTS = 2


def _default_annotator() -> Callable[[Path, str, str | None], dict[str, Any]]:
    # default v2 prompt: the escalated cache then carries the additive v2
    # channels (numerals/margins/rail), which the verified-accept stage reads.
    # backend_for_model routes each tier to ITS provider, so a ladder may mix
    # them (`["claude-sonnet-5", "codex:gpt-5.6-terra"]`).
    backends: dict[tuple[str, str | None], AnnotatorBackend] = {}
    backends_lock = threading.Lock()

    def annotate(img: Path, model: str, variant: str | None = None) -> dict[str, Any]:
        key = (model, variant)
        with backends_lock:
            backend = backends.get(key)
            if backend is None:
                backend = backends[key] = (
                    backend_for_model(model, variant=variant)
                    if variant is not None
                    else backend_for_model(model)
                )
        return backend.annotate_extended(img).raw

    return annotate


_TierAnnotator = Callable[[Path, str, str | None], "Annotation | dict[str, Any]"]
_InjectedAnnotator = Callable[[Path, str], "Annotation | dict[str, Any]"] | _TierAnnotator


def _adapt_annotator(annotate_fn: _InjectedAnnotator | None) -> _TierAnnotator:
    """One ``(image, model, variant)`` arity for the injected and default paths.

    An injected annotator may take ``(image, model)`` or
    ``(image, model, variant)`` — positionally or with ``variant``
    keyword-only; the arity is chosen once, here, so a two-arg injection is
    bridged explicitly rather than silently dropping the tier variant a
    three-arg one receives. A variadic signature (``*args``, mocks) binds
    three and is treated as three-arg.
    """
    if annotate_fn is None:
        return _default_annotator()
    try:
        inspect.signature(annotate_fn).bind(Path("img"), "model", None)
    except (TypeError, ValueError):
        try:
            inspect.signature(annotate_fn).bind(Path("img"), "model", variant=None)
        except (TypeError, ValueError):
            two_arg = cast("Callable[[Path, str], Annotation | dict[str, Any]]", annotate_fn)

            def drop_variant(
                img: Path, model: str, _variant: str | None
            ) -> Annotation | dict[str, Any]:
                return two_arg(img, model)

            return drop_variant

        kw_fn = annotate_fn

        def kw_variant(img: Path, model: str, variant: str | None) -> Annotation | dict[str, Any]:
            return kw_fn(img, model, variant=variant)  # type: ignore[call-arg]

        return kw_variant
    return cast("_TierAnnotator", annotate_fn)


def resolve_tiers(
    model: str | Sequence[str],
    variants: str | Sequence[str | None] | None = None,
) -> tuple[tuple[str, str | None], ...]:
    """Normalize a ladder spec into ``(canonical_model, variant)`` tiers.

    A single model name is a one-tier ladder; a single variant (or ``None``)
    applies to every tier. An empty ladder resolves to no tiers (no stage).
    """
    models: tuple[str, ...] = (model,) if isinstance(model, str) else tuple(model)
    if not models:
        return ()
    if isinstance(variants, str) or variants is None:
        tier_variants: tuple[str | None, ...] = (variants,) * len(models)
    else:
        tier_variants = tuple(variants)
    if len(tier_variants) != len(models):
        raise ValueError("escalation variants must have one entry per model")
    canonical_models = tuple(canonical_model(m) for m in models)
    if len(set(canonical_models)) != len(canonical_models):
        raise ValueError("escalation ladder cannot repeat a model at different reasoning variants")
    return tuple(zip(canonical_models, tier_variants, strict=True))


@dataclass(frozen=True)
class PagePlan:
    """An evidence-gated REJECTED page, ready to walk the ladder."""

    page: str
    info: Any
    image: Path
    result_path: Path


def _gated_pages(
    paths: VolumePaths, manifest: Mapping[str, Any], min_junctions: int
) -> list[PagePlan]:
    """REJECTED pages whose drawn-street evidence clears the junction gate."""
    from .junction_snap import JunctionSnapError, extract_junctions

    planned: list[PagePlan] = []
    for page, r, rp in iter_results(paths):
        status = str(r.get("status", ""))
        # provisional-revoked pages keep their corroboration path; escalate
        # only plain no-model rejections
        if not status.startswith(REJECTED_PREFIX) or status.startswith(REVOKED_PREFIX):
            continue
        # no rotation skip: sheet_input_from maps the re-annotation back into
        # the source frame, so rotation-normalized smalls escalate fine
        entry = small_sheet_entry(paths, manifest, page, stage="escalate", skip_rotated=False)
        if entry is None:
            continue
        info, img = entry
        try:
            n_junc = extract_junctions(img).n_junctions
        except JunctionSnapError:
            continue
        if n_junc < min_junctions:
            continue
        planned.append(PagePlan(page=page, info=info, image=img, result_path=rp))
    return planned


def _tier_paths(paths: VolumePaths, page: str, cache_key: str) -> tuple[Path, Path]:
    """The tier's (annotation cache, failure marker) pair."""
    return (
        paths.annotations / f"p{page}.escalated.{cache_key}.json",
        paths.annotations / f"p{page}.escalated.{cache_key}.failed.json",
    )


def _log_failed_tier(page: str, tier: int, tier_model: str, marker: Path) -> None:
    # a previous run exhausted this tier's attempts; do not re-spend every
    # run — delete the marker to retry deliberately
    logger.info(
        "p%s: tier %d (%s) previously failed (%s), trying next tier",
        page,
        tier,
        tier_model,
        marker,
    )


def _write_failure_marker(
    failed_path: Path, tier_model: str, tier_variant: str | None, error: str
) -> None:
    atomic_write_text(
        failed_path,
        json.dumps(
            {
                "model": tier_model,
                "variant": tier_variant,
                "attempts": MAX_ATTEMPTS,
                "error": error,
            }
        ),
    )


class _Escalator:
    """Stage-wide escalation state: ladder, match context, terminal budget stop.

    A reported budget/usage limit is TERMINAL for the whole stage: every
    further call, at any tier, is a doomed spend. Call admission and the
    terminal stop share one lock (``call_admission``): a worker either
    reserved its bounded in-flight provider attempt before a concurrent stop
    landed, or cannot reserve another attempt after it.
    """

    def __init__(
        self,
        paths: VolumePaths,
        index: CenterlineIndex,
        constraints: VolumeConstraints,
        tiers: tuple[tuple[str, str | None], ...],
        annotate: _TierAnnotator,
        page_scale_multiples: Mapping[str, float] | None,
    ) -> None:
        self.paths = paths
        self.index = index
        self.constraints = constraints
        self.tiers = tiers
        self.annotate = annotate
        self.page_scale_multiples = page_scale_multiples
        self.aliases = index.aliases
        # module-attribute lookups, at call time: tests substitute
        # escalate.threading to observe the stop primitives
        self.budget_hit = threading.Event()
        self.call_admission = threading.Lock()

    def admit_attempt(self) -> bool:
        with self.call_admission:
            return not self.budget_hit.is_set()

    def escalate_page(self, plan: PagePlan) -> str | None:
        """Walk one page up the ladder; the first tier that passes the gates wins."""
        for tier, (tier_model, tier_variant) in enumerate(self.tiers, start=1):
            if self.budget_hit.is_set():
                return None
            ann_dict = self._tier_annotation(plan, tier, tier_model, tier_variant)
            if ann_dict is None:
                if self.budget_hit.is_set():
                    return None
                continue  # this tier produced nothing; try the next
            if self._commit_flip(plan, tier, tier_model, tier_variant, ann_dict):
                return plan.page  # first passing tier wins; frontier tiers stay uncalled
        return None

    def _tier_annotation(
        self, plan: PagePlan, tier: int, tier_model: str, tier_variant: str | None
    ) -> dict[str, Any] | None:
        """One tier's annotation: cached, marker-skipped, or a spent call.

        ``None`` means the tier produced nothing — either skip to the next
        tier, or a terminal budget stop (distinguished by ``budget_hit``).
        """
        page = plan.page
        cache_key = model_cache_key(tier_model, tier_variant)
        esc_path, failed_path = _tier_paths(self.paths, page, cache_key)
        if esc_path.exists():
            return cast("dict[str, Any]", json.loads(esc_path.read_text()))
        if failed_path.exists():
            _log_failed_tier(page, tier, tier_model, failed_path)
            return None
        prior_key = prior_variant_cache_key(tier_model, tier_variant)
        if prior_key is not None:
            prior_path, prior_failed_path = _tier_paths(self.paths, page, prior_key)
            if prior_path.exists():
                return cast("dict[str, Any]", json.loads(prior_path.read_text()))
            if prior_failed_path.exists():
                _log_failed_tier(page, tier, tier_model, prior_failed_path)
                return None
        return self._spend_call(plan, tier, tier_model, tier_variant, esc_path, failed_path)

    def _spend_call(
        self,
        plan: PagePlan,
        tier: int,
        tier_model: str,
        tier_variant: str | None,
        esc_path: Path,
        failed_path: Path,
    ) -> dict[str, Any] | None:
        def _read() -> Annotation | dict[str, Any]:
            return self.annotate(plan.image, tier_model, tier_variant)

        try:
            annotation, last_error = annotate_with_retry(
                _read,
                MAX_ATTEMPTS,
                f"p{plan.page} tier {tier} ({tier_model})",
                cancelled=self.budget_hit.is_set,
                admit_attempt=self.admit_attempt,
            )
        except BudgetLimitError as exc:
            # TERMINAL for the whole stage: a budget/usage limit means every
            # further call, at any tier, is a doomed spend. Stop escalating.
            with self.call_admission:
                self.budget_hit.set()
            logger.error("escalation halted: budget limit reported (%s)", exc)
            return None
        if annotation is None:
            # Another worker may have reported a terminal limit while this
            # page was in a provider call. That interruption is not an
            # exhausted tier and must not become a retry marker.
            if self.budget_hit.is_set():
                return None
            _write_failure_marker(failed_path, tier_model, tier_variant, last_error)
            return None  # this tier cannot read the sheet; try the next
        ann_dict = annotation.to_dict() if isinstance(annotation, Annotation) else annotation
        atomic_write_text(esc_path, json.dumps(ann_dict, indent=2))
        return ann_dict

    def _commit_flip(
        self,
        plan: PagePlan,
        tier: int,
        tier_model: str,
        tier_variant: str | None,
        ann_dict: dict[str, Any],
    ) -> bool:
        # sheet_input_from maps the re-annotation into the source-scan
        # frame and applies the legacy scale rule — same as match/rescue
        sheet = sheet_input_from(plan.page, ann_dict, plan.info)
        record = match_sheet(
            sheet,
            self.index,
            # a page printed at another scale is re-matched against ITS window,
            # exactly as stage_match does — against the volume's it could only
            # ever fail, so escalation would spend model budget on a page that
            # cannot pass by construction (volume.constraints_for_page)
            constraints_for_page(plan.page, self.constraints, self.page_scale_multiples),
            self.aliases,
        )
        if record.get("status") != STATUS_OK:
            logger.info(
                "p%s: tier %d (%s) did not produce a strict accept", plan.page, tier, tier_model
            )
            return False
        record["escalated_model"] = tier_model
        if tier_variant is not None:
            record["escalated_variant"] = tier_variant
        write_result(plan.result_path, record)
        logger.info(
            "p%s: ESCALATION FLIP -> strict accept (%d inliers, %s, tier %d)",
            plan.page,
            record["n_inliers"],
            tier_model,
            tier,
        )
        return True


def stage_escalate(
    paths: VolumePaths,
    index: CenterlineIndex,
    constraints: VolumeConstraints,
    model: str | Sequence[str],
    variants: str | Sequence[str | None] | None = None,
    annotate_fn: _InjectedAnnotator | None = None,
    min_junctions: int = MIN_JUNCTIONS_TO_ESCALATE,
    page_scale_multiples: Mapping[str, float] | None = None,
    jobs: int = 1,
) -> list[str]:
    """Escalate evidence-gated REJECTED pages up a cheap-first model ladder.

    ``model`` is the ladder, cheapest tier first; a single name is a one-tier
    ladder. ``page_scale_multiples`` re-centres a page printed at another
    scale, since re-matching one against the VOLUME window could only fail
    however good the new annotation is. ``annotate_fn`` is the injectable
    annotator; it defaults to the CLI backend with the production prompt,
    cached raw. ``jobs`` bounds concurrent pages — tiers within a page stay
    serial and cheap-first. Returns the pages flipped to strict accepts.
    """
    tiers = resolve_tiers(model, variants)
    if not tiers:
        return []
    manifest = json.loads(paths.manifest.read_text())
    planned = _gated_pages(paths, manifest, min_junctions)
    escalator = _Escalator(
        paths=paths,
        index=index,
        constraints=constraints,
        tiers=tiers,
        annotate=_adapt_annotator(annotate_fn),
        page_scale_multiples=page_scale_multiples,
    )
    # The preflight is intentionally serial: it confines worker concurrency to
    # provider-bound pages. `map` yields plan order, preserving the public result
    # order even though providers and file publication complete independently.
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        flipped = [page for page in pool.map(escalator.escalate_page, planned) if page is not None]
    logger.info("escalation: %d pages flipped to strict accepts", len(flipped))
    return flipped
