"""Annotate prepared sheets and cache their per-page results.

Planning reports a retry-aware call range before spending. Failed reads use
markers so they are not mistaken for annotations; callers must handle those
pages explicitly before continuing.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .annotate.failures import BudgetLimitError
from .annotate.invocation import annotate_with_retry, backend_for_model
from .annotate.providers import model_cache_key, prior_variant_cache_key
from .annotate.schema import prompt_template
from .budget import DEFAULT_GATED_FRACTION, SpendEstimate, estimate_spend
from .paths import VolumePaths, atomic_write_text, sheet_images

logger = logging.getLogger(__name__)


class NotPreppedError(RuntimeError):
    """No sheet manifest: prep has not run, so there is nothing to annotate."""


@dataclass(frozen=True)
class ReadIdentity:
    """What makes one annotation read answer the same question as another.

    Model, reasoning variant and prompt name travel together: they key the
    cache, they are written into every record and marker, and a batch that
    changes any of them must re-read rather than replay. ``None`` variant and
    prompt are the pre-variant model and the frozen prompt.
    """

    model: str
    variant: str | None = None
    prompt: str | None = None

    @property
    def key(self) -> str:
        """The filesystem-safe cache identity for this read."""
        return model_cache_key(self.model, self.variant, self.prompt)

    @property
    def prior_key(self) -> str | None:
        """The pre-variant spelling of this read, or ``None`` when it has none.

        A prompt-keyed read inherits nothing: every read written under the old
        spelling was taken with the frozen prompt.
        """
        if self.prompt is not None:
            return None
        return prior_variant_cache_key(self.model, self.variant)

    def record(self) -> dict[str, Any]:
        """This identity as every cache record and marker spells it.

        ``prompt`` is omitted when it is the frozen one, so records written
        before prompts were selectable stay byte-identical.
        """
        out: dict[str, Any] = {"model": self.model, "variant": self.variant}
        if self.prompt is not None:
            out["prompt"] = self.prompt
        return out

    def label(self) -> str:
        """This identity as a run summary prints it."""
        text = self.model if self.variant is None else f"{self.model} ({self.variant})"
        return text if self.prompt is None else f"{text} [{self.prompt} prompt]"


@dataclass(frozen=True)
class AnnotationPlan:
    """What a batch WOULD spend, computed before it spends anything.

    ``todo`` is the pages that will actually be read; ``uncached`` is how many
    wanted one before ``limit`` was applied. The two differ only when a limit
    capped the batch, and :meth:`summary` says so out loud when they do.
    """

    volume: str
    #: Model, variant and prompt for these reads; see :class:`ReadIdentity`.
    identity: ReadIdentity
    pages: list[str]
    #: Pages with an annotation or failure marker already on disk.
    cached: list[str]
    #: Pages without a small image; they cannot be counted as callable reads.
    no_small: list[str]
    #: pages that want a call, before ``limit``
    uncached: list[str]
    #: pages this batch will actually read (``uncached``, capped by ``limit``)
    todo: list[str]
    #: Maximum attempts per page, including retries.
    attempts: int = 2
    #: Pages whose only annotation is a pre-identity ``p<N>.json``: reused as-is,
    #: re-read only via the explicit ``reread_unattributed`` opt-in.
    legacy: list[str] = field(default_factory=list)
    #: Escalation ladder depth for the run this plan estimates; 0 when the
    #: caller spends on the annotate batch alone.
    escalation_tiers: int = 0
    #: City-configured share of sheets expected to reach the gated pool.
    gated_fraction: float = DEFAULT_GATED_FRACTION

    @property
    def calls(self) -> int:
        """Minimum calls: one read per planned page."""
        return len(self.todo)

    @property
    def estimate(self) -> SpendEstimate:
        """The published bound (`budget.estimate_spend`): retry AND escalation in."""
        return estimate_spend(
            sheets=len(self.pages),
            cached=len(self.cached),
            unread=len(self.todo),
            escalation_tiers=self.escalation_tiers,
            gated_fraction=self.gated_fraction,
            attempts=max(1, self.attempts),
        )

    @property
    def ceiling(self) -> int:
        """The planning ceiling, not a billing guarantee: the escalation term
        stands on the gated-fraction estimate (`budget` module docstring).
        See :meth:`estimate`."""
        return self.estimate.ceiling

    @property
    def capped(self) -> bool:
        return len(self.todo) != len(self.uncached)

    def summary(self) -> str:
        # Report the shared upper bound rather than understating spend.
        spend = self.estimate.render()
        return (
            f"{self.volume}: {len(self.pages)} pages, {len(self.cached)} already cached/failed, "
            f"{spend} to spend ({self.identity.label()})"
            + (f" — capped by --limit from {len(self.uncached)} pages" if self.capped else "")
            + (f"; {len(self.no_small)} page(s) have no small on disk" if self.no_small else "")
            + (
                f"; {len(self.legacy)} unattributed pre-identity read(s) reused "
                "(--reread-unattributed re-reads them under the configured model)"
                if self.legacy
                else ""
            )
        )


@dataclass(frozen=True)
class AnnotationResult:
    plan: AnnotationPlan
    annotated: int
    #: Pages with a failure marker and no annotation after this batch.
    unread: list[str]

    @property
    def failed(self) -> int:
        return self.plan.calls - self.annotated


def _model_cache_path(paths: VolumePaths, page: str, identity: ReadIdentity) -> Path:
    return paths.annotations / f"{page}.annotation.{identity.key}.json"


def _model_failed_path(paths: VolumePaths, page: str, identity: ReadIdentity) -> Path:
    return paths.annotations / f"{page}.annotation.{identity.key}.failed.json"


def _prior_model_cache_path(
    paths: VolumePaths, page: str, identity: ReadIdentity, failed: bool = False
) -> Path | None:
    key = identity.prior_key
    if key is None:
        return None
    suffix = ".failed.json" if failed else ".json"
    return paths.annotations / f"{page}.annotation.{key}{suffix}"


def _active_cache_path(paths: VolumePaths, page: str) -> Path:
    return paths.annotations / f"{page}.annotation.active.json"


def _cached_path(paths: VolumePaths, page: str, identity: ReadIdentity) -> Path | None:
    current = _model_cache_path(paths, page, identity)
    if current.exists():
        return current
    prior = _prior_model_cache_path(paths, page, identity)
    return prior if prior is not None and prior.exists() else None


def _failed_path(paths: VolumePaths, page: str, identity: ReadIdentity) -> Path | None:
    current = _model_failed_path(paths, page, identity)
    if current.exists():
        return current
    prior = _prior_model_cache_path(paths, page, identity, failed=True)
    return prior if prior is not None and prior.exists() else None


def _has_cached_read(paths: VolumePaths, page: str, identity: ReadIdentity) -> bool:
    return _cached_path(paths, page, identity) is not None


def _has_failed_read(paths: VolumePaths, page: str, identity: ReadIdentity) -> bool:
    return _failed_path(paths, page, identity) is not None


def _has_legacy_read(paths: VolumePaths, page: str) -> bool:
    """Bare ``p<N>.json`` with no cache records; any ``.annotation.*`` sibling
    puts the page under per-model semantics instead."""
    if not (paths.annotations / f"{page}.json").exists():
        return False
    return not any(paths.annotations.glob(f"{page}.annotation.*"))


def unread_pages(paths: VolumePaths, pages: list[str], identity: ReadIdentity) -> list[str]:
    """Return pages with a failure marker but no annotation.

    Failure markers prevent automatic re-spending, so callers must surface these
    pages rather than treating the volume as complete.
    """
    return [
        p
        for p in pages
        if _has_failed_read(paths, p, identity) and not _has_cached_read(paths, p, identity)
    ]


def clear_failed_markers(paths: VolumePaths) -> list[str]:
    """Delete every ``*.failed.json`` retry marker; return the deleted names.

    Markers deliberately survive a re-run — :func:`plan` counts a marked page as
    cached — so nothing re-spends until a human decides to. This function IS
    that decision: the caller has read the failure and chosen to pay for the
    re-reads. Successful annotations keep replaying free, and the per-tier
    escalation markers have the same retry-ledger semantics. The ``v2`` markers
    are swept too but are INERT history, since their producer is gone.
    """
    if not paths.annotations.is_dir():
        return []
    markers = sorted(paths.annotations.glob("*.failed.json"))
    for marker in markers:
        marker.unlink()
    return [m.name for m in markers]


def manifest_pages(paths: VolumePaths) -> list[str]:
    """Return addressable pages from the prep manifest.

    Raises :class:`NotPreppedError` when preparation has not created a manifest.
    """
    if not paths.manifest.exists():
        raise NotPreppedError(
            f"no sheet manifest at {paths.manifest} — prep has not run, so there are no "
            "smalls to read. `autogeoref run` does prep first; run `autogeoref prep` to "
            "look before spending."
        )
    manifest = json.loads(paths.manifest.read_text())
    # Underscore-prefixed manifest entries are metadata, not pages.
    return sorted(p for p in manifest if not p.startswith("_"))


def unprepped_summary(
    paths: VolumePaths,
    volume: str,
    model: str,
    variant: str | None,
    limit: int | None,
    *,
    escalation_tiers: int,
    gated_fraction: float,
) -> str:
    """The preamble line for a volume prep has not written a manifest for yet.

    Two states, and saying the wrong one is what makes a correct nonzero exit
    read as a bug: nothing fetched at all, or scans on disk that prep will read
    later in this same invocation. The second counts the addressable scans as
    the upper bound it is — a silent zero would let an operator read "0 calls"
    and get a full batch.
    """
    from .prep import page_of

    scans = sheet_images(paths.regions)
    if not scans:
        # Keyed on the scans the prep stage is keyed on, NOT on the addressable
        # subset: a volume whose scans are all unnameable does run prep, and
        # fails there.
        return (
            f"{volume}: no scans under {paths.regions} — fetch them first "
            f"(scripts/fetch_loc_volume.py {volume}), then this reads about one "
            "sheet per model call. Nothing to prep, annotate or spend yet"
        )
    addressable = [p for p in scans if page_of(p) is not None]
    pages = len(addressable) if limit is None else min(len(addressable), limit)
    est = estimate_spend(
        sheets=pages, escalation_tiers=escalation_tiers, gated_fraction=gated_fraction
    )
    return (
        f"{volume}: not prepped yet — prep runs first. Up to {pages} sheets to read "
        f"= {est.render()} "
        f"({model if variant is None else f'{model} ({variant})'})"
        + (f" — capped by --limit from {len(addressable)} pages" if limit is not None else "")
    )


def plan(
    paths: VolumePaths,
    volume: str,
    *,
    identity: ReadIdentity,
    pages: list[str] | None = None,
    limit: int | None = None,
    attempts: int = 2,
    reread_unattributed: bool = False,
    escalation_tiers: int = 0,
    gated_fraction: float = DEFAULT_GATED_FRACTION,
) -> AnnotationPlan:
    """Plan annotation calls from the filesystem without spending.

    Pages without small images cannot be read and are excluded from the estimate.
    Pre-identity reads are reused, not re-read; ``reread_unattributed`` opts them
    into spend under the configured model. ``escalation_tiers`` widens the
    published ceiling to the whole run's spend; the annotate stage itself
    plans with 0 (it cannot escalate).
    """
    all_pages = manifest_pages(paths)
    wanted = all_pages if pages is None else [p for p in all_pages if p in set(pages)]
    cached = [
        p
        for p in wanted
        if _has_cached_read(paths, p, identity) or _has_failed_read(paths, p, identity)
    ]
    done = set(cached)
    legacy = (
        []
        if reread_unattributed
        else [p for p in wanted if p not in done and _has_legacy_read(paths, p)]
    )
    done |= set(legacy)
    no_small = [
        p for p in wanted if p not in done and not (paths.sheets / f"{p}_small.jpg").exists()
    ]
    unreadable = done | set(no_small)
    uncached = [p for p in wanted if p not in unreadable]
    return AnnotationPlan(
        volume=volume,
        identity=identity,
        pages=wanted,
        cached=cached,
        no_small=no_small,
        uncached=uncached,
        todo=uncached if limit is None else uncached[:limit],
        attempts=attempts,
        legacy=legacy,
        escalation_tiers=escalation_tiers,
        gated_fraction=gated_fraction,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish an annotation cache record without exposing partial JSON."""
    atomic_write_text(path, json.dumps(payload, indent=2))


def annotate_volume(
    paths: VolumePaths,
    volume: str,
    *,
    identity: ReadIdentity,
    pages: list[str] | None = None,
    limit: int | None = None,
    attempts: int = 2,
    jobs: int = 1,
    reread_unattributed: bool = False,
) -> AnnotationResult:
    """Annotate every uncached page. Returns what it planned and what landed.

    ``jobs`` runs that many pages concurrently (each call is its own backend
    subprocess, so the pool is threads waiting on I/O). Keep it small: this is a
    metered backend. A :class:`BudgetLimitError` from any worker is TERMINAL — it
    stops every page not yet started and propagates, because every further call is
    a doomed spend — but the at most ``jobs - 1`` calls already in flight still
    finish, so a large pool means a larger overshoot past the limit.
    """
    batch = plan(
        paths,
        volume,
        identity=identity,
        pages=pages,
        limit=limit,
        attempts=attempts,
        reread_unattributed=reread_unattributed,
    )
    logger.info("%s (annotate stage only; escalation excluded)", batch.summary())
    # Keyed reads can be reused after another model became active. Pre-identity
    # p<N>.json files are reused as-is; no active pointer is forged for them.
    for page in batch.cached:
        cached_path = _cached_path(paths, page, identity)
        if cached_path is not None:
            _write_json(paths.annotations / f"{page}.json", json.loads(cached_path.read_text()))
            _write_json(_active_cache_path(paths, page), identity.record())
    if not batch.todo:
        return AnnotationResult(
            plan=batch, annotated=0, unread=unread_pages(paths, batch.pages, identity)
        )

    paths.annotations.mkdir(parents=True, exist_ok=True)
    budget_hit = threading.Event()
    # One reader for the batch, built after the cached pages are settled so a
    # fully cached volume needs no provider at all. An API-backed provider holds
    # an HTTP client, and one per page would churn its connection pool.
    reader = backend_for_model(
        identity.model,
        variant=identity.variant,
        prompt_template=prompt_template(identity.prompt),
    )

    def annotate_page(page: str) -> bool:
        if budget_hit.is_set():  # a worker already hit the terminal limit
            return False
        img = paths.sheets / f"{page}_small.jpg"
        if not img.exists():
            logger.warning("%s: no small on disk, skipping", page)
            return False
        try:
            result, error = annotate_with_retry(
                lambda: reader.annotate_extended(img),
                attempts,
                f"{volume} {page}",
                # retries WAIT now, so a worker that slept through another
                # worker's limit would wake and spend into an exhausted budget
                cancelled=budget_hit.is_set,
            )
        except BudgetLimitError:
            budget_hit.set()
            raise
        if result is None:
            # a call that did NOT land: a marker, never an empty annotation
            _write_json(
                _model_failed_path(paths, page, identity),
                {"page": page, **identity.record(), "error": error},
            )
            # A failed replacement must never leave a previous model's active
            # annotation available to --allow-failed-reads or the match stage.
            (paths.annotations / f"{page}.json").unlink(missing_ok=True)
            _write_json(_active_cache_path(paths, page), {**identity.record(), "status": "failed"})
            _write_json(
                paths.annotations / f"{page}.failed.json",
                {"page": page, **identity.record(), "error": error},
            )
            return False
        _write_json(_model_cache_path(paths, page, identity), result.raw)
        _write_json(_active_cache_path(paths, page), identity.record())
        _write_json(paths.annotations / f"{page}.json", result.raw)
        streets = len(result.raw.get("streets") or [])
        rails = len(result.raw.get("rail_labels") or [])
        logger.info("%s: %d streets, %d rail labels", page, streets, rails)
        return True

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        annotated = sum(pool.map(annotate_page, batch.todo))
    logger.info("%s: annotated %d/%d", volume, annotated, batch.calls)
    return AnnotationResult(
        plan=batch, annotated=annotated, unread=unread_pages(paths, batch.pages, identity)
    )
