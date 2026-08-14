"""Reviewer review & adjust: ghost-overlay UI server, edit sidecars, and apply.

The UI (``autogeoref review``) walks a volume's flagged pool and records
verdicts; it NEVER mutates pipeline results. Every edit lands in a sidecar
under ``work/<volume>/review/``, and the apply step materializes sidecars into
``OK (reviewer-verified)`` results, re-runs the cutline dry-run for edited
masks, and re-warps.

Import the owning submodule directly (``from autogeoref.review.apply import
apply_reviews``); this package deliberately re-exports nothing, so importing
one part does not pay for the others.

`docs/INTERNALS.md` has the frame conventions, the edit math, the gate semantics,
and the module map. Read it before touching frames.
"""
