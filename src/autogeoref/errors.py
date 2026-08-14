"""Shared domain exceptions with no stage dependencies."""


class PipelineError(RuntimeError):
    """A pipeline stage cannot proceed with the available volume artifacts."""


class ReviewError(ValueError):
    """A malformed review sidecar, op, or save request."""
