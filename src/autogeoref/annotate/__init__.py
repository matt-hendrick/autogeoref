"""Vision annotation: what to ask a model, how to ask it, and how it fails.

Import the owning submodule directly (``from autogeoref.annotate.schema import
Annotation``); this package deliberately re-exports nothing, so importing
one part does not pay for the others. :mod:`.schema` owns the prompts and the
response types, :mod:`.providers` the model-reference routing and cache keys,
:mod:`.cli_call` running one CLI and judging what came back, :mod:`.api_call`
the backends that call a provider over HTTP, :mod:`.invocation` the CLI backends
and the choice between all of them, and :mod:`.failures` the exception taxonomy
the callers branch on. The two ``*_call`` modules are where spending happens.
"""
