"""Sheet cutline masks: pure geometry, the blank-core remedy, and measurement.

Import the owning submodule directly (``from autogeoref.mask.geometry import
heal``); this package deliberately re-exports nothing. :mod:`.geometry` owns
content detection and cutline geometry, :mod:`.move` the blank-core move
remedy, :mod:`.qa` the ``masks-qa.json`` measurement pass.
``autogeoref.bake`` stays top-level — it is a pipeline stage that consumes
this family alongside the mosaic, not a member of it.
"""
