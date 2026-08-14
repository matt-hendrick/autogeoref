"""Viewer manifest generator and static deploy bundle.

Import the owning submodule directly (``from autogeoref.viewer.publish import
publish_volume``); this package deliberately re-exports nothing, so importing
one part does not pay for the others. :mod:`.era` owns the era buckets and
their labelling rule, :mod:`.config` ``[viewer]`` parsing and the site dict,
:mod:`.sources` catalog and layer discovery, :mod:`.bounds` extent probes,
:mod:`.manifest` assembly, :mod:`.publish` transactional landing,
:mod:`.deploy` the public bundle.
"""
