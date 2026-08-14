"""Acquisition, placement, and serving queues.

The network-bound fetch, model-bound placement, and CPU-bound serving tracks drain
independently, each promoting a success to the next: fetch -> place -> serve. Progress derives
from pipeline artifacts, and queue read-modify-write operations hold a file lock so concurrent
drains cannot overwrite membership changes.

Import the owning submodule directly (``from autogeoref.queue.store import add``); this package
deliberately re-exports nothing. :mod:`.store` owns the queue file, :mod:`.command` composes
each leg's child command, :mod:`.run` drains, :mod:`.publish` settles a publish a dead drain
left owed, :mod:`.progress` derives progress from the work tree, and :mod:`.render` is the text
board. The drain lock and log tailing live top-level.
"""
