"""The operator console: what can I run, what is running, what needs a human.

Joins the state index and the queue board into a work list on both tracks,
paste-ready for ``autogeoref queue --add``. It derives and never re-derives:
counts come from :func:`status.build_status` and :func:`queue.progress.board`.
Its buttons act, but every act is somebody else's function — this package owns
no state and no verdict. It binds loopback and is unauthenticated by design.

Import the owning submodule directly (``from autogeoref.console.cli import
main``); this package deliberately re-exports nothing, so importing one part
does not pay for the others.

`docs/OPERATIONS.md` has the derivation rules, the security model, and the
module map.
"""
