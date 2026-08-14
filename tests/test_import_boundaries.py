"""Dependency boundaries that keep back-half and review materialization lightweight."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_review_materialization_imports_without_server_or_cv() -> None:
    code = """
import sys
import autogeoref.review.materialize
assert 'http.server' not in sys.modules
assert 'cv2' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_review_server_imports_without_the_review_model() -> None:
    # the transport must stay importable without .app (numpy/pyproj): the
    # console inherits ReviewHandler at module level and may never draw a map
    code = """
import sys
import autogeoref.review.server
assert 'autogeoref.review.app' not in sys.modules
assert 'pyproj' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bake_import_does_not_load_placement_pipeline() -> None:
    code = """
import sys
import autogeoref.bake
assert 'autogeoref.run_inputs' not in sys.modules
assert 'http.server' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_config_import_does_not_load_annotation_invocation() -> None:
    code = """
import sys
import autogeoref.config
assert 'autogeoref.annotate.invocation' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_annotate_package_import_stays_free_of_invocation() -> None:
    # the package __init__ re-exports nothing, so reaching any annotate
    # submodule must not drag the subprocess/model machinery in behind it
    code = """
import sys
import autogeoref.annotate.schema
assert 'autogeoref.annotate.invocation' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_runplan_import_does_not_load_stage_modules() -> None:
    # The stage bodies import their stage modules (and, transitively, cv2)
    # lazily so a plan can be built on an install without the [cv] extra and
    # without paying every stage's import cost up front. (autogeoref.volume is
    # absent from the list: the stage modules load it eagerly.)
    code = """
import sys
import autogeoref.runplan
assert 'cv2' not in sys.modules
for mod in (
    'autogeoref.verify',
    'autogeoref.prep',
    'autogeoref.annotate_volume',
    'autogeoref.street_index',
    'autogeoref.escalate',
    'autogeoref.verified_accept',
):
    assert mod not in sys.modules, mod
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bake_import_does_not_load_the_placement_stages() -> None:
    """The back half consumes committed records; it must not drag the front half in."""
    code = """
import sys
import autogeoref.bake
assert 'autogeoref.run_inputs' not in sys.modules
loaded = [m for m in sys.modules if m.startswith('autogeoref.stages')]
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_escalate_import_does_not_load_cv() -> None:
    # the junction gate imports cv2 lazily inside the gated-pages preflight
    code = """
import sys
import autogeoref.escalate
assert 'cv2' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "module",
    [
        "autogeoref.escalate",
        "autogeoref.review.app",
        "autogeoref.review.apply",
        "autogeoref.verified_accept",
    ],
)
def test_leaf_imports_do_not_load_placement_pipeline(module: str) -> None:
    # Both halves of what used to be one module: the run inputs, and every
    # stage. The review surfaces are the reason this matters — nothing in the
    # layer contracts places them relative to a stage.
    code = f"""
import importlib
import sys
importlib.import_module({module!r})
assert 'autogeoref.run_inputs' not in sys.modules
loaded = [m for m in sys.modules if m.startswith('autogeoref.stages')]
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_verified_accept_import_does_not_load_the_catalog_client() -> None:
    # the era vocabulary is TYPE_CHECKING-only here; the stage must not pull
    # .era -> .loc -> httpx into the placement import graph
    code = """
import sys
import autogeoref.verified_accept
assert 'httpx' not in sys.modules
assert 'autogeoref.loc' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
