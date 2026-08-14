"""The two machine-readable claims a reuser acts on: who to cite, and the licence.

Neither is exercised by running the pipeline, and both are duplicated by
construction — the citation repeats what pyproject declares, and the licence is
repeated on every exported annotation. A copy that drifts is worse than a
missing one, because it is still believed.

1. CITATION.cff parses, carries the fields the citation button renders, and
   names a real person with a contact rather than a placeholder.
2. It agrees with pyproject.toml on version, licence and author.
3. Every exported annotation and annotation page carries the licence URI.
4. The exports README on disk is the one publish generates, not a hand edit.

These read the working tree, not the index, so staging is still the author's job.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from autogeoref.allmaps import RIGHTS
from autogeoref.exports import README

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CITATION = REPOSITORY_ROOT / "CITATION.cff"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
EXPORTS = REPOSITORY_ROOT / "exports"

#: Values that satisfy a "the field is filled in" check while telling a reader
#: nothing. The runbook that added this file called out exactly this failure:
#: two matching placeholders pass an agreement check and leave both useless.
PLACEHOLDERS = frozenset({"", "maintainer", "maintainers", "autogeoref maintainers", "unknown"})


@pytest.fixture(scope="module")
def citation() -> dict[str, object]:
    return dict(yaml.safe_load(CITATION.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def project() -> dict[str, object]:
    return dict(tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"])


def test_citation_carries_the_fields_the_button_renders(citation: dict[str, object]) -> None:
    # cff-version/message/authors are the schema's own required set; the rest is
    # what turns a valid file into a useful citation.
    for key in ("cff-version", "message", "authors", "title", "repository-code", "license"):
        assert citation.get(key), f"CITATION.cff has no usable {key!r}"
    assert str(citation["cff-version"]).startswith("1.2"), "written against CFF 1.2.0"


def test_citation_names_a_person_who_can_be_reached(citation: dict[str, object]) -> None:
    authors = citation["authors"]
    assert isinstance(authors, list) and authors, "CITATION.cff names no author"
    for author in authors:
        assert isinstance(author, dict)
        family, given = author.get("family-names"), author.get("given-names")
        assert family and given, f"author {author!r} has no personal name"
        assert str(family).strip().lower() not in PLACEHOLDERS, f"{family!r} names no one"
        # A citation with no route back to the author is the dead end this
        # file was written to close.
        assert author.get("email") or author.get("orcid"), f"author {family!r} has no contact"


def test_citation_agrees_with_the_package_metadata(
    citation: dict[str, object], project: dict[str, object]
) -> None:
    assert citation["version"] == project["version"]
    license_field = project["license"]
    declared = license_field["text"] if isinstance(license_field, dict) else license_field
    assert citation["license"] == declared

    authors = citation["authors"]
    assert isinstance(authors, list)
    cff_people = {(f"{a['given-names']} {a['family-names']}", a.get("email")) for a in authors}
    packaged = project["authors"]
    assert isinstance(packaged, list)
    assert cff_people == {(a.get("name"), a.get("email")) for a in packaged}


def test_the_cited_paper_is_a_reference_not_the_preferred_citation(
    citation: dict[str, object],
) -> None:
    # `preferred-citation` asks readers to cite that work INSTEAD of this
    # software, so a third party's paper there misdirects credit for it.
    assert "preferred-citation" not in citation


def _exported_volumes() -> list[Path]:
    return sorted(p for p in EXPORTS.glob("*/allmaps.json"))


def test_the_export_tree_is_not_empty() -> None:
    # The licence assertions below pass vacuously over an empty tree, and the
    # tree is tracked, so an empty one is a deletion rather than a fresh clone.
    assert _exported_volumes(), "no tracked exports/<volume>/allmaps.json"


@pytest.mark.parametrize("annotations", _exported_volumes(), ids=lambda p: p.parent.name)
def test_every_exported_annotation_carries_the_licence(annotations: Path) -> None:
    page = json.loads(annotations.read_text(encoding="utf-8"))
    assert page.get("rights") == RIGHTS, f"{annotations}: page carries no licence"
    items = page["items"]
    assert items, f"{annotations}: an exported page with no annotations"
    # Per annotation as well as per page: Allmaps and friends split a page into
    # its items, and an item that travels alone must carry its own terms.
    unlicensed = [i for i, item in enumerate(items) if item.get("rights") != RIGHTS]
    assert not unlicensed, f"{annotations}: {len(unlicensed)} annotations carry no licence"


def test_the_exports_readme_is_the_generated_one() -> None:
    readme = EXPORTS / "README.md"
    assert readme.is_file(), "exports/README.md is missing; publish generates it"
    assert readme.read_text(encoding="utf-8") == README, (
        "exports/README.md differs from what publish writes — it was hand-edited, "
        "or the generated text changed and the tree was not re-published"
    )
