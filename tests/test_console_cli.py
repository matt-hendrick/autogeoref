"""Wiring the console up: the commands it echoes, and the catalog it defaults to.

The page must not learn a city's name or where its config lives, so the server
composes the `--candidates` and `era` commands from the roots it was started
with — and both have to reparse into the same arguments. The year on a card is
display context, so a `loc_catalog` the city TOML names is read without the
flag, and a stale path there warns instead of breaking the console.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogeoref import cli_context
from console_support import _city_toml, _console_args


def test_the_candidates_command_echoed_to_the_page_is_the_one_that_was_run(
    tmp_path: Path,
) -> None:
    """The text view's per-volume TOML stanzas live behind `queue --candidates`. The
    SERVER composes that command from the roots it is actually reading, because the
    HTML must not learn a city's name or where its config lives."""
    import shlex

    from autogeoref.cli.parser import build_parser
    from autogeoref.console.cli import _candidates_command

    args = build_parser().parse_args(
        [
            "queue",
            "--serve",
            "--city",
            "configs/chicago/chicago.toml",
            "--work",
            str(tmp_path),
            "--fixtures",
            "frozen",
            "--ground-truth",
            "pins",
            "--tiles",
            "served",
            "--viewer-manifest",
            "custom-manifest.json",
        ]
    )
    cmd = _candidates_command(args)
    reparsed = build_parser().parse_args(shlex.split(cmd)[1:])
    assert reparsed.candidates is True
    assert reparsed.city == Path("configs/chicago/chicago.toml")
    assert reparsed.work == tmp_path
    assert reparsed.fixtures == Path("frozen")
    assert reparsed.ground_truth == Path("pins")
    assert reparsed.tiles == Path("served")
    assert reparsed.viewer_manifest == Path("custom-manifest.json")


def test_the_era_command_echoed_to_the_page_reparses_and_carries_the_citys_paths(
    tmp_path: Path,
) -> None:
    """The blocked card's unblock path is `autogeoref era`, not hand-pasted stanzas.
    The page appends only volume ids (already in its payload); the city TOML and
    catalog paths come from the server — same rule as the candidates command."""
    import shlex

    from autogeoref.cli.parser import build_parser
    from autogeoref.console.cli import _era_command

    args = build_parser().parse_args(
        [
            "queue",
            "--serve",
            "--city",
            "configs/chicago/chicago.toml",
            "--work",
            str(tmp_path),
            "--loc-catalog",
            "custom-catalog.json",
        ]
    )
    cmd = _era_command(args)
    assert cmd is not None
    reparsed = build_parser().parse_args([*shlex.split(cmd)[1:], "vol_a", "vol_b"])
    assert reparsed.city == Path("configs/chicago/chicago.toml")
    assert reparsed.loc_catalog == Path("custom-catalog.json")
    assert reparsed.volumes == ["vol_a", "vol_b"]
    # --yes stays off the page's command: the confirm is the safety mechanism, and a
    # copy button that skipped it would hand every operator the rubber stamp.
    assert "--yes" not in cmd

    # No city, no era check, no blocked card — and no command to compose.
    bare = build_parser().parse_args(["queue", "--serve", "--work", str(tmp_path)])
    assert _era_command(bare) is None


def test_console_context_defaults_the_catalog_from_the_city_toml(tmp_path: Path) -> None:
    """The year on a queue card is catalog context; the city TOML already names its
    catalog (`loc_catalog`), so the console reads it without the flag — same
    fallback `autogeoref era` has, minus era's refusal (years are display here)."""
    cfg = _city_toml(tmp_path, loc_catalog="cat.json")
    (tmp_path / "cat.json").write_text(
        json.dumps(
            [
                {
                    "id": "http://www.loc.gov/item/vol_a/",
                    "description": ["Vol. 1, 1896. 100 sheet(s). Bound."],
                    "date": "1896",
                }
            ]
        )
    )

    _, _, catalog = cli_context.console_context(_console_args(tmp_path, cfg))

    assert catalog["vol_a"]["year"] == 1896

    # the flag still wins over the config key
    (tmp_path / "other.json").write_text(
        json.dumps([{"id": "http://www.loc.gov/item/vol_a/", "date": "1912"}])
    )
    args = _console_args(tmp_path, cfg, "--loc-catalog", str(tmp_path / "other.json"))
    _, _, catalog = cli_context.console_context(args)
    assert catalog["vol_a"]["year"] == 1912


def test_queue_publication_defaults_the_catalog_from_the_city_toml(tmp_path: Path) -> None:
    cfg = _city_toml(tmp_path, loc_catalog="cat.json")
    args = _console_args(tmp_path, cfg)
    publication = cli_context.publication_config(args, manifest=args.viewer_manifest)

    assert publication.loc_catalog == tmp_path / "cat.json"


def test_a_stale_config_catalog_path_never_breaks_the_console(tmp_path: Path) -> None:
    """The config fallback is display context: a `loc_catalog` naming a missing or
    unparseable file warns and omits the years. The explicit flag keeps failing
    loudly — the operator named that file, and silence would hide their typo."""
    cfg = _city_toml(tmp_path, loc_catalog="cat.json")

    _, _, catalog = cli_context.console_context(_console_args(tmp_path, cfg))

    assert catalog == {}

    # a catalog that exists but does not parse is the same story: warn, omit
    (tmp_path / "cat.json").write_text("{ not json")
    _, _, catalog = cli_context.console_context(_console_args(tmp_path, cfg))
    assert catalog == {}

    args = _console_args(tmp_path, cfg, "--loc-catalog", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        cli_context.console_context(args)
