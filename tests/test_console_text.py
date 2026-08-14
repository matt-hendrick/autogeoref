"""The printed lines: paste-ready commands, and guidance that comes from the city TOML.

A command the console prints has to be one the parser accepts and the queue
runs, so these round-trip it through both. The era stanza's wording is the
city's own — one city's renumbering narrative must never render for another's
backlog — and the TOML key it prints is quoted, because a real volume id has a
dot in it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from autogeoref.config.model import CityConfig, VolumeConfig
from autogeoref.console import backlog as console_backlog
from autogeoref.console import text as console_text
from autogeoref.queue import store as qstore
from console_support import _BBOX, _city, _images, _results, _status, _tree


def test_the_printed_serve_command_is_complete_and_actually_serves(tmp_path: Path) -> None:
    """It used to stop short of `--reviewed` so a human had to type the sign-off.

    There is no sign-off to type any more: the gate it guarded had already
    been removed, so the console printed a command the queue then refused. The line is
    complete now, and running it works.
    """
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    _results(roots["work"], "vol_a", accepted=3, flagged=1)

    cands = console_backlog.candidates(_status(roots), work=roots["work"])
    cmd = console_text.add_command(cands, "serve")
    assert cmd == "autogeoref queue --track serve --add vol_a"

    # the command as PRINTED is the command that runs — no missing flag, no refusal
    from autogeoref.cli.parser import build_parser

    args = build_parser().parse_args(cmd.split()[1:])
    entry = qstore.add(roots["work"], "vol_a", args.track, then_serve=not args.review)
    assert entry.track == "serve"
    assert entry.then_serve is False  # a serve entry never promotes


def test_the_era_guidance_is_the_city_config_speaking(tmp_path: Path) -> None:
    """The era stanza's parenthetical comes from the city TOML: its own
    `renumbering_note` wording, else a generic line from `renumbering_year`,
    else nothing — one city's 1909/1911 narrative must never render for
    another city's backlog."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    city = _city()  # renumbering table, no note, no year -> blocked candidates
    cands = console_backlog.candidates(_status(roots), work=roots["work"], city=city)
    assert any(not c.runnable for c in cands)

    noted = replace(city, renumbering_note="Springfield renumbered in 1922")
    text = console_text.render_candidates(cands, noted)
    assert "city TOML (Springfield renumbered in 1922):" in text

    yearly = replace(city, renumbering_year=1922)
    text = console_text.render_candidates(cands, yearly)
    assert "city TOML (this city renumbered in 1922):" in text

    for narrative_free in (
        console_text.render_candidates(cands, city),  # table but neither note nor year
        console_text.render_candidates(cands),  # no city at all
    ):
        assert "addresses_modern" in narrative_free, "the stanza itself survives"
        assert "city TOML:" in narrative_free
        assert "1909" not in narrative_free and "Loop" not in narrative_free


def test_chicago_renders_its_1909_1911_narrative_verbatim_from_config(tmp_path: Path) -> None:
    """The Chicago wording is unchanged — it just lives in configs/chicago/chicago.toml
    (`renumbering_note`) instead of this module's source."""
    from autogeoref.config.load import load_city_config

    chicago = load_city_config(
        Path(__file__).resolve().parent.parent / "configs" / "chicago" / "chicago.toml"
    )
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    cands = console_backlog.candidates(_status(roots), work=roots["work"], city=_city())
    text = console_text.render_candidates(cands, chicago)
    assert (
        "Pick ONE line per volume in the city TOML (Chicago renumbered in 1909; "
        "the Loop separately in 1911, and those volumes need the Loop table):" in text
    )


def test_a_city_that_never_renumbered_shows_no_renumbering_guidance(tmp_path: Path) -> None:
    """No renumbering table means no era refusal and therefore no era stanza —
    a city that never renumbered must not be lectured about one that did."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    city = _city(renumbering=False)
    cands = console_backlog.candidates(_status(roots), work=roots["work"], city=city)
    assert all(c.runnable for c in cands)
    text = console_text.render_candidates(cands, city)
    assert "REFUSE to run until you declare the address era" not in text
    assert "renumbered" not in text


def test_the_printed_place_command_is_the_one_that_actually_enqueues(tmp_path: Path) -> None:
    """The paste-ready line is only worth printing if it WORKS. Round-trip it through
    the real argument parser and the real `queue.store.add`."""
    from autogeoref.cli.parser import build_parser

    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)

    bounded = VolumeConfig(identifier="vol_a", bounds_bbox=_BBOX)
    city = CityConfig(**{**_city(renumbering=False).__dict__, "volumes": {"vol_a": bounded}})
    cands = console_backlog.candidates(_status(roots), work=roots["work"], city=city)
    line = next(
        ln.strip()
        for ln in console_text.render_candidates(cands).splitlines()
        if ln.strip().startswith("autogeoref queue --track place")
    )
    args = build_parser().parse_args(line.split()[1:])
    assert args.add == ["vol_a"]
    entry = qstore.add(roots["work"], "vol_a", args.track, then_serve=not args.review)
    assert entry.track == "place"


def test_the_printed_toml_stanza_is_valid_for_a_dotted_volume_id(tmp_path: Path) -> None:
    """`sanborn01790_006.5` is a REAL volume id. A bare `[volumes.sanborn01790_006.5]`
    is a NESTED table in TOML — it would declare an era for a volume that does not
    exist while the operator believed they fixed the one that does, and the run would
    go on refusing. The key is always quoted."""
    import tomllib

    roots = _tree(tmp_path)
    _images(roots["work"], "vol_x.5", 4)

    text = console_text.render_candidates(
        console_backlog.candidates(_status(roots), work=roots["work"], city=_city())
    )
    # take the stanza and uncomment the `true` line, as an operator would
    stanza = "\n".join(
        ln.strip().removeprefix("# ")
        for ln in text.splitlines()
        if ln.strip().startswith(("[volumes.", "# addresses_modern = true"))
    )
    assert tomllib.loads(stanza) == {"volumes": {"vol_x.5": {"addresses_modern": True}}}
