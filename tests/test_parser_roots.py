"""Shared state-root parser contracts for status, queue, and dashboard."""

from pathlib import Path

import pytest

from autogeoref.cli.parser import build_parser


def test_state_root_defaults_match_across_state_views() -> None:
    parser = build_parser()
    commands = (
        ["status"],
        ["queue", "--candidates"],
        ["queue", "--serve"],
        ["dashboard", "--city", "configs/chicago/chicago.toml"],
    )

    for command in commands:
        args = parser.parse_args(command)
        assert args.work == Path("work")
        assert args.fixtures == Path("fixtures")
        assert args.ground_truth is None
        assert args.tiles == Path("deploy/tiles")


def test_catalog_root_is_shared_only_by_queue_and_dashboard(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    queue = parser.parse_args(["queue", "--candidates"])
    dashboard = parser.parse_args(["dashboard", "--city", "configs/chicago/chicago.toml"])
    assert queue.loc_catalog is None
    assert dashboard.loc_catalog is None

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["status", "--loc-catalog", "catalog.json"])
    assert error.value.code == 2
    assert "unrecognized arguments: --loc-catalog catalog.json" in capsys.readouterr().err


def test_city_requiredness_stays_local_to_dashboard(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    assert parser.parse_args(["queue", "--candidates"]).city is None
    assert parser.parse_args(["queue", "--serve"]).city is None
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["dashboard"])
    assert error.value.code == 2
    assert "the following arguments are required: --city" in capsys.readouterr().err


def test_dashboard_usage_leads_with_required_city(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["dashboard", "--help"])
    assert excinfo.value.code == 0
    usage = capsys.readouterr().out.partition("\n\n")[0]
    assert usage.index("--city CITY") < usage.index("[--work WORK]")


def test_dashboard_catalog_help_describes_era_breakdown(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["dashboard", "--help"])

    help_text = capsys.readouterr().out
    assert "era breakdown" in help_text
    assert "reports its era as unknown" in help_text
