"""Fast contracts for run policy resolution, independent of CLI fixtures."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from autogeoref.cli import run as cli_run
from autogeoref.config.model import CityConfig, VolumeConfig
from autogeoref.runpolicy import RunPolicy


@dataclass
class Args:
    city: Path = Path("city.toml")
    volume: str = "vol"
    warp: bool = False
    warp_only: bool = False
    escalate: bool = False
    no_escalate: bool = False
    escalate_model: list[str] | None = None
    no_verify: bool = False
    verify_junctions: bool = False
    verified_accept: bool = False


def _city(*, renumbered: bool = False) -> CityConfig:
    return CityConfig(
        name="Test City",
        centerlines_path=Path("centerlines.geojson"),
        aliases_dir=Path("aliases"),
        renumbering_table_path=Path("renumbering.json") if renumbered else None,
    )


def _resolve(args: Args | None = None, **volume: object) -> RunPolicy:
    vol = VolumeConfig(identifier="vol", **volume)
    return RunPolicy.resolve(args or Args(), _city(), vol)


def _evidence_state(policy: RunPolicy) -> tuple[bool, bool, frozenset[str]]:
    return policy.run_junction, policy.run_verified, policy.allowed_channels


def test_escalation_defaults_overrides_and_opt_out() -> None:
    configured = _resolve(
        escalation_models=("first", "second"), escalation_variants=("high", "medium")
    )
    assert configured.escalation_models == ("first", "second")
    assert configured.escalation_variants == ("high", "medium")
    assert configured.run_escalation
    overridden = _resolve(Args(escalate_model=["override"]), escalation_models=("configured",))
    assert overridden.escalation_models == ("override",)
    assert overridden.escalation_variants == (None,)
    assert overridden.run_escalation
    assert not _resolve(Args(no_escalate=True), escalation_models=("configured",)).run_escalation


def test_escalation_validation_preserves_empty_ladder_precedence() -> None:
    with pytest.raises(SystemExit, match="no escalation ladder configured"):
        _resolve(Args(escalate=True))
    with pytest.raises(SystemExit, match="no escalation ladder configured"):
        _resolve(Args(escalate=True, no_escalate=True))
    with pytest.raises(SystemExit, match="--no-escalate contradicts"):
        _resolve(Args(escalate=True, no_escalate=True), escalation_models=("configured",))
    with pytest.raises(SystemExit, match="--no-escalate contradicts"):
        _resolve(Args(no_escalate=True, escalate_model=["override"]))


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        ((), (False, False, frozenset())),
        (("junction",), (True, True, frozenset({"junction"}))),
        # "addresses" enables NO stage of its own — it is a permission to vote off
        # caches already on disk, since the consensus producer was cut
        (("addresses",), (False, True, frozenset({"addresses"}))),
        (("junction", "addresses"), (True, True, frozenset({"junction", "addresses"}))),
    ],
)
def test_declared_channels_control_stage_enablement(
    channels: tuple[str, ...], expected: tuple[bool, bool, frozenset[str]]
) -> None:
    policy = _resolve(evidence_channels=channels)
    assert _evidence_state(policy) == expected


def test_explicit_channel_flags_preserve_independent_enablement() -> None:
    junction = _resolve(Args(verify_junctions=True))
    assert _evidence_state(junction)[:2] == (True, False)
    verified = _resolve(Args(verified_accept=True))
    assert _evidence_state(verified)[:2] == (True, True)
    assert verified.allowed_channels == frozenset({"junction"})


def test_no_flag_can_turn_the_addresses_channel_on() -> None:
    """The addresses channel is CONFIG-ONLY, and that is load-bearing.

    It is the only channel that may REFUTE, and on a renumbering city it needs a
    declared era to read numerals correctly. A flag that switched it on would let an
    invocation reach past the era gate the config exists to enforce. Its old flag
    (`--consensus-annotate`) went with the producer it named.
    """
    for args in (Args(verified_accept=True), Args(verify_junctions=True)):
        assert "addresses" not in _resolve(args).allowed_channels


@pytest.mark.parametrize("flag", ["verify_junctions", "verified_accept"])
def test_no_verify_rejects_explicit_channel_flags(flag: str) -> None:
    args = Args(no_verify=True)
    setattr(args, flag, True)
    with pytest.raises(SystemExit, match="--no-verify contradicts"):
        _resolve(args)


def test_no_verify_disables_declared_channels() -> None:
    policy = RunPolicy.resolve(
        Args(no_verify=True),
        _city(renumbered=True),
        VolumeConfig(identifier="vol", evidence_channels=("junction", "addresses")),
    )
    assert _evidence_state(policy) == (False, False, frozenset())


@pytest.mark.parametrize("modern", [True, False])
def test_declared_address_era_allows_addresses_channel(modern: bool) -> None:
    args = Args()
    city = _city(renumbered=True)
    vol = VolumeConfig(identifier="vol", addresses_modern=modern, evidence_channels=("addresses",))
    policy = RunPolicy.resolve(args, city, vol)
    assert "addresses" in policy.allowed_channels


def test_undeclared_address_era_refuses_only_when_addresses_run() -> None:
    city = _city(renumbered=True)
    with pytest.raises(SystemExit, match="declares no address era"):
        RunPolicy.resolve(
            Args(), city, VolumeConfig(identifier="vol", evidence_channels=("addresses",))
        )
    policy = RunPolicy.resolve(
        Args(), city, VolumeConfig(identifier="vol", evidence_channels=("junction",))
    )
    assert policy.run_junction and "addresses" not in policy.allowed_channels


def test_warp_modes_are_resolved_without_city_configuration() -> None:
    assert RunPolicy.is_warp_only(Args(warp_only=True))
    assert not RunPolicy.is_warp_only(Args(warp=True))
    with pytest.raises(SystemExit, match="--warp-only contradicts --warp"):
        RunPolicy.is_warp_only(Args(warp_only=True, warp=True))


def test_cli_warp_only_short_circuits_city_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(warp_only=True, warp=False, work=tmp_path, volume="vol", dry_run=False)
    monkeypatch.setattr(
        cli_run, "load_city_config", lambda _path: pytest.fail("loaded city config")
    )
    monkeypatch.setattr(cli_run, "_run_back_half", lambda _args, _paths: 0)
    assert cli_run._cmd_run(args) == 0
    # the serve-only mode still owned its volume while it ran
    assert (tmp_path / "vol" / "volume.lock").exists()


def test_cli_warp_only_conflict_precedes_city_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(warp_only=True, warp=True, work=Path("work"), volume="vol")
    monkeypatch.setattr(
        cli_run, "load_city_config", lambda _path: pytest.fail("loaded city config")
    )
    with pytest.raises(SystemExit) as exc:
        cli_run._cmd_run(args)
    assert str(exc.value) == "--warp-only contradicts --warp"


@pytest.mark.parametrize(
    ("args", "volume", "expected"),
    [
        (
            Args(escalate=True),
            {},
            "--escalate: no escalation ladder configured — set escalation_models "
            "in the city TOML or pass --escalate-model",
        ),
        (
            Args(escalate=True, no_escalate=True),
            {},
            "--escalate: no escalation ladder configured — set escalation_models "
            "in the city TOML or pass --escalate-model",
        ),
        (
            Args(escalate=True, no_escalate=True, no_verify=True, verified_accept=True),
            {"escalation_models": ("configured",)},
            "--no-escalate contradicts --escalate / --escalate-model",
        ),
        (
            Args(no_verify=True, verified_accept=True),
            {},
            "--no-verify contradicts --verify-junctions / --verified-accept",
        ),
    ],
)
def test_policy_errors_have_exact_stable_text(
    args: Args, volume: dict[str, object], expected: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve(args, **volume)
    assert str(exc.value) == expected


def test_undeclared_era_error_has_exact_stable_text() -> None:
    city = _city(renumbered=True)
    vol = VolumeConfig(identifier="vol", evidence_channels=("addresses",))
    with pytest.raises(SystemExit) as exc:
        RunPolicy.resolve(Args(), city, vol)
    assert str(exc.value) == (
        "vol: the addresses channel is ON and this volume declares no address era, but Test City "
        "RENUMBERED its houses (the city config ships a renumbering table). Undeclared means "
        "MODERN, and on a volume printed before the renumbering that reads its numerals against "
        "today's grid and REFUTES correct sheets — the addresses channel is the only one that can "
        "veto.\n"
        "  Set `addresses_modern` in [volumes.vol] of city.toml:\n"
        "    true  = the printed numbers ARE today's numbers (post-renumbering)\n"
        "    false = the volume predates it; numbers convert through the table\n"
        "            (check WHICH book: Chicago's Loop renumbered separately, in 1911)\n"
        "  Or run without the channel: --no-verify, or evidence_channels = [] on the volume."
    )
