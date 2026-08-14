"""Escalation stage: evidence gate, cache separation, full-gate acceptance,
and the cheap-first ladder contract (frontier tiers only after earlier fail).

The flip test replays the measured G1-10 result with the REAL recorded Fable
output for _024 p17 (tests/data/p17_escalated_fable.json) — the page both
strong models flipped to a strict accept with 8 inliers.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from autogeoref.annotate.failures import EmptyResponseError
from autogeoref.annotate.providers import model_cache_key, prior_variant_cache_key
from autogeoref.annotate.schema import Annotation
from autogeoref.bounds import counterpart_bounds
from autogeoref.centerlines import CenterlineIndex
from autogeoref.config.load import load_city_config
from autogeoref.escalate import resolve_tiers, stage_escalate
from autogeoref.names import load_aliases
from autogeoref.paths import VolumePaths
from autogeoref.volume import VolumeConstraints, constraints_from_constants

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
VOL = "sanborn01790_024"

SONNET = "claude-sonnet-5"
OPUS = "claude-opus-4-8"
FABLE = "claude-fable-5"
LADDER = [SONNET, OPUS, FABLE]


def _make_paths(root: Path, fixtures_dir: Path) -> VolumePaths:
    """Fresh volume state: p17 (gated, flips) + p110 (gated, never flips),
    both forced to REJECTED for the replay."""
    paths = VolumePaths(root=root)
    paths.results.mkdir()
    paths.annotations.mkdir()
    paths.sheets.mkdir()
    for page in ("17", "110"):
        shutil.copy2(
            fixtures_dir / VOL / "results" / f"p{page}.json", paths.results / f"p{page}.json"
        )
        shutil.copy2(
            fixtures_dir / VOL / "annotations" / f"p{page}.json",
            paths.annotations / f"p{page}.json",
        )
        shutil.copy2(
            fixtures_dir / VOL / "sheets" / f"p{page}_small.jpg",
            paths.sheets / f"p{page}_small.jpg",
        )
    shutil.copy2(fixtures_dir / VOL / "sheets" / "manifest.json", paths.manifest)
    for page in ("17", "110"):
        rp = paths.results / f"p{page}.json"
        r = json.loads(rp.read_text())
        r["status"] = "REJECTED (no valid RANSAC model)"
        r.pop("layer", None)
        rp.write_text(json.dumps(r))
    return paths


@dataclass(frozen=True)
class _Env:
    """The expensive, immutable half of the test environment.

    Module-scoped on purpose; the mutable half (the volume work tree) is the
    function-scoped ``paths`` fixture, so no test depends on a sibling's
    leftover state."""

    index: CenterlineIndex
    constraints: VolumeConstraints


@pytest.fixture(scope="module")
def env(fixtures_dir: Path, aliases_dir: Path) -> _Env:
    city = load_city_config(ROOT / "configs" / "chicago" / "chicago.toml")
    vol = city.volume(VOL)
    aliases = load_aliases(aliases_dir / f"aliases-{VOL}.json")
    # The goldens replay frozen recordings, and _093's (retired, volunteer)
    # footprint is the frame they were recorded in: bounds pick the
    # CenterlineIndex, which picks the candidate set, which picks the p17
    # inlier count. Production no longer declares bounds_from, so pin the
    # recording frame from the integrity-checked fixture manifest instead.
    manifest = json.loads((fixtures_dir / "viewer-manifest.json").read_text())
    bounds = counterpart_bounds(manifest, "sanborn01790_093")
    index = CenterlineIndex.from_geojson(city.centerlines_path, aliases=aliases, bounds_4326=bounds)
    constraints = constraints_from_constants(vol.scale_m_per_px, vol.rotation_deg)
    return _Env(index=index, constraints=constraints)


@pytest.fixture
def paths(tmp_path: Path, fixtures_dir: Path) -> VolumePaths:
    return _make_paths(tmp_path, fixtures_dir)


def _flip_annotation() -> Annotation:
    return Annotation.from_dict(json.loads((DATA / "p17_escalated_fable.json").read_text()))


def _fable_p17(image_path: Path, model: str) -> Annotation:
    if "p17" in image_path.name:
        return _flip_annotation()
    # p110: recorded fable output did not flip; simulate a failed read
    raise EmptyResponseError("simulated failure for non-flip page")


@pytest.mark.golden
def test_escalation_flips_p17_and_respects_gates(env: _Env, paths: VolumePaths) -> None:
    flipped = stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=FABLE,
        annotate_fn=_fable_p17,
    )
    assert flipped == ["17"]
    r = json.loads((paths.results / "p17.json").read_text())
    assert r["status"] == "OK"
    assert r["escalated_model"] == FABLE
    assert r["n_inliers"] >= 8
    # the v1 annotation cache is untouched; the per-tier escalated cache
    # lives beside it
    assert (paths.annotations / "p17.json").exists()
    assert (paths.annotations / f"p17.escalated.{FABLE}.json").exists()
    # the non-flip page stays honestly rejected (both attempts failed)
    r110 = json.loads((paths.results / "p110.json").read_text())
    assert r110["status"].startswith("REJECTED")


@pytest.mark.golden
def test_escalation_variant_has_a_distinct_cache_key(env: _Env, paths: VolumePaths) -> None:
    def unreadable(_image: Path, _model: str) -> Annotation:
        raise EmptyResponseError("no annotation")

    stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=SONNET,
        variants="high",
        annotate_fn=unreadable,
    )
    high = paths.annotations / f"p17.escalated.{model_cache_key(SONNET, 'high')}.failed.json"
    assert high.exists()

    stage_escalate(paths, env.index, env.constraints, model=SONNET, annotate_fn=unreadable)
    default = paths.annotations / f"p17.escalated.{model_cache_key(SONNET)}.failed.json"
    assert default.exists()


def test_resolve_tiers_requires_one_variant_per_model() -> None:
    with pytest.raises(ValueError, match="one entry per model"):
        resolve_tiers([SONNET, OPUS], variants=["high"])


def test_resolve_tiers_rejects_duplicate_models() -> None:
    with pytest.raises(ValueError, match="cannot repeat a model"):
        resolve_tiers([SONNET, SONNET], variants=[None, "high"])


def test_resolve_tiers_normalizes_ladder_shapes() -> None:
    assert resolve_tiers(SONNET) == ((SONNET, None),)
    assert resolve_tiers([SONNET, OPUS], variants="high") == ((SONNET, "high"), (OPUS, "high"))
    assert resolve_tiers([]) == ()


@pytest.mark.golden
def test_injected_annotator_receives_the_tier_variant(env: _Env, paths: VolumePaths) -> None:
    """A three-arg injected annotator observes each tier's variant; the two-arg
    arity is bridged explicitly (it historically dropped the variant)."""
    calls: list[tuple[str, str | None]] = []

    def capture(image: Path, model: str, variant: str | None) -> Annotation:
        calls.append((model, variant))
        raise EmptyResponseError("capture only")

    stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=[SONNET, OPUS],
        variants=["high", None],
        annotate_fn=capture,
    )
    assert set(calls) == {(SONNET, "high"), (OPUS, None)}


@pytest.mark.golden
def test_escalation_rejects_duplicate_model_variants(env: _Env, paths: VolumePaths) -> None:
    with pytest.raises(ValueError, match="cannot repeat a model"):
        stage_escalate(
            paths,
            env.index,
            env.constraints,
            model=[SONNET, SONNET],
            variants=[None, "high"],
        )


@pytest.mark.golden
def test_escalation_reuses_the_prior_variant_failure_cache(env: _Env, paths: VolumePaths) -> None:
    key = prior_variant_cache_key(SONNET, "high")
    assert key is not None
    for page in ("17", "110"):
        (paths.annotations / f"p{page}.escalated.{key}.failed.json").write_text("{}")
    calls: list[str] = []

    def unreadable(image: Path, _model: str) -> Annotation:
        calls.append(image.name)
        raise EmptyResponseError("must use v1 cache")

    stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=SONNET,
        variants="high",
        annotate_fn=unreadable,
    )
    assert calls == []


@pytest.mark.golden
def test_escalation_is_idempotent_and_reuses_cache(env: _Env, paths: VolumePaths) -> None:
    """A second run over the same tree spends nothing at all."""
    flipped = stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=FABLE,
        annotate_fn=_fable_p17,
    )
    assert flipped == ["17"]
    calls: list[str] = []

    def counting(image_path: Path, model: str) -> Annotation:
        calls.append(image_path.name)
        raise EmptyResponseError("should not be called for cached/flipped pages")

    flipped_again = stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=FABLE,
        annotate_fn=counting,
    )
    # p17 is now OK -> never re-escalated; p110 carries a failure marker from
    # the first run -> skipped without spend. Nothing is called at all.
    assert flipped_again == []
    assert calls == []


@pytest.mark.golden
def test_junction_gate_skips_low_evidence(env: _Env, paths: VolumePaths) -> None:
    calls: list[str] = []

    def counting(image_path: Path, model: str) -> Annotation:
        calls.append(image_path.name)
        raise EmptyResponseError("x")

    stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=FABLE,
        annotate_fn=counting,
        min_junctions=10_000,  # nothing can pass the gate
    )
    assert calls == []


@pytest.mark.golden
def test_budget_limit_halts_the_stage_immediately(env: _Env, paths: VolumePaths) -> None:
    """A budget limit is TERMINAL: no retry, no further pages, no doomed spend."""
    from autogeoref.annotate.failures import BudgetLimitError

    calls: list[str] = []

    def budget_limited(image_path: Path, model: str) -> Annotation:
        calls.append(image_path.name)
        raise BudgetLimitError("usage limit reached")

    flipped = stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=FABLE,
        annotate_fn=budget_limited,
    )
    assert flipped == []
    assert len(calls) == 1  # first attempt only: no retry, no next page
    # a budget stop is not a page failure
    assert not (paths.annotations / f"p110.escalated.{FABLE}.failed.json").exists()


@pytest.mark.golden
def test_exhausted_attempts_write_failure_marker_and_skip_next_run(
    env: _Env, paths: VolumePaths
) -> None:
    calls: list[str] = []

    def always_empty(image_path: Path, model: str) -> Annotation:
        calls.append(image_path.name)
        raise EmptyResponseError("nothing")

    stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=FABLE,
        annotate_fn=always_empty,
    )
    for page in ("17", "110"):
        assert (paths.annotations / f"p{page}.escalated.{FABLE}.failed.json").exists()
    n_first = len(calls)
    assert n_first == 4  # MAX_ATTEMPTS per page, both pages exhausted
    stage_escalate(
        paths,
        env.index,
        env.constraints,
        model=FABLE,
        annotate_fn=always_empty,
    )
    assert len(calls) == n_first  # second run spends nothing on the marked pages


@pytest.mark.golden
def test_ladder_stops_at_first_flipping_tier(env: _Env, paths: VolumePaths) -> None:
    """A page that flips at tier 1 never reaches the frontier tiers; a page
    that fails everywhere walks the full ladder, one failure marker per tier."""
    calls: list[tuple[str, str]] = []

    def tiered(image_path: Path, model: str) -> Annotation:
        calls.append((image_path.name, model))
        if "p17" in image_path.name:
            return _flip_annotation()
        raise EmptyResponseError("tier cannot read this sheet")

    flipped = stage_escalate(paths, env.index, env.constraints, model=LADDER, annotate_fn=tiered)
    assert flipped == ["17"]
    r = json.loads((paths.results / "p17.json").read_text())
    assert r["status"] == "OK"
    assert r["escalated_model"] == SONNET  # cheapest tier won
    assert [m for n, m in calls if "p17" in n] == [SONNET]
    assert (paths.annotations / f"p17.escalated.{SONNET}.json").exists()
    assert not (paths.annotations / f"p17.escalated.{OPUS}.json").exists()
    assert not (paths.annotations / f"p17.escalated.{FABLE}.json").exists()
    # p110 exhausted the whole ladder: MAX_ATTEMPTS per tier, marker per tier
    assert [m for n, m in calls if "p110" in n] == [SONNET] * 2 + [OPUS] * 2 + [FABLE] * 2
    for m in LADDER:
        assert (paths.annotations / f"p110.escalated.{m}.failed.json").exists()
    assert json.loads((paths.results / "p110.json").read_text())["status"].startswith("REJECTED")


@pytest.mark.golden
def test_ladder_reaches_frontier_only_after_lower_tiers_fail(env: _Env, paths: VolumePaths) -> None:
    calls: list[tuple[str, str]] = []

    def frontier_only(image_path: Path, model: str) -> Annotation:
        calls.append((image_path.name, model))
        if "p17" in image_path.name and model == FABLE:
            return _flip_annotation()
        raise EmptyResponseError("lower tier fails")

    flipped = stage_escalate(
        paths, env.index, env.constraints, model=LADDER, annotate_fn=frontier_only
    )
    assert flipped == ["17"]
    r = json.loads((paths.results / "p17.json").read_text())
    assert r["escalated_model"] == FABLE
    # p17 climbed the ladder: two failed attempts each on sonnet and opus
    # (markers written), then fable succeeded first try
    assert [m for n, m in calls if "p17" in n] == [SONNET] * 2 + [OPUS] * 2 + [FABLE]
    assert (paths.annotations / f"p17.escalated.{SONNET}.failed.json").exists()
    assert (paths.annotations / f"p17.escalated.{OPUS}.failed.json").exists()
    # a second run spends nothing: p17 is OK, p110 is marker-covered per tier
    n_before = len(calls)
    flipped2 = stage_escalate(
        paths, env.index, env.constraints, model=LADDER, annotate_fn=frontier_only
    )
    assert flipped2 == []
    assert len(calls) == n_before


@pytest.mark.golden
def test_ladder_advances_past_a_cached_tier_that_does_not_flip(
    env: _Env, paths: VolumePaths
) -> None:
    """A tier whose cached annotation exists but fails the gates costs nothing
    and does not block the next tier."""
    # pre-cache the sonnet tier for p110 with its v1 annotation — the read
    # that already failed to produce a model, so matching it rejects again
    v1 = json.loads((paths.annotations / "p110.json").read_text())
    (paths.annotations / f"p110.escalated.{SONNET}.json").write_text(json.dumps(v1))
    calls: list[tuple[str, str]] = []

    def counting(image_path: Path, model: str) -> Annotation:
        calls.append((image_path.name, model))
        raise EmptyResponseError("x")

    stage_escalate(paths, env.index, env.constraints, model=[SONNET, FABLE], annotate_fn=counting)
    # p110: sonnet consumed the cache (no call), then fable was attempted
    assert [m for n, m in calls if "p110" in n] == [FABLE] * 2
    # p17: no cache anywhere -> both tiers attempted and failed
    assert [m for n, m in calls if "p17" in n] == [SONNET] * 2 + [FABLE] * 2


@pytest.mark.golden
def test_ladder_budget_limit_halts_mid_tier(env: _Env, paths: VolumePaths) -> None:
    """A budget limit at ANY tier stops the whole stage — no frontier calls."""
    from autogeoref.annotate.failures import BudgetLimitError

    calls: list[tuple[str, str]] = []

    def budget_at_opus(image_path: Path, model: str) -> Annotation:
        calls.append((image_path.name, model))
        if model == OPUS:
            raise BudgetLimitError("usage limit reached")
        raise EmptyResponseError("x")

    flipped = stage_escalate(
        paths, env.index, env.constraints, model=LADDER, annotate_fn=budget_at_opus
    )
    assert flipped == []
    # first page (p110 sorts first): sonnet fails twice, opus hits the budget
    # limit on its first attempt -> the stage returns; fable is never called
    # and no further page is attempted
    assert [m for _, m in calls] == [SONNET] * 2 + [OPUS]
    assert all("p110" in n for n, _ in calls)
    # a budget stop is not a page failure for the interrupted tier
    assert not (paths.annotations / f"p110.escalated.{OPUS}.failed.json").exists()


def test_config_escalation_ladder_resolution() -> None:
    city = load_city_config(ROOT / "configs" / "chicago" / "chicago.toml")
    # the city-wide ladder is the default for EVERY volume (owner decision
    # the census-wave hard volumes inherit it rather than
    # carrying their own copy
    ladder = ("codex:gpt-5.6-terra", "codex:gpt-5.6-sol")
    assert city.escalation_models == ladder
    assert city.escalation_variants == ("high", "high")
    assert city.volume("sanborn01790_001").escalation_ladder() == ladder
    assert city.volume("sanborn01790_024").escalation_ladder() == ladder
    # fable is never a default tier anywhere (a budget decision); append it to a
    # volume's ladder deliberately to spend that budget
    assert FABLE not in city.volume("sanborn01790_018").escalation_ladder()


def test_volume_escalation_model_outranks_the_inherited_city_ladder(tmp_path: Path) -> None:
    """A volume's OWN single-tier key must not be silently overridden.

    Regression: when the city gained a default ladder (escalation is now
    default-ON), `escalation_models or city_ladder` made a volume that
    deliberately named one cheaper tier inherit the city's two-tier ladder
    instead — spending on a tier it never asked for.
    """
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        f'escalation_models = ["{SONNET}", "{OPUS}"]\n'
        f'[volumes.volA]\nescalation_model = "{SONNET}"\n'
    )
    assert load_city_config(cfg).volume("volA").escalation_ladder() == (SONNET,)


def test_volume_can_cancel_the_city_ladder(tmp_path: Path) -> None:
    """`escalation_models = []` on a volume CANCELS the city default.

    Presence-based, not truthiness-based: an empty list must not fall through to
    the city ladder, or a volume could never turn the (now default-ON) stage off.
    """
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        f'escalation_models = ["{SONNET}", "{OPUS}"]\n'
        "[volumes.volA]\n"
        "[volumes.volB]\nescalation_models = []\n"
    )
    city = load_city_config(cfg)
    assert city.volume("volA").escalation_ladder() == (SONNET, OPUS)  # inherits
    assert city.volume("volB").escalation_ladder() == ()  # cancelled -> stage off


def test_volume_cancels_a_city_that_uses_the_singular_key(tmp_path: Path) -> None:
    """`escalation_models = []` cancels a city `escalation_model` too.

    Regression: the singular key was inherited with `or`, so an empty plural
    ladder fell through to it and the volume escalated anyway — spending budget
    the config had explicitly turned off.
    """
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        f'escalation_model = "{SONNET}"\n'
        "[volumes.volA]\n"
        "[volumes.volB]\nescalation_models = []\n"
    )
    city = load_city_config(cfg)
    assert city.volume("volA").escalation_ladder() == (SONNET,)  # inherits
    assert city.volume("volB").escalation_ladder() == ()  # cancelled -> stage off
    assert city.volume("volB").escalation_model is None


def test_volume_cancels_a_city_ladder_with_an_empty_singular_key(tmp_path: Path) -> None:
    """`escalation_model = ""` cancels too — the volume's keys are ONE unit.

    `escalation_models = []` is the plural spelling of the same "off", so a config
    author who learned one will write the other; inheriting the city's two-tier
    ladder for it would spend budget on a volume that said no twice.
    """
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        f'escalation_models = ["{SONNET}", "{OPUS}"]\n'
        '[volumes.volA]\nescalation_model = ""\n'
    )
    assert load_city_config(cfg).volume("volA").escalation_ladder() == ()


def test_escalation_tiers_pair_variants_across_inheritance(tmp_path: Path) -> None:
    """Tier model/variant pairing and tier COUNT survive every inheritance shape.

    The count is what the consoles' spend estimate multiplies by, so an
    inherited, singular, cancelled, or unknown volume must each resolve to
    exactly the tiers its config asked for.
    """
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        'escalation_models = ["codex:gpt-5.6-terra", "codex:gpt-5.6-sol"]\n'
        'escalation_variants = ["high", ""]\n'
        "[volumes.volA]\n"
        "[volumes.volB]\n"
        'escalation_model = "codex:gpt-5.6-sol"\nescalation_variant = "medium"\n'
        "[volumes.volC]\nescalation_models = []\n"
    )
    city = load_city_config(cfg)
    inherited = (("codex:gpt-5.6-terra", "high"), ("codex:gpt-5.6-sol", None))
    assert city.volume("volA").escalation_tiers() == inherited  # "" = provider default
    assert city.volume("volB").escalation_tiers() == (("codex:gpt-5.6-sol", "medium"),)
    assert city.volume("volC").escalation_tiers() == ()
    # a volume the config never names inherits the city ladder whole
    assert city.volume("volZ").escalation_tiers() == inherited
    assert len(city.volume("volA").escalation_ladder()) == 2
    assert len(city.volume("volC").escalation_ladder()) == 0


def test_cli_wires_escalate_flags() -> None:
    from autogeoref.cli.parser import build_parser

    args = build_parser().parse_args(["run", "volX", "--city", "c.toml", "--escalate-model", OPUS])
    assert args.escalate_model == [OPUS]
    ladder = build_parser().parse_args(
        ["run", "volX", "--city", "c.toml", "--escalate-model", SONNET, "--escalate-model", FABLE]
    )
    assert ladder.escalate_model == [SONNET, FABLE]
    args2 = build_parser().parse_args(["run", "volX", "--city", "c.toml", "--escalate"])
    assert args2.escalate and not args2.no_escalate
    # the default run neither asserts nor suppresses: the configured ladder decides
    args3 = build_parser().parse_args(["run", "volX", "--city", "c.toml"])
    assert not args3.escalate and not args3.no_escalate and args3.escalate_model is None
    off = build_parser().parse_args(["run", "volX", "--city", "c.toml", "--no-escalate"])
    assert off.no_escalate


def _dry_run_escalate_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> str:
    """The escalate stage's status line from a --dry-run of a minimal city."""
    from autogeoref.cli.entry import main

    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        f'centerlines = "cl.geojson"\nescalation_models = ["{SONNET}"]\n'
        "[volumes.volX]\nbounds_bbox = [-87.7, 41.9, -87.6, 42.0]\n"
    )
    (tmp_path / "cl.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    # the run halts at the first stage with missing inputs, so give match a
    # manifest — a dry run executes nothing, it only reports what WOULD run
    work = tmp_path / "work"
    (work / "volX" / "sheets").mkdir(parents=True)
    (work / "volX" / "sheets" / "manifest.json").write_text("{}")
    main(["run", "volX", "--city", str(cfg), "--work", str(work), "--dry-run", *extra])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("escalate:")]
    return lines[0] if lines else ""


def test_escalation_runs_by_default_when_a_ladder_is_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A configured ladder is the opt-in; no flag needed.

    A plain run must schedule the stage ("skipped" = a dry run's would-run).
    `disabled` was the old opt-in behaviour, and keeping it would mean the
    config key silently never does anything.
    """
    assert _dry_run_escalate_status(tmp_path, capsys, []) == "escalate: skipped"


def test_no_escalate_suppresses_the_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--no-escalate is the opt-out: a configured ladder spends nothing this run."""
    assert _dry_run_escalate_status(tmp_path, capsys, ["--no-escalate"]) == "escalate: disabled"


def test_no_escalate_conflicting_with_escalate_fails_fast(tmp_path: Path) -> None:
    """Contradictory spend flags must never be resolved by silent precedence."""
    from autogeoref.cli.entry import main

    cfg = tmp_path / "city.toml"
    cfg.write_text(f'[city]\nname = "X"\naliases_dir = "a"\nescalation_models = ["{SONNET}"]\n')
    with pytest.raises(SystemExit, match="contradicts"):
        main(
            [
                "run",
                "volX",
                "--city",
                str(cfg),
                "--work",
                str(tmp_path / "work"),
                "--escalate",
                "--no-escalate",
                "--dry-run",
            ]
        )


def test_escalate_without_a_ladder_fails_fast(tmp_path: Path) -> None:
    """--escalate with no configured ladder must error BEFORE any stage runs
    (even under --dry-run) — never silently default to a frontier model."""
    import pytest

    from autogeoref.cli.entry import main

    cfg = tmp_path / "city.toml"
    cfg.write_text('[city]\nname = "X"\naliases_dir = "aliases"\n')
    with pytest.raises(SystemExit, match="no escalation ladder configured"):
        main(
            [
                "run",
                "volX",
                "--city",
                str(cfg),
                "--work",
                str(tmp_path / "work"),
                "--escalate",
                "--dry-run",
            ]
        )


def test_adapt_annotator_arity_bridging() -> None:
    """The arity choice is made once at construction: positional three-arg
    passes through, keyword-only variant is bridged as a keyword, two-arg
    drops the variant explicitly, and an unsignaturable callable still runs."""
    from autogeoref.escalate import _adapt_annotator

    seen: list[tuple[str, ...]] = []

    def kw_only(img: Path, model: str, *, variant: str | None = None) -> dict[str, object]:
        seen.append(("kw", model, str(variant)))
        return {}

    _adapt_annotator(kw_only)(Path("i"), "m", "high")
    assert seen == [("kw", "m", "high")]

    def two_arg(img: Path, model: str) -> dict[str, object]:
        seen.append(("two", model))
        return {}

    _adapt_annotator(two_arg)(Path("i"), "m", "high")
    assert seen[-1] == ("two", "m")
