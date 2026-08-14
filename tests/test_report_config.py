"""Reporter aggregation against recorded fixtures + city-config loading."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from autogeoref.config.load import load_city_config
from autogeoref.config.model import ConfigError
from autogeoref.paths import VolumePaths
from autogeoref.report import build_report, load_results_dir, report_markdown
from autogeoref.runcontext import RunContext

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_report_024_matches_recorded_funnel(fixtures_dir: Path) -> None:
    """The recorded funnel at fixture freeze: 67 strict OK,
    23 rescued, 7 corroborated (=97 accepted), 17 flagged of which 12 revoked."""
    results = load_results_dir(fixtures_dir / "sanborn01790_024" / "results")
    report = build_report("sanborn01790_024", results, None)
    assert report.n_sheets == 114
    assert report.strict_accepted == 67
    assert report.rescued == 23
    assert report.corroborated == 7
    assert report.revoked == 12
    assert report.accepted_total == 97
    assert report.flagged == 114 - 97
    md = report_markdown(report)
    assert "sanborn01790_024" in md and "97" in md


def test_report_ref_volume(fixtures_dir: Path) -> None:
    results = load_results_dir(fixtures_dir / "ref-volume" / "results")
    report = build_report("sanborn01790_006.5", results)
    assert report.n_sheets == 97
    assert report.accepted_total == 78
    # unscored until someone scores it: the frozen records still CARRY the field a
    # run used to write, and the report must not read it off them — a number that
    # only a scoring pass can produce must not appear because a record is old
    assert report.median_rmse_vs_human_m is None
    assert report.accepts_scored_vs_human == 0
    assert report.accepts_unscored_vs_human == report.accepted_total


def test_report_reads_its_ground_truth_counters_from_the_sidecar(fixtures_dir: Path) -> None:
    """The four GT counters, over the scores the scoring pass hands the report."""
    results = load_results_dir(fixtures_dir / "ref-volume" / "results")
    accepts = [p for p, r in results.items() if str(r.get("status", "")).startswith("OK")]
    scores = dict.fromkeys(accepts[:4], 4.0)
    scores[accepts[0]] = 20.0  # one beyond the 15 m gate
    report = build_report("sanborn01790_006.5", results, scores=scores)
    assert report.accepts_scored_vs_human == 4
    assert report.accepts_unscored_vs_human == report.accepted_total - 4
    assert report.accepts_over_commit_gate == 1
    assert report.median_rmse_vs_human_m == 4.0
    # ...and it is a finding, not a withdrawal: the over-gate sheet still commits
    assert report.committed == report.accepted_total


def test_chicago_config_loads() -> None:
    cfg = load_city_config(CONFIGS / "chicago" / "chicago.toml")
    assert cfg.name.startswith("Chicago")
    assert cfg.centerlines_path.name == "street_center_lines.geojson"
    # tracked and config-relative, like every other city's
    assert cfg.aliases_dir == CONFIGS / "chicago" / "aliases"
    v = cfg.volume("sanborn01790_024")
    # both constants are baked into the golden suite's replay constraints
    # (tests/test_escalate.py) — this CI-visible pin is what fails first if a
    # config edit would silently re-baseline those recordings
    assert v.scale_m_per_px == pytest.approx(0.0669)
    assert v.rotation_deg == pytest.approx(1.20)
    # `_024` and `_021` took bounds_from `_093`/`_090`, both volunteer-placed
    # layers now retired. counterpart_bounds reads the viewer manifest and
    # raises BoundsError on a missing id — which _resolve_or_derive_bounds does
    # NOT catch (it catches only NoBoundsSourceError) — so a surviving reference
    # is a hard run failure, not a fallthrough. No Chicago volume may name one.
    assert v.bounds_from_counterpart is None
    assert [vid for vid, vc in cfg.volumes.items() if vc.bounds_from_counterpart is not None] == []
    assert cfg.volume("sanborn01790_034").bounds_areas == (
        "UPTOWN",
        "LINCOLN SQUARE",
        "EDGEWATER",
        "LAKE VIEW",
    )
    # unknown volume -> defaults (two-pass derivation)
    assert cfg.volume("sanborn01790_999").scale_m_per_px is None
    assert cfg.aliases_path("sanborn01790_024").name == "aliases-sanborn01790_024.json"
    rail_extract = CONFIGS.parent / "fixtures/reference/rail-chicago-cta-overpass.json"
    assert cfg.rail_geojson_path == rail_extract
    assert cfg.rail_gazetteer_path == CONFIGS / "chicago" / "rail-gazetteer-chicago.json"


def test_the_documented_first_config_is_runnable_from_a_cold_clone() -> None:
    """The config README's Quick Start names must need nothing outside the clone."""
    cfg = load_city_config(CONFIGS / "staunton" / "staunton.toml")
    assert cfg.centerlines_from_osm is True
    # the cache lands beside the config (gitignored), with no probe-local override
    assert cfg.centerlines_path.parent == CONFIGS / "staunton" / "cache"
    assert cfg.aliases_dir == CONFIGS / "staunton" / "aliases"
    # every declared channel must be one the OSM default can actually feed:
    # junction reads derived node ids, addresses would need ranges OSM lacks
    assert cfg.evidence_channels == ("junction",)
    # the catalog is what gives a published layer a year, and so an era chip
    # instead of the "undated" bucket; it has to be tracked, not fetched
    assert cfg.loc_catalog_path is not None
    assert cfg.loc_catalog_path.is_file()
    volume = cfg.volume("sanborn02165_007")
    minx, miny, maxx, maxy = volume.bounds_bbox or (0.0, 0.0, 0.0, 0.0)
    # the box is drawn round the volume's key map, so it holds the town centre
    # with room; bounds_bbox is used verbatim and a tight one drops sheets
    assert minx < -89.792 < maxx and miny < 39.0117 < maxy
    # and it reaches past the town to the north-east, where the printed index
    # puts sheet 14's water works and coal shafts — the detached-area trap
    assert maxy > 39.05 and maxx > -89.757


def test_the_second_small_town_config_is_runnable_from_a_cold_clone() -> None:
    """The other OSM-default city config carries the same cold-clone contract."""
    cfg = load_city_config(CONFIGS / "crystal-lake" / "crystal-lake.toml")
    assert cfg.centerlines_from_osm is True
    # the cache lands beside the config (gitignored), with no probe-local override
    assert cfg.centerlines_path.parent == CONFIGS / "crystal-lake" / "cache"
    assert cfg.aliases_dir == CONFIGS / "crystal-lake" / "aliases"
    # every declared channel must be one the OSM default can actually feed:
    # junction reads derived node ids, addresses would need ranges OSM lacks
    assert cfg.evidence_channels == ("junction",)
    volume = cfg.volume("sanborn01810_007")
    minx, miny, maxx, maxy = volume.bounds_bbox or (0.0, 0.0, 0.0, 0.0)
    # the box is drawn round the volume's key map, so it holds the town centre
    # with room; bounds_bbox is used verbatim and a tight one drops sheets
    assert minx < -88.31 < maxx and miny < 42.24 < maxy
    assert maxx - minx > 0.05 and maxy - miny > 0.03


def test_every_alias_dir_resolves_inside_the_clone() -> None:
    """The path is what has to stay in the clone, table or no table.

    One pointing out to a gitignored tree would make a stranger-runnable config
    stop resolving — that is the regression this exists to catch, and it bites
    hardest on a city with nothing in the directory to notice by. A README
    keeps the directory tracked; whether a city also has tables is its own
    business, and `alias-sweep` may add some at any time.
    """
    for city in ("cleveland", "crystal-lake", "staunton", "chicago"):
        cfg = load_city_config(CONFIGS / city / f"{city}.toml")
        assert cfg.aliases_dir == CONFIGS / city / "aliases"
        assert (cfg.aliases_dir / "README.md").is_file()


def test_rail_config_requires_a_paired_gazetteer(tmp_path: Path) -> None:
    for key, value in (
        ("rail_geojson", '"rail.json"'),
        ("rail_gazetteer", '"gazetteer.json"'),
    ):
        cfg = tmp_path / f"{key}.toml"
        cfg.write_text(f'[city]\nname = "X"\naliases_dir = "a"\n{key} = {value}\n')
        with pytest.raises(ConfigError, match="must be configured together"):
            load_city_config(cfg)


def test_run_context_loads_the_configured_rail_gazetteer(tmp_path: Path) -> None:
    rail_path = tmp_path / "rail.geojson"
    rail_path.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"properties":{"name":"Test Rail"},"geometry":{"type":"LineString",'
        '"coordinates":[[-87.7,41.8],[-87.6,41.9]]}}]}'
    )
    cfg_path = tmp_path / "city.toml"
    cfg_path.write_text(
        f'[city]\nname = "X"\naliases_dir = "a"\n'
        f'rail_geojson = "{rail_path}"\n'
        f'rail_gazetteer = "{CONFIGS / "chicago" / "rail-gazetteer-chicago.json"}"\n'
    )
    city = load_city_config(cfg_path)
    ctx = RunContext(
        args=SimpleNamespace(),
        city=city,
        vol=city.volume("sanborn01790_034"),
        paths=VolumePaths(root=tmp_path),
        bounds=(-87.7, 41.8, -87.6, 41.9),
    )
    assert ctx.rail_index is not None
    assert set(ctx.rail_index.groups) == {"TEST RAIL"}
    assert ctx.rail_index.gazetteer is not None


def test_volume_renumbering_table_overrides_the_city_one(tmp_path: Path) -> None:
    """A district renumbered on its own date needs its own book.

    Chicago's 1909 table excludes the Loop (renumbered 1911), so the 1906 Loop
    volumes carry a merged Loop table and every other pre-1909 volume keeps the
    city's — converting a Loop numeral through the outside-the-Loop book lands
    it a measured median 934 m away.
    """
    cfg = load_city_config(CONFIGS / "chicago" / "chicago.toml")
    loop = cfg.volume("sanborn01790_017").renumbering_table_path
    assert loop is not None and loop.name == "renumbering-chicago-loop-merged.json"
    assert cfg.volume("sanborn01790_018").renumbering_table_path == loop
    # every other volume defers to the city table (None = no override)
    assert cfg.volume("sanborn01790_034").renumbering_table_path is None
    assert cfg.volume("sanborn01790_999").renumbering_table_path is None
    assert cfg.renumbering_table_path is not None
    assert cfg.renumbering_table_path.name == "renumbering-chicago-1909.json"
    # the override resolves relative to the config file, like every other path
    toml = tmp_path / "city.toml"
    toml.write_text(
        '[city]\nname = "X"\naliases_dir = "a"\n[volumes.v1]\nrenumbering_table = "t.json"\n'
    )
    assert load_city_config(toml).volume("v1").renumbering_table_path == tmp_path / "t.json"


def test_address_block_size_default_and_validation(tmp_path: Path) -> None:
    """Block size defaults to the common US 100-numbers convention; the
    addresses channel's along-street tolerance is 0.75 x block (calibrated:
    truth median 2 vs one-block-shift median 99 on 100-number blocks)."""
    from autogeoref.address_channel import ADDR_TOL_NUMBERS, addr_tol_numbers

    cfg = load_city_config(CONFIGS / "chicago" / "chicago.toml")
    assert cfg.address_block_size == 100
    assert addr_tol_numbers(cfg.address_block_size) == ADDR_TOL_NUMBERS == 75.0

    toml = tmp_path / "city.toml"
    toml.write_text('[city]\nname = "X"\naliases_dir = "a"\naddress_block_size = 200\n')
    assert load_city_config(toml).address_block_size == 200
    assert addr_tol_numbers(200) == 150.0

    for bad_value in ('"100"', "0", "-100", "true"):
        bad = tmp_path / "bad.toml"
        bad.write_text(f'[city]\nname = "X"\naliases_dir = "a"\naddress_block_size = {bad_value}\n')
        with pytest.raises(ConfigError):
            load_city_config(bad)


def test_city_measured_defaults_are_config_not_code(tmp_path: Path) -> None:
    """The Chicago-frozen defaults are CITY facts now: chicago.toml declares
    today's values, a second city declares its own, and a city that declares
    nothing gets the labeled defaults — never another city's numbers changing
    behavior invisibly."""
    from autogeoref.budget import DEFAULT_GATED_FRACTION

    cfg = load_city_config(CONFIGS / "chicago" / "chicago.toml")
    assert cfg.gated_fraction == DEFAULT_GATED_FRACTION == 0.41
    assert cfg.loc_catalog_path is not None
    assert cfg.loc_catalog_path.name == "loc-catalog-chicago.json"
    assert cfg.renumbering_note is not None and "1911" in cfg.renumbering_note

    toml = tmp_path / "city.toml"
    toml.write_text(
        '[city]\nname = "X"\naliases_dir = "a"\ngated_fraction = 0.5\n'
        'loc_catalog = "catalog.json"\nrenumbering_note = "the docks renumbered in 1922"\n'
    )
    other = load_city_config(toml)
    assert other.gated_fraction == 0.5
    assert other.loc_catalog_path == tmp_path / "catalog.json"
    assert other.renumbering_note == "the docks renumbered in 1922"

    plain = tmp_path / "plain.toml"
    plain.write_text('[city]\nname = "X"\naliases_dir = "a"\n')
    defaults = load_city_config(plain)
    assert defaults.gated_fraction == DEFAULT_GATED_FRACTION
    assert defaults.loc_catalog_path is None and defaults.renumbering_note is None

    for bad_line in (
        "gated_fraction = 0",
        "gated_fraction = 1.5",
        'gated_fraction = "x"',
        "gated_fraction = true",
        'renumbering_note = ""',
        "renumbering_note = 3",
        'loc_catalog = ""',
        "loc_catalog = 3",
    ):
        bad = tmp_path / "bad.toml"
        bad.write_text(f'[city]\nname = "X"\naliases_dir = "a"\n{bad_line}\n')
        with pytest.raises(ConfigError):
            load_city_config(bad)


def test_config_missing_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[city]\nname = "X"\n')
    with pytest.raises(ConfigError):
        load_city_config(bad)


def test_valid_bounds_from_still_parses(tmp_path: Path) -> None:
    """The POSITIVE half of bounds_from parsing.

    No city declares `bounds_from` any more — Chicago's last two references were
    volunteer volumes are retired — so this synthetic case is the only
    thing left asserting that a valid value reaches `bounds_from_counterpart`
    at all. Without it the rejecting branch is the only covered one, and the
    feature could rot to a silent no-op for the next city that wants it.
    """
    toml = _city_toml(tmp_path, '[volumes.v1]\nbounds_from = "other_001"\n')
    assert load_city_config(toml).volume("v1").bounds_from_counterpart == "other_001"


@pytest.mark.parametrize(
    "volume_body",
    [
        'quadrant_rescue = "false"',
        'content_masks = "false"',
        'content_mask_exempt = ["57"]',  # requires content_masks = true
        'content_masks = true\ncontent_mask_exempt = "57"',  # not a list
        "content_masks = true\ncontent_mask_exempt = [57]",  # not strings
        'content_masks = true\ncontent_mask_exempt = [""]',  # empty page id
        "scale_m_per_px = nan\nrotation_deg = 0",
        "scale_m_per_px = inf\nrotation_deg = 0",
        "scale_m_per_px = 0\nrotation_deg = 0",
        "scale_m_per_px = 1\nrotation_deg = nan",
        "bounds_bbox = [-87.6, 41.9, -87.7, 42.0]",
        "bounds_bbox = [-87.6, 41.9, -87.5, 91]",
        'bounds_from = "../other"',
        'bounds_areas = "UPTOWN"',
    ],
)
def test_config_rejects_invalid_volume_domains(tmp_path: Path, volume_body: str) -> None:
    toml = _city_toml(tmp_path, f"[volumes.v1]\n{volume_body}\n")
    with pytest.raises(ConfigError):
        load_city_config(toml)


@pytest.mark.parametrize(
    "body",
    [
        '[volumes."../v1"]\nbounds_bbox = [-87.7, 41.8, -87.6, 41.9]',
        "[volumes]\nv1 = []",
        "centerlines = 7",
        "osm_cache_dir = false",
        "[volumes.v1.extra]",
    ],
)
def test_config_rejects_invalid_table_shapes_and_identifiers(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError):
        load_city_config(_city_toml(tmp_path, body))


def test_config_rejects_a_falsy_volume_table(tmp_path: Path) -> None:
    toml = tmp_path / "city.toml"
    toml.write_text('volumes = []\n[city]\nname = "X"\naliases_dir = "a"\n')
    with pytest.raises(ConfigError):
        load_city_config(toml)


def _city_toml(tmp_path: Path, body: str) -> Path:
    toml = tmp_path / "city.toml"
    toml.write_text(f'[city]\nname = "X"\naliases_dir = "a"\n{body}')
    return toml


def test_provider_qualified_models_load(tmp_path: Path) -> None:
    """A ladder may mix providers; a bare name still means Anthropic."""
    toml = _city_toml(
        tmp_path,
        'escalation_models = ["claude-sonnet-5", "codex:gpt-5.6-terra", '
        '"opencode:openai/gpt-5.6-terra"]\n',
    )
    assert load_city_config(toml).escalation_models == (
        "claude-sonnet-5",
        "codex:gpt-5.6-terra",
        "opencode:openai/gpt-5.6-terra",
    )


def test_model_variants_load_and_volume_overrides(tmp_path: Path) -> None:
    toml = _city_toml(
        tmp_path,
        'annotation_model = "codex:gpt-5.6-terra"\n'
        'annotation_variant = "high"\n'
        'escalation_models = ["codex:gpt-5.6-terra", "codex:gpt-5.6-sol"]\n'
        'escalation_variants = ["high", "high"]\n'
        "[volumes.v1]\n"
        'annotation_variant = "medium"\n'
        'escalation_models = ["codex:gpt-5.6-sol"]\n'
        'escalation_variants = ["high"]\n',
    )
    cfg = load_city_config(toml)
    assert cfg.annotation_variant == "high"
    assert cfg.volume("v1").annotation_variant == "medium"
    assert cfg.volume("v1").escalation_tiers() == (("codex:gpt-5.6-sol", "high"),)

    switched = _city_toml(
        tmp_path,
        'annotation_model = "codex:gpt-5.6-terra"\n'
        'annotation_variant = "high"\n'
        "[volumes.v2]\n"
        'annotation_model = "claude-sonnet-5"\n',
    )
    assert load_city_config(switched).volume("v2").annotation_variant is None


def test_a_direct_api_model_loads_like_any_other_reference(tmp_path: Path) -> None:
    """A provider is a model reference. The OpenAI API takes the same reasoning
    variant the Codex CLI does; the Anthropic API and Ollama take none."""
    toml = _city_toml(
        tmp_path,
        'annotation_model = "openai-api:gpt-5.6-terra"\n'
        'annotation_variant = "high"\n'
        'escalation_models = ["anthropic-api:claude-sonnet-5", "ollama:gemma4:12b"]\n',
    )
    cfg = load_city_config(toml)
    assert cfg.annotation_model == "openai-api:gpt-5.6-terra"
    assert cfg.escalation_models == ("anthropic-api:claude-sonnet-5", "ollama:gemma4:12b")

    no_variant = _city_toml(
        tmp_path,
        'annotation_model = "anthropic-api:claude-sonnet-5"\nannotation_variant = "high"\n',
    )
    with pytest.raises(ConfigError, match="requires one of"):
        load_city_config(no_variant)


def test_model_variants_are_validated_at_config_load(tmp_path: Path) -> None:
    anthopic = _city_toml(tmp_path, 'annotation_variant = "high"\n')
    with pytest.raises(ConfigError, match="requires one of"):
        load_city_config(anthopic)

    mismatched = _city_toml(
        tmp_path,
        'escalation_models = ["codex:gpt-5.6-terra", "codex:gpt-5.6-sol"]\n'
        'escalation_variants = ["high"]\n',
    )
    with pytest.raises(ConfigError, match="one variant string per"):
        load_city_config(mismatched)

    duplicate = _city_toml(
        tmp_path,
        'escalation_models = ["codex:gpt-5.6-terra", "codex:gpt-5.6-terra"]\n'
        'escalation_variants = ["", "high"]\n',
    )
    with pytest.raises(ConfigError, match="cannot repeat a model"):
        load_city_config(duplicate)

    orphan_variant = _city_toml(tmp_path, 'escalation_variant = "high"\n')
    with pytest.raises(ConfigError, match="requires escalation_model"):
        load_city_config(orphan_variant)


def test_bad_model_reference_fails_at_config_load(tmp_path: Path) -> None:
    """Fail at CONFIGURE time, not after earlier stages have spent calls.

    A mistyped provider and a small model are both unusable; catching them at
    load means a run dies before it costs anything, not four stages in.
    """
    typo = _city_toml(tmp_path, 'escalation_models = ["codx:gpt-5.6-terra"]\n')
    with pytest.raises(ConfigError, match="unknown provider"):
        load_city_config(typo)

    small = _city_toml(tmp_path, 'escalation_model = "codex:gpt-5.4-mini"\n')
    with pytest.raises(ConfigError):
        load_city_config(small)


# page_scale_multiples parsing/validation: tests/test_page_scale_override.py


def test_the_retired_consensus_fallback_key_is_refused_not_ignored(tmp_path: Path) -> None:
    """`consensus_fallback_model` named the third voice of a stage that no longer
    exists.

    A retired key must FAIL, never be silently ignored: an operator who still writes it believes
    a third model is reading their gated pool, and a quiet no-op leaves them reading a funnel as
    evidence about a voice that never spoke. BOTH levels, and the city one is the whole reason
    `_RETIRED_KEYS` exists: a volume table has an allow-list so a stale key errors on its own,
    while the CITY table has none, so without an explicit refusal a key that used to be parsed
    and validated becomes exactly the silent no-op this test forbids.
    """
    for table in ("[city]", "[volumes.v1]"):
        toml = tmp_path / f"{table.strip('[]').replace('.', '_')}.toml"
        body = '[city]\nname = "X"\naliases_dir = "a"\n'
        if table == "[city]":
            body += 'consensus_fallback_model = "claude-fable-5"\n'
        else:
            body += '[volumes.v1]\nconsensus_fallback_model = "claude-fable-5"\n'
        toml.write_text(body)
        with pytest.raises(ConfigError, match="consensus_fallback_model is RETIRED"):
            load_city_config(toml)

    cfg = load_city_config(CONFIGS / "chicago" / "chicago.toml")
    assert not hasattr(cfg, "consensus_fallback_model")
    assert not hasattr(cfg.volume("sanborn01790_024"), "consensus_fallback_model")


def test_escalation_error_order_is_the_contract_city_vs_volume(tmp_path: Path) -> None:
    """When two escalation keys are bad at once, WHICH error wins is part of
    the CLI contract — and the city and volume blocks check in different
    orders (see resolve_escalation). These pin the winner on each side so an
    order unification cannot land silently."""
    # city block: models-shape check precedes the orphan-variant pairing check
    bad_city = _city_toml(
        tmp_path,
        'escalation_models = 3\nescalation_variant = "high"\n',
    )
    with pytest.raises(ConfigError, match="escalation_models must be a list of model names"):
        load_city_config(bad_city)

    # volume block: the orphan-variant pairing check wins over the models shape
    bad_volume = tmp_path / "volume.toml"
    bad_volume.write_text(
        '[city]\nname = "X"\naliases_dir = "a"\n'
        "[volumes.v1]\n"
        "escalation_models = 3\n"
        'escalation_variant = "high"\n'
    )
    with pytest.raises(ConfigError, match="escalation_variant requires escalation_model"):
        load_city_config(bad_volume)

    # city block: with every pairing satisfied, a bad variant string wins
    # over a variants length mismatch
    bad_variant = tmp_path / "variant.toml"
    bad_variant.write_text(
        '[city]\nname = "X"\naliases_dir = "a"\n'
        'annotation_model = "codex:gpt-5.6-terra"\n'
        'escalation_model = "codex:gpt-5.6-sol"\n'
        'escalation_models = ["codex:gpt-5.6-terra", "codex:gpt-5.6-sol"]\n'
        "escalation_variant = 3\n"
        'escalation_variants = ["high"]\n'
    )
    with pytest.raises(ConfigError, match="escalation_variant must be a variant string"):
        load_city_config(bad_variant)
