"""The ``[viewer]`` config block: what parses, what is refused, what is emitted.

Chicago's TOML carries the live site's wording and behaviour verbatim, so it is
read here as the production contract. The fictional city proves the same loader
produces a valid config with generic fallbacks, and the refusal tests pin the
typos that would otherwise publish a silently wrong or blank page.
"""

from __future__ import annotations

import tomllib
from itertools import pairwise
from pathlib import Path

import pytest

from autogeoref.config.model import ConfigError
from autogeoref.viewer.config import ViewerConfig, load_viewer_config, site_dict
from autogeoref.viewer.era import era_label
from viewer_support import CHICAGO_TOML, ROOT, VIREO_TOML


class TestChicagoViewerConfig:
    def test_the_configured_story_parses_and_cites_real_volumes(self) -> None:
        """Chicago's own story is configuration, so it is checked like the rest.

        The coverage gate itself needs the served layers and cannot run here;
        what this pins is that the sidecar parses, that every stop names an era
        the city actually declares, and that the volumes the captions cite are
        the ones the manifest build reported — so retiring a volume out from
        under a caption fails here rather than shipping a blank pane.
        """
        viewer = load_viewer_config(CHICAGO_TOML)
        assert viewer.stories is not None
        story = viewer.stories.stories[0]
        assert len(story.stops) >= 6
        declared = {b.label for b in viewer.era_buckets}
        for stop in story.stops:
            assert stop.eras, f"{stop.id} names no era, so the gate cannot check it"
            assert set(stop.eras) <= declared, f"{stop.id} names an undeclared era"
            assert stop.title and stop.body_html and stop.sources
        # more than one survey across the set: the strongest stops only turned
        # up once the search stopped looking at a single decade, and a story
        # that never leaves one is not showing change
        assert len({era for stop in story.stops for era in stop.eras}) >= 3
        # No two stops on literally the same ground. Whether two nearby stops
        # are really one stop told twice is a judgement about subject, not
        # distance, and stays with review; this only catches a copied camera.
        cameras = [
            (round(stop.camera.center[0], 3), round(stop.camera.center[1], 3))
            for stop in story.stops
        ]
        assert len(set(cameras)) == len(cameras), f"stops share a camera: {cameras}"

    def test_production_wording_verbatim(self) -> None:
        v = load_viewer_config(CHICAGO_TOML)
        assert v.title == "Chicago {era} / Now — Sanborn Atlas"
        assert v.kicker == "Sanborn Fire Insurance Atlas"
        assert v.heading == "Chicago {era}"
        assert v.heading_note == "against now"
        assert v.loading_text == "unrolling the atlas…"
        # every era opens together, so the atlas starts at its full extent;
        # each chip stays individually toggleable
        assert v.default_eras == (
            "1890s",
            "1900s",
            "1910s",
            "1920s",
            "1930s",
            "1940s",
            "1950s",
        )
        assert v.home_point == (-87.6280, 41.8820)  # State & Madison

    def test_geocoder_and_basemap(self) -> None:
        v = load_viewer_config(CHICAGO_TOML)
        assert v.geocoder is not None
        assert v.geocoder.suffix == ", Chicago, IL"
        assert v.geocoder.bbox == (-87.95, 41.62, -87.5, 42.05)
        # The public viewer serves its own basemap: OSMF's raster tiles are
        # forbidden at this traffic shape, and the failure mode is a referer
        # block and a blank map.
        assert v.basemap is not None
        assert v.basemap.type == "vector"
        assert v.basemap.tiles is None
        assert v.basemap.pmtiles is not None and v.basemap.pmtiles.endswith(".pmtiles")
        assert v.basemap.style_atlas is not None and v.basemap.style_now is not None
        for style in (v.basemap.style_atlas, v.basemap.style_now):
            assert (ROOT / "viewer" / style).is_file(), f"unvendored basemap style {style}"

    def test_vector_basemap_needs_its_archive_and_styles(self, tmp_path: Path) -> None:
        """Half a vector basemap renders as a blank map with the atlas floating
        on it and no browser-side clue why — refuse the config instead."""
        toml = tmp_path / "half.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer.basemap]\ntype = "vector"\npmtiles = "b.pmtiles"\n'
        )
        with pytest.raises(ConfigError, match="style_atlas, style_now"):
            load_viewer_config(toml)

    def test_a_configured_geocoder_provider_is_rejected(self, tmp_path: Path) -> None:
        """The field existed, was serialized into every manifest, and was read
        by nothing — so setting it to `mapbox` silently did nothing while the
        published manifest advertised whatever it said. Refuse it rather than
        ignore it: which geocoder answers is the deployed token's business."""
        toml = tmp_path / "geo.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer.geocoder]\nprovider = "mapbox"\n'
        )
        with pytest.raises(ConfigError, match="provider is not configurable"):
            load_viewer_config(toml)

    def test_unknown_basemap_type_is_rejected(self, tmp_path: Path) -> None:
        toml = tmp_path / "bad.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer.basemap]\ntype = "wms"\n'
        )
        with pytest.raises(ConfigError, match="raster or vector"):
            load_viewer_config(toml)

    def test_era_buckets_are_decades(self) -> None:
        """The viewer groups every map year into its calendar decade."""
        v = load_viewer_config(CHICAGO_TOML)
        spans = {b.label: (b.first_year, b.last_year) for b in v.era_buckets}
        assert spans == {
            "1890s": (1890, 1899),
            "1900s": (1900, 1909),
            "1910s": (1910, 1919),
            "1920s": (1920, 1929),
            "1930s": (1930, 1939),
            "1940s": (1940, 1949),
            "1950s": (1950, 1959),
        }
        # Contiguous: no served year between the first and last bucket can
        # fall through to a self-labelled chip.
        ordered = sorted(v.era_buckets, key=lambda b: b.first_year)
        for prev, nxt in pairwise(ordered):
            assert nxt.first_year == prev.last_year + 1

    def test_one_credit_line_and_it_names_nobody_else(self) -> None:
        """This city serves what this pipeline placed, so it needs ONE credit
        line and no composition. What must hold is that nothing this city can
        emit — the site-wide line, or any per-era override — names another
        party for georeferencing that is ours."""
        v = load_viewer_config(CHICAGO_TOML)
        assert v.optional_credits_html
        emitted = [v.optional_credits_html, *(b.credits_html or "" for b in v.era_buckets)]
        for line in emitted:
            assert "oldinsurancemaps" not in line.lower()
            assert "volunteer" not in line.lower()

    def test_the_retired_credits_key_is_refused_not_ignored(self, tmp_path: Path) -> None:
        """`optional_credits` was `default_credits`. The key is OPTIONAL, so an
        unrecognized spelling parses clean and publishes a site with no credit
        line at all — a silence nobody notices. Refuse it by name instead."""
        toml = tmp_path / "old-key.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer]\ndefault_credits = "placed by someone"\n'
        )
        with pytest.raises(ConfigError, match=r"now viewer\.optional_credits"):
            load_viewer_config(toml)

    def test_a_non_string_credit_is_refused(self, tmp_path: Path) -> None:
        """The viewer joins this as a string. A number throws in `applySelection`
        BEFORE the footer is written, so the page boots with an empty footer,
        sources panel and district list — a blank page from one config typo."""
        toml = tmp_path / "num.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            "[viewer]\noptional_credits = 42\n"
        )
        with pytest.raises(ConfigError, match="must be a string"):
            load_viewer_config(toml)

    def test_a_credit_may_be_omitted_entirely(self, tmp_path: Path) -> None:
        """Optional means optional: a city that declares no credit publishes
        none, and nothing refuses. That is a choice, not a missing default."""
        toml = tmp_path / "no-credit.toml"
        toml.write_text('[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n[viewer]\n')
        v = load_viewer_config(toml)
        assert v.optional_credits_html is None
        assert "optional_credits_html" not in site_dict("X", v)

    def test_the_serving_directories_are_declared(self) -> None:
        """The one credit line is only honest while nothing else can be served
        under it, and `serving_dirs` is what enforces that.

        Read from the RAW TOML, not from the parsed config: the parsed value
        equals the default, so asserting it would pass just as well with the
        key deleted or mis-nested under another table.
        """
        raw = tomllib.loads(CHICAGO_TOML.read_text(encoding="utf-8"))
        assert raw["viewer"]["serving_dirs"] == ["autogeoref"]
        assert load_viewer_config(CHICAGO_TOML).serving_dirs == ("autogeoref",)

    def test_a_serving_dirs_path_is_rejected(self, tmp_path: Path) -> None:
        """These are NAMES. `deploy/tiles/autogeoref` is the natural mistake and
        matches no directory's last component, so it would declare nothing and
        refuse every build with a message about the directory, never the typo."""
        toml = tmp_path / "path.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer]\nserving_dirs = ["deploy/tiles/autogeoref"]\n'
        )
        with pytest.raises(ConfigError, match="bare directory NAMES"):
            load_viewer_config(toml)

    # written as TOML source, not Python values: an inline table is `{a = "b"}`,
    # so a json.dumps of one would fail to PARSE and never reach the check
    @pytest.mark.parametrize(
        "literal", ['"autogeoref"', "3", '{a = "b"}', '[["autogeoref"]]', '[""]']
    )
    def test_a_serving_dirs_that_is_not_a_list_of_names_is_rejected(
        self, tmp_path: Path, literal: str
    ) -> None:
        """A bare string is the shape TOML makes easiest to write by mistake,
        and iterating it would declare one directory per CHARACTER."""
        toml = tmp_path / "shape.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            f"[viewer]\nserving_dirs = {literal}\n"
        )
        with pytest.raises(ConfigError, match="non-empty list"):
            load_viewer_config(toml)

    def test_an_empty_serving_dirs_list_is_rejected(self, tmp_path: Path) -> None:
        """`serving_dirs = []` would read as "declare nothing" and disable the
        refusal outright. A city that wants the default omits the key."""
        toml = tmp_path / "empty.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            "[viewer]\nserving_dirs = []\n"
        )
        with pytest.raises(ConfigError, match="non-empty list"):
            load_viewer_config(toml)

    def test_region_labels_port_side_name(self) -> None:
        """The lat/lng bands reproduce the old sideName() thresholds."""
        v = load_viewer_config(CHICAGO_TOML)
        assert v.region_labels is not None
        lat = [(b.above, b.label) for b in v.region_labels.lat_bands]
        assert lat == [
            (41.94, "Far North"),
            (41.885, "North"),
            (41.84, "Central"),
            (41.76, "South"),
            (None, "Far South"),
        ]
        lng = [(b.below, b.above, b.label) for b in v.region_labels.lng_bands]
        assert lng == [
            (-87.72, None, "West Side"),
            (None, -87.63, "Lakefront"),
            (None, None, "Side"),
        ]
        assert v.region_labels.collapse == (("Central Side", "Central"),)

    def test_absent_viewer_block_is_all_defaults(self, tmp_path: Path) -> None:
        toml = tmp_path / "bare.toml"
        toml.write_text('[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n')
        assert load_viewer_config(toml) == ViewerConfig()

    def test_viewer_labels_declared_per_volume(self, tmp_path: Path) -> None:
        toml = tmp_path / "labels.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            "[viewer.labels]\n"
            'sanborn01790_190 = " 1933 Century of Progress World\'s Fair "\n'
        )
        v = load_viewer_config(toml)
        assert dict(v.volume_labels) == {
            "sanborn01790_190": "1933 Century of Progress World's Fair"
        }

    def test_blank_or_nonstring_viewer_label_is_rejected(self, tmp_path: Path) -> None:
        """A blank or non-string value would render literally in the district
        list — refuse the config instead."""
        for value in ('""', "1933"):
            toml = tmp_path / "bad-label.toml"
            toml.write_text(
                '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
                f"[viewer.labels]\nsanborn01790_190 = {value}\n"
            )
            with pytest.raises(ConfigError, match="non-empty string"):
                load_viewer_config(toml)

    def test_viewer_labels_must_be_a_table(self, tmp_path: Path) -> None:
        # falsy scalars included: `labels = false` must not silently pass
        for value in ('"Stockyards"', "false"):
            toml = tmp_path / "bad-labels.toml"
            toml.write_text(
                '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
                f"[viewer]\nlabels = {value}\n"
            )
            with pytest.raises(ConfigError, match="must be a table"):
                load_viewer_config(toml)

    def test_legacy_default_era_string_still_parses(self, tmp_path: Path) -> None:
        """A single-era config predates multi-select and must keep working."""
        toml = tmp_path / "single.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer]\ndefault_era = "1900s"\n'
            '[[viewer.era]]\nyears = [1900, 1909]\nlabel = "1900s"\n'
        )
        assert load_viewer_config(toml).default_eras == ("1900s",)

    def test_default_era_and_default_eras_are_mutually_exclusive(self, tmp_path: Path) -> None:
        toml = tmp_path / "both.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer]\ndefault_era = "1900s"\ndefault_eras = ["1900s"]\n'
        )
        with pytest.raises(ConfigError, match="mutually exclusive"):
            load_viewer_config(toml)

    def test_default_era_typo_is_rejected(self, tmp_path: Path) -> None:
        """A label that names no bucket would silently start the viewer on its
        fallback era instead of the configured one."""
        toml = tmp_path / "typo.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer]\ndefault_eras = ["1890"]\n'  # bucket label is "1890s"
            '[[viewer.era]]\nyears = [1890, 1899]\nlabel = "1890s"\n'
        )
        with pytest.raises(ConfigError, match="not an era bucket label"):
            load_viewer_config(toml)

    def test_default_era_accepts_self_labelled_unbucketed_year(self, tmp_path: Path) -> None:
        """era_label lets a year outside every bucket label itself, so that
        year is a real chip and a legitimate default; only a year a bucket
        covers is rejected (it renders under the bucket's label instead)."""
        toml = tmp_path / "bare-year.toml"
        toml.write_text(
            '[city]\nname = "X"\ncenterlines = "c"\naliases_dir = "a"\n'
            '[viewer]\ndefault_eras = ["1895"]\n'  # outside the only bucket
            '[[viewer.era]]\nyears = [1900, 1909]\nlabel = "1900s"\n'
        )
        assert load_viewer_config(toml).default_eras == ("1895",)


def test_era_label_buckets_and_fallthrough() -> None:
    buckets = load_viewer_config(CHICAGO_TOML).era_buckets
    assert era_label(1894, buckets) == "1890s"
    assert era_label(1917, buckets) == "1910s"
    assert era_label(1950, buckets) == "1950s"
    assert era_label(1901, buckets) == "1900s"
    assert era_label(1927, buckets) == "1920s"
    # outside every bucket: the year labels itself, exactly as the original
    # behaved (nothing surveyed lands here since the buckets went contiguous)
    assert era_label(1885, buckets) == "1885"
    assert era_label(None, buckets) is None


def test_site_dict_chicago_round_trip() -> None:
    site = site_dict("Chicago, Ill.", load_viewer_config(CHICAGO_TOML))
    assert site["name"] == "Chicago, Ill."
    assert site["title"] == "Chicago {era} / Now — Sanborn Atlas"
    assert site["default_eras"] == [
        "1890s",
        "1900s",
        "1910s",
        "1920s",
        "1930s",
        "1940s",
        "1950s",
    ]
    assert "default_era" not in site  # the single-era key is parse-only now
    assert site["home_point"] == [-87.628, 41.882]
    assert site["geocoder"]["suffix"] == ", Chicago, IL"
    # bias only: which geocoder answers is the page's call, from the deployed
    # token and the serving host, and a published `provider` said otherwise
    assert "provider" not in site["geocoder"]
    # the data is still OSM's, whoever cut the tiles
    assert site["basemap"]["attribution"].startswith("© OpenStreetMap contributors")
    assert site["basemap"]["type"] == "vector"
    assert set(site["basemap"]["styles"]) == {"atlas", "now"}
    # Chicago declares no per-era credits, so site_dict emits none and every
    # era falls through to the one site-wide line
    assert "era_credits" not in site
    # the credit carries its own method link, and reaches the page as markup:
    # a page that says "auto-georeferenced" owes the reader somewhere to check
    assert site["optional_credits_html"] == (
        'auto-georeferenced by <a href="walkthrough.html">this pipeline</a>'
    )
    assert site["region_labels"]["collapse"] == [["Central Side", "Central"]]
    assert site["footer_source_html"] == "Maps: Library of Congress, Sanborn Maps Collection"


def test_site_dict_minimal_emits_only_name() -> None:
    assert site_dict("Nowhere", ViewerConfig()) == {"name": "Nowhere"}


def test_fictional_city_config_loads_with_generic_fallbacks() -> None:
    v = load_viewer_config(VIREO_TOML)
    assert v.title is None  # -> HTML generic fallback
    assert v.region_labels is None
    assert v.home_point == (172.64, -43.53)
    assert v.geocoder is not None and v.geocoder.suffix == ", Port Vireo"
    site = site_dict("Port Vireo", v)
    assert site["era_credits"] == {"1905": "auto-georeferenced by this pipeline"}
    # legacy single default_era config lands in the manifest as a one-era list
    assert site["default_eras"] == ["1905"]
