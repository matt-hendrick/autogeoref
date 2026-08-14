"""The whole deploy sequence for one city.

Bundle, archives, page, verify — in an order where a new volume actually
becomes visible, since the volume list lives in the manifest inside the
bundle rather than beside the archives. The verification is a gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy_support import DEPLOY, GOOD_HEADERS, Deployment


@pytest.fixture
def deployment(tmp_path: Path) -> Deployment:
    return Deployment(tmp_path)


def test_deploy_runs_the_four_steps_and_verifies_the_range_path(deployment: Deployment) -> None:
    deployment.archive("basemap", "basemap-testcity-20260808", 100)
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"], basemap="basemap-testcity-20260808")
    deployment.publish({})

    result = deployment.run(DEPLOY, "--city", str(deployment.city))

    assert result.returncode == 0, result.stdout + result.stderr
    bundle = deployment.data / "deploy" / "test-city"
    published = json.loads((bundle / "manifest.json").read_text())
    assert published["volumes"][0]["pmtiles"] == (
        "https://tiles.example.com/testcity/vol_a.pmtiles"
    )
    assert 'window.MAPBOX_TOKEN = "pk.testtoken";' in (bundle / "config.js").read_text()
    calls = deployment.calls().splitlines()
    order = [line.split()[0] for line in calls if line and not line.startswith(" ")]
    assert order.index("rclone") < order.index("npx") < order.index("curl")
    assert f"pages deploy {bundle} --project-name=example-project" in "\n".join(calls)
    probes = [line for line in calls if "tiles.example.com" in line]
    # the smallest archive is the one probed: it is certainly under the edge
    # cache's size limit. Twice, because the first fetch may legitimately miss
    assert len(probes) == 2
    for probe in probes:
        assert probe.endswith(
            "https://tiles.example.com/testcity/basemap-testcity-20260808.pmtiles"
        )
        # without an Origin, R2 answers without the CORS header it would send a
        # browser, and the check would pass on a bucket that fails in the page
        assert "Origin: https://example-project.pages.dev" in probe
    assert "serves the manifest this run built" in result.stdout


def test_deploy_says_so_when_the_page_is_not_yet_serving_this_bundle(
    deployment: Deployment,
) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})
    stale = deployment.root / "stale-manifest.json"
    stale.write_text('{"volumes": []}')

    result = deployment.run(DEPLOY, "--city", str(deployment.city), STUB_PAGE=str(stale))

    assert result.returncode == 0, result.stderr
    assert "not yet serving the manifest this run built" in result.stderr


def test_deploy_warns_about_a_baked_volume_that_was_never_published(
    deployment: Deployment,
) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.archive("autogeoref", "vol_unpublished", 300)
    deployment.archive("autogeoref", "vol_a-overview", 50)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})

    result = deployment.run(DEPLOY, "--city", str(deployment.city), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "vol_unpublished.pmtiles" in result.stderr
    assert "autogeoref publish" in result.stderr
    # the overview companion is not a volume and nothing serves it
    assert "overview" not in result.stderr


def test_deploy_passes_a_replace_through_to_the_upload(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 180})

    result = deployment.run(DEPLOY, "--city", str(deployment.city), "--replace", "vol_a")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "rclone copyto" in deployment.calls()
    assert "purge_cache" in deployment.calls()


def test_deploy_can_republish_the_page_alone(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})

    result = deployment.run(DEPLOY, "--city", str(deployment.city), "--skip-upload")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "rclone" not in deployment.calls()
    assert "npx" in deployment.calls()


def test_deploy_dry_run_touches_nothing(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({})

    result = deployment.run(DEPLOY, "--city", str(deployment.city), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert not (deployment.data / "deploy" / "test-city").exists()
    calls = deployment.calls()
    assert "npx" not in calls
    assert "curl" not in calls
    assert "rclone copy" not in calls
    assert "new       vol_a.pmtiles" in result.stdout


@pytest.mark.parametrize(
    "dropped",
    ["etag", "access-control-allow-origin", "access-control-expose-headers", "cache-control"],
)
def test_deploy_fails_when_the_archives_answer_without_a_needed_header(
    deployment: Deployment, dropped: str
) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})
    deployment.headers.write_text(
        "\r\n".join(
            line for line in GOOD_HEADERS.split("\r\n") if not line.startswith(f"{dropped}:")
        )
    )

    result = deployment.run(DEPLOY, "--city", str(deployment.city))

    assert result.returncode == 1
    # the message, not just the name: the dumped headers name most of these too
    assert f"did not answer with a {dropped} header" in result.stderr


def test_deploy_says_why_when_the_repeat_fetch_answers_without_a_cache_status(
    deployment: Deployment,
) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})

    result = deployment.run(DEPLOY, "--city", str(deployment.city), STUB_SECOND_UNCACHED="1")

    assert result.returncode == 1
    assert "without a cf-cache-status header" in result.stderr


def test_deploy_fails_when_a_range_request_is_not_partial_content(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})
    deployment.headers.write_text(GOOD_HEADERS.replace("HTTP/2 206", "HTTP/2 404"))

    result = deployment.run(DEPLOY, "--city", str(deployment.city))

    assert result.returncode == 1
    assert "not uploaded" in result.stderr


def test_deploy_fails_when_the_archives_are_not_cached_at_the_edge(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})
    deployment.headers.write_text(
        GOOD_HEADERS.replace("cf-cache-status: HIT", "cf-cache-status: DYNAMIC")
    )

    result = deployment.run(DEPLOY, "--city", str(deployment.city))

    assert result.returncode == 1
    assert "Cache Rule" in result.stderr


def test_deploy_without_a_mapbox_token_stops_before_rebuilding_the_bundle(
    deployment: Deployment,
) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])

    result = deployment.run(DEPLOY, "--city", str(deployment.city), AUTOGEOREF_MAPBOX_TOKEN="")

    assert result.returncode == 1
    assert "AUTOGEOREF_MAPBOX_TOKEN" in result.stderr
    assert not (deployment.data / "deploy" / "test-city").exists()


def test_deploy_refuses_to_replace_an_archive_it_is_not_uploading(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 180})

    result = deployment.run(
        DEPLOY, "--city", str(deployment.city), "--skip-upload", "--replace", "vol_a"
    )

    assert result.returncode == 1
    assert "--skip-upload" in result.stderr
    assert deployment.calls() == ""
