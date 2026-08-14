"""Uploading the archives one city's page fetches.

The whole job is to add what is missing and refuse to overwrite what is
published: an archive keeps its name across a re-bake, so a plain copy
replaces a published object and a cache holding ranges of the old one
corrupts the reader's tiles. Replacing one is deliberate, and purged."""

from __future__ import annotations

from pathlib import Path

import pytest

from autogeoref.viewer.deploy import public_tiles_base
from deploy_support import PUSH, Deployment


@pytest.fixture
def deployment(tmp_path: Path) -> Deployment:
    return Deployment(tmp_path)


def test_dry_run_classifies_and_uploads_nothing(deployment: Deployment) -> None:
    deployment.archive("basemap", "basemap-testcity-20260808", 100)
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.archive("autogeoref", "vol_b", 300)
    deployment.write_manifest(["vol_a", "vol_b"], basemap="basemap-testcity-20260808")
    deployment.publish({"vol_a": 200, "basemap-testcity-20260808": 100})

    result = deployment.run(PUSH, "--city", str(deployment.city), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "new       vol_b.pmtiles" in result.stdout
    assert "2 already published unchanged, 1 new, 0 differing" in result.stdout
    assert "copy" not in deployment.calls()


def test_a_rebaked_archive_is_refused_by_name(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 180})

    result = deployment.run(PUSH, "--city", str(deployment.city))

    assert result.returncode == 2
    assert "DIFFERS   vol_a.pmtiles" in result.stdout
    assert "--replace vol_a" in result.stderr
    assert "copy" not in deployment.calls()


def test_new_archives_upload_basemap_first_and_cannot_overwrite(deployment: Deployment) -> None:
    deployment.archive("basemap", "basemap-testcity-20260808", 100)
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"], basemap="basemap-testcity-20260808")
    # no published listing at all: the first deploy, where the prefix is empty

    result = deployment.run(PUSH, "--city", str(deployment.city))

    assert result.returncode == 0, result.stderr
    copies = [line for line in deployment.calls().splitlines() if line.startswith("rclone copy")]
    assert len(copies) == 2
    assert "/deploy/tiles/basemap " in copies[0]
    assert "/deploy/tiles/autogeoref " in copies[1]
    for call in copies:
        assert "--ignore-existing" in call
        assert "Cache-Control: public, max-age=31536000, immutable" in call
        assert "--s3-chunk-size 64M" in call
    assert "  from: vol_a.pmtiles" in deployment.calls()


def test_replace_overwrites_that_archive_and_purges_its_url(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.archive("autogeoref", "vol_b", 300)
    deployment.write_manifest(["vol_a", "vol_b"])
    deployment.publish({"vol_a": 180, "vol_b": 300})

    result = deployment.run(PUSH, "--city", str(deployment.city), "--replace", "vol_a")

    assert result.returncode == 0, result.stderr
    calls = deployment.calls().splitlines()
    upload = [i for i, line in enumerate(calls) if line.startswith("rclone copyto")]
    purge = [i for i, line in enumerate(calls) if "purge_cache" in line]
    assert len(upload) == 1 and len(purge) == 1
    # purging first would leave the edge to re-cache the old object
    assert upload[0] < purge[0]
    # the destination key, not just the name somewhere in the line: this is the
    # one path that deliberately overwrites a published object
    assert (
        calls[upload[0]] == f"rclone copyto {deployment.tiles}/autogeoref/vol_a.pmtiles"
        " r2:example-bucket/testcity/vol_a.pmtiles"
        " --header-upload Cache-Control: public, max-age=31536000, immutable"
        " --s3-chunk-size 64M --progress"
    )
    assert "https://tiles.example.com/testcity/vol_a.pmtiles" in calls[purge[0]]
    assert "vol_b" not in "\n".join(calls)


def test_a_failed_purge_is_not_a_successful_replace(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 180})

    result = deployment.run(
        PUSH, "--city", str(deployment.city), "--replace", "vol_a", STUB_PURGE_FAILS="1"
    )

    assert result.returncode == 1
    assert "stale at the edge" in result.stderr


def test_replace_can_skip_the_purge_only_out_loud(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 180})

    result = deployment.run(
        PUSH, "--city", str(deployment.city), "--replace", "vol_a", "--no-purge"
    )

    assert result.returncode == 0, result.stderr
    assert "rclone copyto" in deployment.calls()
    assert "purge_cache" not in deployment.calls()
    assert "NOT purging" in result.stderr


def test_a_dry_run_replace_needs_no_purge_credentials(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 180})

    result = deployment.run(
        PUSH,
        "--city",
        str(deployment.city),
        "--replace",
        "vol_a",
        "--dry-run",
        CLOUDFLARE_API_TOKEN="",
        AUTOGEOREF_CF_ZONE_ID="",
    )

    assert result.returncode == 0, result.stderr
    assert "REPLACE   vol_a.pmtiles" in result.stdout
    assert "copyto" not in deployment.calls()


def test_a_bucket_that_does_not_list_is_not_an_empty_prefix(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])

    result = deployment.run(PUSH, "--city", str(deployment.city), STUB_NO_BUCKET="1")

    assert result.returncode == 1
    assert "AUTOGEOREF_R2_BUCKET" in result.stderr
    assert "rclone copy" not in deployment.calls()


def test_two_archives_sharing_a_basename_are_refused(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.archive("basemap", "vol_a", 100)
    deployment.write_manifest(["vol_a"], basemap="vol_a")

    result = deployment.run(PUSH, "--city", str(deployment.city))

    assert result.returncode == 1
    assert "share the basename vol_a.pmtiles" in result.stderr
    assert deployment.calls() == ""


@pytest.mark.parametrize(
    "base",
    [
        "https://tiles.example.com/testcity",
        "https://tiles.example.com/testcity/",
        "https://tiles.example.com//testcity",
        "https://tiles.example.com/test//city/",
        "https://tiles.example.com",
    ],
)
def test_the_upload_prefix_matches_the_url_the_manifest_will_carry(
    deployment: Deployment, base: str
) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    # the archive's public URL is this base plus the basename, so the upload key
    # has to be the same prefix or the page asks for something nobody uploaded
    prefix = public_tiles_base(base).split("//", 1)[1].split("/", 1)
    expected = f"r2:example-bucket/{prefix[1]}/" if len(prefix) > 1 else "r2:example-bucket/"

    result = deployment.run(
        PUSH, "--city", str(deployment.city), "--dry-run", AUTOGEOREF_TILES_BASE=base
    )

    assert result.returncode == 0, result.stderr
    assert f"rclone lsf --format ps --separator | --files-only --include *.pmtiles {expected}" in (
        deployment.calls()
    )


def test_a_tiles_base_with_credentials_is_refused(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])

    result = deployment.run(
        PUSH,
        "--city",
        str(deployment.city),
        "--dry-run",
        AUTOGEOREF_TILES_BASE="https://user:pw@tiles.example.com/testcity",
    )

    assert result.returncode == 1
    assert "credentials" in result.stderr


def test_replace_without_purge_settings_fails_before_uploading(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 180})

    result = deployment.run(
        PUSH, "--city", str(deployment.city), "--replace", "vol_a", CLOUDFLARE_API_TOKEN=""
    )

    assert result.returncode == 1
    assert "CLOUDFLARE_API_TOKEN" in result.stderr
    assert deployment.calls() == ""


def test_replace_naming_no_archive_is_a_typo_not_a_no_op(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})

    result = deployment.run(PUSH, "--city", str(deployment.city), "--replace", "vol_typo")

    assert result.returncode == 1
    assert "vol_typo" in result.stderr


def test_a_manifest_naming_an_absent_archive_is_refused(deployment: Deployment) -> None:
    deployment.write_manifest(["vol_a"])
    deployment.publish({})

    result = deployment.run(PUSH, "--city", str(deployment.city))

    assert result.returncode != 0
    assert "not on disk" in result.stderr
    assert deployment.calls() == ""


def test_a_missing_setting_is_named_before_any_upload(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])

    result = deployment.run(PUSH, "--city", str(deployment.city), AUTOGEOREF_R2_BUCKET="")

    assert result.returncode == 1
    assert "AUTOGEOREF_R2_BUCKET" in result.stderr
    assert deployment.calls() == ""


def test_settings_come_from_the_env_file_but_the_environment_wins(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})
    env_file = deployment.root / "deploy.env"
    env_file.write_text(
        "# a comment\n\nexport AUTOGEOREF_R2_BUCKET=from-the-file\n"
        'AUTOGEOREF_R2_REMOTE="from-the-file"\n'
    )

    result = deployment.run(
        PUSH,
        "--city",
        str(deployment.city),
        "--dry-run",
        AUTOGEOREF_ENV_FILE=str(env_file),
        AUTOGEOREF_R2_BUCKET="",
        AUTOGEOREF_R2_REMOTE="from-the-environment",
    )

    assert result.returncode == 0, result.stderr
    # the bucket was unset, so it came from the file; the remote was set for
    # this run, so the file's value for it is ignored
    assert "from-the-environment:from-the-file/testcity/" in deployment.calls()


def test_an_env_file_saved_on_the_windows_side_still_reads(deployment: Deployment) -> None:
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    deployment.publish({"vol_a": 200})
    env_file = deployment.root / "crlf.env"
    env_file.write_bytes(b"AUTOGEOREF_R2_BUCKET=crlf-bucket\r\n")

    result = deployment.run(
        PUSH,
        "--city",
        str(deployment.city),
        "--dry-run",
        AUTOGEOREF_ENV_FILE=str(env_file),
        AUTOGEOREF_R2_BUCKET="",
    )

    assert result.returncode == 0, result.stderr
    assert "r2:crlf-bucket/testcity/" in deployment.calls()


def test_a_missing_project_interpreter_stops_with_its_own_message(
    deployment: Deployment,
) -> None:
    """The refusal has to be the thing that stops the run, not the thing before it.

    `die` inside a helper reached through `$(...)` exits only that subshell, and
    `set -e` does not fire on a failed assignment in a function already running
    inside one. Left alone, the script ran on and tripped over an empty command,
    reporting 127 and a shell error on top of the message that explains the fix.
    """
    deployment.archive("autogeoref", "vol_a", 200)
    deployment.write_manifest(["vol_a"])
    bare = deployment.scripts_without_an_environment()

    result = deployment.run(bare / "push_archives.sh", "--city", str(deployment.city))

    assert result.returncode == 1, result.stderr
    assert "build the environment with make setup" in result.stderr
    assert "command not found" not in result.stderr
