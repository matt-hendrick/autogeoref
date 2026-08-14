"""No test may spawn a real model CLI: the conftest tripwires block every
binary in the provider registry, and parametrizing over the registry extends
the proof to each new backend automatically."""

import subprocess

import pytest

from autogeoref.annotate.providers import CLI_PROVIDERS


@pytest.mark.parametrize("provider", sorted(CLI_PROVIDERS))
def test_every_registered_backend_spawn_trips(provider: str) -> None:
    binary = CLI_PROVIDERS[provider].default_executable
    with pytest.raises(AssertionError, match="no test may spend"):
        subprocess.run([binary, "--version"])


def test_non_model_subprocesses_pass_through() -> None:
    proc = subprocess.run(["true"], check=False)
    assert proc.returncode == 0
