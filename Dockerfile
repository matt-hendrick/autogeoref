# Cold-clone verification image. Build from the repo root; the context must be
# the tracked tree only (.dockerignore is load-bearing, not hygiene).
#
# ubuntu:24.04 rather than python:3.12-slim: 24.04 ships Python 3.12 *and*
# GDAL 3.8, the pair the development box runs, so a failure here is the
# project's fault rather than the base image's.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# gdal-bin carries gdalwarp/gdal_translate/gdalinfo/gdalbuildvrt AND
# gdal2tiles.py on 24.04. libgomp1/libglib2.0-0t64 are what
# opencv-python-headless and scikit-image want at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git make gdal-bin libgomp1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

# Run as a normal user. As root, the test asserting a PermissionError on a
# 0o444 result file gets no error and fails. 24.04 already ships a uid-1000
# `ubuntu`; reuse it rather than adding one, which collides.
# `install -d` before WORKDIR: WORKDIR would create /app owned by root, and
# COPY --chown sets ownership only on what it copies, not on the parent. The
# tree must be writable — the suite creates cache/ in the working directory.
RUN install -d -o ubuntu -g ubuntu /app
WORKDIR /app
COPY --chown=ubuntu:ubuntu . /app
USER ubuntu

RUN uv sync --locked --all-extras --dev --python 3.12

# Pre-warm the build backend into uv's cache, so package-smoke's isolated wheel
# build does not resolve hatchling over the network on every run.
RUN uv build --wheel --out-dir /tmp/prewarm && rm -rf /tmp/prewarm

# Without this, every `uv run` revalidates the project, rebuilds it, and
# re-resolves over the network. It warns harmlessly under `uv run --no-project`.
ENV UV_NO_SYNC=1

# lint-py, not lint: `lint` also runs the npm-installed frontend linter, which
# this image has no node for and should not — see the Makefile.
CMD ["bash","-lc","set -eux; gdalinfo --version; \
  uv run python -c 'import autogeoref, cv2, pmtiles, anthropic'; \
  make lint-py typecheck test-fast package-smoke; \
  uv run autogeoref status; echo COLD-START-OK"]
