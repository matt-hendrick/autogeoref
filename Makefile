.DEFAULT_GOAL := help

UV ?= uv
PORT ?= 8123
CITY ?= configs/chicago/chicago.toml
TEST_WORKERS ?= 8
UV_ALL = $(UV) run --all-extras

.PHONY: help setup setup-js test-fast test test-golden test-gdal test-file coverage coverage-fast \
	lint lint-js lint-py docstrings module-sizes typecheck check status package-smoke candidates \
	prep report viewer

help:
	@printf '%s\n' \
		'Run from the Linux/WSL checkout, then bootstrap with make setup.' \
		'' \
		'Common development commands:' \
		'  make setup      Install the locked development environment.' \
		'  make setup-js   Install the frontend linter (dev only; never deployed).' \
		'  make test-fast  Run the fast test suite.' \
		'  make test       Run the complete test suite in 8 workers.' \
		'  make test-golden Run golden tests (requires fixtures).' \
		'  make test-gdal  Run GDAL tests.' \
		'  make test-file TEST=tests/test_queue_drain.py  Run one test file.' \
		'  make coverage   Run the complete suite with coverage (term + html + xml reports).' \
		'  make coverage-fast  Run the fast suite with coverage and the CI floor.' \
		'  make lint       Run Ruff, the docstring and module-size budgets, the import contracts, and lint-js.' \
		'  make lint-py    The Python half of lint alone (no npm; what the container runs).' \
		'  make lint-js    Run ESLint over the hand-written frontend (needs make setup-js).' \
		'  make docstrings Report every docstring-budget site, experiments included.' \
		'  make module-sizes  Report every module size and its headroom.' \
		'  make typecheck  Run strict mypy over the source and the live scripts.' \
		'  make check      Run lint, typecheck, and the fast suite.' \
		'  make package-smoke  Build the wheel and smoke the installed console script.' \
		'  make status     Show filesystem-derived processing status.' \
		'  make candidates CITY=configs/chicago/chicago.toml  List runnable volumes.' \
		'  make prep VOLUME=sanborn01790_024  Prepare one volume.' \
		'  make report VOLUME=sanborn01790_024  Rebuild one volume report.' \
		'  make viewer     Serve the local viewer (PORT=8123 by default).' \
		'' \
		'Model-spending runs, queue drains, review, and deployment stay explicit.' \
		'See docs/OPERATIONS.md.'

# --python is redundant beside .python-version and stated anyway: CI, the
# container and a contributor's checkout must resolve the same interpreter, and
# an unpinned one diverges silently.
setup:
	$(UV) sync --all-extras --dev --python 3.12

# Dev-only, and separate from `make setup` on purpose: this is a Python
# project, npm is not required to run or deploy anything, and the deploy bundle
# is file copies that no toolchain can break. See package.json.
setup-js:
	npm ci

# No `-q` on any pytest line below: pyproject already sets it in addopts, and a
# second one is `-qq`, which suppresses the summary. Every target then ends
# mid-dots and a green run is indistinguishable from a truncated one.
test-fast:
	$(UV_ALL) pytest -m "not golden and not gdal"

test:
	$(UV_ALL) pytest -n $(TEST_WORKERS) --dist loadfile

coverage:
	$(UV_ALL) pytest -n $(TEST_WORKERS) --dist loadfile \
		--cov=autogeoref --cov-report=term --cov-report=html --cov-report=xml

# Erosion backstop, not a precision gate. CI skips fixture-dependent tests and
# measures lowest; keep the floor ~2pts under CI's number when raising it.
coverage-fast:
	$(UV_ALL) pytest -m "not golden and not gdal" \
		--cov=autogeoref --cov-report=term --cov-report=html --cov-report=xml \
		--cov-fail-under=75

test-golden:
	$(UV_ALL) pytest -m golden

test-gdal:
	$(UV_ALL) pytest -m gdal

test-file:
	@test -n "$(TEST)" || { printf '%s\n' 'TEST is required: make test-file TEST=tests/test_queue_drain.py' >&2; exit 2; }
	$(UV_ALL) pytest $(TEST)

LINT_PATHS = src tests scripts

lint: lint-js lint-py

# Split out so the cold-clone container can lint without npm. Nothing in that
# image runs the frontend, and installing node there would put a host-shaped
# toolchain into the one check whose whole point is starting with nothing.
lint-py:
	$(UV_ALL) ruff check $(LINT_PATHS)
	$(UV_ALL) ruff format --check $(LINT_PATHS)
	$(UV_ALL) python scripts/lint/check_docstrings.py
	$(UV_ALL) python scripts/lint/check_module_size.py
	$(UV_ALL) python scripts/lint/check_symbol_reachability.py
	PYTHONPATH=scripts/lint $(UV_ALL) lint-imports

# Invoked through node rather than node_modules/.bin: on a Windows-hosted
# checkout npm writes .cmd shims there with no exec bit, and they cannot run.
lint-js:
	@test -x "$$(command -v node)" || { printf '%s\n' 'node is not installed; the frontend linter needs it' >&2; exit 2; }
	@test -f node_modules/eslint/bin/eslint.js || { printf '%s\n' 'frontend linter not installed: run make setup-js' >&2; exit 2; }
	node node_modules/eslint/bin/eslint.js .

docstrings:
	$(UV_ALL) python scripts/lint/check_docstrings.py --report --include-experiments

module-sizes:
	$(UV_ALL) python scripts/lint/check_module_size.py --report

# One strict pass over the package AND the live script tiers (pyproject.toml).
# It is what catches a script left behind by an API change, which ruff misses.
typecheck:
	$(UV_ALL) mypy

check: lint typecheck test-fast

package-smoke:
	$(UV) run --no-project --python 3.12 python scripts/wheel_smoke.py

status:
	$(UV) run autogeoref status

candidates:
	$(UV) run autogeoref queue --candidates --city $(CITY)

prep:
	@test -n "$(VOLUME)" || { printf '%s\n' 'VOLUME is required: make prep VOLUME=sanborn01790_024' >&2; exit 2; }
	$(UV) run autogeoref prep $(VOLUME) --work work

report:
	@test -n "$(VOLUME)" || { printf '%s\n' 'VOLUME is required: make report VOLUME=sanborn01790_024' >&2; exit 2; }
	$(UV) run autogeoref report $(VOLUME) --work work

viewer:
	$(UV) run python scripts/serve_viewer.py $(PORT)
