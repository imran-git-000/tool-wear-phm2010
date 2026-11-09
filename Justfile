set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

src_dir := "src"
REPOSITORY_NAME := "tool-wear-phm2010"

# Show the available recipes.
default:
    just --list

# == USEFUL COMMANDS


# Serve TensorBoard over the artifacts directory (one run per channel).
board dir="reports":
    uv run tensorboard --logdir {{dir}}


# == SETUP REPOSITORY AND DEPENDENCIES

# Install the repository git hooks into .git/hooks.
set-hooks:
    cp .hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    cp .hooks/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    cp .hooks/post-merge .git/hooks/post-merge
    chmod +x .git/hooks/post-merge

# Create or update the virtual environment. The project is relocked before syncing. This installs all extras and the development group.
dev-sync:
    uv sync --cache-dir .uv_cache --all-extras

# Sync environment as in dev-sync but also refreshes a package, which might be a local version.
dev-sync-refresh-package lib:
    uv sync --cache-dir .uv_cache --all-extras --refresh-package {{lib}} --refresh-install {{lib}}

# Install hooks and sync the development environment.
setup: set-hooks dev-sync

# === CODE VALIDATION

# Format source and test files with Ruff.
format:
    uv run ruff format {{src_dir}} scripts tests

# Check whether formatting changes would be required.
format-on-commit:
    uv run ruff format {{src_dir}} --exit-non-zero-on-format

# Run linting with Ruff autofix and type checks with ty.
lint:
    uv run ruff check {{src_dir}} --fix
    uv run ty check {{src_dir}}
    uv run ruff check scripts --fix
    uv run ty check scripts
    uv run ruff check tests --fix
    uv run ty check tests

# Run non-mutating lint and type checks.
lint-on-push:
    uv run ruff check {{src_dir}} tests
    uv run ty check {{src_dir}}

# Run the test suite with coverage and xdist.
test:
    uv run pytest --verbose --color=yes --cov={{src_dir}} --exitfirst -n auto

# Run formatting, linting, and tests.
all-validation: format lint test
