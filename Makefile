# OrganizeMe — dev workflow.
#
#   make            # setup venv + build native extension (the common path)
#   make setup      # create .venv and install Python deps via uv
#   make native     # build the Rust extension into .venv (maturin develop)
#   make test       # run the test suite
#   make mcp        # run the MCP server over stdio
#   make clean      # remove build artifacts and caches
#
# The native build needs cargo on PATH. brew's rustup installs no ~/.cargo/bin
# shims, so we resolve the toolchain via `rustup which cargo`.

VENV ?= .venv
PY   := $(VENV)/bin/python

# Directory holding cargo/rustc (e.g. ~/.rustup/toolchains/<triple>/bin).
CARGO_DIR := $(shell rustup which cargo 2>/dev/null | xargs dirname 2>/dev/null)
ifneq ($(CARGO_DIR),)
export PATH := $(CARGO_DIR):$(PATH)
endif

.PHONY: all setup native build test mcp run clean tools

all: setup native

## setup: create the venv and install Python dependencies (uv).
# --inexact so syncing never prunes the editable native extension (it isn't in
# the lockfile); `make native` manages that separately.
setup:
	uv sync --inexact

## native: build the Rust extension into the venv.
# `maturin develop` refuses to run when both VIRTUAL_ENV and CONDA_PREFIX are
# set, so we drop CONDA_PREFIX and point it explicitly at this project's venv.
native:
	@command -v cargo >/dev/null 2>&1 || { echo "error: cargo not found. Run 'rustup default stable' (and ensure rustup is installed)."; exit 1; }
	@command -v maturin >/dev/null 2>&1 || { echo "error: maturin not found. Install with 'uv pip install maturin' or 'brew install maturin'."; exit 1; }
	cd native && env -u CONDA_PREFIX VIRTUAL_ENV="$(CURDIR)/$(VENV)" maturin develop --release

## build: alias for a full setup + native build.
build: all

## test: run the test suite in the venv.
test:
	$(PY) -m pytest -q

## mcp: run the MCP server (stdio).
mcp:
	$(PY) -m mcp_server.server

## run: launch the interactive CLI.
run:
	$(PY) main.py

## clean: remove Rust build artifacts and Python caches.
clean:
	rm -rf native/target
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
