# =============================================================================
# RAG Study Buddy — Backend Makefile
# =============================================================================
# Usage (from the backend/ directory):
#
#   make install          Install all dependencies via Poetry
#   make run              Start the FastAPI dev server (hot-reload)
#   make test             Run the full pytest suite
#   make test-fast        Run tests, stop on first failure (-x)
#   make test-cov         Run tests with terminal coverage report
#   make lint             Run ruff linter (auto-fix safe issues)
#   make typecheck        Run mypy static type checker
#   make eval             Run the offline evaluation harness (dummy providers)
#   make index-sample     POST a sample PDF to the running server for manual QA
#   make clean            Remove __pycache__, .pytest_cache, coverage artefacts
#
# All targets assume you are inside backend/ and Poetry is on $PATH.
# Install Poetry once with:
#   curl -sSL https://install.python-poetry.org | python3 -
# =============================================================================

SHELL := /bin/bash
POETRY := poetry
PYTHON := $(POETRY) run python
PYTEST := $(POETRY) run pytest
APP_MODULE := app.main:app
HOST := 0.0.0.0
PORT := 8000

# --------------------------------------------------------------------------- #
# Default target                                                                #
# --------------------------------------------------------------------------- #
.DEFAULT_GOAL := help

.PHONY: help
help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Dependencies                                                                  #
# --------------------------------------------------------------------------- #
.PHONY: install
install:       ## Install all dependencies (prod + dev) via Poetry
	$(POETRY) install
	@echo "✓ Dependencies installed"

.PHONY: install-prod
install-prod:  ## Install production dependencies only (no dev extras)
	$(POETRY) install --only main
	@echo "✓ Production dependencies installed"

# --------------------------------------------------------------------------- #
# Dev server                                                                    #
# --------------------------------------------------------------------------- #
.PHONY: run
run:           ## Start FastAPI dev server with hot-reload
	$(POETRY) run uvicorn $(APP_MODULE) \
		--reload \
		--host $(HOST) \
		--port $(PORT)

.PHONY: run-prod
run-prod:      ## Start FastAPI with multiple workers (no reload)
	$(POETRY) run uvicorn $(APP_MODULE) \
		--host $(HOST) \
		--port $(PORT) \
		--workers 2

# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #
.PHONY: test
test:          ## Run the full pytest suite
	$(PYTEST) app/tests/ -v

.PHONY: test-fast
test-fast:     ## Run tests — stop at first failure
	$(PYTEST) app/tests/ -v -x

.PHONY: test-cov
test-cov:      ## Run tests with terminal + HTML coverage report
	$(PYTEST) app/tests/ \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v
	@echo "HTML coverage report → backend/htmlcov/index.html"

.PHONY: test-file
test-file:     ## Run a single test file  (usage: make test-file FILE=app/tests/test_chat.py)
	$(PYTEST) $(FILE) -v

# --------------------------------------------------------------------------- #
# Lint / type-check                                                             #
# --------------------------------------------------------------------------- #
.PHONY: lint
lint:          ## Run ruff linter and auto-fix safe issues
	@$(POETRY) run ruff check app/ --fix --show-fixes || true
	@echo "✓ Lint complete"

.PHONY: lint-check
lint-check:    ## Run ruff in check-only mode (no fixes — for CI)
	$(POETRY) run ruff check app/

.PHONY: typecheck
typecheck:     ## Run mypy static type checker
	$(POETRY) run mypy app/ --ignore-missing-imports

# --------------------------------------------------------------------------- #
# Evaluation harness                                                            #
# --------------------------------------------------------------------------- #
.PHONY: eval
eval:          ## Run the offline evaluation harness (no API keys needed)
	EMBEDDINGS_PROVIDER=dummy \
	LLM_PROVIDER=dummy \
	$(PYTHON) -m app.eval.eval_runner
	@echo "✓ Eval report written → eval_reports/report.json"

# --------------------------------------------------------------------------- #
# Sample data helper (requires running server)                                  #
# --------------------------------------------------------------------------- #
.PHONY: index-sample
index-sample:  ## Upload + index a sample PDF for manual QA (server must be running)
	@if [ -z "$(PDF)" ]; then \
		echo "Usage: make index-sample PDF=/path/to/file.pdf WS=1"; \
		exit 1; \
	fi
	@WS_ID=$${WS:-1}; \
	echo "→ Uploading $(PDF) to workspace $$WS_ID …"; \
	DOC_ID=$$(curl -s -X POST \
		"http://$(HOST):$(PORT)/api/v1/workspaces/$$WS_ID/documents/upload" \
		-F "file=@$(PDF);type=application/pdf" \
		| python3 -c "import sys,json; print(json.load(sys.stdin)['document_id'])"); \
	echo "  Uploaded document_id=$$DOC_ID"; \
	echo "→ Indexing document_id=$$DOC_ID …"; \
	curl -s -X POST \
		"http://$(HOST):$(PORT)/api/v1/documents/$$DOC_ID/index" \
		| python3 -m json.tool; \
	echo "✓ Done — document_id=$$DOC_ID is indexed"

# --------------------------------------------------------------------------- #
# Clean                                                                         #
# --------------------------------------------------------------------------- #
.PHONY: clean
clean:         ## Remove build artefacts, caches, and coverage output
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Clean complete"