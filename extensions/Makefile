# Pre-compiled CUDA wheel build pipeline
#
# Usage:
#   make wheel             # Build wheel into ./dist/
#   make source-hash       # Print source hash
#   make cache-key         # Print full wheel filename (hash-keyed)
#   make clean             # Remove build artifacts
#
# This is the open-source-friendly build half. The internal "upload to
# Alipay OSS + check-remote + build-and-upload-or-skip" wrapper lives in
# internal cluster-ops infrastructure (separate from this repo).

SHELL := /bin/bash
.ONESHELL:

CUDA_ARCH := 9.0+PTX
EXT_DIR   := $(shell pwd)

# Override torch version for cross-building (e.g., local torch 2.11 but cluster has 2.9):
#   make wheel TORCH_VERSION_OVERRIDE=2.9
TORCH_VERSION_OVERRIDE ?=

# ── Derived values ──────────────────────────────────────────────

# Source hash: SHA256 of all CUDA/C++ source + setup.py, truncated to 8 chars.
# The hash is the cache key for wheel reuse — bumping any source bumps the key.
SOURCE_HASH := $(shell (find sparse_engines_cuda/csrc -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' \) | sort | xargs sha256sum && sha256sum setup.py) | sha256sum | cut -c1-8)

# Torch version: major.minor (override with TORCH_VERSION_OVERRIDE for cross-building)
TORCH_VERSION := $(or $(TORCH_VERSION_OVERRIDE),$(shell python -c "import torch; v=torch.__version__.split('.')[:2]; print(f'{v[0]}.{v[1]}')" 2>/dev/null || echo "unknown"))

# CUDA arch flattened: "9.0+PTX" -> "90"
CUDA_ARCH_FLAT := $(shell echo "$(CUDA_ARCH)" | tr -d '.+' | sed 's/PTX//')

# Full wheel filename (cache-keyed by source hash + torch + cuda)
WHEEL_NAME := sparse_engines_cuda-0.0.1-torch$(TORCH_VERSION)-cu$(CUDA_ARCH_FLAT)-$(SOURCE_HASH)-cp39-abi3-linux_x86_64.whl
WHEEL_PATH := dist/$(WHEEL_NAME)

# ── Targets ─────────────────────────────────────────────────────

.PHONY: source-hash cache-key wheel clean

source-hash:
	@echo $(SOURCE_HASH)

cache-key:
	@echo $(WHEEL_NAME)

wheel: $(WHEEL_PATH)

$(WHEEL_PATH):
	@echo "Building wheel: $(WHEEL_NAME)"
	@echo "  torch=$(TORCH_VERSION) cuda_arch=$(CUDA_ARCH) src_hash=$(SOURCE_HASH)"
	mkdir -p dist
	TORCH_CUDA_ARCH_LIST="$(CUDA_ARCH)" pip wheel --no-build-isolation --no-deps -w dist/ .
	@# Rename from generic name to cache-keyed name
	@generic=$$(ls -1 dist/sparse_engines_cuda-*.whl 2>/dev/null | grep -v "torch" | head -1); \
	if [ -n "$$generic" ] && [ "$$generic" != "$(WHEEL_PATH)" ]; then \
		mv "$$generic" "$(WHEEL_PATH)"; \
	fi
	@echo "Built: $(WHEEL_PATH)"

clean:
	rm -rf dist/ build/ *.egg-info
	@echo "Cleaned build artifacts"
