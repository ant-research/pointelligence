# Pre-compiled CUDA wheel build & upload pipeline
#
# Usage:
#   make build-and-upload   # Build wheel if needed, upload to OSS
#   make wheel              # Build wheel locally
#   make upload             # Upload to OSS
#   make check-remote       # Check if wheel exists on OSS
#   make source-hash        # Print source hash
#   make cache-key          # Print full wheel filename
#   make clean              # Remove build artifacts

SHELL := /bin/bash
.ONESHELL:

OSS_PREFIX := oss://antsys-robbys3-prod/pointelligence/wheels
CUDA_ARCH  := 9.0+PTX
OSSUTIL    := ossutil64
EXT_DIR    := $(shell pwd)

# Override torch version for cross-building (e.g., local torch 2.11 but cluster has 2.9):
#   make wheel TORCH_VERSION_OVERRIDE=2.9
TORCH_VERSION_OVERRIDE ?=

# ── Derived values ──────────────────────────────────────────────

# Source hash: SHA256 of all CUDA/C++ source + setup.py, truncated to 8 chars
SOURCE_HASH := $(shell (find sparse_engines_cuda/csrc -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' \) | sort | xargs sha256sum && sha256sum setup.py) | sha256sum | cut -c1-8)

# Torch version: major.minor (override with TORCH_VERSION_OVERRIDE for cross-building)
TORCH_VERSION := $(or $(TORCH_VERSION_OVERRIDE),$(shell python -c "import torch; v=torch.__version__.split('.')[:2]; print(f'{v[0]}.{v[1]}')" 2>/dev/null || echo "unknown"))

# CUDA arch flattened: "9.0+PTX" -> "90"
CUDA_ARCH_FLAT := $(shell echo "$(CUDA_ARCH)" | tr -d '.+' | sed 's/PTX//')

# Full wheel filename
WHEEL_NAME := sparse_engines_cuda-0.0.1-torch$(TORCH_VERSION)-cu$(CUDA_ARCH_FLAT)-$(SOURCE_HASH)-cp39-abi3-linux_x86_64.whl
WHEEL_PATH := dist/$(WHEEL_NAME)
OSS_KEY    := $(OSS_PREFIX)/$(WHEEL_NAME)

# ── Targets ─────────────────────────────────────────────────────

.PHONY: source-hash cache-key wheel upload check-remote build-and-upload clean

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

upload: $(WHEEL_PATH)
	@echo "Uploading $(WHEEL_NAME) to OSS..."
	$(OSSUTIL) cp "$(WHEEL_PATH)" "$(OSS_KEY)" -f
	@echo "Uploaded: $(OSS_KEY)"

check-remote:
	@if $(OSSUTIL) stat "$(OSS_KEY)" >/dev/null 2>&1; then \
		echo "EXISTS: $(OSS_KEY)"; \
	else \
		echo "MISSING: $(OSS_KEY)"; \
		exit 1; \
	fi

build-and-upload:
	@if $(OSSUTIL) stat "$(OSS_KEY)" >/dev/null 2>&1; then \
		echo "Wheel already on OSS, skipping build: $(WHEEL_NAME)"; \
	else \
		echo "Wheel not found on OSS, building..."; \
		$(MAKE) wheel; \
		$(MAKE) upload; \
	fi

clean:
	rm -rf dist/ build/ *.egg-info
	@echo "Cleaned build artifacts"
