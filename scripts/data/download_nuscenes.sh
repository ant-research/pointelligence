#!/bin/bash
# Prepare NuScenes dataset for Pointcept training.
# NuScenes requires a license agreement at https://www.nuscenes.org/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${POINTELLIGENCE_DATA:-$REPO_ROOT/data}"
OUTPUT_DIR="$DATA_DIR/nuscenes_processed"
POINTCEPT_DIR="$REPO_ROOT/examples/Pointcept"

echo "=== NuScenes Dataset Preparation ==="

usage() {
    echo ""
    echo "Usage: $0 --raw-dir PATH [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --raw-dir PATH       Path to raw NuScenes dataset (required)"
    echo "                       Must contain: v1.0-trainval/, samples/, sweeps/, lidarseg/"
    echo "  --max-sweeps N       Number of sweeps to aggregate (default: 10)"
    echo "  --skip-preprocess    Skip preprocessing, just create symlinks"
    echo ""
    echo "NuScenes must be downloaded manually from https://www.nuscenes.org/"
    echo "Required files:"
    echo "  - Full dataset (v1.0 trainval): ~300GB"
    echo "  - lidarseg annotations"
}

RAW_DIR=""
MAX_SWEEPS=10
SKIP_PREPROCESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --raw-dir) RAW_DIR="$2"; shift 2 ;;
        --max-sweeps) MAX_SWEEPS="$2"; shift 2 ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ -z "$RAW_DIR" ]; then
    echo "ERROR: --raw-dir is required"
    usage
    exit 1
fi

RAW_DIR="$(cd "$RAW_DIR" && pwd)"

# Verify raw data
echo "Checking raw NuScenes data at: $RAW_DIR"
ERRORS=0
for subdir in v1.0-trainval samples sweeps; do
    if [ ! -d "$RAW_DIR/$subdir" ]; then
        echo "  MISSING: $subdir/"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: $subdir/"
    fi
done
if [ -d "$RAW_DIR/lidarseg" ]; then
    echo "  OK: lidarseg/"
else
    echo "  WARN: lidarseg/ not found (needed for semantic segmentation)"
fi

if [ $ERRORS -gt 0 ]; then
    echo "ERROR: Raw NuScenes data is incomplete"
    exit 1
fi

# Create output dir and symlink raw data
mkdir -p "$OUTPUT_DIR"
ln -sfn "$RAW_DIR" "$OUTPUT_DIR/raw"
echo "  Symlinked: $OUTPUT_DIR/raw -> $RAW_DIR"

if [ "$SKIP_PREPROCESS" = true ]; then
    echo "Skipping preprocessing (--skip-preprocess)"
else
    # Check for nuscenes-devkit
    if ! python -c "from nuscenes.nuscenes import NuScenes" 2>/dev/null; then
        echo "Installing nuscenes-devkit..."
        pip install nuscenes-devkit
    fi

    PREPROCESS_SCRIPT="$POINTCEPT_DIR/pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py"
    if [ ! -f "$PREPROCESS_SCRIPT" ]; then
        echo "ERROR: Preprocessing script not found: $PREPROCESS_SCRIPT"
        echo "Make sure examples/Pointcept is on the pointcnnpp-version branch"
        exit 1
    fi

    echo ""
    echo "Running NuScenes preprocessing (max_sweeps=$MAX_SWEEPS)..."
    echo "  Input:  $RAW_DIR"
    echo "  Output: $OUTPUT_DIR"

    PYTHONPATH="$REPO_ROOT:$POINTCEPT_DIR:$PYTHONPATH" \
        python "$PREPROCESS_SCRIPT" \
            --dataset_root "$RAW_DIR" \
            --output_root "$OUTPUT_DIR" \
            --max_sweeps "$MAX_SWEEPS"
fi

# Verify output
echo ""
echo "Verifying preprocessed data..."
INFO_DIR="$OUTPUT_DIR/info"
if [ -d "$INFO_DIR" ]; then
    echo "  Info files:"
    ls -lh "$INFO_DIR"/*.pkl 2>/dev/null || echo "  WARN: No .pkl files in $INFO_DIR"
else
    echo "  WARN: No info/ directory found"
fi

echo ""
echo "=== NuScenes preparation complete ==="
echo "  Processed path: $OUTPUT_DIR"
echo "  Use in config:  data_root = \"$OUTPUT_DIR\""
