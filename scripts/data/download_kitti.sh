#!/bin/bash
# Download/prepare KITTI Odometry dataset for FCGF training.
# KITTI requires manual registration at http://www.cvlibs.net/datasets/kitti/eval_odometry.php
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${POINTELLIGENCE_DATA:-$REPO_ROOT/data}"
OUTPUT_DIR="$DATA_DIR/kitti_odometry"
PROXY_PORT="${ANTFAMILY_PORT:-13659}"

echo "=== KITTI Odometry Dataset Preparation ==="
echo "  Output: $OUTPUT_DIR"

usage() {
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --velodyne-tar PATH   Path to downloaded data_odometry_velodyne.zip"
    echo "  --calib-tar PATH      Path to downloaded data_odometry_calib.zip"
    echo "  --poses-tar PATH      Path to downloaded data_odometry_labels.zip (or poses)"
    echo "  --raw-dir PATH        Path to already-extracted KITTI odometry dir"
    echo ""
    echo "KITTI requires manual download from:"
    echo "  http://www.cvlibs.net/datasets/kitti/eval_odometry.php"
    echo ""
    echo "Download these files:"
    echo "  1. data_odometry_velodyne.zip  (~80GB) - velodyne point clouds"
    echo "  2. data_odometry_calib.zip     (~1MB)  - calibration files"
    echo "  3. data_odometry_labels.zip    or poses from devkit"
    echo ""
    echo "Alternatively, use --raw-dir if you already have the extracted data."
}

VELODYNE_TAR=""
CALIB_TAR=""
POSES_TAR=""
RAW_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --velodyne-tar) VELODYNE_TAR="$2"; shift 2 ;;
        --calib-tar) CALIB_TAR="$2"; shift 2 ;;
        --poses-tar) POSES_TAR="$2"; shift 2 ;;
        --raw-dir) RAW_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

if [ -n "$RAW_DIR" ]; then
    echo "Using existing data at: $RAW_DIR"
    if [ ! -d "$RAW_DIR/sequences" ] && [ ! -d "$RAW_DIR/dataset/sequences" ]; then
        echo "ERROR: Expected sequences/ or dataset/sequences/ under $RAW_DIR"
        exit 1
    fi
    # Symlink to expected structure
    if [ -d "$RAW_DIR/dataset/sequences" ]; then
        ln -sfn "$RAW_DIR/dataset" "$OUTPUT_DIR/dataset"
    elif [ -d "$RAW_DIR/sequences" ]; then
        mkdir -p "$OUTPUT_DIR/dataset"
        ln -sfn "$RAW_DIR/sequences" "$OUTPUT_DIR/dataset/sequences"
    fi
else
    if [ -z "$VELODYNE_TAR" ]; then
        echo "ERROR: No data source specified."
        usage
        exit 1
    fi

    echo "Extracting velodyne data..."
    cd "$OUTPUT_DIR"
    unzip -q -o "$VELODYNE_TAR" -d .

    if [ -n "$CALIB_TAR" ]; then
        echo "Extracting calibration data..."
        unzip -q -o "$CALIB_TAR" -d .
    fi

    if [ -n "$POSES_TAR" ]; then
        echo "Extracting poses..."
        unzip -q -o "$POSES_TAR" -d .
    fi
fi

# Download ground truth poses if not present (these are publicly available)
POSES_URL="https://www.cvlibs.net/datasets/kitti/data_odometry_poses.zip"
SEQ_DIR="$OUTPUT_DIR/dataset/sequences"
if [ -d "$SEQ_DIR/00" ] && [ ! -f "$SEQ_DIR/00/poses.txt" ]; then
    echo "Downloading ground truth poses..."
    POSES_ZIP="$OUTPUT_DIR/data_odometry_poses.zip"
    if [ ! -f "$POSES_ZIP" ]; then
        ALL_PROXY="socks5h://127.0.0.1:$PROXY_PORT" \
            curl -L -o "$POSES_ZIP" "$POSES_URL" 2>/dev/null || \
            curl -L -o "$POSES_ZIP" "$POSES_URL"
    fi
    unzip -q -o "$POSES_ZIP" -d "$OUTPUT_DIR/poses_tmp"
    # Copy poses into sequence dirs
    for f in "$OUTPUT_DIR"/poses_tmp/dataset/poses/*.txt; do
        SEQ=$(basename "$f" .txt)
        if [ -d "$SEQ_DIR/$SEQ" ]; then
            cp "$f" "$SEQ_DIR/$SEQ/poses.txt"
        fi
    done
    rm -rf "$OUTPUT_DIR/poses_tmp"
fi

# Verify
echo ""
echo "Verifying directory structure..."
ERRORS=0
TOTAL_SEQS=0
for seq in $(seq -w 0 10); do
    SEQ_PATH="$SEQ_DIR/$seq"
    if [ ! -d "$SEQ_PATH/velodyne" ]; then
        echo "  MISSING: $SEQ_PATH/velodyne"
        ERRORS=$((ERRORS + 1))
    else
        NFRAMES=$(ls "$SEQ_PATH/velodyne"/*.bin 2>/dev/null | wc -l)
        HAS_POSES="no"
        [ -f "$SEQ_PATH/poses.txt" ] && HAS_POSES="yes"
        echo "  OK: seq $seq ($NFRAMES frames, poses=$HAS_POSES)"
        TOTAL_SEQS=$((TOTAL_SEQS + 1))
    fi
done

if [ $ERRORS -gt 0 ]; then
    echo "WARNING: $ERRORS sequences missing velodyne data"
fi

echo ""
echo "=== KITTI preparation complete ==="
echo "  Path: $OUTPUT_DIR"
echo "  Sequences: $TOTAL_SEQS"
echo "  Set: export KITTI_PATH=$OUTPUT_DIR"
