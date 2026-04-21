#!/bin/bash
# Download 3DMatch dataset for FCGF training.
# Uses antfamily SOCKS5 proxy for Google Drive access.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${POINTELLIGENCE_DATA:-$REPO_ROOT/data}"
OUTPUT_DIR="$DATA_DIR/3dmatch_processed"

GDRIVE_ID="1zsZbJSID5AL4diJuhC0gZDYJsz-PidhH"
PROXY_PORT="${ANTFAMILY_PORT:-13659}"
GDOWN="${GDOWN:-gdown}"

echo "=== 3DMatch Dataset Download ==="
echo "  Output: $OUTPUT_DIR"
echo "  Proxy:  socks5h://127.0.0.1:$PROXY_PORT"

# Check proxy
if ! curl -s --socks5-hostname "127.0.0.1:$PROXY_PORT" https://drive.google.com -o /dev/null; then
    echo "ERROR: antfamily proxy not reachable on port $PROXY_PORT"
    echo "Start it with: /home/yangyan/antfamily -l $PROXY_PORT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

ARCHIVE="3dmatch_data.zip"
if [ -f "$ARCHIVE" ]; then
    echo "Archive already exists: $ARCHIVE (skipping download)"
else
    echo "Downloading from Google Drive..."
    ALL_PROXY="socks5h://127.0.0.1:$PROXY_PORT" \
        "$GDOWN" "$GDRIVE_ID" -O "$ARCHIVE"
fi

# Extract
if [ -d "indoor" ] && [ -d "indoor/train" ]; then
    echo "Data already extracted (indoor/ exists)"
else
    echo "Extracting..."
    unzip -q -o "$ARCHIVE"
    # The archive may extract to indoor/ directly or to current dir
    if [ ! -d "indoor" ]; then
        # If train/ exists at top level, wrap in indoor/
        if [ -d "train" ]; then
            mkdir -p indoor
            mv train val test *.pkl indoor/ 2>/dev/null || true
        else
            echo "ERROR: Unexpected archive structure. Contents:"
            ls -la
            exit 1
        fi
    fi
fi

# Verify
echo "Verifying directory structure..."
ERRORS=0
for dir in indoor/train indoor/val; do
    if [ ! -d "$dir" ]; then
        echo "  MISSING: $dir"
        ERRORS=$((ERRORS + 1))
    else
        NSCENES=$(ls -d "$dir"/*/ 2>/dev/null | wc -l)
        echo "  OK: $dir ($NSCENES scenes)"
    fi
done

# Check for pkl files (may be in indoor/ or in FCGF configs/)
if [ -f "indoor/train_info.pkl" ]; then
    echo "  OK: indoor/train_info.pkl"
elif [ -f "$REPO_ROOT/examples/FCGF/configs/indoor/train_info.pkl" ]; then
    echo "  OK: train_info.pkl found in FCGF configs/indoor/ (will be used as fallback)"
else
    echo "  WARN: No train_info.pkl found. Training will use auto-scan mode."
fi

if [ $ERRORS -gt 0 ]; then
    echo "ERROR: $ERRORS missing directories"
    exit 1
fi

echo ""
echo "=== 3DMatch download complete ==="
echo "  Path: $OUTPUT_DIR/indoor"
echo "  Set: export THREEDMATCH_ROOT=$OUTPUT_DIR/indoor"
