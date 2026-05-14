#!/bin/bash
echo "KITTI PointCNN++ Training Script"
export OMP_NUM_THREADS=4
export DATA_ROOT="./outputs/Experiments"
export DATASET=${DATASET:-KITTINMPairDataset}
export TRAINER=${TRAINER:-HardestContrastiveLossTrainer}
export MODEL=${MODEL:-ResUNetBN2C}
export MODEL_N_OUT=${MODEL_N_OUT:-64}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-1}
export MAX_EPOCH=${MAX_EPOCH:-200}
export BATCH_SIZE=${BATCH_SIZE:-4}
export ITER_SIZE=${ITER_SIZE:-1}
export VOXEL_SIZE=${VOXEL_SIZE:-0.3}
export POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER=${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER:-1.5}
export CONV1_KERNEL_SIZE=${CONV1_KERNEL_SIZE:-5}
export EXP_GAMMA=${EXP_GAMMA:-0.99}
export RANDOM_SCALE=${RANDOM_SCALE:-False}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export PATH_POSTFIX=${1:-""}
export MISC_ARGS=${2:-""}
if command -v git &> /dev/null; then
    export VERSION=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
else
    export VERSION="unknown"
fi
export OUT_DIR=${DATA_ROOT}/${DATASET}-v${VOXEL_SIZE}/${TRAINER}/${MODEL}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}i${ITER_SIZE}-modelnout${MODEL_N_OUT}${PATH_POSTFIX}/${TIME}
export PYTHONUNBUFFERED="True"
echo "Configuration"
echo "  Dataset: $DATASET"
echo "  Trainer: $TRAINER"
echo "  Model: $MODEL"
echo "  Model N Out: $MODEL_N_OUT"
echo "  Optimizer: $OPTIMIZER"
echo "  Learning Rate: $LR"
echo "  Max Epoch: $MAX_EPOCH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Voxel Size: $VOXEL_SIZE"
echo "  Pre-downsample Voxel Size: $PRE_DOWNSAMPLE_VOXEL_SIZE"
echo "  KITTI Root: $KITTI_PATH"
echo "  Output Dir: $OUT_DIR"
if [ ! -d "$KITTI_PATH" ]; then
    echo "Error: KITTI dataset not found: $KITTI_PATH"
    exit 1
fi
if [ ! -d "$KITTI_PATH/dataset/sequences" ]; then
    echo "Error: KITTI sequences dir not found: $KITTI_PATH/dataset/sequences"
    exit 1
fi
echo "Data check passed"
mkdir -m 755 -p "$OUT_DIR"
LOG="$OUT_DIR/log_${TIME}.txt"
echo "Host: $(hostname)" | tee -a "$LOG"
echo "Conda: $(which conda 2>/dev/null || echo 'not found')" | tee -a "$LOG"
echo "Working Directory: $(pwd)" | tee -a "$LOG"
echo "Version: $VERSION" | tee -a "$LOG"
echo "Git diff:" | tee -a "$LOG"
echo "" | tee -a "$LOG"
if command -v git &> /dev/null; then
    git diff | tee -a "$LOG" 2>/dev/null || echo "No git diff available" | tee -a "$LOG"
else
    echo "Git not available" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"
nvidia-smi | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Start training"
echo "Training command will be logged to: $LOG"
python train.py \
    --dataset ${DATASET} \
    --trainer ${TRAINER} \
    --model ${MODEL} \
    --model_n_out ${MODEL_N_OUT} \
    --conv1_kernel_size ${CONV1_KERNEL_SIZE} \
    --optimizer ${OPTIMIZER} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --iter_size ${ITER_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --voxel_size ${VOXEL_SIZE} \
    --out_dir ${OUT_DIR} \
    --use_random_scale ${RANDOM_SCALE} \
    --positive_pair_search_voxel_size_multiplier ${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER} \
    --kitti_root ${KITTI_PATH} \
    --hit_ratio_thresh 0.3 \
    --exp_gamma ${EXP_GAMMA} \
    ${MISC_ARGS} 2>&1 | tee "$LOG"