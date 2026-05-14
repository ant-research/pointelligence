#!/bin/bash
echo "3DMatch PointCNN++ Training Script"
echo "Based on OverlapPredator approach"
export OMP_NUM_THREADS=4
export DATA_ROOT="./outputs/Experiments"
export DATASET=${DATASET:-ThreeDMatchNewPairDatasetPure}
export TRAINER=${TRAINER:-HardestContrastiveLossTrainer}
export MODEL=${MODEL:-ResUNetBN2C}
export MODEL_N_OUT=${MODEL_N_OUT:-64}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-0.0006}
export MAX_EPOCH=${MAX_EPOCH:-40}
export BATCH_SIZE=${BATCH_SIZE:-4}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-1}
export ITER_SIZE=${ITER_SIZE:-1}
export VOXEL_SIZE=${VOXEL_SIZE:-0.025}
export PRE_DOWNSAMPLE_VOXEL_SIZE=${PRE_DOWNSAMPLE_VOXEL_SIZE:-0.02}
export POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER=${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER:-1.5}
export CONV1_KERNEL_SIZE=${CONV1_KERNEL_SIZE:-5}
export EXP_GAMMA=${EXP_GAMMA:-0.95}
export MOMENTUM=${MOMENTUM:-0.98}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
export RANDOM_SCALE=${RANDOM_SCALE:-False}
export RANDOM_ROTATION=${RANDOM_ROTATION:-False}
export ROTATION_RANGE=${ROTATION_RANGE:-360}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export PATH_POSTFIX=${1:-""}
export MISC_ARGS=${2:-""}
export OVERLAP_RADIUS=${OVERLAP_RADIUS:-0.0375}
export AUGMENT_NOISE=${AUGMENT_NOISE:-0.005}
export MAX_POINTS=${MAX_POINTS:-1000000}
export ROT_FACTOR=${ROT_FACTOR:-1.0}
export NUM_POS_PER_BATCH=${NUM_POS_PER_BATCH:-512}
export NUM_HN_SAMPLES_PER_BATCH=${NUM_HN_SAMPLES_PER_BATCH:-128}
export USE_HARD_NEGATIVE=${USE_HARD_NEGATIVE:-True}
export VAL_EPOCH_FREQ=${VAL_EPOCH_FREQ:-1}
export VAL_MAX_ITER=${VAL_MAX_ITER:-400}
export TEST_FREQ_EPOCH=${TEST_FREQ_EPOCH:-5}
export STAT_FREQ=${STAT_FREQ:-40}
export START_EPOCH=${START_EPOCH:-0}
export TRAIN_NUM_THREAD=${TRAIN_NUM_THREAD:-6}
export VAL_NUM_THREAD=${VAL_NUM_THREAD:-1}
export TEST_NUM_THREAD=${TEST_NUM_THREAD:-2}
export NUM_THREADS=${NUM_THREADS:-6}
TRAIN_INFO_PKL="${THREEDMATCH_ROOT}/train_info.pkl"
VAL_INFO_PKL="${THREEDMATCH_ROOT}/val_info.pkl"
if [ ! -f "$TRAIN_INFO_PKL" ]; then
    if [ -f "${CONFIG_PATH}/train_info.pkl" ]; then
        TRAIN_INFO_PKL="${CONFIG_PATH}/train_info.pkl"
        VAL_INFO_PKL="${CONFIG_PATH}/val_info.pkl"
    else
        TRAIN_INFO_PKL=""
        VAL_INFO_PKL=""
    fi
fi
if command -v git &> /dev/null; then
    export VERSION=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
else
    export VERSION="unknown"
fi
export OUT_DIR=${DATA_ROOT}/${DATASET}-v${VOXEL_SIZE}/${TRAINER}/${MODEL}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}i${ITER_SIZE}-modelnout${MODEL_N_OUT}${PATH_POSTFIX}/${TIME}
export LOG_DIR=${OUT_DIR}
export PYTHONUNBUFFERED="True"
echo "Configuration"
echo "  Dataset: $DATASET"
echo "  Trainer: $TRAINER"
echo "  Model: $MODEL"
echo "  Model N Out: $MODEL_N_OUT"
echo "  Optimizer: $OPTIMIZER"
echo "  Learning Rate: $LR"
echo "  Momentum: $MOMENTUM"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Max Epoch: $MAX_EPOCH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Val Batch Size: $VAL_BATCH_SIZE"
echo "  Test Batch Size: $TEST_BATCH_SIZE"
echo "  Voxel Size: $VOXEL_SIZE"
echo "  Pre-downsample Voxel Size: $PRE_DOWNSAMPLE_VOXEL_SIZE"
echo "  3DMatch Root: $THREEDMATCH_ROOT"
echo "  Train Info: ${TRAIN_INFO_PKL:-auto scan}"
echo "  Val Info: ${VAL_INFO_PKL:-auto scan}"
echo "  Overlap Radius: $OVERLAP_RADIUS"
echo "  Augment Noise: $AUGMENT_NOISE"
echo "  Max Points: $MAX_POINTS"
echo "  Num Pos Per Batch: $NUM_POS_PER_BATCH"
echo "  Num HN Samples Per Batch: $NUM_HN_SAMPLES_PER_BATCH"
echo "  Output Dir: $OUT_DIR"
if [ ! -d "$THREEDMATCH_ROOT" ]; then
    echo "Error: 3DMatch dataset dir not found: $THREEDMATCH_ROOT"
    exit 1
fi
if [ ! -d "$THREEDMATCH_ROOT/train" ]; then
    echo "Error: 3DMatch train data dir not found: $THREEDMATCH_ROOT/train"
    exit 1
fi
if [ -n "$TRAIN_INFO_PKL" ] && [ -f "$TRAIN_INFO_PKL" ]; then
    echo "Found train_info.pkl: $TRAIN_INFO_PKL"
    if [ -n "$VAL_INFO_PKL" ] && [ -f "$VAL_INFO_PKL" ]; then
        echo "Found val_info.pkl: $VAL_INFO_PKL"
    fi
else
    echo "Warning: pkl not found, using auto scan mode"
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
TRAIN_CMD="python train.py"
TRAIN_CMD="$TRAIN_CMD --dataset ${DATASET}"
TRAIN_CMD="$TRAIN_CMD --threed_match_dir ${THREEDMATCH_ROOT}"
if [ -n "$TRAIN_INFO_PKL" ] && [ -f "$TRAIN_INFO_PKL" ]; then
    TRAIN_CMD="$TRAIN_CMD --train_info $TRAIN_INFO_PKL"
    if [ -n "$VAL_INFO_PKL" ] && [ -f "$VAL_INFO_PKL" ]; then
        TRAIN_CMD="$TRAIN_CMD --val_info $VAL_INFO_PKL"
    fi
    echo "Using OverlapPredator format (pkl)"
else
    echo "Using auto scan mode"
fi
TRAIN_CMD="$TRAIN_CMD --overlap_radius ${OVERLAP_RADIUS}"
TRAIN_CMD="$TRAIN_CMD --augment_noise ${AUGMENT_NOISE}"
TRAIN_CMD="$TRAIN_CMD --max_points ${MAX_POINTS}"
TRAIN_CMD="$TRAIN_CMD --rot_factor ${ROT_FACTOR}"
TRAIN_CMD="$TRAIN_CMD --use_random_rotation ${RANDOM_ROTATION}"
TRAIN_CMD="$TRAIN_CMD --rotation_range ${ROTATION_RANGE}"
TRAIN_CMD="$TRAIN_CMD --use_random_scale ${RANDOM_SCALE}"
TRAIN_CMD="$TRAIN_CMD --voxel_size ${VOXEL_SIZE}"
if [ -n "$PRE_DOWNSAMPLE_VOXEL_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --pre_downsample_voxel_size ${PRE_DOWNSAMPLE_VOXEL_SIZE}"
fi
TRAIN_CMD="$TRAIN_CMD --positive_pair_search_voxel_size_multiplier ${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER}"
TRAIN_CMD="$TRAIN_CMD --batch_size ${BATCH_SIZE}"
TRAIN_CMD="$TRAIN_CMD --val_batch_size ${VAL_BATCH_SIZE}"
TRAIN_CMD="$TRAIN_CMD --iter_size ${ITER_SIZE}"
TRAIN_CMD="$TRAIN_CMD --optimizer ${OPTIMIZER}"
TRAIN_CMD="$TRAIN_CMD --lr ${LR}"
TRAIN_CMD="$TRAIN_CMD --momentum ${MOMENTUM}"
TRAIN_CMD="$TRAIN_CMD --weight_decay ${WEIGHT_DECAY}"
TRAIN_CMD="$TRAIN_CMD --scheduler ExpLR"
TRAIN_CMD="$TRAIN_CMD --exp_gamma ${EXP_GAMMA}"
TRAIN_CMD="$TRAIN_CMD --max_epoch ${MAX_EPOCH}"
TRAIN_CMD="$TRAIN_CMD --model ${MODEL}"
TRAIN_CMD="$TRAIN_CMD --model_n_out ${MODEL_N_OUT}"
TRAIN_CMD="$TRAIN_CMD --conv1_kernel_size ${CONV1_KERNEL_SIZE}"
TRAIN_CMD="$TRAIN_CMD --normalize_feature True"
TRAIN_CMD="$TRAIN_CMD --pos_thresh 0.1"
TRAIN_CMD="$TRAIN_CMD --neg_thresh 1.4"
TRAIN_CMD="$TRAIN_CMD --hit_ratio_thresh 0.3"
TRAIN_CMD="$TRAIN_CMD --num_pos_per_batch ${NUM_POS_PER_BATCH}"
TRAIN_CMD="$TRAIN_CMD --num_hn_samples_per_batch ${NUM_HN_SAMPLES_PER_BATCH}"
TRAIN_CMD="$TRAIN_CMD --use_hard_negative ${USE_HARD_NEGATIVE}"
TRAIN_CMD="$TRAIN_CMD --trainer ${TRAINER}"
TRAIN_CMD="$TRAIN_CMD --val_epoch_freq ${VAL_EPOCH_FREQ}"
TRAIN_CMD="$TRAIN_CMD --val_max_iter ${VAL_MAX_ITER}"
TRAIN_CMD="$TRAIN_CMD --save_freq_epoch 1"
TRAIN_CMD="$TRAIN_CMD --stat_freq ${STAT_FREQ}"
TRAIN_CMD="$TRAIN_CMD --test_valid True"
TRAIN_CMD="$TRAIN_CMD --best_val_metric feat_match_ratio"
TRAIN_CMD="$TRAIN_CMD --train_num_thread ${TRAIN_NUM_THREAD}"
TRAIN_CMD="$TRAIN_CMD --val_num_thread ${VAL_NUM_THREAD}"
TRAIN_CMD="$TRAIN_CMD --test_num_thread ${TEST_NUM_THREAD}"
TRAIN_CMD="$TRAIN_CMD --use_gpu True"
TRAIN_CMD="$TRAIN_CMD --out_dir ${OUT_DIR}"
TRAIN_CMD="$TRAIN_CMD ${MISC_ARGS}"
echo ""
echo "Training command:"
echo "$TRAIN_CMD"
echo ""
$TRAIN_CMD 2>&1 | tee "$LOG"