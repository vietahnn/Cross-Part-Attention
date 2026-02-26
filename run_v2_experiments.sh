#!/bin/bash
# run_v2_experiments.sh
# Script to run experiments comparing Normalize V1 vs V2

# ============================================================================
# Configuration
# ============================================================================
TRAIN_PATH="datasets/WLASL100/train.csv"
TEST_PATH="datasets/WLASL100/test.csv"
NUM_CLASSES=100
EPOCHS=100
BATCH_SIZE=24
LR=0.0001

# Base arguments
BASE_ARGS="--training_set_path $TRAIN_PATH \
           --testing_set_path $TEST_PATH \
           --num_classes $NUM_CLASSES \
           --epochs $EPOCHS \
           --batch_size $BATCH_SIZE \
           --lr $LR"

# ============================================================================
# Experiment 1: V1 Baseline (Standard Normalization)
# ============================================================================
echo "========================================================================"
echo "Running Experiment 1: V1 Baseline (Standard Normalization)"
echo "========================================================================"
python train.py $BASE_ARGS \
    --experiment_name "exp1_v1_baseline" \
    --use_normalize_v2 False \
    --use_position True

# ============================================================================
# Experiment 2: V2 with Neck Reference
# ============================================================================
echo ""
echo "========================================================================"
echo "Running Experiment 2: V2 with Neck Reference"
echo "========================================================================"
python train.py $BASE_ARGS \
    --experiment_name "exp2_v2_neck" \
    --use_normalize_v2 True \
    --body_ref_key neck \
    --use_position True

# ============================================================================
# Experiment 3: V2 with Nose Reference
# ============================================================================
echo ""
echo "========================================================================"
echo "Running Experiment 3: V2 with Nose Reference"
echo "========================================================================"
python train.py $BASE_ARGS \
    --experiment_name "exp3_v2_nose" \
    --use_normalize_v2 True \
    --body_ref_key nose \
    --use_position True

# ============================================================================
# Experiment 4: V2 with Left Shoulder Reference
# ============================================================================
echo ""
echo "========================================================================"
echo "Running Experiment 4: V2 with Left Shoulder Reference"
echo "========================================================================"
python train.py $BASE_ARGS \
    --experiment_name "exp4_v2_leftshoulder" \
    --use_normalize_v2 True \
    --body_ref_key leftShoulder \
    --use_position True

# ============================================================================
# Experiment 5: V2 without Position Features (Ablation)
# ============================================================================
echo ""
echo "========================================================================"
echo "Running Experiment 5: V2 with Neck, No Position Features"
echo "========================================================================"
python train.py $BASE_ARGS \
    --experiment_name "exp5_v2_neck_no_position" \
    --use_normalize_v2 True \
    --body_ref_key neck \
    --use_position False

echo ""
echo "========================================================================"
echo "All experiments completed!"
echo "========================================================================"
echo "Results saved in:"
echo "  - Checkpoints: out-checkpoints/"
echo "  - Plots: out-img/"
echo ""
echo "To compare results, check the generated plots and model checkpoints."
