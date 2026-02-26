#!/bin/bash
# Quick Start Training Script with Temporal Augmentation
# This script runs multiple experiments to evaluate temporal augmentation effectiveness

export EXPERIMENT_BASE="WLASL100_temporal_aug"
export TRAIN_DATA="datasets/WLASL100/WLASL100_train_25fps.csv"
export VAL_DATA="datasets/WLASL100/WLASL100_val_25fps.csv"
export NUM_CLASSES=100
export EPOCHS=60
export BATCH_SIZE=16
export LR=0.0001

echo "=================================================="
echo "  Temporal Augmentation Experiments for Paper"
echo "=================================================="
echo ""

# Experiment 1: Baseline (no temporal augmentation)
echo "[1/5] Running Baseline (no temporal augmentation)..."
python train.py \
  --experiment_name ${EXPERIMENT_BASE}_baseline \
  --training_set_path ${TRAIN_DATA} \
  --validation_set from-file \
  --validation_set_path ${VAL_DATA} \
  --num_classes ${NUM_CLASSES} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --use_temporal_aug False

echo ""
echo "Baseline completed!"
echo ""

# Experiment 2: Temporal Masking Only
echo "[2/5] Running Temporal Masking Only..."
python train.py \
  --experiment_name ${EXPERIMENT_BASE}_temporal_only \
  --training_set_path ${TRAIN_DATA} \
  --validation_set from-file \
  --validation_set_path ${VAL_DATA} \
  --num_classes ${NUM_CLASSES} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --use_temporal_aug True \
  --temporal_mask_prob 0.5 \
  --keypoint_dropout_prob 0.0 \
  --mask_ratio 0.15

echo ""
echo "Temporal masking completed!"
echo ""

# Experiment 3: Keypoint Dropout Only
echo "[3/5] Running Keypoint Dropout Only..."
python train.py \
  --experiment_name ${EXPERIMENT_BASE}_keypoint_only \
  --training_set_path ${TRAIN_DATA} \
  --validation_set from-file \
  --validation_set_path ${VAL_DATA} \
  --num_classes ${NUM_CLASSES} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --use_temporal_aug True \
  --temporal_mask_prob 0.0 \
  --keypoint_dropout_prob 0.5 \
  --dropout_type random \
  --max_keypoints_drop 5

echo ""
echo "Keypoint dropout completed!"
echo ""

# Experiment 4: Full Method (Both)
echo "[4/5] Running Full Method (Temporal + Keypoint)..."
python train.py \
  --experiment_name ${EXPERIMENT_BASE}_full \
  --training_set_path ${TRAIN_DATA} \
  --validation_set from-file \
  --validation_set_path ${VAL_DATA} \
  --num_classes ${NUM_CLASSES} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --use_temporal_aug True \
  --temporal_mask_prob 0.3 \
  --keypoint_dropout_prob 0.3 \
  --mask_ratio 0.15 \
  --dropout_type random \
  --max_keypoints_drop 5

echo ""
echo "Full method completed!"
echo ""

# Experiment 5: Aggressive Augmentation
echo "[5/5] Running Aggressive Augmentation..."
python train.py \
  --experiment_name ${EXPERIMENT_BASE}_aggressive \
  --training_set_path ${TRAIN_DATA} \
  --validation_set from-file \
  --validation_set_path ${VAL_DATA} \
  --num_classes ${NUM_CLASSES} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --use_temporal_aug True \
  --temporal_mask_prob 0.5 \
  --keypoint_dropout_prob 0.5 \
  --mask_ratio 0.2 \
  --dropout_type bodypart \
  --max_keypoints_drop 7

echo ""
echo "Aggressive augmentation completed!"
echo ""

echo "=================================================="
echo "  All Experiments Completed!"
echo "=================================================="
echo ""
echo "Results saved in:"
echo "  - Checkpoints: out-checkpoints/${EXPERIMENT_BASE}_*/"
echo "  - Logs: ${EXPERIMENT_BASE}_*.log"
echo "  - Plots: out-img/${EXPERIMENT_BASE}_*.png"
echo ""
echo "Next steps:"
echo "  1. Analyze training logs to compare overfitting"
echo "  2. Compare validation accuracies"
echo "  3. Generate plots for paper"
echo ""
