#!/bin/bash

# Contrastive Learning Experiments for WLASL100
# This script runs comprehensive experiments for Option B (Supervised Contrastive + Center Loss)

echo "================================================="
echo "  Contrastive Learning Experiments - Option B   "
echo "================================================="
echo ""

# Configuration
TRAIN_CSV="datasets/WLASL100/WLASL100_train_25fps.csv"
VAL_CSV="datasets/WLASL100/WLASL100_val_25fps.csv"
NUM_CLASSES=100
EPOCHS=60
BATCH_SIZE=16
LR=0.0001

echo "[INFO] Configuration:"
echo "  Training CSV: $TRAIN_CSV"
echo "  Validation CSV: $VAL_CSV"
echo "  Classes: $NUM_CLASSES"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo ""

# Check if CSV files exist
if [ ! -f "$TRAIN_CSV" ]; then
    echo "[ERROR] Training CSV not found: $TRAIN_CSV"
    exit 1
fi

if [ ! -f "$VAL_CSV" ]; then
    echo "[ERROR] Validation CSV not found: $VAL_CSV"
    exit 1
fi

# ============================================================
# EXPERIMENT 1: BASELINE (No Contrastive Learning)
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 1: Baseline (No Contrastive Learning)"
echo "============================================================"
echo ""

python train.py \
  --experiment_name "WLASL100_baseline_contrastive" \
  --training_set_path "$TRAIN_CSV" \
  --validation_set from-file \
  --validation_set_path "$VAL_CSV" \
  --num_classes $NUM_CLASSES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --use_contrastive False

if [ $? -ne 0 ]; then
    echo "[ERROR] Baseline experiment failed!"
    exit 1
fi

echo "[SUCCESS] Baseline experiment completed!"
sleep 2

# ============================================================
# EXPERIMENT 2: CONTRASTIVE ONLY (No Center Loss)
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 2: Contrastive Loss Only (beta=0.5, gamma=0.0)"
echo "============================================================"
echo ""

python train.py \
  --experiment_name "WLASL100_contrastive_only" \
  --training_set_path "$TRAIN_CSV" \
  --validation_set from-file \
  --validation_set_path "$VAL_CSV" \
  --num_classes $NUM_CLASSES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --use_contrastive True \
  --contrastive_temperature 0.07 \
  --contrastive_beta 0.5 \
  --center_lambda 0.003 \
  --center_gamma 0.0

if [ $? -ne 0 ]; then
    echo "[ERROR] Contrastive-only experiment failed!"
    exit 1
fi

echo "[SUCCESS] Contrastive-only experiment completed!"
sleep 2

# ============================================================
# EXPERIMENT 3: CENTER LOSS ONLY (No Contrastive)
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 3: Center Loss Only (beta=0.0, gamma=0.5)"
echo "============================================================"
echo ""

python train.py \
  --experiment_name "WLASL100_center_only" \
  --training_set_path "$TRAIN_CSV" \
  --validation_set from-file \
  --validation_set_path "$VAL_CSV" \
  --num_classes $NUM_CLASSES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --use_contrastive True \
  --contrastive_temperature 0.07 \
  --contrastive_beta 0.0 \
  --center_lambda 0.003 \
  --center_gamma 0.5

if [ $? -ne 0 ]; then
    echo "[ERROR] Center-only experiment failed!"
    exit 1
fi

echo "[SUCCESS] Center-only experiment completed!"
sleep 2

# ============================================================
# EXPERIMENT 4: FULL METHOD (Both Losses)
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 4: Full Method (beta=0.5, gamma=0.1)"
echo "============================================================"
echo ""

python train.py \
  --experiment_name "WLASL100_full_contrastive" \
  --training_set_path "$TRAIN_CSV" \
  --validation_set from-file \
  --validation_set_path "$VAL_CSV" \
  --num_classes $NUM_CLASSES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --use_contrastive True \
  --contrastive_temperature 0.07 \
  --contrastive_beta 0.5 \
  --center_lambda 0.003 \
  --center_gamma 0.1

if [ $? -ne 0 ]; then
    echo "[ERROR] Full method experiment failed!"
    exit 1
fi

echo "[SUCCESS] Full method experiment completed!"
sleep 2

# ============================================================
# EXPERIMENT 5: HYPERPARAMETER TUNING - Contrastive Weight
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 5: Contrastive Weight Sensitivity (beta)"
echo "============================================================"
echo ""

for beta in 0.3 0.8 1.0; do
    echo "[INFO] Testing beta = $beta"
    
    python train.py \
      --experiment_name "WLASL100_beta_${beta}" \
      --training_set_path "$TRAIN_CSV" \
      --validation_set from-file \
      --validation_set_path "$VAL_CSV" \
      --num_classes $NUM_CLASSES \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --use_contrastive True \
      --contrastive_temperature 0.07 \
      --contrastive_beta $beta \
      --center_lambda 0.003 \
      --center_gamma 0.1
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Beta=$beta experiment failed!"
        exit 1
    fi
    
    echo "[SUCCESS] Beta=$beta experiment completed!"
    sleep 2
done

# ============================================================
# EXPERIMENT 6: HYPERPARAMETER TUNING - Center Weight
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 6: Center Weight Sensitivity (gamma)"
echo "============================================================"
echo ""

for gamma in 0.05 0.2 0.3; do
    echo "[INFO] Testing gamma = $gamma"
    
    python train.py \
      --experiment_name "WLASL100_gamma_${gamma}" \
      --training_set_path "$TRAIN_CSV" \
      --validation_set from-file \
      --validation_set_path "$VAL_CSV" \
      --num_classes $NUM_CLASSES \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --use_contrastive True \
      --contrastive_temperature 0.07 \
      --contrastive_beta 0.5 \
      --center_lambda 0.003 \
      --center_gamma $gamma
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Gamma=$gamma experiment failed!"
        exit 1
    fi
    
    echo "[SUCCESS] Gamma=$gamma experiment completed!"
    sleep 2
done

# ============================================================
# EXPERIMENT 7: TEMPERATURE SENSITIVITY
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 7: Temperature Sensitivity"
echo "============================================================"
echo ""

for temp in 0.05 0.10 0.15; do
    echo "[INFO] Testing temperature = $temp"
    
    python train.py \
      --experiment_name "WLASL100_temp_${temp}" \
      --training_set_path "$TRAIN_CSV" \
      --validation_set from-file \
      --validation_set_path "$VAL_CSV" \
      --num_classes $NUM_CLASSES \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --use_contrastive True \
      --contrastive_temperature $temp \
      --contrastive_beta 0.5 \
      --center_lambda 0.003 \
      --center_gamma 0.1
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Temperature=$temp experiment failed!"
        exit 1
    fi
    
    echo "[SUCCESS] Temperature=$temp experiment completed!"
    sleep 2
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "================================================="
echo "         ALL EXPERIMENTS COMPLETED!              "
echo "================================================="
echo ""

echo "Experiments run:"
echo "  1. Baseline (no contrastive learning)"
echo "  2. Contrastive Loss Only"
echo "  3. Center Loss Only"
echo "  4. Full Method (both losses)"
echo "  5. Beta sensitivity: 0.3, 0.5 (default), 0.8, 1.0"
echo "  6. Gamma sensitivity: 0.05, 0.1 (default), 0.2, 0.3"
echo "  7. Temperature sensitivity: 0.05, 0.07 (default), 0.10, 0.15"
echo ""

echo "Total experiments: 13"
echo ""

echo "[NEXT STEPS]"
echo "1. Compare results in TensorBoard:"
echo "   tensorboard --logdir=logs"
echo ""
echo "2. Analyze training logs for each experiment"
echo ""
echo "3. Create comparison table:"
echo "   | Method         | Train Acc | Val Acc | Gap    |"
echo "   | Baseline       | XX.X%     | XX.X%   | XX.X%  |"
echo "   | Contrastive    | XX.X%     | XX.X%   | XX.X%  |"
echo "   | Center         | XX.X%     | XX.X%   | XX.X%  |"
echo "   | Full Method    | XX.X%     | XX.X%   | XX.X%  |"
echo ""
echo "4. Visualize features with t-SNE (see README)"
echo ""

echo "================================================="
