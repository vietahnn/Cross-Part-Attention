# PowerShell Script for Temporal Augmentation Experiments
# This script runs multiple experiments to evaluate temporal augmentation effectiveness

$EXPERIMENT_BASE = "WLASL100_temporal_aug"
$TRAIN_DATA = "datasets/WLASL100/WLASL100_train_25fps.csv"
$VAL_DATA = "datasets/WLASL100/WLASL100_val_25fps.csv"
$NUM_CLASSES = 100
$EPOCHS = 60
$BATCH_SIZE = 16
$LR = 0.0001

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Temporal Augmentation Experiments for Paper" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Experiment 1: Baseline (no temporal augmentation)
Write-Host "[1/5] Running Baseline (no temporal augmentation)..." -ForegroundColor Yellow
python train.py `
  --experiment_name "${EXPERIMENT_BASE}_baseline" `
  --training_set_path $TRAIN_DATA `
  --validation_set from-file `
  --validation_set_path $VAL_DATA `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_temporal_aug False

Write-Host ""
Write-Host "Baseline completed!" -ForegroundColor Green
Write-Host ""

# Experiment 2: Temporal Masking Only
Write-Host "[2/5] Running Temporal Masking Only..." -ForegroundColor Yellow
python train.py `
  --experiment_name "${EXPERIMENT_BASE}_temporal_only" `
  --training_set_path $TRAIN_DATA `
  --validation_set from-file `
  --validation_set_path $VAL_DATA `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_temporal_aug True `
  --temporal_mask_prob 0.5 `
  --keypoint_dropout_prob 0.0 `
  --mask_ratio 0.15

Write-Host ""
Write-Host "Temporal masking completed!" -ForegroundColor Green
Write-Host ""

# Experiment 3: Keypoint Dropout Only
Write-Host "[3/5] Running Keypoint Dropout Only..." -ForegroundColor Yellow
python train.py `
  --experiment_name "${EXPERIMENT_BASE}_keypoint_only" `
  --training_set_path $TRAIN_DATA `
  --validation_set from-file `
  --validation_set_path $VAL_DATA `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_temporal_aug True `
  --temporal_mask_prob 0.0 `
  --keypoint_dropout_prob 0.5 `
  --dropout_type random `
  --max_keypoints_drop 5

Write-Host ""
Write-Host "Keypoint dropout completed!" -ForegroundColor Green
Write-Host ""

# Experiment 4: Full Method (Both)
Write-Host "[4/5] Running Full Method (Temporal + Keypoint)..." -ForegroundColor Yellow
python train.py `
  --experiment_name "${EXPERIMENT_BASE}_full" `
  --training_set_path $TRAIN_DATA `
  --validation_set from-file `
  --validation_set_path $VAL_DATA `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_temporal_aug True `
  --temporal_mask_prob 0.3 `
  --keypoint_dropout_prob 0.3 `
  --mask_ratio 0.15 `
  --dropout_type random `
  --max_keypoints_drop 5

Write-Host ""
Write-Host "Full method completed!" -ForegroundColor Green
Write-Host ""

# Experiment 5: Aggressive Augmentation
Write-Host "[5/5] Running Aggressive Augmentation..." -ForegroundColor Yellow
python train.py `
  --experiment_name "${EXPERIMENT_BASE}_aggressive" `
  --training_set_path $TRAIN_DATA `
  --validation_set from-file `
  --validation_set_path $VAL_DATA `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_temporal_aug True `
  --temporal_mask_prob 0.5 `
  --keypoint_dropout_prob 0.5 `
  --mask_ratio 0.2 `
  --dropout_type bodypart `
  --max_keypoints_drop 7

Write-Host ""
Write-Host "Aggressive augmentation completed!" -ForegroundColor Green
Write-Host ""

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  All Experiments Completed!" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in:" -ForegroundColor White
Write-Host "  - Checkpoints: out-checkpoints/${EXPERIMENT_BASE}_*/" -ForegroundColor Gray
Write-Host "  - Logs: ${EXPERIMENT_BASE}_*.log" -ForegroundColor Gray
Write-Host "  - Plots: out-img/${EXPERIMENT_BASE}_*.png" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Analyze training logs to compare overfitting" -ForegroundColor Gray
Write-Host "  2. Compare validation accuracies" -ForegroundColor Gray
Write-Host "  3. Generate plots for paper" -ForegroundColor Gray
Write-Host ""
