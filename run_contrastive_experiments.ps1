# Contrastive Learning Experiments for WLASL100
# This script runs comprehensive experiments for Option B (Supervised Contrastive + Center Loss)

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Contrastive Learning Experiments - Option B   " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$TRAIN_CSV = "datasets/WLASL100/WLASL100_train_25fps.csv"
$VAL_CSV = "datasets/WLASL100/WLASL100_val_25fps.csv"
$NUM_CLASSES = 100
$EPOCHS = 60
$BATCH_SIZE = 16
$LR = 0.0001

Write-Host "[INFO] Configuration:" -ForegroundColor Green
Write-Host "  Training CSV: $TRAIN_CSV"
Write-Host "  Validation CSV: $VAL_CSV"
Write-Host "  Classes: $NUM_CLASSES"
Write-Host "  Epochs: $EPOCHS"
Write-Host "  Batch Size: $BATCH_SIZE"
Write-Host "  Learning Rate: $LR"
Write-Host ""

# Check if CSV files exist
if (-not (Test-Path $TRAIN_CSV)) {
    Write-Host "[ERROR] Training CSV not found: $TRAIN_CSV" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VAL_CSV)) {
    Write-Host "[ERROR] Validation CSV not found: $VAL_CSV" -ForegroundColor Red
    exit 1
}

# ============================================================
# EXPERIMENT 1: BASELINE (No Contrastive Learning)
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "EXPERIMENT 1: Baseline (No Contrastive Learning)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

python train.py `
  --experiment_name "WLASL100_baseline_contrastive" `
  --training_set_path $TRAIN_CSV `
  --validation_set from-file `
  --validation_set_path $VAL_CSV `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_contrastive False

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Baseline experiment failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] Baseline experiment completed!" -ForegroundColor Green
Start-Sleep -Seconds 2

# ============================================================
# EXPERIMENT 2: CONTRASTIVE ONLY (No Center Loss)
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "EXPERIMENT 2: Contrastive Loss Only (beta=0.5, gamma=0.0)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

python train.py `
  --experiment_name "WLASL100_contrastive_only" `
  --training_set_path $TRAIN_CSV `
  --validation_set from-file `
  --validation_set_path $VAL_CSV `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_contrastive True `
  --contrastive_temperature 0.07 `
  --contrastive_beta 0.5 `
  --center_lambda 0.003 `
  --center_gamma 0.0

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Contrastive-only experiment failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] Contrastive-only experiment completed!" -ForegroundColor Green
Start-Sleep -Seconds 2

# ============================================================
# EXPERIMENT 3: CENTER LOSS ONLY (No Contrastive)
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "EXPERIMENT 3: Center Loss Only (beta=0.0, gamma=0.5)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

python train.py `
  --experiment_name "WLASL100_center_only" `
  --training_set_path $TRAIN_CSV `
  --validation_set from-file `
  --validation_set_path $VAL_CSV `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_contrastive True `
  --contrastive_temperature 0.07 `
  --contrastive_beta 0.0 `
  --center_lambda 0.003 `
  --center_gamma 0.5

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Center-only experiment failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] Center-only experiment completed!" -ForegroundColor Green
Start-Sleep -Seconds 2

# ============================================================
# EXPERIMENT 4: FULL METHOD (Both Losses)
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "EXPERIMENT 4: Full Method (beta=0.5, gamma=0.1)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

python train.py `
  --experiment_name "WLASL100_full_contrastive" `
  --training_set_path $TRAIN_CSV `
  --validation_set from-file `
  --validation_set_path $VAL_CSV `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_contrastive True `
  --contrastive_temperature 0.07 `
  --contrastive_beta 0.5 `
  --center_lambda 0.003 `
  --center_gamma 0.1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Full method experiment failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] Full method experiment completed!" -ForegroundColor Green
Start-Sleep -Seconds 2

# ============================================================
# EXPERIMENT 5: HYPERPARAMETER TUNING - Contrastive Weight
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "EXPERIMENT 5: Contrastive Weight Sensitivity (beta)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$beta_values = @(0.3, 0.8, 1.0)

foreach ($beta in $beta_values) {
    Write-Host "[INFO] Testing beta = $beta" -ForegroundColor Cyan
    
    python train.py `
      --experiment_name "WLASL100_beta_$beta" `
      --training_set_path $TRAIN_CSV `
      --validation_set from-file `
      --validation_set_path $VAL_CSV `
      --num_classes $NUM_CLASSES `
      --epochs $EPOCHS `
      --batch_size $BATCH_SIZE `
      --lr $LR `
      --use_contrastive True `
      --contrastive_temperature 0.07 `
      --contrastive_beta $beta `
      --center_lambda 0.003 `
      --center_gamma 0.1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Beta=$beta experiment failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[SUCCESS] Beta=$beta experiment completed!" -ForegroundColor Green
    Start-Sleep -Seconds 2
}

# ============================================================
# EXPERIMENT 6: HYPERPARAMETER TUNING - Center Weight
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "EXPERIMENT 6: Center Weight Sensitivity (gamma)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$gamma_values = @(0.05, 0.2, 0.3)

foreach ($gamma in $gamma_values) {
    Write-Host "[INFO] Testing gamma = $gamma" -ForegroundColor Cyan
    
    python train.py `
      --experiment_name "WLASL100_gamma_$gamma" `
      --training_set_path $TRAIN_CSV `
      --validation_set from-file `
      --validation_set_path $VAL_CSV `
      --num_classes $NUM_CLASSES `
      --epochs $EPOCHS `
      --batch_size $BATCH_SIZE `
      --lr $LR `
      --use_contrastive True `
      --contrastive_temperature 0.07 `
      --contrastive_beta 0.5 `
      --center_lambda 0.003 `
      --center_gamma $gamma
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Gamma=$gamma experiment failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[SUCCESS] Gamma=$gamma experiment completed!" -ForegroundColor Green
    Start-Sleep -Seconds 2
}

# ============================================================
# EXPERIMENT 7: TEMPERATURE SENSITIVITY
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "EXPERIMENT 7: Temperature Sensitivity" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$temp_values = @(0.05, 0.10, 0.15)

foreach ($temp in $temp_values) {
    Write-Host "[INFO] Testing temperature = $temp" -ForegroundColor Cyan
    
    python train.py `
      --experiment_name "WLASL100_temp_$temp" `
      --training_set_path $TRAIN_CSV `
      --validation_set from-file `
      --validation_set_path $VAL_CSV `
      --num_classes $NUM_CLASSES `
      --epochs $EPOCHS `
      --batch_size $BATCH_SIZE `
      --lr $LR `
      --use_contrastive True `
      --contrastive_temperature $temp `
      --contrastive_beta 0.5 `
      --center_lambda 0.003 `
      --center_gamma 0.1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Temperature=$temp experiment failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[SUCCESS] Temperature=$temp experiment completed!" -ForegroundColor Green
    Start-Sleep -Seconds 2
}

# ============================================================
# SUMMARY
# ============================================================
Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "         ALL EXPERIMENTS COMPLETED!              " -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Experiments run:" -ForegroundColor Cyan
Write-Host "  1. Baseline (no contrastive learning)"
Write-Host "  2. Contrastive Loss Only"
Write-Host "  3. Center Loss Only"
Write-Host "  4. Full Method (both losses)"
Write-Host "  5. Beta sensitivity: 0.3, 0.5 (default), 0.8, 1.0"
Write-Host "  6. Gamma sensitivity: 0.05, 0.1 (default), 0.2, 0.3"
Write-Host "  7. Temperature sensitivity: 0.05, 0.07 (default), 0.10, 0.15"
Write-Host ""

Write-Host "Total experiments: 13" -ForegroundColor Yellow
Write-Host ""

Write-Host "[NEXT STEPS]" -ForegroundColor Magenta
Write-Host "1. Compare results in TensorBoard:"
Write-Host "   tensorboard --logdir=logs"
Write-Host ""
Write-Host "2. Analyze training logs for each experiment"
Write-Host ""
Write-Host "3. Create comparison table:"
Write-Host "   | Method         | Train Acc | Val Acc | Gap    |"
Write-Host "   | Baseline       | XX.X%     | XX.X%   | XX.X%  |"
Write-Host "   | Contrastive    | XX.X%     | XX.X%   | XX.X%  |"
Write-Host "   | Center         | XX.X%     | XX.X%   | XX.X%  |"
Write-Host "   | Full Method    | XX.X%     | XX.X%   | XX.X%  |"
Write-Host ""
Write-Host "4. Visualize features with t-SNE (see README)"
Write-Host ""

Write-Host "=================================================" -ForegroundColor Green
