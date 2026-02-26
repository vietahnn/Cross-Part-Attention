# Quick Start: Combined Training (Option A + B)
# Run a quick test with both temporal augmentation and contrastive learning

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "   Quick Start - Combined Training (A + B)      " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$TRAIN_CSV = "datasets/WLASL100/WLASL100_train_25fps.csv"
$VAL_CSV = "datasets/WLASL100/WLASL100_val_25fps.csv"
$NUM_CLASSES = 100
$EPOCHS = 5  # Quick test
$BATCH_SIZE = 16
$LR = 0.0001

Write-Host "[INFO] Testing BOTH methods together:" -ForegroundColor Yellow
Write-Host "  ✓ Temporal Augmentation (Option A)" -ForegroundColor Green
Write-Host "  ✓ Contrastive Learning (Option B)" -ForegroundColor Green
Write-Host ""
Write-Host "[INFO] Running short test (5 epochs)" -ForegroundColor Yellow
Write-Host ""

# Check CSV files
if (-not (Test-Path $TRAIN_CSV)) {
    Write-Host "[ERROR] Training CSV not found: $TRAIN_CSV" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VAL_CSV)) {
    Write-Host "[ERROR] Validation CSV not found: $VAL_CSV" -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] Dataset files found" -ForegroundColor Green
Write-Host ""

# Run combined training
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "Training with COMBINED approach..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Temporal Augmentation (Option A):" -ForegroundColor Cyan
Write-Host "  - Aug Probability: 0.5"
Write-Host "  - Temporal Masking: 0.3"
Write-Host "  - Keypoint Dropout: 0.15"
Write-Host ""
Write-Host "Contrastive Learning (Option B):" -ForegroundColor Cyan
Write-Host "  - Temperature: 0.07"
Write-Host "  - Beta (contrastive): 0.5"
Write-Host "  - Gamma (center): 0.1"
Write-Host ""

python train.py `
  --experiment_name "WLASL100_quickstart_combined" `
  --training_set_path $TRAIN_CSV `
  --validation_set from-file `
  --validation_set_path $VAL_CSV `
  --num_classes $NUM_CLASSES `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LR `
  --use_temporal_aug True `
  --temporal_aug_prob 0.5 `
  --temporal_masking_prob 0.3 `
  --temporal_masking_ratio 0.15 `
  --keypoint_dropout_prob 0.15 `
  --keypoint_dropout_max 5 `
  --sequential_cutout_prob 0.2 `
  --use_contrastive True `
  --contrastive_temperature 0.07 `
  --contrastive_beta 0.5 `
  --center_lambda 0.003 `
  --center_gamma 0.1

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Combined training failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "[TROUBLESHOOTING]" -ForegroundColor Yellow
    Write-Host "1. Test Option A separately:"
    Write-Host "   .\quickstart_temporal_aug.ps1"
    Write-Host ""
    Write-Host "2. Test Option B separately:"
    Write-Host "   .\quickstart_contrastive.ps1"
    Write-Host ""
    Write-Host "3. Check individual test scripts:"
    Write-Host "   python test_temporal_augmentation.py"
    Write-Host "   python test_contrastive_learning.py"
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "     COMBINED TRAINING TEST COMPLETED!          " -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""

Write-Host "[SUCCESS] Both methods working together!" -ForegroundColor Green
Write-Host ""

Write-Host "[EXPECTED BENEFITS]" -ForegroundColor Magenta
Write-Host "✓ Temporal robustness from augmentation"
Write-Host "✓ Better features from contrastive learning"
Write-Host "✓ Maximum overfitting reduction"
Write-Host "✓ Best validation accuracy"
Write-Host ""

Write-Host "[NEXT STEPS]" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run full training (60 epochs):" -ForegroundColor Yellow
Write-Host "   python train.py \"
Write-Host "     --experiment_name WLASL100_combined_full \"
Write-Host "     --training_set_path $TRAIN_CSV \"
Write-Host "     --validation_set from-file \"
Write-Host "     --validation_set_path $VAL_CSV \"
Write-Host "     --num_classes 100 \"
Write-Host "     --epochs 60 \"
Write-Host "     --batch_size 16 \"
Write-Host "     --use_temporal_aug True \"
Write-Host "     --use_contrastive True"
Write-Host ""
Write-Host "2. Run ablation study:" -ForegroundColor Yellow
Write-Host "   Compare: Baseline | Option A | Option B | Combined"
Write-Host "   See COMBINED_TRAINING_README.md for commands"
Write-Host ""
Write-Host "3. Tune hyperparameters:" -ForegroundColor Yellow
Write-Host "   - If still overfitting: Increase aug_prob and beta"
Write-Host "   - If underfitting: Decrease aug_prob and beta"
Write-Host ""
Write-Host "4. Expected results:" -ForegroundColor Yellow
Write-Host "   | Method   | Train | Val   | Gap   |"
Write-Host "   | Baseline | 97.4% | 87.1% | 10.3% |"
Write-Host "   | Combined | ~92%  | ~90%  | ~2%   | ← Target"
Write-Host ""

Write-Host "=================================================" -ForegroundColor Green
Write-Host ""
Write-Host "See COMBINED_TRAINING_README.md for full guide!" -ForegroundColor Cyan
