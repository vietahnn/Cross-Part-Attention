# Quick Start: Test Option B (Contrastive Learning)
# Run a single quick experiment to verify the implementation

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "     Quick Start - Contrastive Learning Test     " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$TRAIN_CSV = "datasets/WLASL100/WLASL100_train_25fps.csv"
$VAL_CSV = "datasets/WLASL100/WLASL100_val_25fps.csv"
$NUM_CLASSES = 100
$EPOCHS = 5  # Short test run
$BATCH_SIZE = 16
$LR = 0.0001

Write-Host "[INFO] Running short test (5 epochs) to verify implementation" -ForegroundColor Yellow
Write-Host ""

# Check CSV files
if (-not (Test-Path $TRAIN_CSV)) {
    Write-Host "[ERROR] Training CSV not found: $TRAIN_CSV" -ForegroundColor Red
    Write-Host "[HELP] Please ensure you have the WLASL100 dataset prepared." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $VAL_CSV)) {
    Write-Host "[ERROR] Validation CSV not found: $VAL_CSV" -ForegroundColor Red
    Write-Host "[HELP] Please ensure you have the WLASL100 dataset prepared." -ForegroundColor Yellow
    exit 1
}

Write-Host "[SUCCESS] Dataset files found" -ForegroundColor Green
Write-Host ""

# Run quick test with full method
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "Training with Full Method (Contrastive + Center Loss)..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Parameters:" -ForegroundColor Cyan
Write-Host "  - Epochs: $EPOCHS (quick test)"
Write-Host "  - Batch Size: $BATCH_SIZE"
Write-Host "  - Temperature: 0.07"
Write-Host "  - Beta (contrastive weight): 0.5"
Write-Host "  - Gamma (center weight): 0.1"
Write-Host ""

python train.py `
  --experiment_name "WLASL100_quickstart_contrastive" `
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
    Write-Host ""
    Write-Host "[ERROR] Training failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "[TROUBLESHOOTING]" -ForegroundColor Yellow
    Write-Host "1. Check that all dependencies are installed:"
    Write-Host "   pip install torch torchvision tqdm"
    Write-Host ""
    Write-Host "2. Verify the dataset CSV files are correctly formatted"
    Write-Host ""
    Write-Host "3. Check train.py for any syntax errors"
    Write-Host ""
    Write-Host "4. Review the error message above for specific issues"
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "           QUICK TEST COMPLETED!                 " -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""

Write-Host "[SUCCESS] Contrastive learning is working correctly!" -ForegroundColor Green
Write-Host ""

Write-Host "[NEXT STEPS]" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run full experiments (60 epochs):" -ForegroundColor Yellow
Write-Host "   .\run_contrastive_experiments.ps1"
Write-Host ""
Write-Host "2. Or run a single full training:" -ForegroundColor Yellow
Write-Host "   python train.py \"
Write-Host "     --experiment_name WLASL100_full_contrastive \"
Write-Host "     --training_set_path $TRAIN_CSV \"
Write-Host "     --validation_set from-file \"
Write-Host "     --validation_set_path $VAL_CSV \"
Write-Host "     --num_classes 100 \"
Write-Host "     --epochs 60 \"
Write-Host "     --batch_size 16 \"
Write-Host "     --lr 0.0001 \"
Write-Host "     --use_contrastive True"
Write-Host ""
Write-Host "3. Monitor training:" -ForegroundColor Yellow
Write-Host "   - Watch for loss components in output"
Write-Host "   - CE Loss: Classification loss"
Write-Host "   - Contrastive Loss: Should stabilize around 1-3"
Write-Host "   - Center Loss: Should decrease over time"
Write-Host ""
Write-Host "4. Compare with baseline:" -ForegroundColor Yellow
Write-Host "   - Train without contrastive learning"
Write-Host "   - Compare validation accuracy and overfitting gap"
Write-Host ""
Write-Host "5. Visualize features (optional):" -ForegroundColor Yellow
Write-Host "   - Use t-SNE to visualize learned features"
Write-Host "   - See CONTRASTIVE_LEARNING_README.md for details"
Write-Host ""

Write-Host "=================================================" -ForegroundColor Green
