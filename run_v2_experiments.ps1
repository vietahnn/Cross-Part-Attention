# run_v2_experiments.ps1
# PowerShell script to run experiments comparing Normalize V1 vs V2

# ============================================================================
# Configuration
# ============================================================================
$TRAIN_PATH = "datasets/WLASL100/train.csv"
$TEST_PATH = "datasets/WLASL100/test.csv"
$NUM_CLASSES = 100
$EPOCHS = 100
$BATCH_SIZE = 24
$LR = 0.0001

# Base arguments
$BASE_ARGS = @(
    "--training_set_path", $TRAIN_PATH,
    "--testing_set_path", $TEST_PATH,
    "--num_classes", $NUM_CLASSES,
    "--epochs", $EPOCHS,
    "--batch_size", $BATCH_SIZE,
    "--lr", $LR
)

Write-Host "========================================================================"
Write-Host "Starting Normalize V2 Experiments"
Write-Host "========================================================================"
Write-Host ""

# ============================================================================
# Experiment 1: V1 Baseline (Standard Normalization)
# ============================================================================
Write-Host "========================================================================"
Write-Host "Running Experiment 1: V1 Baseline (Standard Normalization)"
Write-Host "========================================================================"
python train.py @BASE_ARGS `
    --experiment_name "exp1_v1_baseline" `
    --use_normalize_v2 False `
    --use_position True

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in Experiment 1. Exiting..." -ForegroundColor Red
    exit 1
}

# ============================================================================
# Experiment 2: V2 with Neck Reference
# ============================================================================
Write-Host ""
Write-Host "========================================================================"
Write-Host "Running Experiment 2: V2 with Neck Reference"
Write-Host "========================================================================"
python train.py @BASE_ARGS `
    --experiment_name "exp2_v2_neck" `
    --use_normalize_v2 True `
    --body_ref_key "neck" `
    --use_position True

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in Experiment 2. Continuing..." -ForegroundColor Yellow
}

# ============================================================================
# Experiment 3: V2 with Nose Reference
# ============================================================================
Write-Host ""
Write-Host "========================================================================"
Write-Host "Running Experiment 3: V2 with Nose Reference"
Write-Host "========================================================================"
python train.py @BASE_ARGS `
    --experiment_name "exp3_v2_nose" `
    --use_normalize_v2 True `
    --body_ref_key "nose" `
    --use_position True

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in Experiment 3. Continuing..." -ForegroundColor Yellow
}

# ============================================================================
# Experiment 4: V2 with Left Shoulder Reference
# ============================================================================
Write-Host ""
Write-Host "========================================================================"
Write-Host "Running Experiment 4: V2 with Left Shoulder Reference"
Write-Host "========================================================================"
python train.py @BASE_ARGS `
    --experiment_name "exp4_v2_leftshoulder" `
    --use_normalize_v2 True `
    --body_ref_key "leftShoulder" `
    --use_position True

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in Experiment 4. Continuing..." -ForegroundColor Yellow
}

# ============================================================================
# Experiment 5: V2 without Position Features (Ablation)
# ============================================================================
Write-Host ""
Write-Host "========================================================================"
Write-Host "Running Experiment 5: V2 with Neck, No Position Features"
Write-Host "========================================================================"
python train.py @BASE_ARGS `
    --experiment_name "exp5_v2_neck_no_position" `
    --use_normalize_v2 True `
    --body_ref_key "neck" `
    --use_position False

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in Experiment 5. Continuing..." -ForegroundColor Yellow
}

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "========================================================================"
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "========================================================================"
Write-Host "Results saved in:"
Write-Host "  - Checkpoints: out-checkpoints/"
Write-Host "  - Plots: out-img/"
Write-Host ""
Write-Host "Experiments run:"
Write-Host "  1. exp1_v1_baseline          - V1 baseline"
Write-Host "  2. exp2_v2_neck              - V2 with neck reference"
Write-Host "  3. exp3_v2_nose              - V2 with nose reference"
Write-Host "  4. exp4_v2_leftshoulder      - V2 with left shoulder reference"
Write-Host "  5. exp5_v2_neck_no_position  - V2 with neck, no position features"
Write-Host ""
Write-Host "To compare results, check the generated plots and model checkpoints."
