# Quick Start: Training với Normalize V2

## Chạy với V1 (Mặc định - Standard Normalization)

```bash
python train.py \
    --training_set_path datasets/WLASL100/train.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --epochs 100 \
    --batch_size 24
```

**Hoặc không cần chỉ định gì** (V1 là mặc định):
```bash
python train.py \
    --training_set_path datasets/WLASL100/train.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --use_normalize_v2 False  # Mặc định rồi
```

---

## Chạy với V2 (Semi-Isolated Normalization)

### V2 với neck reference (Recommended)

```bash
python train.py \
    --training_set_path datasets/WLASL100/train.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --epochs 100 \
    --batch_size 24 \
    --use_normalize_v2 True \
    --body_ref_key neck
```

### V2 với nose reference

```bash
python train.py \
    --training_set_path datasets/WLASL100/train.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --use_normalize_v2 True \
    --body_ref_key nose
```

### V2 với shoulder reference

```bash
python train.py \
    --training_set_path datasets/WLASL100/train.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --use_normalize_v2 True \
    --body_ref_key leftShoulder
```

---

## Arguments mới đã thêm

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--use_normalize_v2` | bool | `False` | True/False | Bật normalize V2 (semi-isolated) |
| `--body_ref_key` | str | `"neck"` | neck, nose, leftShoulder, rightShoulder | Điểm tham chiếu cơ thể |

---

## Experiments: So sánh V1 vs V2

### Experiment 1: Baseline Comparison

```bash
# V1 (Baseline)
python train.py \
    --experiment_name "baseline_v1" \
    --training_set_path datasets/WLASL100/train.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --use_normalize_v2 False

# V2 (Semi-isolated)
python train.py \
    --experiment_name "v2_neck" \
    --training_set_path datasets/WLASL100/train.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --use_normalize_v2 True \
    --body_ref_key neck
```

### Experiment 2: Ablation - Body Reference Points

```bash
# V2 with Neck
python train.py \
    --experiment_name "v2_neck" \
    --use_normalize_v2 True \
    --body_ref_key neck \
    [... other args ...]

# V2 with Nose
python train.py \
    --experiment_name "v2_nose" \
    --use_normalize_v2 True \
    --body_ref_key nose \
    [... other args ...]

# V2 with Left Shoulder
python train.py \
    --experiment_name "v2_leftshoulder" \
    --use_normalize_v2 True \
    --body_ref_key leftShoulder \
    [... other args ...]

# V2 with Right Shoulder
python train.py \
    --experiment_name "v2_rightshoulder" \
    --use_normalize_v2 True \
    --body_ref_key rightShoulder \
    [... other args ...]
```

### Experiment 3: Combined with Position Features

```bash
# V1 + Position Features (Current default)
python train.py \
    --experiment_name "v1_with_position" \
    --use_normalize_v2 False \
    --use_position True \
    [... other args ...]

# V2 + Position Features (Recommended)
python train.py \
    --experiment_name "v2_with_position" \
    --use_normalize_v2 True \
    --body_ref_key neck \
    --use_position True \
    [... other args ...]

# V2 without Position Features
python train.py \
    --experiment_name "v2_no_position" \
    --use_normalize_v2 True \
    --body_ref_key neck \
    --use_position False \
    [... other args ...]
```

---

## Validation from File

Nếu dùng validation set từ file:

```bash
python train.py \
    --training_set_path datasets/WLASL100/train.csv \
    --validation_set from-file \
    --validation_set_path datasets/WLASL100/val.csv \
    --testing_set_path datasets/WLASL100/test.csv \
    --num_classes 100 \
    --use_normalize_v2 True \
    --body_ref_key neck
```

---

## Shell Script Template

Tạo file `run_v2_experiments.sh`:

```bash
#!/bin/bash

# Base arguments
BASE_ARGS="--training_set_path datasets/WLASL100/train.csv \
           --testing_set_path datasets/WLASL100/test.csv \
           --num_classes 100 \
           --epochs 100 \
           --batch_size 24 \
           --lr 0.0001"

# Experiment 1: V1 Baseline
echo "Running V1 Baseline..."
python train.py $BASE_ARGS \
    --experiment_name "v1_baseline" \
    --use_normalize_v2 False

# Experiment 2: V2 with Neck
echo "Running V2 with Neck..."
python train.py $BASE_ARGS \
    --experiment_name "v2_neck" \
    --use_normalize_v2 True \
    --body_ref_key neck

# Experiment 3: V2 with Nose
echo "Running V2 with Nose..."
python train.py $BASE_ARGS \
    --experiment_name "v2_nose" \
    --use_normalize_v2 True \
    --body_ref_key nose

echo "All experiments completed!"
```

Chạy:
```bash
chmod +x run_v2_experiments.sh
./run_v2_experiments.sh
```

---

## Lưu ý quan trọng

### ⚠️ Consistency giữa Train và Test
- Nếu train với V2, phải test với V2 (cùng `body_ref_key`)
- Nếu train với V1, phải test với V1
- **KHÔNG được mix V1 và V2**

### ✅ Recommended Setup
Theo experiments của Teledeaf, setup tốt nhất:
```bash
--use_normalize_v2 True \
--body_ref_key neck \
--use_position True
```

### 📊 Monitor Logs
Khi chạy, bạn sẽ thấy output như:
```
✓ CzechSLRDataset: Position features ENABLED (hands will have 4 channels: x, y, rel_x, rel_y)
✓ CzechSLRDataset: Using NORMALIZE V2 (semi-isolated, relative to neck)
```

Hoặc:
```
✓ CzechSLRDataset: Position features ENABLED (hands will have 4 channels: x, y, rel_x, rel_y)
  CzechSLRDataset: Using NORMALIZE V1 (standard bounding box)
```

---

## Troubleshooting

### Dataset không có body landmarks
```
KeyError: 'neck'
```
**Solution**: Thử body_ref_key khác hoặc quay lại V1:
```bash
--use_normalize_v2 False
```

### Performance không tốt hơn V1
- Thử các body_ref_key khác
- Kiểm tra quality của body landmarks
- V2 cần body landmarks chất lượng tốt

---

## Summary

| Scenario | Command |
|----------|---------|
| **Mặc định (V1)** | `python train.py [args]` |
| **Enable V2** | `python train.py [args] --use_normalize_v2 True` |
| **V2 + Neck** | `python train.py [args] --use_normalize_v2 True --body_ref_key neck` |
| **V2 + Nose** | `python train.py [args] --use_normalize_v2 True --body_ref_key nose` |

**Default behavior**: V1 (backward compatible)
**Recommended for new experiments**: V2 with neck reference
