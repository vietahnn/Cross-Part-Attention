# Semi-Isolated Normalization (V2) for Siformer

## Tổng quan

Đã implement phương pháp **normalize v2** (semi-isolated normalization) từ Teledeaf vào Siformer, được điều chỉnh cho dữ liệu 2D.

## Sự khác biệt giữa V1 và V2

### V1 - Standard Normalization (Mặc định)
```python
# Chỉ normalize trong bounding box của riêng tay
normalize_single_dict(row)
```

**Đặc điểm:**
- Normalize trực tiếp dựa trên bounding box của tay
- Mất thông tin về vị trí tay so với cơ thể
- Chỉ giữ lại thông tin về shape của bàn tay

### V2 - Semi-Isolated Normalization (Mới)
```python
# Tính tọa độ tương đối so với điểm tham chiếu cơ thể, sau đó normalize
normalize_single_dict_v2(row, body_ref_key="neck")
```

**Đặc điểm:**
- **Bước 1**: Chuyển tọa độ tay thành **tọa độ tương đối** so với điểm tham chiếu cơ thể (mặc định là "neck")
  ```python
  x_relative = x_hand - x_neck
  y_relative = y_hand - y_neck
  ```
- **Bước 2**: Normalize trong bounding box của tay (như V1)
- **Ưu điểm**: Giữ được mối quan hệ không gian giữa tay và cơ thể
- Phù hợp cho Sign Language Recognition vì vị trí tay so với cơ thể rất quan trọng

## Cách sử dụng

### 1. Trong Dataset

```python
from datasets.czech_slr_dataset import CzechSLRDataset

# Sử dụng V1 (mặc định)
dataset_v1 = CzechSLRDataset(
    dataset_filename="path/to/data.csv",
    normalize=True,
    use_normalize_v2=False  # V1
)

# Sử dụng V2 (semi-isolated)
dataset_v2 = CzechSLRDataset(
    dataset_filename="path/to/data.csv",
    normalize=True,
    use_normalize_v2=True,  # V2
    body_ref_key="neck"     # Điểm tham chiếu (có thể thay đổi)
)
```

### 2. Sử dụng trực tiếp function

```python
from normalization.hand_normalization import normalize_single_dict_v2

# Row là dict chứa landmarks
# Format: {"wrist_0": [[x1,y1], [x2,y2], ...], "neck": [[x1,y1], ...], ...}
normalized_row = normalize_single_dict_v2(row, body_ref_key="neck")
```

### 3. Các tham số

- **body_ref_key** (str): Tên key của điểm tham chiếu cơ thể
  - Mặc định: `"neck"`
  - Các tùy chọn khác: `"nose"`, `"leftShoulder"`, `"rightShoulder"`, hoặc bất kỳ body landmark nào

## Test Script

Chạy script test để so sánh V1 và V2:

```bash
cd Siformer
python test_normalize_v2.py
```

Script này sẽ:
- Tạo dữ liệu mẫu với hand landmarks và neck position
- Áp dụng cả V1 và V2 normalization
- So sánh kết quả và hiển thị sự khác biệt

## Training với V2

Để train model với normalize v2, update training script:

```python
# train.py
train_dataset = CzechSLRDataset(
    dataset_filename=train_file,
    augmentations=True,
    normalize=True,
    use_normalize_v2=True,  # Enable V2
    body_ref_key="neck"
)

val_dataset = CzechSLRDataset(
    dataset_filename=val_file,
    normalize=True,
    use_normalize_v2=True,  # Enable V2
    body_ref_key="neck"
)
```

**Lưu ý**: Nếu train với V2, phải inference với V2. Không nên mix V1 và V2.

## So sánh hiệu quả

### Khi nào nên dùng V1?
- Dataset có hand landmarks quality tốt
- Không quan tâm đến vị trí tay so với cơ thể
- Muốn model học hand shape thuần túy

### Khi nào nên dùng V2?
- Sign language recognition (vị trí tay quan trọng)
- Dataset có body landmarks chất lượng tốt
- Muốn model học cả shape và spatial relationship
- Có nhiều signs phân biệt bởi vị trí tay (gần/xa cơ thể)

## Kỹ thuật chi tiết

### V2 Algorithm

```python
for each frame:
    # Step 1: Convert to relative coordinates
    for each hand landmark:
        x_rel = x - x_neck
        y_rel = y - y_neck
    
    # Step 2: Find bounding box
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    
    # Step 3: Add padding with aspect ratio adjustment
    if width > height:
        delta_x = 0.1 * width
        delta_y = delta_x + (width - height) / 2
    else:
        delta_y = 0.1 * height
        delta_x = delta_y + (height - width) / 2
    
    # Step 4: Normalize
    x_norm = (x_rel - min_x + delta_x) / (max_x - min_x + 2*delta_x)
    y_norm = (y_rel - min_y + delta_y) / (max_y - min_y + 2*delta_y)
```

## Tham khảo

- **Teledeaf**: [teledeaf-care-model/src/data/dataset.py](../teledeaf-care-model/src/data/dataset.py) (normalize_hand_v2)
- **Siformer Original**: Sign Pose-based Transformer for Word-level Sign Language Recognition
- **Implementation**: Adapted from Teledeaf's 3D approach to Siformer's 2D format

## Troubleshooting

### Lỗi: KeyError 'neck'
- Đảm bảo dataset có body landmarks
- Thử đổi `body_ref_key` sang landmark khác như `"nose"`

### Performance không cải thiện
- Thử các body_ref_key khác
- Kiểm tra quality của body landmarks trong dataset
- V2 có thể không phù hợp nếu body landmarks bị noise nhiều

## Kết luận

Phương pháp V2 implement thành công cách normalize của Teledeaf vào Siformer với các điều chỉnh cho 2D:
- ✅ Tính tọa độ tương đối so với body reference
- ✅ Aspect ratio adjustment
- ✅ Tương thích với data pipeline hiện tại
- ✅ Dễ dàng switch giữa V1 và V2
