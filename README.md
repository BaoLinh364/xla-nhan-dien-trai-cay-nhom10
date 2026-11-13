# Nhận diện trái cây (Nhóm 10)

Ứng dụng desktop (Tkinter): chọn một ảnh, app nhận diện 1 trong 5 loại trái cây
Táo (Apple), Chuối (Banana), Nho (Grape), Xoài (Mango), Dâu tây (Strawberry).

Ảnh đi qua 2 mô hình MobileNetV2 (transfer learning):

1. **Gate** (`models/best_gate_model.keras`): ảnh có phải trái cây không.
   Nếu xác suất NonFruit ≥ 0.5, app từ chối ảnh.
2. **Phân loại** (`models/best_fruit_model.keras`): 5 lớp, hiện top-3.
   Nếu độ tin cậy < 0.6, app báo "Chưa chắc chắn".

## Chạy nhanh (Windows)

Cần **Python 3.12** (https://www.python.org/downloads/, tick "Add python.exe to PATH").
TensorFlow chưa hỗ trợ Python 3.14+.

Bấm đúp **`run_app.bat`**. Lần đầu file này tự tạo `.venv` và cài thư viện
(cần Internet, khoảng 5–10 phút); các lần sau mở app ngay.

## Chạy thủ công

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Thử nhanh một ảnh bằng dòng lệnh: `python predictor.py duong_dan_anh.jpg`

## Cấu trúc

```
app.py               Giao diện Tkinter
predictor.py         Nạp 2 mô hình, predict_image()
train_model.py       Train mô hình 5 lớp
train_gate_model.py  Train gate Fruit / NonFruit
evaluate_model.py    Đánh giá mô hình 5 lớp, gate và cả chuỗi như trong app
check_dataset.py     Kiểm tra dataset: ảnh hỏng, ảnh trùng, rò rỉ giữa các tập
models/              Mô hình đã train + báo cáo (app chỉ cần thư mục này)
dataset/             train | valid | test × 5 lớp        (chỉ cần khi train/đánh giá)
gate_dataset/        train | valid | test × Fruit, NonFruit (chỉ cần khi train gate)
```

## Đóng gói gửi người khác

App chỉ cần code + `models/` (~34MB). Khi nén zip:

- **Nên bỏ** `.venv/` (người nhận tự tạo bằng `run_app.bat`) và `gate_dataset/` (~1,3GB).
- `dataset/` (~82MB) chỉ cần nếu người nhận muốn chạy `evaluate_model.py` hoặc train lại.
  Thiếu `gate_dataset/` thì `evaluate_model.py` tự bỏ qua phần đánh giá gate.

Mọi đường dẫn trong code đều tính theo vị trí file, nên giải nén ở đâu cũng chạy được.

## Dữ liệu

- **Trái cây**: từ dataset Kaggle `utkarshsaxenadn/fruits-classification` (10.000 ảnh).
  Bản gốc chia 97/2/1 và có ảnh trùng, ảnh rò rỉ giữa các tập, ảnh nhãn sai.
  Nhóm đã loại 632 ảnh (263 gần trùng, 200 trùng hoàn toàn, 127 nhãn sai/ảnh nhiễu
  duyệt tay, 40 ảnh nhỏ hơn 64px, 2 ảnh mang 2 nhãn) và chia lại 80/10/10:
  7.494 / 937 / 937 ảnh.
- **Gate**: Fruit (10.000 ảnh) và NonFruit (48.618 ảnh: động vật, phong cảnh, xe).

## Kết quả (tập test của dataset đã dọn, 937 ảnh)

| | Kết quả |
|---|---|
| Mô hình phân loại 5 lớp | **93.81%** (macro F1 0.938) |
| Cả chuỗi như app hiển thị (gate + phân loại + ngưỡng 0.6) | **89.54%** hiện đúng tên; 2.7% bị gate chặn nhầm, 3.5% "Chưa chắc chắn", 4.3% sai |
| Gate trên `gate_dataset/test` (8.809 ảnh) | **99.44%** |

Chi tiết: `models/evaluation_results.txt`, `models/confusion_matrix.png`, `models/training_history.png`.

## Giới hạn đã biết

- Dữ liệu NonFruit chủ yếu là động vật và phong cảnh; chưa có người, đồ ăn,
  rau củ hay trái cây ngoài 5 lớp. Ảnh loại này có thể lọt qua gate và bị gán
  nhầm vào 5 lớp.
- Lớp Apple dễ nhầm nhất (F1 0.905), chủ yếu với Grape và Mango.
- TensorFlow không dùng GPU trên Windows native nên train trên CPU khá chậm.
