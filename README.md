# Nhận diện trái cây (Nhóm 10)

Ứng dụng desktop (Tkinter): chọn một ảnh, app cho biết ảnh thuộc lớp nào trong **6 lớp**:
Táo (Apple), Chuối (Banana), Nho (Grape), Xoài (Mango), Dâu tây (Strawberry) và
**unknown** — ảnh không thuộc 5 loại quả trên.

## Thiết kế: một mô hình duy nhất

- Một mạng **MobileNetV2** (học chuyển giao từ ImageNet) với lớp đầu ra **Softmax 6 lớp**.
- Một lần suy luận quyết định tất cả: nhãn cuối cùng là lớp có xác suất cao nhất;
  ảnh không thuộc 5 loại quả được xếp vào lớp `unknown`.
- App hiển thị nhãn cuối cùng và **xác suất của cả 6 lớp** (lấy trực tiếp từ softmax, tổng đúng 100%).
- Thứ tự lớp dùng chung ở mọi nơi (lưu trong `models/class_names.json`):
  `Apple, Banana, Grape, Mango, Strawberry, unknown`.

## Chạy nhanh (Windows)

Cần **Python 3.12** (https://www.python.org/downloads/, tick "Add python.exe to PATH").

Bấm đúp **`run_app.bat`**: lần đầu tự tạo `.venv` và cài thư viện (cần Internet, vài phút);
các lần sau mở app ngay.

Chạy thủ công:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Nhận diện một ảnh bằng dòng lệnh: `python predictor.py duong_dan_anh.jpg`

## Cấu trúc dự án

```
app.py               Giao diện Tkinter: nhãn cuối cùng + xác suất 6 lớp
predictor.py         Nạp mô hình, tiền xử lý ảnh, predict_image(), to_percentages()
prepare_dataset.py   Dựng dataset/ từ dữ liệu gốc: lọc trùng, chia tập, kiểm tra rò rỉ, resize
train_model.py       Huấn luyện best_fruits_model, lưu lịch sử và siêu tham số
evaluate_model.py    Đánh giá trên tập test, lưu chỉ số, ma trận nhầm lẫn
check_dataset.py     Kiểm tra dataset/: ảnh hỏng, trùng MD5, rò rỉ giữa các tập
test_utils.py        Kiểm tra nhanh các hàm tiện ích (python test_utils.py)
dataset/             Dữ liệu đã xử lý (được commit cùng code)
data/excluded_images.csv  Ảnh gốc đã duyệt bằng mắt: chuyển sang unknown hoặc bỏ (cột xu_ly)
models/              best_fruits_model.keras + class_names.json
reports/             Mọi kết quả do code sinh ra (JSON, CSV, PNG)
```

## Dữ liệu

```
dataset/
├── train/  Apple/  Banana/  Grape/  Mango/  Strawberry/  unknown/
├── val/    (6 thư mục lớp như trên)
└── test/   (6 thư mục lớp như trên)
```

- **5 loại quả**: bộ dữ liệu công khai *Fruits Classification* trên Kaggle
  (`utkarshsaxenadn/fruits-classification`, 2.000 ảnh mỗi lớp).
- **unknown**: 3.000 ảnh không phải trái cây, lấy đều từ 22 nhóm (10 loài động vật, chó, mèo,
  6 loại phong cảnh, 4 loại phương tiện), cộng 556 ảnh bánh, đồ uống, món ăn có trái cây
  lấy ra từ bộ 5 loại quả (xem bước 3 bên dưới).
- `prepare_dataset.py` xử lý trên **ảnh gốc** (trước khi resize):
  1. Bỏ ảnh trùng MD5 (ảnh giống hệt ở nhiều lớp thì bỏ hết).
  2. Bỏ ảnh gần trùng trong cùng lớp bằng dHash (cách nhau ≤ 4/64 bit).
  3. Xử lý các ảnh trong `data/excluded_images.csv` (ảnh đã duyệt bằng mắt mà trái cây không phải
     chủ thể chính), theo cột `xu_ly`:
     - `unknown`: bánh, kem, dâu nhúng chocolate, đồ uống, món ăn → chuyển sang lớp `unknown`
       để mô hình biết từ chối ảnh món ăn có trái cây;
     - `bo`: nhiều loại quả, sạp/giỏ quả, bao bì, chữ, hình vẽ, ảnh người cầm quả → bỏ.

     Ứng viên được khoanh vùng bằng MobileNetV2 ImageNet, sau đó duyệt từng ảnh;
     riêng lớp Dâu tây được duyệt toàn bộ.
  4. Chia mỗi lớp 80/10/10 (seed 42).
  5. Kiểm tra rò rỉ: ảnh train trùng/gần trùng với bất kỳ ảnh val/test nào bị loại khỏi train.
  6. Resize cạnh dài ≤ 256 px, lưu JPEG chất lượng 90.

  Kết quả chi tiết: `reports/data_audit.json`, `reports/removed_images.csv`, `reports/dataset_manifest.csv`.

Dựng lại dataset (chỉ cần khi muốn làm lại từ đầu; `dataset/` đã có sẵn trong repo):

```bash
python prepare_dataset.py --fruits-zip archive.zip --gate-dir <thư mục có */NonFruit>
```

## Huấn luyện và đánh giá

```bash
python train_model.py      # -> models/best_fruits_model.keras, reports/training_*.csv/png, hyperparameters.json
python evaluate_model.py   # -> reports/metrics.json, classification_report.csv, confusion_matrix.png/csv
```

`train_model.py` dùng `class_weight` tỷ lệ nghịch với số ảnh mỗi lớp để lớp `unknown`
không lấn át và cũng không bị bỏ qua. TensorFlow không dùng GPU trên Windows native,
nên train trên CPU khá chậm; có thể chạy cùng script trên máy có GPU.

## Kết quả

Đánh giá trên tập test (1.067 ảnh chưa dùng khi huấn luyện; `python evaluate_model.py`):

- Accuracy: **93,72%**; macro F1: 0,9291; weighted F1: 0,9375.
- Recall lớp `unknown` (ảnh không thuộc 5 loại quả bị từ chối đúng): **97,73%**.
- Ảnh bánh, đồ uống, món ăn có trái cây trong tập test được nhận đúng là `unknown`: 44/50 ảnh.
- Ảnh trái cây thật bị xếp nhầm vào `unknown`: 1,40% (10 ảnh).
- Huấn luyện trên Kaggle (GPU T4): 8 epoch đầu phân loại + 13 epoch tinh chỉnh, tốt nhất ở epoch 17 (val_accuracy 94,18%), khoảng 7 phút. Đánh giá lại trên CPU cho kết quả trùng khớp từng ảnh.

| Lớp | Precision | Recall | F1 | Số ảnh |
|---|---|---|---|---|
| Apple | 0,8323 | 0,9306 | 0,8787 | 144 |
| Banana | 0,9517 | 0,9583 | 0,9550 | 144 |
| Grape | 0,9392 | 0,8968 | 0,9175 | 155 |
| Mango | 0,9484 | 0,8855 | 0,9159 | 166 |
| Strawberry | 0,9417 | 0,9238 | 0,9327 | 105 |
| unknown | 0,9718 | 0,9773 | 0,9746 | 353 |
| macro avg | 0,9309 | 0,9287 | 0,9291 | 1067 |
| weighted avg | 0,9389 | 0,9372 | 0,9375 | 1067 |

Đồ thị huấn luyện: `reports/training_curves.png`; ma trận nhầm lẫn: `reports/confusion_matrix.png`;
lịch sử từng epoch: `reports/training_history.csv`; siêu tham số: `reports/hyperparameters.json`.

## Git

- `dataset/` và `models/best_fruits_model.keras` **được commit** cùng code.
- `.gitignore` chỉ loại `.venv/`, `__pycache__/`, thư mục IDE, file tạm và các file `.zip`.
- Không file nào vượt 100 MB (mô hình ~24 MB, dataset ~154 MB gồm hàng nghìn ảnh nhỏ),
  nên không cần Git LFS.

## Giới hạn

- Lớp `unknown` chưa có ảnh các loại quả khác (cam, chanh, cà chua…) và ảnh người,
  nên những ảnh này có thể bị nhận nhầm thành một trong 5 lớp.
- Chỉ lớp Dâu tây được duyệt bằng mắt toàn bộ; 4 lớp quả còn lại chỉ duyệt các ảnh được
  MobileNetV2 ImageNet khoanh vùng, nên có thể còn sót một ít ảnh không phù hợp.
