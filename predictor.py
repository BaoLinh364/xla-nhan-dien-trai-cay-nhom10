"""
predictor.py - Nhận diện một ảnh bằng best_fruits_model (6 lớp: 5 loại quả + unknown).

Một mô hình, một lần suy luận: nhãn cuối cùng là lớp có xác suất softmax cao nhất.
Nếu lớp đó là unknown, ảnh không thuộc 5 loại quả được hỗ trợ.

Dùng thử: python predictor.py duong_dan_anh.jpg
"""

import json
import os

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_fruits_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
IMG_SIZE = (224, 224)  # phải trùng IMG_SIZE trong train_model.py

VI_NAMES = {
    "Apple": "Táo (Apple)",
    "Banana": "Chuối (Banana)",
    "Grape": "Nho (Grape)",
    "Mango": "Xoài (Mango)",
    "Strawberry": "Dâu tây (Strawberry)",
    "unknown": "Không xác định (unknown)",
}

_model = None
_class_names = None


def load_model():
    """Nạp mô hình và danh sách lớp một lần, dùng lại cho các lần sau."""
    global _model, _class_names
    if _model is None:
        for path in (MODEL_PATH, CLASS_NAMES_PATH):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Không tìm thấy:\n{path}")
        from tensorflow import keras  # nhập muộn để app hiện giao diện trước
        _model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            _class_names = json.load(f)
        if len(_class_names) != int(_model.output_shape[-1]):
            raise ValueError("Số lớp trong class_names.json khác số đầu ra của mô hình.")
    return _model, _class_names


def display_name(name):
    return VI_NAMES.get(name, name)


def load_image_array(image_path):
    """Đọc ảnh thành mảng (1, 224, 224, 3), giá trị 0..255.

    - exif_transpose: xoay đúng ảnh chụp dọc từ điện thoại.
    - BILINEAR: giống image_dataset_from_directory lúc huấn luyện.
    Chuẩn hóa về [-1, 1] nằm sẵn trong mô hình nên không làm ở đây.
    """
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def to_percentages(probs, decimals=2):
    """Đổi xác suất softmax sang phần trăm làm tròn sao cho tổng hiển thị đúng 100.

    Dùng phương pháp phần dư lớn nhất: làm tròn xuống, rồi cộng phần còn thiếu
    cho các lớp có phần lẻ lớn nhất. Giá trị gốc vẫn là đầu ra softmax của mô hình.
    """
    scale = 10 ** decimals
    raw = np.asarray(probs, dtype=np.float64)
    raw = raw / raw.sum() * 100 * scale            # softmax có sai số float nhỏ, chuẩn về đúng 100%
    floors = np.floor(raw).astype(np.int64)
    missing = 100 * scale - floors.sum()
    for i in np.argsort(-(raw - floors))[:missing]:
        floors[i] += 1
    return [v / scale for v in floors]


def predict_image(image_path):
    """
    Trả về:
        label:  tên lớp có xác suất cao nhất (một trong class_names)
        probs:  danh sách (tên lớp, xác suất softmax) cho CẢ 6 lớp, theo đúng thứ tự class_names
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh:\n{image_path}")
    model, class_names = load_model()
    output = np.asarray(model.predict(load_image_array(image_path), verbose=0)[0], dtype=np.float64)
    label = class_names[int(output.argmax())]
    return label, list(zip(class_names, output.tolist()))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # tránh lỗi in tiếng Việt trên console không phải UTF-8
    if len(sys.argv) < 2:
        print("Cách dùng: python predictor.py duong_dan_anh.jpg")
        raise SystemExit(1)
    label, probs = predict_image(sys.argv[1])
    print("Kết quả:", display_name(label))
    for (name, _), pct in zip(probs, to_percentages([p for _, p in probs])):
        print(f"  {display_name(name):28} {pct:6.2f}%")
