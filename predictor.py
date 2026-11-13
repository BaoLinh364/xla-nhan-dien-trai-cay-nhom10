import os
import json

import numpy as np
from PIL import Image, ImageOps

# =========================
# CẤU HÌNH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Mô hình nhận diện trái cây
MODEL_PATH = os.path.join(
    MODEL_DIR, "best_fruit_model.keras"
)
CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR, "class_names.json"
)

# Mô hình lọc ảnh
GATE_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_gate_model.keras"
)
GATE_CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR, "gate_class_names.json"
)

IMG_SIZE = (224, 224)
TOP_K = 3

# Ngưỡng cảnh báo dự đoán trái cây không chắc chắn
CONFIDENCE_THRESHOLD = 0.60

# Ngưỡng lọc ảnh không phải trái cây
GATE_NONFRUIT_THRESHOLD = 0.50

VI_NAMES = {
    "Apple": "Táo",
    "Banana": "Chuối",
    "Grape": "Nho",
    "Mango": "Xoài",
    "Strawberry": "Dâu tây",
}

_fruit_model = None
_class_names = None

_gate_model = None
_gate_class_names = None


# =========================
# TẢI MÔ HÌNH TRÁI CÂY
# =========================

def load_model():
    global _fruit_model, _class_names

    if _fruit_model is None:

        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy mô hình trái cây:\n{MODEL_PATH}"
            )

        if not os.path.isfile(CLASS_NAMES_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy danh sách lớp:\n{CLASS_NAMES_PATH}"
            )

        from tensorflow import keras

        _fruit_model = keras.models.load_model(MODEL_PATH)

        with open(
            CLASS_NAMES_PATH, "r", encoding="utf-8"
        ) as file:
            _class_names = json.load(file)

        if len(_class_names) != int(
            _fruit_model.output_shape[-1]
        ):
            raise ValueError(
                "Số lớp trong class_names.json không khớp "
                "với số lớp đầu ra của mô hình trái cây."
            )

    return _fruit_model, _class_names


# =========================
# TẢI MÔ HÌNH LỌC ẢNH
# =========================

def load_gate_model():
    global _gate_model, _gate_class_names

    if _gate_model is None:

        if not os.path.isfile(GATE_MODEL_PATH):
            raise FileNotFoundError(
                "Không tìm thấy mô hình lọc ảnh:\n"
                f"{GATE_MODEL_PATH}"
            )

        if not os.path.isfile(GATE_CLASS_NAMES_PATH):
            raise FileNotFoundError(
                "Không tìm thấy danh sách lớp của mô hình lọc:\n"
                f"{GATE_CLASS_NAMES_PATH}"
            )

        from tensorflow import keras

        _gate_model = keras.models.load_model(
            GATE_MODEL_PATH
        )

        with open(
            GATE_CLASS_NAMES_PATH, "r", encoding="utf-8"
        ) as file:
            _gate_class_names = json.load(file)

        if int(_gate_model.output_shape[-1]) != 1:
            raise ValueError(
                "Mô hình lọc ảnh phải có một đầu ra sigmoid."
            )

        if isinstance(_gate_class_names, dict):
            if (
                _gate_class_names.get("0") != "Fruit"
                or _gate_class_names.get("1") != "NonFruit"
            ):
                raise ValueError(
                    "gate_class_names.json cần có nội dung "
                    '{"0": "Fruit", "1": "NonFruit"}.'
                )

    return _gate_model, _gate_class_names


# =========================
# CHUYỂN TÊN LỚP SANG TIẾNG VIỆT
# =========================

def display_name(name):

    if name in VI_NAMES:
        return f"{VI_NAMES[name]} ({name})"

    return name


# =========================
# ĐỌC ẢNH
# =========================

def load_image_array(image_path):
    """
    Đọc ảnh thành mảng (1, 224, 224, 3), giá trị 0..255.

    - exif_transpose: xoay đúng ảnh chụp dọc từ điện thoại.
    - BILINEAR: giống image_dataset_from_directory lúc train.
    Không cần preprocess_input: cả hai mô hình đã có lớp
    tiền xử lý bên trong.
    """

    with Image.open(image_path) as image:

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)

        image_array = np.asarray(
            image, dtype=np.float32
        )

    return np.expand_dims(image_array, axis=0)


# =========================
# KIỂM TRA ẢNH CÓ PHẢI TRÁI CÂY
# =========================

def check_fruit_gate(image_array):
    """
    Trả về:
        is_fruit: True nếu bộ lọc chấp nhận ảnh là trái cây.
        nonfruit_probability: xác suất ảnh thuộc lớp NonFruit.

    Mô hình gate được huấn luyện với:
        0 = Fruit
        1 = NonFruit

    Đầu ra sigmoid biểu thị xác suất của lớp NonFruit.
    """

    gate_model, _ = load_gate_model()

    prediction = gate_model.predict(
        image_array, verbose=0
    )

    nonfruit_probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    if not 0.0 <= nonfruit_probability <= 1.0:
        raise ValueError(
            "Mô hình lọc ảnh trả về xác suất không hợp lệ."
        )

    is_fruit = (
        nonfruit_probability < GATE_NONFRUIT_THRESHOLD
    )

    return is_fruit, nonfruit_probability


# =========================
# NHẬN DIỆN ẢNH
# =========================

def predict_image(image_path):
    """
    Quy trình:
        1. Kiểm tra ảnh bằng mô hình gate.
        2. Nếu không phải trái cây, từ chối phân loại.
        3. Nếu được chấp nhận, nhận diện loại trái cây.

    Trả về:
        status: "rejected" (không phải trái cây),
                "uncertain" (độ tin cậy thấp) hoặc "ok".
        label: nhãn dự đoán.
        confidence: độ tin cậy. Với "rejected" là xác suất NonFruit.
        topk: danh sách tối đa 3 dự đoán.
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Không tìm thấy ảnh:\n{image_path}"
        )

    image_array = load_image_array(image_path)

    # BƯỚC 1: LỌC ẢNH
    is_fruit, nonfruit_probability = check_fruit_gate(
        image_array
    )

    if not is_fruit:

        # Nhãn ngắn để vừa ô kết quả; app hiện lời nhắc chi tiết riêng
        return "rejected", "Không phải trái cây", nonfruit_probability, []

    # BƯỚC 2: NHẬN DIỆN LOẠI TRÁI CÂY
    model, class_names = load_model()

    predictions = model.predict(
        image_array, verbose=0
    )[0]

    predictions = np.asarray(
        predictions, dtype=float
    )

    if len(predictions) != len(class_names):
        raise ValueError(
            "Kết quả dự đoán không khớp với danh sách lớp."
        )

    # Sắp xếp xác suất từ cao xuống thấp
    sorted_indices = np.argsort(predictions)[::-1]

    best_index = int(sorted_indices[0])

    confidence = float(predictions[best_index])

    predicted_class = class_names[best_index]

    # Lấy tối đa 3 dự đoán cao nhất
    topk = []

    for index in sorted_indices[:TOP_K]:

        index = int(index)

        topk.append((
            display_name(class_names[index]),
            float(predictions[index])
        ))

    # Kiểm tra độ tin cậy
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", "Chưa chắc chắn", confidence, topk

    return "ok", display_name(predicted_class), confidence, topk


# =========================
# KIỂM TRA ĐỘC LẬP
# =========================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Cách sử dụng:\n"
            "python predictor.py duong_dan_anh.jpg"
        )

    else:
        try:
            # Tránh crash khi console không phải UTF-8
            sys.stdout.reconfigure(encoding="utf-8")

            status, label, confidence, topk = predict_image(sys.argv[1])

            print("Nhãn dự đoán:", label)

            print(
                "Xác suất không phải trái cây:"
                if status == "rejected" else "Độ tin cậy:",
                f"{confidence * 100:.2f}%"
            )

            if topk:
                print("Các dự đoán khác:")

                for name, probability in topk:
                    print(
                        f"- {name}: "
                        f"{probability * 100:.2f}%"
                    )

        except Exception as error:
            print("Lỗi:", error)