"""
predictor.py - Cau noi giua app.py (Tkinter) va mo hinh MobileNetV2 da huan luyen.

app.py goi: predict_image(image_path) -> (nhan, do_tin_cay 0..1, top_k)
Dat file nay CUNG THU MUC voi app.py. Mo hinh doc tu models/.
"""

import os
import json

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_fruit_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

IMG_SIZE = (224, 224)
TOP_K = 3
USE_VIETNAMESE = True  # True: hien "Tao (Apple)"; False: chi hien ten tieng Anh

VI_NAMES = {
    "Apple": "Táo",
    "Banana": "Chuối",
    "Grape": "Nho",
    "Mango": "Xoài",
    "Strawberry": "Dâu tây",
}

_model = None
_class_names = None


def _load():
    """Nap mo hinh 1 lan duy nhat (lan nhan dien dau tien se hoi cham)."""
    global _model, _class_names
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Khong tim thay mo hinh: {MODEL_PATH}")
        if not os.path.isfile(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"Khong tim thay: {CLASS_NAMES_PATH}")
        from tensorflow import keras  # import tre de app mo nhanh hon
        _model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            _class_names = json.load(f)
    return _model, _class_names


def _display_name(name):
    if USE_VIETNAMESE and name in VI_NAMES:
        return f"{VI_NAMES[name]} ({name})"
    return name


def predict_image(image_path):
    model, class_names = _load()

    # RGB, 224x224, pixel 0-255 (preprocess_input da nam trong mo hinh)
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype="float32")
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    best = int(order[0])
    label = _display_name(class_names[best])
    conf = float(probs[best])
    topk = [(_display_name(class_names[int(i)]), float(probs[int(i)]))
            for i in order[:TOP_K]]
    return label, conf, topk


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_image(sys.argv[1]))
    else:
        print("Cach dung: python predictor.py duong_dan_anh.jpg")
