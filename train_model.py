"""
train_model.py - Huấn luyện MỘT mô hình MobileNetV2 phân loại 6 lớp:
5 loại quả (Apple, Banana, Grape, Mango, Strawberry) + unknown (ảnh không thuộc 5 loại quả).

Đầu vào : dataset/train, dataset/val (tạo bởi prepare_dataset.py)
Đầu ra  : models/best_fruits_model.keras   mô hình có val_accuracy cao nhất (dùng cho app)
          models/class_names.json           danh sách lớp đúng thứ tự đầu ra của mô hình
          reports/training_history.csv      loss/accuracy train-val, learning rate, thời gian từng epoch
          reports/training_curves.png       đồ thị accuracy, loss, learning rate
          reports/hyperparameters.json      siêu tham số, kích thước các tập, trọng số lớp
"""

import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # lưu ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================ CẤU HÌNH ============================
CLASS_NAMES = ["Apple", "Banana", "Grape", "Mango", "Strawberry", "unknown"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 8            # giai đoạn 1: chỉ huấn luyện phần đầu phân loại
EPOCHS_FINE = 15           # giai đoạn 2: tinh chỉnh các lớp cuối (có dừng sớm)
LR_HEAD = 1e-3
LR_FINE = 1e-5
FINE_TUNE_AT = 100         # mở khóa các lớp của MobileNetV2 từ chỉ số này trở đi
PATIENCE = 4
SEED = 42
AUGMENTATION = {"RandomFlip": "horizontal", "RandomRotation": 0.1, "RandomZoom": 0.1, "RandomContrast": 0.1}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(MODELS_DIR, "best_fruits_model.keras")
# ==================================================================


def load_split(split, shuffle):
    """Đọc dataset/<split>; nhãn one-hot theo đúng thứ tự CLASS_NAMES."""
    return keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, split), class_names=CLASS_NAMES, label_mode="categorical",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED)


def count_images(split):
    return {c: len(os.listdir(os.path.join(DATASET_DIR, split, c))) for c in CLASS_NAMES}


def build_model():
    """MobileNetV2 (ImageNet) + đầu phân loại 6 lớp. Augmentation và chuẩn hóa nằm trong mô hình."""
    augmentation = keras.Sequential([
        layers.RandomFlip(AUGMENTATION["RandomFlip"]),
        layers.RandomRotation(AUGMENTATION["RandomRotation"]),
        layers.RandomZoom(AUGMENTATION["RandomZoom"]),
        layers.RandomContrast(AUGMENTATION["RandomContrast"]),
    ], name="augmentation")
    base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)                                   # chỉ hoạt động khi huấn luyện
    x = keras.applications.mobilenet_v2.preprocess_input(x)    # [0, 255] -> [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return keras.Model(inputs, outputs, name="fruits_mobilenetv2"), base


class EpochLogger(keras.callbacks.Callback):
    """Ghi thêm learning rate và thời gian mỗi epoch vào lịch sử."""

    def __init__(self, phase, rows):
        super().__init__()
        self.phase, self.rows = phase, rows

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.rows.append({
            "epoch": len(self.rows) + 1, "phase": self.phase,
            "loss": logs["loss"], "accuracy": logs["accuracy"],
            "val_loss": logs["val_loss"], "val_accuracy": logs["val_accuracy"],
            "learning_rate": float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)),
            "time_s": round(time.time() - self.t0, 1),
        })


def plot_history(rows, boundary, path):
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(ep, [r[key] for r in rows], marker="o", ms=3, label="Train")
        ax.plot(ep, [r["val_" + key] for r in rows], marker="o", ms=3, label="Validation")
        ax.set_title(title)
    axes[2].plot(ep, [r["learning_rate"] for r in rows], marker="o", ms=3, color="tab:green")
    axes[2].set_yscale("log")
    axes[2].set_title("Learning rate")
    for ax in axes:
        ax.axvline(boundary + 0.5, color="gray", linestyle="--", label="Bắt đầu fine-tune")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    print("TensorFlow:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU") or "không (CPU)")

    train_ds = load_split("train", shuffle=True).prefetch(tf.data.AUTOTUNE)
    val_ds = load_split("val", shuffle=False).prefetch(tf.data.AUTOTUNE)
    with open(os.path.join(MODELS_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)

    # Trọng số lớp tỷ lệ nghịch với số ảnh: unknown không lấn át, cũng không bị bỏ qua
    train_counts = count_images("train")
    total = sum(train_counts.values())
    class_weight = {i: total / (len(CLASS_NAMES) * train_counts[c]) for i, c in enumerate(CLASS_NAMES)}
    print("Số ảnh train:", train_counts)
    print("class_weight:", {CLASS_NAMES[i]: round(w, 3) for i, w in class_weight.items()})

    model, base = build_model()
    rows = []
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", mode="max",
                                                 save_best_only=True, verbose=1)
    t_start = time.time()

    # Giai đoạn 1: đóng băng MobileNetV2, chỉ học phần đầu phân loại
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, class_weight=class_weight,
              callbacks=[checkpoint, EpochLogger("head", rows),
                         keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                                       patience=PATIENCE, restore_best_weights=True)])
    boundary = len(rows)

    # Giai đoạn 2: mở khóa các lớp cuối, giữ BatchNorm ở chế độ suy luận
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(LR_FINE), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, class_weight=class_weight,
              callbacks=[checkpoint, EpochLogger("fine_tune", rows),
                         keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                                       patience=PATIENCE, restore_best_weights=True)])
    total_time = time.time() - t_start

    # Lưu lịch sử, đồ thị, siêu tham số
    with open(os.path.join(REPORT_DIR, "training_history.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_history(rows, boundary, os.path.join(REPORT_DIR, "training_curves.png"))
    best = max(rows, key=lambda r: r["val_accuracy"])
    params = {
        "model": "MobileNetV2 (ImageNet) + GAP + Dropout(0.3) + Dense(6, softmax)",
        "class_names": CLASS_NAMES, "image_size": list(IMG_SIZE), "batch_size": BATCH_SIZE,
        "optimizer": "Adam", "loss": "categorical_crossentropy",
        "learning_rate": {"head": LR_HEAD, "fine_tune": LR_FINE},
        "epochs_max": {"head": EPOCHS_HEAD, "fine_tune": EPOCHS_FINE},
        "epochs_run": {"head": boundary, "fine_tune": len(rows) - boundary},
        "fine_tune_from_layer": FINE_TUNE_AT, "early_stopping_patience": PATIENCE,
        "augmentation": AUGMENTATION, "dropout": 0.3, "seed": SEED,
        "split_sizes": {s: count_images(s) for s in ("train", "val", "test")},
        "class_weight": {CLASS_NAMES[i]: round(w, 4) for i, w in class_weight.items()},
        "trainable_params": int(sum(np.prod(v.shape) for v in model.trainable_weights)),
        "total_params": int(model.count_params()),
        "best_epoch": best["epoch"], "best_val_accuracy": best["val_accuracy"], "best_val_loss": best["val_loss"],
        "training_time_s": round(total_time, 1),
        "tensorflow": tf.__version__,
    }
    with open(os.path.join(REPORT_DIR, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"\nXong sau {total_time / 60:.1f} phút. Epoch tốt nhất: {best['epoch']} "
          f"(val_accuracy {best['val_accuracy']:.4f}). Mô hình: {MODEL_PATH}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # in tiếng Việt cả khi output bị chuyển hướng trên Windows
    main()
