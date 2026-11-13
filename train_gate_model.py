
import json
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Đường dẫn dự án
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "gate_dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Thông số huấn luyện
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


def load_dataset(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["Fruit", "NonFruit"],
        shuffle=shuffle,
        seed=SEED
    )


def main():
    # Tránh crash khi in tiếng Việt ra console/file không phải UTF-8
    sys.stdout.reconfigure(encoding="utf-8")

    train_ds = load_dataset("train", True)
    valid_ds = load_dataset("valid", False)
    test_ds = load_dataset("test", False)

    # Tối ưu tốc độ đọc dữ liệu
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    # Dùng MobileNetV2 đã được huấn luyện trước
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Ban đầu giữ nguyên trọng số của mô hình nền
    base_model.trainable = False

    # Augmentation chỉ hoạt động khi training
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    best_model_path = MODEL_DIR / "best_gate_model.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # Fruit:NonFruit khoảng 1:5 -> tăng trọng số lớp Fruit
    counts = [
        len(list((DATA_DIR / "train" / name).iterdir()))
        for name in ("Fruit", "NonFruit")
    ]
    class_weight = {
        i: sum(counts) / (2 * count) for i, count in enumerate(counts)
    }
    print("Số ảnh train (Fruit, NonFruit):", counts)
    print("class_weight:", class_weight)

    print("\nBẮT ĐẦU HUẤN LUYỆN...")

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight
    )

    print("\nĐÁNH GIÁ TRÊN TẬP TEST:")
    results = model.evaluate(test_ds, return_dict=True)

    lines = [f"{name}: {value:.4f}" for name, value in results.items()]
    print("\n".join(lines))

    with open(MODEL_DIR / "gate_training_results.txt", "w", encoding="utf-8") as file:
        file.write("KẾT QUẢ GATE TRÊN TẬP TEST\n" + "\n".join(lines) + "\n")

    # Lưu thông tin lớp và thứ tự nhãn
    class_names = {
        "0": "Fruit",
        "1": "NonFruit"
    }

    with open(
        MODEL_DIR / "gate_class_names.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(class_names, file, ensure_ascii=False, indent=4)

    print(f"\nĐÃ LƯU MÔ HÌNH TẠI: {best_model_path}")


if __name__ == "__main__":
    main()
