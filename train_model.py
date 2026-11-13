"""
train_model.py - Huan luyen MobileNetV2 phan loai 5 lop trai cay (Transfer Learning)

Dau vao : dataset/train, dataset/valid, dataset/test
Dau ra  : models/best_fruit_model.keras   (mo hinh tot nhat theo val_accuracy)
          models/fruit_model.keras       (mo hinh o cuoi qua trinh train)
          models/class_names.json        (danh sach lop dung thu tu)
          models/training_history.png    (bieu do Accuracy/Loss)
          models/training_results.txt    (ket qua valid va test)

"""

import os
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")  # luu ra file, khong can mo cua so
import matplotlib.pyplot as plt

# ============================ CAU HINH ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 8          # giai doan 1: chi train phan phan loai
EPOCHS_FINE = 15         # giai doan 2: fine-tune, co EarlyStopping nen dung som neu het cai thien
FINE_TUNE_AT = 100       # mo khoa cac lop tu chi so nay tro di
SEED = 42

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fruit_model.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "fruit_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
HISTORY_PNG_PATH = os.path.join(MODELS_DIR, "training_history.png")
RESULTS_TXT_PATH = os.path.join(MODELS_DIR, "training_results.txt")
# ==================================================================


def load_datasets():
    """Nap train/valid/test tu thu muc. Nhan lay tu ten thu muc con (theo ABC)."""
    common = dict(image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, shuffle=True, seed=SEED, **common)
    valid_ds = keras.utils.image_dataset_from_directory(
        VALID_DIR, shuffle=False, **common)
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, **common)

    class_names = train_ds.class_names
    if valid_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError("Ten lop o train/valid/test khong khop nhau!")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, valid_ds, test_ds, class_names


def build_model(num_classes):
    """MobileNetV2 (ImageNet) + dau phan loai. Augmentation va preprocess nam trong model."""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)  # chi hoat dong khi training
    x = keras.applications.mobilenet_v2.preprocess_input(x)  # [0,255] -> [-1,1]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="fruit_mobilenetv2")
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_checkpoint(threshold=None):
    kwargs = dict(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    if threshold is not None:
        kwargs["initial_value_threshold"] = threshold
    return keras.callbacks.ModelCheckpoint(**kwargs)


def plot_history(histories, path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    boundary = len(histories[0].history["accuracy"])

    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train accuracy")
    plt.plot(val_acc, label="Valid accuracy")
    plt.axvline(boundary - 1, color="gray", linestyle="--", label="Bat dau fine-tune")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train loss")
    plt.plot(val_loss, label="Valid loss")
    plt.axvline(boundary - 1, color="gray", linestyle="--", label="Bat dau fine-tune")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU") or "khong co (chay bang CPU)")

    train_ds, valid_ds, test_ds, class_names = load_datasets()
    print("Cac lop (dung thu tu):", class_names)

    # Luu class_names.json ngay de khong bi mat neu train bi ngat
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    model, base_model = build_model(len(class_names))

    #Giai doan 1: train phan phan loai, dong bang MobileNetV2
    compile_model(model, lr=1e-3)
    print("\n=== GIAI DOAN 1: Transfer Learning (dong bang backbone) ===")
    h1 = model.fit(
        train_ds, validation_data=valid_ds, epochs=EPOCHS_HEAD,
        callbacks=[make_checkpoint(),
                   # restore_best_weights: fine-tune bat dau tu epoch tot nhat, khong phai epoch cuoi
                   keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                                 patience=4, restore_best_weights=True)],
    )
    best_so_far = max(h1.history["val_accuracy"])

    #Giai doan 2: fine-tune cac lop cuoi voi learning rate nho
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    # Giu BatchNorm o che do inference de fine-tune on dinh
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, lr=1e-5)
    print("\n=== GIAI DOAN 2: Fine-tune ===")
    h2 = model.fit(
        train_ds, validation_data=valid_ds, epochs=EPOCHS_FINE,
        callbacks=[make_checkpoint(threshold=best_so_far),
                   keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                                 patience=4, restore_best_weights=True)],
    )

    # Luu ket qua
    model.save(FINAL_MODEL_PATH)
    plot_history([h1, h2], HISTORY_PNG_PATH)

    best_model = keras.models.load_model(BEST_MODEL_PATH)
    val_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)

    result = (
        "KET QUA HUAN LUYEN - MobileNetV2\n"
        f"Cac lop: {class_names}\n"
        f"Valid accuracy: {val_acc * 100:.2f}% | Valid loss: {val_loss:.4f}\n"
        f"Test  accuracy: {test_acc * 100:.2f}% | Test  loss: {test_loss:.4f}\n"
    )
    print("\n" + result)
    with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(result)

    print("Da luu:")
    for p in (BEST_MODEL_PATH, FINAL_MODEL_PATH, CLASS_NAMES_PATH,
              HISTORY_PNG_PATH, RESULTS_TXT_PATH):
        print(" -", p)


if __name__ == "__main__":
    main()