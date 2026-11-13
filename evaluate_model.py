"""
evaluate_model.py
Danh gia mo hinh MobileNetV2 nhan dien 5 loai trai cay,
mo hinh gate va ca chuoi xu ly giong app (gate + phan loai + nguong).

Dau vao:
    dataset/test/
    gate_dataset/test/            (neu co)
    models/best_fruit_model.keras
    models/best_gate_model.keras
    models/class_names.json

Dau ra:
    models/evaluation_results.txt
    models/confusion_matrix.png
"""

import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

from predictor import CONFIDENCE_THRESHOLD, GATE_NONFRUIT_THRESHOLD

# ======================== CAU HINH ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TEST_DIR = os.path.join(DATASET_DIR, "test")
GATE_TEST_DIR = os.path.join(BASE_DIR, "gate_dataset", "test")
MODEL_PATH = os.path.join(MODELS_DIR, "best_fruit_model.keras")
GATE_MODEL_PATH = os.path.join(MODELS_DIR, "best_gate_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

RESULTS_PATH = os.path.join(
    MODELS_DIR, "evaluation_results.txt"
)
CONFUSION_MATRIX_PATH = os.path.join(
    MODELS_DIR, "confusion_matrix.png"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================================


def main():
    print("=" * 65)
    print("DANH GIA MO HINH NHAN DIEN TRAI CAY")
    print("=" * 65)

    # 1. Kiểm tra các đường dẫn bắt buộc
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Khong tim thay tap Test: {TEST_DIR}"
        )

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {MODEL_PATH}"
        )

    if not os.path.isfile(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Khong tim thay class_names.json: {CLASS_NAMES_PATH}"
        )

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Đọc danh sách lớp đã lưu khi huấn luyện
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        saved_class_names = json.load(file)

    # 3. Nạp tập Test
    # label_mode="int" giúp lấy nhãn thực tế dưới dạng số nguyên.
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    dataset_class_names = test_ds.class_names

    # Thứ tự lớp phải giống thứ tự đầu ra của mô hình
    if dataset_class_names != saved_class_names:
        raise ValueError(
            "Thu tu lop trong dataset/test khong khop "
            "voi models/class_names.json.\n"
            f"Dataset: {dataset_class_names}\n"
            f"Model:   {saved_class_names}"
        )

    class_names = saved_class_names

    print("\nCac lop:", class_names)

    # 4. Nạp mô hình
    print("\nDang nap mo hinh...")
    model = keras.models.load_model(MODEL_PATH)

    # 5. Không gọi model.evaluate() vì kiểu nhãn
    # có thể không tương thích với hàm loss đã lưu.
    print("\nDang danh gia tren tap Test...")
    print("Se tinh cac chi so tu nhan thuc te va du doan.")

    # 6. Lấy nhãn thực tế
    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    # 7. Dự đoán xác suất và lấy lớp có xác suất cao nhất
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    # 8. Tính các chỉ số tổng quát
    accuracy = accuracy_score(y_true, y_pred)

    # Macro average: tính chỉ số từng lớp rồi lấy trung bình đều
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            average="macro",
            zero_division=0,
        )
    )

    # 9. Báo cáo chi tiết cho từng lớp
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    # 10. Tạo ma trận nhầm lẫn
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    # 11. Vẽ và lưu hình ảnh Confusion Matrix
    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Confusion Matrix - Fruit Recognition")
    ax.set_xlabel("Nhan du doan")
    ax.set_ylabel("Nhan thuc te")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(fig)

    # 12. Đánh giá cả chuỗi như trong app: gate -> phân loại -> ngưỡng
    print("\nDang danh gia gate va ca chuoi xu ly...")
    gate_model = keras.models.load_model(GATE_MODEL_PATH)
    gate_p = gate_model.predict(test_ds, verbose=0).reshape(-1)

    rejected = gate_p >= GATE_NONFRUIT_THRESHOLD
    uncertain = ~rejected & (probabilities.max(axis=1) < CONFIDENCE_THRESHOLD)
    shown = ~rejected & ~uncertain
    pipeline = (
        f"Hien dung ten trai cay:      {int((shown & (y_pred == y_true)).sum())}/{len(y_true)} "
        f"({(shown & (y_pred == y_true)).mean() * 100:.2f}%)\n"
        f"Bi gate tu choi nham:        {int(rejected.sum())}\n"
        f"'Chua chac chan':            {int(uncertain.sum())}\n"
        f"Hien SAI ten trai cay:       {int((shown & (y_pred != y_true)).sum())}"
    )

    # 13. Đánh giá gate trên gate_dataset/test (nếu có)
    gate_report = "Khong co gate_dataset/test - bo qua."
    if os.path.isdir(GATE_TEST_DIR):
        gate_ds = keras.utils.image_dataset_from_directory(
            GATE_TEST_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int",
            class_names=["Fruit", "NonFruit"],
            shuffle=False,
        )
        g_true = np.concatenate([labels.numpy() for _, labels in gate_ds])
        g_p = gate_model.predict(gate_ds, verbose=1).reshape(-1)
        lines = [f"So anh: Fruit {int((g_true == 0).sum())}, NonFruit {int((g_true == 1).sum())}",
                 "Nguong | Accuracy | NonFruit lot qua | Fruit bi chan"]
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            pred = (g_p >= thr).astype(int)
            mark = "  <- dang dung" if thr == GATE_NONFRUIT_THRESHOLD else ""
            lines.append(
                f"{thr:6.1f} | {(pred == g_true).mean() * 100:7.2f}% | "
                f"{int(((pred == 0) & (g_true == 1)).sum()):16} | "
                f"{int(((pred == 1) & (g_true == 0)).sum()):13}{mark}"
            )
        gate_report = "\n".join(lines)

    # 14. Tạo nội dung báo cáo
    results = f"""
============================================================
BAO CAO DANH GIA MO HINH NHAN DIEN TRAI CAY
============================================================

Mo hinh: MobileNetV2
Tap danh gia: Test
So luong lop: {len(class_names)}
Danh sach lop: {class_names}

SO MAU TEST: {len(y_true)}

CAC CHI SO TONG QUAT
------------------------------------------------------------
Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)

Macro Precision:    {precision:.4f}
Macro Recall:       {recall:.4f}
Macro F1-score:     {f1:.4f}

BAO CAO THEO TUNG LOP
------------------------------------------------------------
{report}

CONFUSION MATRIX
------------------------------------------------------------
Hang = nhan thuc te
Cot  = nhan du doan

{cm}

CA CHUOI XU LY NHU TRONG APP (tap Test)
------------------------------------------------------------
Nguong gate: {GATE_NONFRUIT_THRESHOLD} | Nguong tin cay: {CONFIDENCE_THRESHOLD}
{pipeline}

MO HINH GATE (gate_dataset/test)
------------------------------------------------------------
{gate_report}

Ghi chu:
- Cac chi so duoc tinh tren tap Test.
- Macro Precision/Recall/F1 tinh trung binh deu giua cac lop.
- Ket qua phu thuoc vao dataset va mo hinh dang su dung.
============================================================
"""

    print(results)

    # 15. Lưu báo cáo vào file
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write(results)

    print("\nDa luu ket qua:")
    print(" -", RESULTS_PATH)
    print(" -", CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()