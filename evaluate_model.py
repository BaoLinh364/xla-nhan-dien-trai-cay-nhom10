"""
evaluate_model.py - Đánh giá best_fruits_model trên tập test (dataset/test).

Đầu ra (reports/):
  metrics.json                accuracy, macro/weighted, chỉ số riêng cho lớp unknown, số ảnh
  classification_report.csv   precision / recall / F1 / support của từng lớp
  confusion_matrix.csv/.png   ma trận nhầm lẫn (hàng = nhãn thật, cột = nhãn dự đoán)
  test_predictions.csv        dự đoán và xác suất 6 lớp của từng ảnh test
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix)
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_fruits_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    unknown = class_names.index("unknown")

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, class_names=class_names, label_mode="int",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    model = keras.models.load_model(MODEL_PATH)
    assert model.output_shape[-1] == len(class_names), "Số lớp của mô hình khác class_names.json"

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=1)
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))

    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    is_fruit = y_true != unknown
    metrics = {
        "so_anh_test": int(len(y_true)),
        "so_anh_theo_lop": {c: int((y_true == i).sum()) for i, c in enumerate(class_names)},
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_avg": {k: report["macro avg"][k] for k in ("precision", "recall", "f1-score")},
        "weighted_avg": {k: report["weighted avg"][k] for k in ("precision", "recall", "f1-score")},
        # Khả năng từ chối ảnh không thuộc 5 loại quả = recall của unknown
        "unknown_recall_tu_choi_anh_la": report["unknown"]["recall"],
        "unknown_precision": report["unknown"]["precision"],
        # Ảnh lạ bị gọi nhầm thành một loại quả
        "ty_le_anh_la_bi_nhan_thanh_qua": float((y_pred[~is_fruit] != unknown).mean()),
        # Trái cây thật bị đẩy nhầm sang unknown
        "ty_le_qua_that_bi_nhan_thanh_unknown": float((y_pred[is_fruit] == unknown).mean()),
        "so_qua_that_bi_nhan_thanh_unknown": int((y_pred[is_fruit] == unknown).sum()),
        # Độ chính xác chỉ tính trên ảnh trái cây (5 lớp quả)
        "accuracy_tren_anh_trai_cay": float((y_pred[is_fruit] == y_true[is_fruit]).mean()),
        "ghi_chu_unknown": "Lớp unknown gồm động vật, phong cảnh, phương tiện và bánh, đồ uống, món ăn có trái cây; "
                           "chưa có ảnh các loại quả khác nên chưa đo được khả năng từ chối quả lạ.",
    }

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(REPORT_DIR, "classification_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name in class_names + ["macro avg", "weighted avg"]:
            r = report[name]
            w.writerow([name, r["precision"], r["recall"], r["f1-score"], int(r["support"])])
    with open(os.path.join(REPORT_DIR, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + class_names)
        for name, row in zip(class_names, cm):
            w.writerow([name] + row.tolist())
    with open(os.path.join(REPORT_DIR, "test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "pred"] + [f"p_{c}" for c in class_names])
        for path, t, p, pr in zip(test_ds.file_paths, y_true, y_pred, probs):
            w.writerow([os.path.relpath(path, TEST_DIR), class_names[t], class_names[p]] + [f"{v:.6f}" for v in pr])

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion matrix - tập test")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall unknown (từ chối ảnh lạ): {metrics['unknown_recall_tu_choi_anh_la']:.4f}")
    print(f"Ảnh lạ bị nhận thành quả: {metrics['ty_le_anh_la_bi_nhan_thanh_qua']:.4f}")
    print(f"Quả thật bị nhận thành unknown: {metrics['ty_le_qua_that_bi_nhan_thanh_unknown']:.4f} "
          f"({metrics['so_qua_that_bi_nhan_thanh_unknown']} ảnh)")
    print("Đã lưu kết quả vào", REPORT_DIR)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # in tiếng Việt cả khi output bị chuyển hướng trên Windows
    main()
