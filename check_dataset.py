"""
check_dataset.py
Kiểm tra cấu trúc và chất lượng cơ bản của dataset nhận diện trái cây.
"""

import os
from collections import defaultdict
from PIL import Image

# ======================== CAU HINH ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SPLITS = ["train", "valid", "test"]

EXPECTED_CLASSES = [
    "Apple",
    "Banana",
    "Grape",
    "Mango",
    "Strawberry",
]

# Các định dạng ảnh được chấp nhận
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ==========================================================


def check_dataset():
    print("=" * 65)
    print("KIEM TRA DATASET NHAN DIEN TRAI CAY")
    print("=" * 65)

    if not os.path.isdir(DATASET_DIR):
        print(f"\n[LOI] Khong tim thay thu muc dataset:")
        print(DATASET_DIR)
        print("\nHay dat dataset cung cap voi app.py va train_model.py.")
        return

    total_images = 0
    total_invalid = 0
    split_totals = {}
    all_ok = True

    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)

        print(f"\n--- TAP {split.upper()} ---")

        if not os.path.isdir(split_dir):
            print(f"[LOI] Thieu thu muc: {split_dir}")
            all_ok = False
            continue

        # Kiểm tra tên các thư mục lớp
        actual_classes = sorted(
            name for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
        )

        missing_classes = sorted(
            set(EXPECTED_CLASSES) - set(actual_classes)
        )
        extra_classes = sorted(
            set(actual_classes) - set(EXPECTED_CLASSES)
        )

        if missing_classes:
            print("[LOI] Thieu lop:", ", ".join(missing_classes))
            all_ok = False

        if extra_classes:
            print("[CANH BAO] Co lop khong nam trong danh sach:",
                  ", ".join(extra_classes))
            all_ok = False

        split_count = 0

        for class_name in EXPECTED_CLASSES:
            class_dir = os.path.join(split_dir, class_name)

            if not os.path.isdir(class_dir):
                print(f"{class_name:12} | KHONG CO THU MUC")
                continue

            valid_count = 0
            invalid_count = 0
            unsupported_count = 0

            # Duyệt cả thư mục con nếu có
            for root, _, files in os.walk(class_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    extension = os.path.splitext(filename)[1].lower()

                    if extension not in IMAGE_EXTENSIONS:
                        unsupported_count += 1
                        continue

                    try:
                        # Kiểm tra file có phải ảnh đọc được không
                        with Image.open(file_path) as img:
                            img.verify()
                        valid_count += 1

                    except Exception:
                        invalid_count += 1
                        print(f"[ANH LOI] {file_path}")

            split_count += valid_count
            total_invalid += invalid_count

            print(
                f"{class_name:12} | "
                f"Anh hop le: {valid_count:5} | "
                f"Anh loi: {invalid_count:4} | "
                f"File khong phai anh: {unsupported_count:4}"
            )

            if valid_count == 0:
                print(f"[CANH BAO] Lop {class_name} khong co anh hop le.")
                all_ok = False

            if invalid_count > 0:
                all_ok = False

        split_totals[split] = split_count
        total_images += split_count

        print(f"Tong so anh hop le trong {split}: {split_count}")

    # Tổng kết
    print("\n" + "=" * 65)
    print("TONG KET")
    print("=" * 65)

    for split in SPLITS:
        if split in split_totals:
            print(f"{split:10}: {split_totals[split]} anh")

    print(f"Tong anh hop le : {total_images}")
    print(f"Tong anh loi    : {total_invalid}")

    if all_ok:
        print("\n[DAT] Khong phat hien loi cau truc co ban.")
    else:
        print("\n[CAN KIEM TRA] Dataset co van de can xem lai.")

    print("\nLuu y:")
    print("- File khong phai anh duoc thong ke rieng.")
    print("- Hay kiem tra so luong anh moi lop co du de huan luyen.")
    print("- Chuong trinh nay chi kiem tra, khong di chuyen hoac xoa anh.")


if __name__ == "__main__":
    check_dataset()