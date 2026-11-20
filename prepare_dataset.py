"""
prepare_dataset.py - Dựng thư mục dataset/ 6 lớp (5 loại quả + unknown) từ dữ liệu gốc.

Các bước (chạy một lần):
  1. Đọc ảnh GỐC: 5 loại quả từ archive.zip (bộ "Fruits Classification" trên Kaggle)
     và ảnh không phải trái cây từ gate_dataset/*/NonFruit (làm lớp unknown).
  2. Tính mã băm trên ảnh gốc, trước khi resize: MD5 (trùng hoàn toàn) và dHash 64 bit (gần trùng).
  3. Bỏ ảnh trùng: cùng MD5 ở nhiều lớp (nhãn mâu thuẫn) thì bỏ hết; cùng MD5 trong một lớp
     thì giữ một bản; trong cùng lớp, ảnh có dHash cách ảnh đã giữ <= 4 bit (gần trùng) cũng bỏ.
  3b. Xử lý các ảnh trong data/excluded_images.csv (đã duyệt bằng mắt, trái cây không phải chủ thể chính):
     cột xu_ly = "unknown": bánh, kem, đồ uống, món ăn -> chuyển sang lớp unknown;
     cột xu_ly = "bo": nhiều loại quả, sạp/giỏ quả, bao bì, chữ, hình vẽ, ảnh người -> bỏ.
  4. Chia mỗi lớp thành train/val/test = 80/10/10 với seed cố định.
  5. Kiểm tra rò rỉ: ảnh train nào trùng hoặc gần trùng với bất kỳ ảnh val/test nào
     (ở mọi lớp) thì bị loại khỏi train; val/test giữ nguyên.
  6. Resize về cạnh dài tối đa 256 px, lưu JPEG chất lượng 90 vào dataset/<split>/<lớp>/.
  7. Ghi báo cáo vào reports/: data_audit.json, removed_images.csv, dataset_manifest.csv.

Cách chạy:
  python prepare_dataset.py --fruits-zip <đường dẫn archive.zip> --gate-dir <thư mục gate_dataset>
  (mặc định đọc danh sách loại ở data/excluded_images.csv; đổi bằng --exclude)
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

FRUITS = ["Apple", "Banana", "Grape", "Mango", "Strawberry"]
CLASSES = FRUITS + ["unknown"]          # thứ tự lớp dùng chung cho train / evaluate / app
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
SEED = 42
UNKNOWN_TOTAL = 3000                    # số ảnh không phải trái cây lấy từ gate_dataset
NEAR_DUP_MAX_DISTANCE = 4               # dHash cách nhau <= 4 bit (trên 64) coi là gần trùng
MAX_SIDE = 256
JPEG_QUALITY = 90
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "dataset")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
EXCLUDE_PATH = os.path.join(BASE_DIR, "data", "excluded_images.csv")


def dhash(image):
    """dHash 64 bit: thu ảnh xám về 9x8, so từng cặp điểm ảnh liền kề theo hàng."""
    px = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def nonfruit_category(filename):
    """Nhóm ảnh NonFruit theo tiền tố tên tệp, ví dụ animal_cane, nature_sea, vehicle_Bus, dogs, cats."""
    m = re.match(r"(animal_[a-z]+|nature_[a-z]+|vehicle_[A-Za-z]+|dogs|cats)", filename)
    return m.group(1).lower() if m else "khac"


def load_items(fruits_zip, gate_dir):
    """Trả về danh sách ảnh gốc: dict(cls, source, data) với data là bytes của tệp gốc."""
    items = []
    with zipfile.ZipFile(fruits_zip) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if len(parts) == 4 and parts[2] in FRUITS and name.lower().endswith(IMAGE_EXT):
                items.append(dict(cls=parts[2], source=f"archive.zip:{name}", data=z.read(name)))

    # unknown: lấy mẫu đều theo từng nhóm ảnh để lớp đa dạng (chó, mèo, nhện, phong cảnh, xe...)
    groups = defaultdict(list)
    for split in sorted(os.listdir(gate_dir)):
        folder = os.path.join(gate_dir, split, "NonFruit")
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(IMAGE_EXT):
                    groups[nonfruit_category(f)].append(os.path.join(folder, f))
    rng = random.Random(SEED)
    for paths in groups.values():
        rng.shuffle(paths)
    picked, round_idx = [], 0
    while len(picked) < UNKNOWN_TOTAL and any(round_idx < len(p) for p in groups.values()):
        for cat in sorted(groups):
            if round_idx < len(groups[cat]) and len(picked) < UNKNOWN_TOTAL:
                picked.append(groups[cat][round_idx])
        round_idx += 1
    for path in picked:
        with open(path, "rb") as f:
            items.append(dict(cls="unknown", source="gate_dataset:" + os.path.relpath(path, gate_dir).replace("\\", "/"),
                              data=f.read()))
    return items, Counter(nonfruit_category(os.path.basename(p)) for p in picked)


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def distances(hash_list, h):
    """Khoảng cách Hamming từ dHash h tới từng phần tử của hash_list (tính vector hóa)."""
    a = np.array(hash_list, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    return POPCOUNT[np.bitwise_xor(a, np.array([h], dtype=np.uint64).view(np.uint8))].sum(1)


def remove_near_duplicates(items):
    """Trong từng lớp, ảnh có dHash cách một ảnh đã giữ <= NEAR_DUP_MAX_DISTANCE bit thì bỏ."""
    kept, removed = [], []
    for cls in CLASSES:
        kept_cls = []
        for it in (x for x in items if x["cls"] == cls):
            if kept_cls:
                dist = distances([k["dhash"] for k in kept_cls], it["dhash"])
                j = int(dist.argmin())
                if dist[j] <= NEAR_DUP_MAX_DISTANCE:
                    removed.append(("gan_trung", it, f"gần trùng trong cùng lớp (dHash cách {int(dist[j])} bit)",
                                    kept_cls[j]))
                    continue
            kept_cls.append(it)
        kept += kept_cls
    return kept, removed


def find_leaks(items):
    """Trả về {chỉ số ảnh train bị rò rỉ: (loại, nguồn ảnh val/test trùng)}."""
    evals = [it for it in items if it["split"] != "train"]
    md5_eval = {it["md5"]: it for it in evals}
    eval_hashes = [it["dhash"] for it in evals]
    leaks = {}
    for i, it in enumerate(items):
        if it["split"] != "train":
            continue
        if it["md5"] in md5_eval:
            leaks[i] = ("trùng hoàn toàn (MD5)", md5_eval[it["md5"]])
            continue
        dist = distances(eval_hashes, it["dhash"])
        j = int(dist.argmin())
        if dist[j] <= NEAR_DUP_MAX_DISTANCE:
            leaks[i] = (f"gần trùng (dHash cách {int(dist[j])} bit)", evals[j])
    return leaks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fruits-zip", required=True, help="archive.zip của bộ Fruits Classification (ảnh gốc)")
    ap.add_argument("--gate-dir", required=True, help="thư mục gate_dataset có các thư mục */NonFruit")
    ap.add_argument("--exclude", default=EXCLUDE_PATH, help="CSV các ảnh gốc cần loại (cột source)")
    args = ap.parse_args()
    with open(args.exclude, encoding="utf-8") as f:
        excluded = {r["source"]: r.get("xu_ly", "bo") for r in csv.DictReader(f)}
    if os.path.exists(OUT_DIR):
        raise SystemExit(f"{OUT_DIR} đã tồn tại - hãy đổi tên hoặc xóa trước khi dựng lại.")

    items, unknown_groups = load_items(args.fruits_zip, args.gate_dir)
    print(f"Đọc {len(items)} ảnh gốc. unknown lấy từ {len(unknown_groups)} nhóm: {dict(unknown_groups)}")

    # Băm trên ảnh gốc (trước khi resize)
    for it in items:
        it["md5"] = hashlib.md5(it["data"]).hexdigest()
        with Image.open(io.BytesIO(it["data"])) as im:
            it["dhash"] = dhash(im)

    # Bỏ ảnh trùng MD5 trước khi chia (tránh một ảnh xuất hiện 2 lần hoặc mang 2 nhãn)
    original = Counter(it["cls"] for it in items)
    removed = []  # (bước, ảnh bị bỏ, lý do, ảnh trùng với)
    by_md5 = defaultdict(list)
    for it in items:
        by_md5[it["md5"]].append(it)
    unique = []
    for group in by_md5.values():
        if len({it["cls"] for it in group}) > 1:
            removed += [("nhan_mau_thuan", it, "cùng ảnh nằm ở nhiều lớp", group[0] if it is not group[0] else group[1])
                        for it in group]
            continue
        unique.append(group[0])
        removed += [("trung_lap", it, "trùng MD5 trong cùng lớp", group[0]) for it in group[1:]]
    items, near = remove_near_duplicates(sorted(unique, key=lambda it: it["source"]))
    removed += near
    # Loại sau bước lọc trùng: bản sao của ảnh bị loại đã bị bỏ ở trên nên không quay lại được
    removed += [("loai_thu_cong", it, "trái cây không phải chủ thể chính hoặc nhiều loại quả", it)
                for it in items if excluded.get(it["source"]) == "bo"]
    items = [it for it in items if excluded.get(it["source"]) != "bo"]
    moved = Counter()  # bánh, đồ uống, món ăn: trái cây chỉ là thành phần -> lớp unknown
    for it in items:
        if excluded.get(it["source"]) == "unknown":
            moved[it["cls"]] += 1
            it["cls"] = "unknown"
    unknown_groups["do_an_do_uong"] = sum(moved.values())
    seen = {it["source"] for st, it, _, _ in removed if st == "loai_thu_cong"} | {
        it["source"] for it in items if it["source"] in excluded}
    unmatched = excluded.keys() - seen
    if unmatched:
        print(f"Lưu ý: {len(unmatched)} ảnh trong danh sách loại không còn sau bước lọc trùng (đã bị bỏ trước đó).")
    items.sort(key=lambda it: it["source"])

    # Chia 80/10/10 theo từng lớp
    rng = random.Random(SEED)
    for cls in CLASSES:
        idx = [i for i, it in enumerate(items) if it["cls"] == cls]
        rng.shuffle(idx)
        n_train, n_val = round(len(idx) * RATIOS[0]), round(len(idx) * RATIOS[1])
        for k, i in enumerate(idx):
            items[i]["split"] = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
    before = Counter((it["cls"], it["split"]) for it in items)

    leaks = find_leaks(items)
    removed += [("ro_ri", items[i], reason, other) for i, (reason, other) in sorted(leaks.items())]
    kept = [it for i, it in enumerate(items) if i not in leaks]

    # Ghi ảnh đã resize
    counter = Counter()
    manifest = []
    for it in kept:
        counter[(it["split"], it["cls"])] += 1
        folder = os.path.join(OUT_DIR, it["split"], it["cls"])
        os.makedirs(folder, exist_ok=True)
        name = f"{it['cls'].lower()}_{counter[(it['split'], it['cls'])]:05d}.jpg"
        with Image.open(io.BytesIO(it["data"])) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)  # chỉ thu nhỏ, giữ tỷ lệ
            im.save(os.path.join(folder, name), "JPEG", quality=JPEG_QUALITY)
        manifest.append([it["split"], it["cls"], name, it["source"], it["md5"]])

    # Báo cáo
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "dataset_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["split", "class", "file", "source", "md5"]] + manifest)
    with open(os.path.join(REPORT_DIR, "removed_images.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "class", "source", "reason", "matched_class", "matched_split", "matched_source"])
        for step, it, reason, other in removed:
            w.writerow([step, it["cls"], it["source"], reason, other["cls"], other.get("split", ""), other["source"]])

    size_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT_DIR) for f in fs) / 1e6
    per_class = {}
    for cls in CLASSES:
        found = [leaks[i] for i in leaks if items[i]["cls"] == cls]
        per_class[cls] = {
            "anh_goc": original[cls],
            "bo_trung_lap_md5": sum(st == "trung_lap" and it["cls"] == cls for st, it, _, _ in removed),
            "bo_nhan_mau_thuan": sum(st == "nhan_mau_thuan" and it["cls"] == cls for st, it, _, _ in removed),
            "bo_gan_trung_dhash": sum(st == "gan_trung" and it["cls"] == cls for st, it, _, _ in removed),
            "bo_loai_thu_cong": sum(st == "loai_thu_cong" and it["cls"] == cls for st, it, _, _ in removed),
            "chuyen_sang_unknown": moved[cls],
            "truoc_kiem_tra_ro_ri": {s: before[(cls, s)] for s in SPLITS},
            "ro_ri_trung_hoan_toan": sum(r.startswith("trùng") for r, _ in found),
            "ro_ri_gan_trung": sum(r.startswith("gần") for r, _ in found),
            "da_loai_khoi_train": len(found),
            "sau_cung": {s: counter[(s, cls)] for s in SPLITS},
        }
    audit = {
        "nguon": {"5_loai_qua": os.path.basename(args.fruits_zip), "unknown": "gate_dataset/*/NonFruit"},
        "ty_le_chia": dict(zip(SPLITS, RATIOS)), "seed": SEED,
        "nguong_gan_trung_dhash": NEAR_DUP_MAX_DISTANCE, "unknown_theo_nhom": dict(sorted(unknown_groups.items())),
        "theo_lop": per_class,
        "tong_sau_cung": {s: sum(counter[(s, c)] for c in CLASSES) for s in SPLITS},
        "tong_anh_goc": sum(original.values()),
        "tong_bo_trung_lap_md5": sum(st == "trung_lap" for st, *_ in removed),
        "tong_bo_nhan_mau_thuan": sum(st == "nhan_mau_thuan" for st, *_ in removed),
        "tong_bo_gan_trung_dhash": sum(st == "gan_trung" for st, *_ in removed),
        "tong_bo_loai_thu_cong": sum(st == "loai_thu_cong" for st, *_ in removed),
        "tong_chuyen_sang_unknown": sum(moved.values()),
        "tong_ro_ri_da_loai_khoi_train": len(leaks), "dung_luong_dataset_mb": round(size_mb, 1),
        "resize": f"cạnh dài <= {MAX_SIDE} px, JPEG chất lượng {JPEG_QUALITY}",
    }
    with open(os.path.join(REPORT_DIR, "data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print("\nKIỂM TRA RÒ RỈ (train so với val/test, trên ảnh gốc)")
    print(f"{'Lớp':12} {'Gốc':>5} {'Trùng MD5':>9} {'2 nhãn':>6} {'Gần trùng':>9} {'Loại tay':>8} {'Rò MD5':>6} "
          f"{'Rò gần':>6} {'train':>6} {'val':>5} {'test':>5}")
    for cls, r in per_class.items():
        s = r["sau_cung"]
        print(f"{cls:12} {r['anh_goc']:5} {r['bo_trung_lap_md5']:9} {r['bo_nhan_mau_thuan']:6} {r['bo_gan_trung_dhash']:9} "
              f"{r['bo_loai_thu_cong']:8} {r['ro_ri_trung_hoan_toan']:6} {r['ro_ri_gan_trung']:6} {s['train']:6} {s['val']:5} {s['test']:5}")
    t = audit["tong_sau_cung"]
    print(f"Bỏ trùng MD5: {audit['tong_bo_trung_lap_md5']} | bỏ nhãn mâu thuẫn: {audit['tong_bo_nhan_mau_thuan']} | "
          f"bỏ gần trùng (dHash): {audit['tong_bo_gan_trung_dhash']} | loại theo danh sách: {audit['tong_bo_loai_thu_cong']} | "
          f"chuyển sang unknown: {audit['tong_chuyen_sang_unknown']} | "
          f"rò rỉ loại khỏi train: {len(leaks)} | train {t['train']} | val {t['val']} | test {t['test']}")
    print(f"Dung lượng dataset sau resize: {size_mb:.1f} MB -> {OUT_DIR}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # in tiếng Việt cả khi output bị chuyển hướng trên Windows
    main()
