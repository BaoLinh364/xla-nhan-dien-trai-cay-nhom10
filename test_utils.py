"""Kiểm tra nhanh các hàm tiện ích (không cần mô hình). Chạy: python test_utils.py"""

import numpy as np
from PIL import Image

from predictor import to_percentages
from prepare_dataset import dhash, hamming

# Phần trăm hiển thị luôn cộng đúng 100 và lệch giá trị gốc không quá 1 đơn vị làm tròn
for probs in ([1 / 3] * 3, [0.125, 0.125, 0.25, 0.5, 0, 0], np.random.default_rng(0).dirichlet(np.ones(6), 50)):
    for p in np.atleast_2d(probs):
        pct = to_percentages(p)
        assert abs(sum(pct) - 100) < 1e-9, pct
        assert all(abs(a - b * 100) <= 0.01 + 1e-9 for a, b in zip(pct, p / np.sum(p)))
assert to_percentages([1 / 3] * 3) == [33.34, 33.33, 33.33]

# dHash: ảnh thu nhỏ của chính nó gần trùng, ảnh khác hẳn thì cách xa
rng = np.random.default_rng(1)
img = Image.fromarray(rng.integers(0, 255, (9, 12, 3), dtype=np.uint8)).resize((160, 120), Image.Resampling.BICUBIC)
assert hamming(dhash(img), dhash(img.resize((80, 60)))) <= 4
assert hamming(dhash(img), dhash(Image.fromarray(255 - np.asarray(img)))) > 20
# Lọc gần trùng: chỉ bỏ ảnh gần trùng trong CÙNG lớp, giữ ảnh đầu tiên
from prepare_dataset import remove_near_duplicates
h = dhash(img)
items = [dict(cls="Apple", dhash=h, source="a"), dict(cls="Apple", dhash=h ^ 0b11, source="b"),
         dict(cls="Banana", dhash=h, source="c"), dict(cls="Apple", dhash=~h & (2**64 - 1), source="d")]
kept, removed = remove_near_duplicates(items)
assert [k["source"] for k in kept] == ["a", "d", "c"] and [r[1]["source"] for r in removed] == ["b"]
print("test_utils: OK")
