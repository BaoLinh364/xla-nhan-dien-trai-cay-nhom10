"""
app.py - Giao diện Tkinter

Chức năng:
- Cửa sổ chính, 3 nút: Chọn ảnh / Nhận diện / Làm mới
- Vùng xem trước ảnh đã chọn
- Khu vực hiển thị kết quả + độ tin cậy (thanh tiến trình + %)
- Thông báo khi chưa chọn ảnh hoặc thao tác không hợp lệ

Cách tích hợp với model :
Tạo file predictor.py cùng thư mục, trong đó có hàm:

    def predict_image(image_path: str):
        # trả về (nhan, do_tin_cay) hoặc (nhan, do_tin_cay, top_k)
        # do_tin_cay: số từ 0..1; top_k: list[(nhan, xac_suat)]
        ...

Chưa có predictor.py thì bấm "Nhận diện" sẽ báo "chưa có model". Thêm file vào
cùng thư mục là chạy luôn, không cần sửa app.py.
"""

import importlib
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, UnidentifiedImageError

PREVIEW_SIZE = (420, 420)
IMAGE_TYPES = [
    ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("Tất cả file", "*.*"),
]


class ModelNotReady(Exception):
    """Chưa có file predictor.py."""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ứng dụng nhận diện ảnh")
        self.geometry("960x600")
        self.minsize(860, 560)

        self.image_path = None
        self.preview_photo = None  # giữ tham chiếu để ảnh không bị xóa

        self._setup_style()
        self._build_ui()
        self._set_status("Sẵn sàng. Hãy chọn một ảnh để bắt đầu.")

    # ------------------------------------------------------------------ UI
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 20, "bold"),
                        foreground="#1a5fb4")
        style.configure("Conf.TLabel", font=("Segoe UI", 12))
        style.configure("Action.TButton", font=("Segoe UI", 11), padding=8)
        style.configure("Status.TLabel", font=("Segoe UI", 9),
                        foreground="#555555")

    def _build_ui(self):
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="NHẬN DIỆN ẢNH", style="Title.TLabel").pack(side="left")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # --- Cột trái: xem trước ảnh
        left = ttk.LabelFrame(body, text=" Ảnh xem trước ", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            left, text="Chưa có ảnh", bg="#eeeeee", fg="#777777",
            font=("Segoe UI", 12), width=1, height=1,
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.file_label = ttk.Label(left, text="", style="Status.TLabel")
        self.file_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        # --- Cột phải: nút + kết quả
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        btns = ttk.Frame(right)
        btns.grid(row=0, column=0, sticky="ew")
        btns.columnconfigure((0, 1, 2), weight=1)
        self.btn_choose = ttk.Button(btns, text="Chọn ảnh", style="Action.TButton",
                                     command=self.choose_image)
        self.btn_predict = ttk.Button(btns, text="Nhận diện", style="Action.TButton",
                                      command=self.recognize)
        self.btn_reset = ttk.Button(btns, text="Làm mới", style="Action.TButton",
                                    command=self.reset)
        self.btn_choose.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_predict.grid(row=0, column=1, sticky="ew", padx=4)
        self.btn_reset.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        result = ttk.LabelFrame(right, text=" Kết quả ", padding=12)
        result.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        result.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(result, text="Nhãn dự đoán:", style="Head.TLabel").grid(
            row=0, column=0, sticky="w")
        self.result_var = tk.StringVar(value="—")
        ttk.Label(result, textvariable=self.result_var,
                  style="Result.TLabel", wraplength=330).grid(
            row=1, column=0, sticky="w", pady=(2, 14))

        ttk.Label(result, text="Độ tin cậy:", style="Head.TLabel").grid(
            row=2, column=0, sticky="w")
        self.conf_bar = ttk.Progressbar(result, maximum=100, length=300)
        self.conf_bar.grid(row=3, column=0, sticky="ew", pady=(4, 2))
        self.conf_var = tk.StringVar(value="—")
        ttk.Label(result, textvariable=self.conf_var,
                  style="Conf.TLabel").grid(row=4, column=0, sticky="w")

        ttk.Label(result, text="Các dự đoán khác:", style="Head.TLabel").grid(
            row=5, column=0, sticky="w", pady=(14, 2))
        self.topk_var = tk.StringVar(value="")
        ttk.Label(result, textvariable=self.topk_var, justify="left").grid(
            row=6, column=0, sticky="w")

        # --- Thanh trạng thái
        self.status_var = tk.StringVar()
        ttk.Separator(root).pack(fill="x", pady=(10, 4))
        ttk.Label(root, textvariable=self.status_var,
                  style="Status.TLabel").pack(anchor="w")

    # ------------------------------------------------------------- Helpers
    def _set_status(self, text):
        self.status_var.set(text)

    def _clear_result(self):
        self.result_var.set("—")
        self.conf_var.set("—")
        self.conf_bar["value"] = 0
        self.topk_var.set("")

    def _show_preview(self, path):
        """Mở ảnh và hiển thị. Ném exception nếu file không phải ảnh hợp lệ."""
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(img.copy())
        self.preview_label.config(image=self.preview_photo, text="")

    # ------------------------------------------------------------- Actions
    def choose_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=IMAGE_TYPES)
        if not path:
            return  # người dùng bấm Hủy
        try:
            self._show_preview(path)
        except (UnidentifiedImageError, OSError):
            messagebox.showerror(
                "File không hợp lệ",
                "Không thể mở file này như một ảnh.\n"
                "Vui lòng chọn file .jpg, .png, .bmp hoặc .webp.")
            self._set_status("Chọn ảnh thất bại: file không hợp lệ.")
            return

        self.image_path = path
        self._clear_result()
        self.file_label.config(text=os.path.basename(path))
        self._set_status("Đã chọn ảnh. Bấm 'Nhận diện' để dự đoán.")

    def recognize(self):
        if not self.image_path:
            messagebox.showwarning("Chưa chọn ảnh",
                                   "Bạn chưa chọn ảnh. Hãy bấm 'Chọn ảnh' trước.")
            self._set_status("Chưa có ảnh để nhận diện.")
            return
        if not os.path.isfile(self.image_path):
            messagebox.showerror("Không tìm thấy ảnh",
                                 "File ảnh đã bị di chuyển hoặc xóa. Hãy chọn lại.")
            self.reset()
            return

        self._set_status("Đang nhận diện...")
        self.btn_predict.state(["disabled"])
        self.update_idletasks()
        try:
            label, conf, topk = self._run_prediction(self.image_path)
        except ModelNotReady:
            messagebox.showinfo("Chưa có model",
                                "Chưa có file predictor.py.\n"
                                "Hãy thêm file này vào cùng thư mục với app.py.")
            self._set_status("Chưa có model để nhận diện.")
            return
        except Exception as e:  # lỗi từ model
            messagebox.showerror("Lỗi nhận diện", f"Không thể nhận diện ảnh:\n{e}")
            self._set_status("Nhận diện thất bại.")
            return
        finally:
            self.btn_predict.state(["!disabled"])

        self.show_result(label, conf, topk)
        self._set_status("Nhận diện xong.")

    def reset(self):
        self.image_path = None
        self.preview_photo = None
        self.preview_label.config(image="", text="Chưa có ảnh")
        self.file_label.config(text="")
        self._clear_result()
        self._set_status("Đã làm mới. Hãy chọn một ảnh để bắt đầu.")

    # ------------------------------------------------------ Kết quả / model
    def _run_prediction(self, path):
        """Gọi predictor.predict_image; chuẩn hóa về (label, conf 0..1, topk)."""
        try:
            predictor = importlib.import_module("predictor")
        except ModuleNotFoundError as e:
            if e.name == "predictor":
                raise ModelNotReady() from e
            raise  # predictor có mặt nhưng thiếu thư viện bên trong

        out = predictor.predict_image(path)
        if len(out) == 2:
            return out[0], out[1], []
        return out[0], out[1], list(out[2])

    def show_result(self, label, conf, topk=None):
        """API để module khác cũng có thể gọi: conf trong khoảng 0..1."""
        conf = max(0.0, min(1.0, float(conf)))
        self.result_var.set(str(label))
        self.conf_bar["value"] = conf * 100
        self.conf_var.set(f"{conf * 100:.1f}%")
        if topk:
            self.topk_var.set("\n".join(f"• {n}: {p * 100:.1f}%" for n, p in topk))
        else:
            self.topk_var.set("")


if __name__ == "__main__":
    App().mainloop()
