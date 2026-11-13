import os
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError

from predictor import load_gate_model, load_model, predict_image

# =========================
# CẤU HÌNH GIAO DIỆN
# =========================

PREVIEW_SIZE = (420, 420)

IMAGE_TYPES = [
    ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("Tất cả file", "*.*"),
]


# =========================
# LỚP GIAO DIỆN
# =========================

class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Ứng dụng nhận diện ảnh")
        self.geometry("960x600")
        self.minsize(860, 560)

        self.image_path = None
        self.preview_photo = None

        # Một luồng nền duy nhất: nạp mô hình rồi nhận diện lần lượt,
        # để giao diện không bị đơ.
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.setup_style()
        self.build_ui()

        self.set_status(
            "Đang tải mô hình... Bạn vẫn có thể chọn ảnh."
        )

        self.wait_for(
            self.executor.submit(self.preload_models),
            self.on_models_loaded
        )

    # =========================
    # NẠP MÔ HÌNH Ở NỀN
    # =========================

    @staticmethod
    def preload_models():
        load_gate_model()
        load_model()

    def on_models_loaded(self, future):

        try:
            future.result()

        except Exception as error:
            messagebox.showerror(
                "Lỗi tải mô hình",
                f"Không thể tải mô hình:\n{error}"
            )
            self.set_status("Tải mô hình thất bại.")
            return

        if not self.image_path:
            self.set_status(
                "Sẵn sàng. Hãy chọn một ảnh để bắt đầu."
            )

    def wait_for(self, future, callback):
        # Tkinter không an toàn với đa luồng: chỉ luồng chính
        # được cập nhật giao diện, nên kiểm tra định kỳ.
        if future.done():
            callback(future)
        else:
            self.after(100, self.wait_for, future, callback)

    # =========================
    # THIẾT LẬP GIAO DIỆN
    # =========================

    def setup_style(self):

        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 16, "bold")
        )

        style.configure(
            "Head.TLabel",
            font=("Segoe UI", 11, "bold")
        )

        style.configure(
            "Result.TLabel",
            font=("Segoe UI", 20, "bold"),
            foreground="#1a5fb4"
        )

        style.configure(
            "Conf.TLabel",
            font=("Segoe UI", 12)
        )

        style.configure(
            "Action.TButton",
            font=("Segoe UI", 11),
            padding=8
        )

        style.configure(
            "Status.TLabel",
            font=("Segoe UI", 9),
            foreground="#555555"
        )

        style.configure(
            "Warning.TLabel",
            font=("Segoe UI", 9),
            foreground="#9c5700"
        )

    # =========================
    # TẠO CÁC THÀNH PHẦN GIAO DIỆN
    # =========================

    def build_ui(self):

        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        # Tiêu đề
        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))

        ttk.Label(
            header,
            text="NHẬN DIỆN ẢNH",
            style="Title.TLabel"
        ).pack(side="left")

        # Chia giao diện thành hai cột
        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # -------------------------
        # CỘT TRÁI: ẢNH XEM TRƯỚC
        # -------------------------

        left = ttk.LabelFrame(
            body,
            text=" Ảnh xem trước ",
            padding=10
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            left,
            text="Chưa có ảnh",
            bg="#eeeeee",
            fg="#777777",
            font=("Segoe UI", 12)
        )

        self.preview_label.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.file_label = ttk.Label(
            left,
            text="",
            style="Status.TLabel"
        )

        self.file_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(6, 0)
        )

        # -------------------------
        # CỘT PHẢI
        # -------------------------

        right = ttk.Frame(body)

        right.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        right.columnconfigure(0, weight=1)

        # Ba nút chức năng
        buttons = ttk.Frame(right)
        buttons.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        for col in range(3):
            buttons.columnconfigure(col, weight=1)

        self.btn_choose = ttk.Button(
            buttons,
            text="Chọn ảnh",
            style="Action.TButton",
            command=self.choose_image
        )

        self.btn_predict = ttk.Button(
            buttons,
            text="Nhận diện",
            style="Action.TButton",
            command=self.recognize
        )

        self.btn_reset = ttk.Button(
            buttons,
            text="Làm mới",
            style="Action.TButton",
            command=self.reset
        )

        self.btn_choose.grid(
            row=0, column=0,
            sticky="ew",
            padx=(0, 4)
        )

        self.btn_predict.grid(
            row=0, column=1,
            sticky="ew",
            padx=4
        )

        self.btn_reset.grid(
            row=0, column=2,
            sticky="ew",
            padx=(4, 0)
        )

        # Khung kết quả
        result = ttk.LabelFrame(
            right,
            text=" Kết quả ",
            padding=12
        )

        result.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=(12, 0)
        )

        result.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # Nhãn dự đoán
        ttk.Label(
            result,
            text="Nhãn dự đoán:",
            style="Head.TLabel"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.result_var = tk.StringVar(value="—")

        ttk.Label(
            result,
            textvariable=self.result_var,
            style="Result.TLabel",
            wraplength=330
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(2, 14)
        )

        # Độ tin cậy
        ttk.Label(
            result,
            text="Độ tin cậy của mô hình:",
            style="Head.TLabel"
        ).grid(
            row=2,
            column=0,
            sticky="w"
        )

        self.conf_bar = ttk.Progressbar(
            result,
            maximum=100,
            length=300
        )

        self.conf_bar.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(4, 2)
        )

        self.conf_var = tk.StringVar(value="—")

        ttk.Label(
            result,
            textvariable=self.conf_var,
            style="Conf.TLabel"
        ).grid(
            row=4,
            column=0,
            sticky="w"
        )

        # Các dự đoán khác
        ttk.Label(
            result,
            text="Các dự đoán khác:",
            style="Head.TLabel"
        ).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(14, 2)
        )

        self.topk_var = tk.StringVar(value="")

        ttk.Label(
            result,
            textvariable=self.topk_var,
            justify="left"
        ).grid(
            row=6,
            column=0,
            sticky="w"
        )

        # Cảnh báo giới hạn mô hình
        self.warning_var = tk.StringVar(
            value=(
                "Lưu ý: mô hình chỉ phân loại 5 loại trái cây. "
                "Ảnh ngoài phạm vi vẫn có thể bị nhận nhầm."
            )
        )

        ttk.Label(
            result,
            textvariable=self.warning_var,
            style="Warning.TLabel",
            wraplength=360,
            justify="left"
        ).grid(
            row=7,
            column=0,
            sticky="w",
            pady=(16, 0)
        )

        # Thanh trạng thái
        self.status_var = tk.StringVar()

        ttk.Separator(root).pack(
            fill="x",
            pady=(10, 4)
        )

        ttk.Label(
            root,
            textvariable=self.status_var,
            style="Status.TLabel"
        ).pack(anchor="w")

    # =========================
    # CẬP NHẬT TRẠNG THÁI
    # =========================

    def set_status(self, text):
        self.status_var.set(text)

    # =========================
    # XÓA KẾT QUẢ CŨ
    # =========================

    def clear_result(self):

        self.result_var.set("—")
        self.conf_var.set("—")
        self.conf_bar["value"] = 0
        self.topk_var.set("")

        self.warning_var.set(
            "Lưu ý: mô hình chỉ phân loại 5 loại trái cây. "
            "Ảnh ngoài phạm vi vẫn có thể bị nhận nhầm."
        )

    # =========================
    # HIỂN THỊ ẢNH
    # =========================

    def show_preview(self, path):

        with Image.open(path) as image:

            image = ImageOps.exif_transpose(image).convert("RGB")

            image.thumbnail(
                PREVIEW_SIZE,
                Image.Resampling.LANCZOS
            )

            self.preview_photo = ImageTk.PhotoImage(
                image.copy()
            )

        self.preview_label.config(
            image=self.preview_photo,
            text=""
        )

    # =========================
    # CHỌN ẢNH
    # =========================

    def choose_image(self):

        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=IMAGE_TYPES
        )

        if not path:
            return

        try:
            self.show_preview(path)

        except (UnidentifiedImageError, OSError):

            messagebox.showerror(
                "File không hợp lệ",
                "Không thể mở file này như một ảnh. "
                "Vui lòng chọn file ảnh hợp lệ."
            )

            self.set_status(
                "Chọn ảnh thất bại: file không hợp lệ."
            )

            return

        self.image_path = path

        self.clear_result()

        self.file_label.config(
            text=os.path.basename(path)
        )

        self.set_status(
            "Đã chọn ảnh. Bấm 'Nhận diện' để dự đoán."
        )

    # =========================
    # NHẬN DIỆN ẢNH
    # =========================

    def recognize(self):

        if not self.image_path:

            messagebox.showwarning(
                "Chưa chọn ảnh",
                "Bạn chưa chọn ảnh. "
                "Hãy bấm 'Chọn ảnh' trước."
            )

            self.set_status(
                "Chưa có ảnh để nhận diện."
            )

            return

        if not os.path.isfile(self.image_path):

            messagebox.showerror(
                "Không tìm thấy ảnh",
                "File ảnh đã bị di chuyển hoặc xóa. "
                "Hãy chọn lại."
            )

            self.reset()
            return

        self.set_status("Đang nhận diện...")

        # Khóa các nút để kết quả không bị gán nhầm sang ảnh khác
        self.set_buttons_enabled(False)

        self.wait_for(
            self.executor.submit(predict_image, self.image_path),
            self.on_prediction
        )

    def set_buttons_enabled(self, enabled):

        flag = ["!disabled"] if enabled else ["disabled"]

        for button in (self.btn_choose, self.btn_predict, self.btn_reset):
            button.state(flag)

    def on_prediction(self, future):

        self.set_buttons_enabled(True)

        try:
            status, label, confidence, topk = future.result()

        except Exception as error:

            messagebox.showerror(
                "Lỗi nhận diện",
                f"Không thể nhận diện ảnh:\n{error}"
            )

            self.set_status("Nhận diện thất bại.")
            return

        self.show_result(
            label,
            confidence,
            topk
        )

        if status == "rejected":

            # Đây là xác suất NonFruit, không phải độ tin cậy trái cây
            self.conf_var.set(
                f"{confidence * 100:.1f}% (xác suất không phải trái cây)"
            )

            self.warning_var.set(
                "Mô hình lọc xác định đây không phải ảnh trái cây. "
                "Vui lòng chọn ảnh có táo, chuối, nho, xoài hoặc dâu tây."
            )

            self.set_status(
                "Đã từ chối ảnh không được xác định là trái cây."
            )

            messagebox.showwarning(
                "Ảnh không phải trái cây",
                "Ảnh đầu vào không phải trái cây. "
                "Vui lòng chọn ảnh trái cây!"
            )

        elif status == "uncertain":

            self.set_status(
                "Dự đoán có độ tin cậy thấp. "
                "Hãy thử ảnh trái cây rõ nét hơn."
            )

        else:

            self.warning_var.set(
                "Ảnh đã vượt qua bộ lọc. "
                "Kết quả nhận diện vẫn có thể sai."
            )

            self.set_status(
                "Đã hoàn tất nhận diện trái cây."
            )

    # =========================
    # HIỂN THỊ KẾT QUẢ
    # =========================

    def show_result(
        self,
        label,
        confidence,
        topk=None
    ):

        confidence = max(
            0.0,
            min(1.0, float(confidence))
        )

        self.result_var.set(str(label))

        self.conf_bar["value"] = confidence * 100

        self.conf_var.set(
            f"{confidence * 100:.1f}%"
        )

        if topk:

            lines = []

            for name, probability in topk:

                lines.append(
                    f"• {name}: {probability * 100:.1f}%"
                )

            self.topk_var.set("\n".join(lines))

        else:

            self.topk_var.set("")

    # =========================
    # LÀM MỚI GIAO DIỆN
    # =========================

    def reset(self):

        self.image_path = None
        self.preview_photo = None

        self.preview_label.config(
            image="",
            text="Chưa có ảnh"
        )

        self.file_label.config(text="")

        self.clear_result()

        self.set_status(
            "Đã làm mới. Hãy chọn một ảnh để bắt đầu."
        )


# =========================
# CHẠY CHƯƠNG TRÌNH
# =========================

if __name__ == "__main__":

    app = App()
    app.mainloop()