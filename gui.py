"""
Thai Speech-to-Text GUI
อัปโหลดไฟล์เสียง (.mp3 .wav .m4a .flac ฯลฯ) แล้วถอดเป็นข้อความภาษาไทย
ใช้ faster-whisper บน GPU (รองรับตั้งแต่ GTX 1650 จนถึง RTX 5070+)

รัน:  python gui.py
"""

import os
import subprocess
import sys
from pathlib import Path

# === ทำให้ Windows หา cublas64_12.dll / cudnn64_9.dll เจอ ===
def _setup_cuda_dlls():
    if sys.platform != "win32":
        return
    import ctypes
    bin_dirs: list[Path] = []
    try:
        import nvidia  # type: ignore
        for root in list(getattr(nvidia, "__path__", [])):
            base = Path(root)
            for sub in ("cublas/bin", "cudnn/bin", "cuda_runtime/bin", "cuda_nvrtc/bin"):
                p = base / sub
                if p.is_dir():
                    bin_dirs.append(p)
    except ImportError:
        pass
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and (Path(cuda_path) / "bin").is_dir():
        bin_dirs.append(Path(cuda_path) / "bin")
    if not bin_dirs:
        return
    os.environ["PATH"] = os.pathsep.join(str(p) for p in bin_dirs) + os.pathsep + os.environ.get("PATH", "")
    for p in bin_dirs:
        try: os.add_dll_directory(str(p))
        except OSError: pass
    for name in ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll",
                 "cudart64_12.dll", "nvrtc64_120_0.dll"]:
        for d in bin_dirs:
            full = d / name
            if full.is_file():
                try: ctypes.WinDLL(str(full))
                except OSError: pass
                break

_setup_cuda_dlls()

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from faster_whisper import WhisperModel


# ===== ตรวจสอบ GPU =====
def detect_gpu() -> dict | None:
    """คืน {'name','vram_gb','compute_cap'} หรือ None ถ้าไม่มี NVIDIA GPU"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip()
        line = out.splitlines()[0]
        parts = [x.strip() for x in line.split(",")]
        return {"name": parts[0], "vram_gb": float(parts[1]) / 1024, "compute_cap": parts[2]}
    except Exception:
        return None


# ===== Preset =====
# (model, compute_type, device, beam_size, num_workers, cpu_threads)
PRESETS: dict[str, dict] = {
    "🚀 คุณภาพสูง (VRAM 10+ GB)":
        dict(model="large-v3", compute="float16", device="cuda", beam=5, workers=1, cpu_threads=0),
    "⚡ สมดุล (VRAM 6–10 GB)":
        dict(model="medium", compute="float16", device="cuda", beam=5, workers=1, cpu_threads=0),
    "🪶 เบา (VRAM 4–6 GB)":
        dict(model="small", compute="int8_float16", device="cuda", beam=3, workers=1, cpu_threads=0),
    "🐢 การ์ดเก่า / GTX 1650 (4 GB)":
        dict(model="small", compute="int8_float32", device="cuda", beam=1, workers=1, cpu_threads=2),
    "🌱 ประหยัดพลัง (ไม่กระตุก)":
        dict(model="base", compute="int8", device="cuda", beam=1, workers=1, cpu_threads=2),
    "💻 CPU เท่านั้น":
        dict(model="small", compute="int8", device="cpu", beam=1, workers=1, cpu_threads=4),
}


def auto_preset_for(gpu: dict | None) -> str:
    """เลือก preset ที่เหมาะตาม VRAM และ compute capability"""
    if gpu is None:
        return "💻 CPU เท่านั้น"
    vram = gpu["vram_gb"]
    try:
        cc = float(gpu["compute_cap"])
    except ValueError:
        cc = 7.5
    if cc < 7.0:  # ก่อน Volta - ไม่มี tensor core, ห้ามใช้ fp16
        return "🐢 การ์ดเก่า / GTX 1650 (4 GB)"
    if vram >= 10:
        return "🚀 คุณภาพสูง (VRAM 10+ GB)"
    if vram >= 6:
        return "⚡ สมดุล (VRAM 6–10 GB)"
    if vram >= 4:
        # GTX 1650 (cc=7.5, 4GB) ไม่มี fp16 tensor core ใช้ int8_float32 ดีกว่า
        return "🐢 การ์ดเก่า / GTX 1650 (4 GB)" if vram < 5 else "🪶 เบา (VRAM 4–6 GB)"
    return "💻 CPU เท่านั้น"


def set_low_priority():
    """ลด priority ของ process นี้ให้ Windows scheduler ให้ความสำคัญน้อยลง"""
    if sys.platform == "win32":
        try:
            import ctypes
            BELOW_NORMAL = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL)
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Thai Speech-to-Text  —  faster-whisper")
        root.geometry("960x720")
        root.minsize(720, 540)

        self.model: WhisperModel | None = None
        self._model_sig: tuple | None = None
        self._segments: list = []

        self.gpu_info = detect_gpu()
        default_preset = auto_preset_for(self.gpu_info)
        default = PRESETS[default_preset]

        self.preset = tk.StringVar(value=default_preset)
        self.model_size = tk.StringVar(value=default["model"])
        self.device = tk.StringVar(value=default["device"])
        self.compute_type = tk.StringVar(value=default["compute"])
        self.beam_size = tk.IntVar(value=default["beam"])
        self.cpu_threads = tk.IntVar(value=default["cpu_threads"])
        self.language = tk.StringVar(value="th")
        self.show_timestamps = tk.BooleanVar(value=False)
        self.low_priority = tk.BooleanVar(value=False)

        self._build_ui()
        self._show_gpu_info()

    def _build_ui(self):
        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass

        # ===== ปุ่มอัปโหลดใหญ่ =====
        upload_frame = ttk.Frame(self.root, padding=20)
        upload_frame.pack(fill="x")

        self.drop_label = tk.Label(
            upload_frame,
            text="📁  คลิกเพื่อเลือกไฟล์เสียง (.mp3 / .wav / .m4a / .flac ...)",
            font=("Segoe UI", 14), bg="#f3f6fb", fg="#2c3e50",
            relief="ridge", bd=2, cursor="hand2", height=4,
        )
        self.drop_label.pack(fill="x")
        self.drop_label.bind("<Button-1>", lambda e: self.pick_file())

        path_row = ttk.Frame(self.root, padding=(20, 0))
        path_row.pack(fill="x")
        ttk.Label(path_row, text="ไฟล์:").pack(side="left")
        self.file_path = tk.StringVar(value="")
        ttk.Entry(path_row, textvariable=self.file_path, state="readonly").pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(path_row, text="เปลี่ยน...", command=self.pick_file).pack(side="left")

        # ===== Preset =====
        preset_row = ttk.LabelFrame(self.root, text="โหมด (preset)", padding=10)
        preset_row.pack(fill="x", padx=20, pady=(10, 0))
        preset_cb = ttk.Combobox(preset_row, textvariable=self.preset, width=40, state="readonly",
                                  values=list(PRESETS.keys()))
        preset_cb.pack(side="left")
        preset_cb.bind("<<ComboboxSelected>>", self._apply_preset)
        ttk.Checkbutton(preset_row, text="โหมดประหยัดพลัง (ลด priority — ไม่กระตุก)",
                        variable=self.low_priority).pack(side="left", padx=20)

        # ===== ตัวเลือกละเอียด =====
        opt = ttk.LabelFrame(self.root, text="ตัวเลือกละเอียด (override preset)", padding=10)
        opt.pack(fill="x", padx=20, pady=10)

        ttk.Label(opt, text="โมเดล:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(opt, textvariable=self.model_size, width=12, state="readonly",
                     values=["tiny", "base", "small", "medium", "large-v2", "large-v3"]
                     ).grid(row=0, column=1, padx=4)

        ttk.Label(opt, text="Device:").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Combobox(opt, textvariable=self.device, width=7, state="readonly",
                     values=["cuda", "cpu"]).grid(row=0, column=3, padx=4)

        ttk.Label(opt, text="Compute:").grid(row=0, column=4, sticky="w", padx=4)
        ttk.Combobox(opt, textvariable=self.compute_type, width=14, state="readonly",
                     values=["float16", "int8_float16", "int8_float32", "int8", "float32"]
                     ).grid(row=0, column=5, padx=4)

        ttk.Label(opt, text="ภาษา:").grid(row=0, column=6, sticky="w", padx=4)
        ttk.Combobox(opt, textvariable=self.language, width=6, state="readonly",
                     values=["th", "en", "auto"]).grid(row=0, column=7, padx=4)

        ttk.Label(opt, text="Beam:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(opt, from_=1, to=10, textvariable=self.beam_size, width=5
                    ).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(opt, text="CPU threads (0=auto):").grid(row=1, column=2, columnspan=2, sticky="w", padx=4)
        ttk.Spinbox(opt, from_=0, to=32, textvariable=self.cpu_threads, width=5
                    ).grid(row=1, column=4, sticky="w", padx=4)

        ttk.Checkbutton(opt, text="แสดง timestamp", variable=self.show_timestamps
                        ).grid(row=1, column=5, columnspan=2, sticky="w", padx=10)

        # ===== ปุ่มแอ็คชั่น =====
        action = ttk.Frame(self.root, padding=(20, 0))
        action.pack(fill="x")
        self.run_btn = ttk.Button(action, text="▶  ถอดเสียง", command=self.start_transcribe)
        self.run_btn.pack(side="left")
        ttk.Button(action, text="💾 บันทึก .txt", command=lambda: self.save_text(".txt")).pack(side="left", padx=5)
        ttk.Button(action, text="💾 บันทึก .srt", command=self.save_srt).pack(side="left")
        ttk.Button(action, text="ล้างผลลัพธ์", command=self.clear).pack(side="right")

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=20, pady=(10, 0))

        text_frame = ttk.Frame(self.root, padding=20)
        text_frame.pack(fill="both", expand=True)
        self.text = tk.Text(text_frame, wrap="word", font=("Sarabun", 13), undo=True)
        scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.status = ttk.Label(self.root, text="พร้อมใช้งาน", relief="sunken", anchor="w", padding=4)
        self.status.pack(fill="x", side="bottom")

    def _show_gpu_info(self):
        if self.gpu_info:
            g = self.gpu_info
            self.set_status(
                f"พบ GPU: {g['name']}  |  VRAM {g['vram_gb']:.1f} GB  |  "
                f"compute {g['compute_cap']}  →  preset: {self.preset.get()}"
            )
        else:
            self.set_status("ไม่พบ NVIDIA GPU — จะใช้ CPU (ช้ากว่า)")

    def _apply_preset(self, *_):
        cfg = PRESETS[self.preset.get()]
        self.model_size.set(cfg["model"])
        self.device.set(cfg["device"])
        self.compute_type.set(cfg["compute"])
        self.beam_size.set(cfg["beam"])
        self.cpu_threads.set(cfg["cpu_threads"])

    # ---------- utility ----------
    def log(self, msg: str):
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def set_status(self, msg: str):
        self.status.config(text=msg)
        self.root.update_idletasks()

    @staticmethod
    def _ts(seconds: float, srt: bool = False) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        sep = "," if srt else "."
        return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int((s - int(s)) * 1000):03d}"

    # ---------- callbacks ----------
    def pick_file(self):
        path = filedialog.askopenfilename(
            title="เลือกไฟล์เสียง",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.webm *.mp4 *.aac *.opus"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_path.set(path)
            self.drop_label.config(text=f"✓  {Path(path).name}", bg="#e8f5e9", fg="#1b5e20")

    def clear(self):
        self.text.delete("1.0", "end")
        self._segments = []

    def save_text(self, ext: str = ".txt"):
        content = self.text.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("ว่าง", "ยังไม่มีข้อความให้บันทึก")
            return
        default = Path(self.file_path.get()).stem + ext if self.file_path.get() else f"transcript{ext}"
        path = filedialog.asksaveasfilename(
            defaultextension=ext, initialfile=default,
            filetypes=[("Text", f"*{ext}")],
        )
        if path:
            Path(path).write_text(content, encoding="utf-8")
            messagebox.showinfo("บันทึกแล้ว", path)

    def save_srt(self):
        if not self._segments:
            messagebox.showinfo("ว่าง", "ยังไม่ได้ถอดเสียง")
            return
        default = Path(self.file_path.get()).stem + ".srt" if self.file_path.get() else "transcript.srt"
        path = filedialog.asksaveasfilename(
            defaultextension=".srt", initialfile=default,
            filetypes=[("SubRip", "*.srt")],
        )
        if not path:
            return
        lines = []
        for i, seg in enumerate(self._segments, 1):
            lines.append(str(i))
            lines.append(f"{self._ts(seg.start, srt=True)} --> {self._ts(seg.end, srt=True)}")
            lines.append(seg.text.strip())
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        messagebox.showinfo("บันทึกแล้ว", path)

    def start_transcribe(self):
        path = self.file_path.get().strip()
        if not path or not Path(path).exists():
            messagebox.showerror("ผิดพลาด", "กรุณาเลือกไฟล์เสียงก่อน")
            return
        self.run_btn.config(state="disabled")
        self.progress.start(10)
        self.clear()
        threading.Thread(target=self._run_transcribe, args=(path,), daemon=True).start()

    # ---------- worker ----------
    def _ensure_model(self):
        sig = (self.model_size.get(), self.device.get(), self.compute_type.get(), self.cpu_threads.get())
        if self.model is not None and self._model_sig == sig:
            return
        size, dev, ct, threads = sig
        self.set_status(f"กำลังโหลดโมเดล {size} ({dev}/{ct}) ครั้งแรกอาจช้าเพราะต้องดาวน์โหลด...")
        t0 = time.time()
        kwargs = dict(device=dev, compute_type=ct, num_workers=1)
        if threads > 0:
            kwargs["cpu_threads"] = threads
        self.model = WhisperModel(size, **kwargs)
        self._model_sig = sig
        self.set_status(f"โหลดโมเดลเสร็จใน {time.time() - t0:.1f}s")

    def _run_transcribe(self, path: str):
        try:
            if self.low_priority.get():
                set_low_priority()
                self.set_status("ตั้ง process priority ต่ำลง (ลดอาการกระตุก)")

            self._ensure_model()
            lang = self.language.get()
            lang = None if lang == "auto" else lang

            self.set_status(f"กำลังถอดเสียง: {Path(path).name}")
            t0 = time.time()
            segments, info = self.model.transcribe(
                path,
                language=lang,
                beam_size=self.beam_size.get(),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
            )

            header = (f"--- {Path(path).name}  |  ภาษา={info.language} "
                      f"({info.language_probability:.0%})  |  ยาว={info.duration:.1f}s ---")
            self.log(header)
            self.log("")

            self._segments = []
            for seg in segments:
                self._segments.append(seg)
                text = seg.text.strip()
                if self.show_timestamps.get():
                    self.log(f"[{self._ts(seg.start)} → {self._ts(seg.end)}]  {text}")
                else:
                    self.log(text)
                self.root.update_idletasks()
                # หายใจสั้นๆ ระหว่าง segment เพื่อปล่อย GPU/CPU ให้แอปอื่น (low-power mode)
                if self.low_priority.get():
                    time.sleep(0.01)

            elapsed = time.time() - t0
            rtf = elapsed / info.duration if info.duration else 0
            self.set_status(f"เสร็จสิ้น  ({elapsed:.1f}s, RTF={rtf:.2f}x, {len(self._segments)} segments)")
        except Exception as e:
            self.set_status("ผิดพลาด")
            messagebox.showerror("ผิดพลาด", f"{type(e).__name__}: {e}")
        finally:
            self.progress.stop()
            self.run_btn.config(state="normal")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
