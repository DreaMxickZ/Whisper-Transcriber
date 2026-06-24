# Whistper — Thai Speech-to-Text GUI

โปรแกรม **ถอดเสียงเป็นข้อความภาษาไทย** ทำงานบนเครื่องตัวเอง (offline) ด้วย [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
รองรับ GPU NVIDIA ตั้งแต่ **GTX 1650 (4 GB)** จนถึง **RTX 5070+** (Blackwell)

---

## ✨ Features

- 🖱️ **GUI ใช้งานง่าย** — กดอัปโหลดไฟล์ .mp3 / .wav / .m4a / .flac แล้วถอดเสียงได้เลย
- 🇹🇭 **เน้นภาษาไทย** — ตั้ง default `language=th` พร้อม VAD filter กรองช่วงเงียบ
- ⚡ **เร็วบน GPU** — ใช้ CUDA + cuDNN, รองรับ Blackwell (sm_120) ของ RTX 50 series
- 🎛️ **6 preset อัตโนมัติ** — auto-detect VRAM แล้วเลือก preset ที่เหมาะ
- 🌱 **โหมดประหยัดพลัง** — ลด GPU/CPU usage ระหว่างถอดเสียงเพื่อไม่ให้คอมกระตุก
- 💾 **บันทึกเป็น .txt หรือ .srt** (subtitle พร้อม timestamp)
- 🛠️ **CLI version** ([`transcribe.py`](transcribe.py)) สำหรับเรียกจาก script

---

## 📦 ติดตั้ง

### 1. Clone และสร้าง venv

```powershell
git clone <repo-url> whistper
cd whistper
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows PowerShell
# หรือ:  source venv/bin/activate    # macOS/Linux
```

### 2. ติดตั้ง dependencies

```powershell
pip install -r requirements.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   # สำหรับใช้ GPU
```

### 3. (เฉพาะ RTX 5070 / Blackwell)
ดู [setup_rtx5070.md](setup_rtx5070.md) สำหรับรายละเอียด — สั้นๆ คือต้องใช้ **NVIDIA Driver 570+** และ CUDA 12.8+

---

## 🚀 ใช้งาน

### GUI (แนะนำ)

```powershell
python gui.py
```

1. คลิกกล่อง 📁 ใหญ่ๆ → เลือกไฟล์เสียง
2. โปรแกรมจะ detect GPU และเลือก **preset** ที่เหมาะให้อัตโนมัติ
3. กด **▶ ถอดเสียง**
4. กด **💾 บันทึก .txt / .srt**

### CLI

```powershell
# พื้นฐาน
python transcribe.py audio.mp3 --output result.txt

# เลือกโมเดลและบันทึก subtitle
python transcribe.py audio.mp3 --model large-v3 --srt result.srt

# ใช้ CPU (กรณีไม่มี GPU)
python transcribe.py audio.mp3 --device cpu --compute-type int8
```

ดู option ทั้งหมด: `python transcribe.py --help`

---

## 🎛️ Preset แนะนำ

| Preset | โมเดล | Compute | เหมาะกับ |
|---|---|---|---|
| 🚀 คุณภาพสูง | large-v3 | float16 | RTX 4070/5070 (VRAM 10+ GB) |
| ⚡ สมดุล | medium | float16 | RTX 3060 (VRAM 6–10 GB) |
| 🪶 เบา | small | int8_float16 | VRAM 4–6 GB |
| 🐢 การ์ดเก่า / GTX 1650 | small | int8_float32 | GTX 1650/1660 |
| 🌱 ประหยัดพลัง | base | int8 | ใช้คอมไปด้วยตอนถอดเสียง |
| 💻 CPU เท่านั้น | small | int8 | ไม่มี GPU |

> 💡 **เคล็ดลับ:** ถ้าอยากใช้คอมระหว่างถอดเสียงโดยไม่กระตุก → ติ๊ก ☑ **โหมดประหยัดพลัง**
> จะลด process priority และหายใจสั้นๆ ระหว่าง segment → GPU usage ~30% แทน 100%

---

## 🧩 Troubleshooting

### `RuntimeError: Library cublas64_12.dll is not found`
ติดตั้ง cuBLAS/cuDNN runtime ผ่าน pip:
```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
โปรแกรมจะ register DLL path ให้อัตโนมัติตอนเปิด

### โหลดโมเดลครั้งแรกช้ามาก
ครั้งแรกจะดาวน์โหลดโมเดลจาก HuggingFace:
- `tiny` ~75 MB
- `base` ~140 MB
- `small` ~460 MB
- `medium` ~1.5 GB
- `large-v3` ~3 GB

ครั้งต่อๆ ไปจะใช้ cache ที่ `~/.cache/huggingface/hub/`

### GPU ไม่ถูก detect
- ตรวจสอบ driver: `nvidia-smi` (ต้องเห็นชื่อการ์ดและ CUDA version)
- ลองเลือก preset **💻 CPU เท่านั้น** ดูก่อน — ถ้ารันได้แปลว่าโค้ดทำงานปกติ ปัญหาอยู่ที่ CUDA setup

### ภาษาไทยถอดผิดเพี้ยน
- ใช้โมเดลใหญ่ขึ้น (`large-v3` คุณภาพดีที่สุดสำหรับไทย)
- เพิ่ม `beam_size` เป็น 5
- ตรวจสอบคุณภาพเสียง (เสียงเบา/มีเสียงรบกวนเยอะจะถอดยาก)

---

## 📁 โครงสร้างไฟล์

```
whistper/
├── gui.py                # โปรแกรมหลัก GUI
├── transcribe.py         # CLI version
├── requirements.txt      # dependencies
├── setup_rtx5070.md      # คู่มือ RTX 5070 / Blackwell
├── README.md
└── .gitignore
```

---

## 🛠️ Tech stack

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper ที่ optimize ด้วย CTranslate2 (เร็วกว่า openai-whisper ~4x)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — runtime สำหรับ neural network inference
- Python 3.10+ / Tkinter (มากับ Python)
- CUDA 12.x + cuDNN 9.x (สำหรับ GPU)

---

## 📜 License

MIT — ใช้ฟรี แก้ไขได้ตามต้องการ
