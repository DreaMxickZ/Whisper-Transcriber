# การติดตั้งสำหรับ RTX 5070 (Blackwell)

RTX 5070 เป็นการ์ด **Blackwell (sm_120)** ต้องใช้ CUDA 12.8+ และ cuDNN 9.x

## 1. ติดตั้ง Driver + CUDA

- NVIDIA Driver: **570+** (สำหรับ Blackwell)
- ตรวจสอบ: `nvidia-smi` ต้องเห็น "RTX 5070" และ CUDA Version >= 12.8

## 2. สร้าง venv

```powershell
cd C:\Users\DreaMxickZen\Desktop\whistper
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. ติดตั้ง dependencies

```powershell
pip install -r requirements.txt
```

`faster-whisper` จะดึง `ctranslate2` >= 4.5 มาให้เอง ซึ่งรองรับ Blackwell แล้ว
(ถ้ารัน CUDA แล้วเจอ error เรื่อง compute capability → อัปเดต ctranslate2: `pip install -U ctranslate2`)

## 4. ติดตั้ง cuDNN 9.x runtime (จำเป็นสำหรับ ctranslate2 GPU)

ดาวน์โหลด cuDNN 9.x สำหรับ CUDA 12 จาก https://developer.nvidia.com/cudnn
แตกไฟล์ `bin/*.dll` ใส่โฟลเดอร์ที่ `PATH` มองเห็น เช่น `C:\Program Files\NVIDIA\CUDNN\v9\bin`

ทางลัด: ติดตั้งผ่าน pip
```powershell
pip install nvidia-cudnn-cu12
```

## 5. ทดสอบ

```powershell
# ถอดไฟล์เสียง
python transcribe.py "C:\path\to\audio.mp3" --output result.txt --srt result.srt

# ไมโครโฟนเรียลไทม์
python realtime_mic.py --model medium

# GUI
python gui.py
```

## เคล็ดลับเลือก compute_type บน RTX 5070 (12 GB)

| compute_type    | VRAM (large-v3) | ความเร็ว | คุณภาพ |
|-----------------|-----------------|----------|--------|
| `float16`       | ~5 GB           | เร็ว     | ดีสุด  |
| `int8_float16`  | ~3 GB           | เร็วสุด  | ใกล้เคียง |
| `int8`          | ~2 GB           | เร็ว     | ลดลงเล็กน้อย |

แนะนำ `float16` สำหรับงานทั่วไป — RTX 5070 มี VRAM เหลือพอ
