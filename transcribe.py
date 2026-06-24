"""
ถอดเสียงเป็นข้อความ (Thai Speech-to-Text)
ใช้ faster-whisper บน GPU RTX 5070 (Blackwell, CUDA 12.8+)

ตัวอย่างการใช้งาน:
    python transcribe.py audio.mp3
    python transcribe.py audio.wav --model large-v3 --output result.txt
    python transcribe.py audio.mp3 --srt subtitle.srt
"""

import argparse
import os
import sys
import time
from pathlib import Path


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

from faster_whisper import WhisperModel


def format_timestamp(seconds: float, srt: bool = False) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int((s - int(s)) * 1000):03d}"


def load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    print(f"[*] โหลดโมเดล: {model_size} | device={device} | compute_type={compute_type}")
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"[+] โหลดเสร็จใน {time.time() - t0:.1f}s")
    return model


def transcribe_file(
    audio_path: Path,
    model: WhisperModel,
    language: str = "th",
    beam_size: int = 5,
    vad: bool = True,
):
    print(f"[*] กำลังถอดเสียง: {audio_path.name}")
    t0 = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=500) if vad else None,
        condition_on_previous_text=False,
    )

    print(f"[+] ตรวจพบภาษา: {info.language} (prob={info.language_probability:.2f})")
    print(f"[+] ความยาวเสียง: {info.duration:.1f}s")
    print("-" * 60)

    results = []
    for seg in segments:
        line = f"[{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}] {seg.text.strip()}"
        print(line)
        results.append(seg)

    elapsed = time.time() - t0
    rtf = elapsed / info.duration if info.duration > 0 else 0
    print("-" * 60)
    print(f"[+] เสร็จใน {elapsed:.1f}s (RTF={rtf:.2f}x)")
    return results


def write_txt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")
    print(f"[+] บันทึก text: {out_path}")


def write_srt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start, srt=True)} --> {format_timestamp(seg.end, srt=True)}\n")
            f.write(seg.text.strip() + "\n\n")
    print(f"[+] บันทึก SRT: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Thai Speech-to-Text ด้วย faster-whisper")
    p.add_argument("audio", help="ไฟล์เสียง (.mp3 .wav .m4a .flac ฯลฯ)")
    p.add_argument("--model", default="large-v3",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="ขนาดโมเดล (default: large-v3)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--compute-type", default="float16",
                   choices=["float16", "int8_float16", "int8", "float32"],
                   help="float16 = เร็วและคุณภาพดีบน RTX 5070")
    p.add_argument("--language", default="th", help="รหัสภาษา (th/en/auto)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--no-vad", action="store_true", help="ปิด VAD filter")
    p.add_argument("--output", "-o", help="path บันทึกผล .txt")
    p.add_argument("--srt", help="path บันทึก subtitle .srt")
    args = p.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[!] ไม่พบไฟล์: {audio_path}", file=sys.stderr)
        sys.exit(1)

    lang = None if args.language == "auto" else args.language
    model = load_model(args.model, args.device, args.compute_type)
    segments = transcribe_file(
        audio_path, model,
        language=lang,
        beam_size=args.beam_size,
        vad=not args.no_vad,
    )

    if args.output:
        write_txt(segments, Path(args.output))
    if args.srt:
        write_srt(segments, Path(args.srt))


if __name__ == "__main__":
    main()
