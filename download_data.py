#!/usr/bin/env python3
"""
Dataset tayyorlash — Kaggle'dan audio yuklab olish
=======================================================
dataset/0  ->  NON-SNORING: nutq, shovqin, turli ovozlar
dataset/1  ->  SNORING: xurrak ovozlar

Maqsad: Model faqat xurrakni tanishi kerak.
        Nutq (o'zbek, rus, ingliz), yo'tal, musiqa,
        shovqin — bularni XURRAK EMAS deb bilishi kerak.
"""

import os
import sys
import shutil
import random
import subprocess
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ------------------------------------------------------------------
# SOZLAMALAR
# ------------------------------------------------------------------
OUT_0       = Path("dataset/0")
OUT_1       = Path("dataset/1")
SR          = 16000
WIN_LEN     = SR          # 1 soniya
TARGET_0    = 2000
TARGET_1    = 2000
TMP         = Path("_tmp_download")

# ------------------------------------------------------------------
# KAGGLE DATASET SLUGLARI
#
# Har bir kategoriya uchun bir nechta variant —
# biri ishlamasa keyingisi avtomatik sinab ko'riladi
# ------------------------------------------------------------------
SNORING_SLUGS = [
    "tareqkhanemu/snoring",          # 1000 klip, 500+500, tasdiqlangan
]

SPEECH_SLUGS = [
    "yashdogra/speech-commands",     # Google Speech Commands v0.02
]

NOISE_SLUGS = [
    "mmoreaux/environmental-sound-classification-50",  # ESC-50, 16kHz WAV
    "chrisfilo/urbansound8k",                          # UrbanSound8K
]


# ------------------------------------------------------------------
# YORDAMCHI: KAGGLE YUKLAB OLISH
# ------------------------------------------------------------------

def check_kaggle() -> bool:
    try:
        r = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        print(f"  [OK] Kaggle CLI: {r.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("  [!!] kaggle topilmadi: pip install kaggle")
        return False


def kaggle_download(slug: str, out_dir: Path) -> bool:
    """
    Dataset yuklaydi. Progress to'g'ridan-to'g'ri terminalda ko'rinadi.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  >>> Yuklanmoqda: kaggle datasets download -d {slug}")
    print(f"  >>> Papka: {out_dir}")
    print()

    # capture_output=False => progress terminaldа ko'rinadi
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug,
         "-p", str(out_dir), "--unzip"],
    )
    print()
    if result.returncode != 0:
        print(f"  [!!] Xato: {slug} — qaytdi {result.returncode}")
        return False

    files = list(out_dir.rglob("*"))
    n = sum(1 for f in files if f.is_file())
    print(f"  [OK] Yuklandi: {n} ta fayl -> {out_dir}")
    return True


def try_slugs(slugs: list, out_dir: Path, label: str) -> bool:
    """Ro'yxatdagi sluglarni birma-bir sinab ko'radi."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for i, slug in enumerate(slugs, 1):
        print(f"\n  [{i}/{len(slugs)}] Sinab ko'rilmoqda: {slug}")
        if kaggle_download(slug, out_dir):
            return True
    print(f"\n  [!!] {label} — hech biri ishlamadi, o'tkazildi")
    return False


# ------------------------------------------------------------------
# YORDAMCHI: AUDIO PROCESSING
# ------------------------------------------------------------------

def process_audio(src: str, dst_folder: Path, prefix: str,
                  max_clips: int = 3) -> int:
    """
    Istalgan formatdagi audio faylni:
      - 16kHz mono ga o'tkazadi
      - 1 soniyalik kliplaarga bo'ladi
      - dataset papkasiga saqlaydi
    Qaytaradi: qo'shilgan kliplar soni
    """
    dst_folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(dst_folder.glob("*.wav")))
    try:
        y, _ = librosa.load(src, sr=SR, mono=True)
    except Exception:
        return 0

    rms = np.sqrt(np.mean(y ** 2))
    if rms < 5e-5:          # juda jim — o'tkazib yuborish
        return 0

    clips = 0

    if len(y) < WIN_LEN:
        y = np.pad(y, (0, WIN_LEN - len(y)), mode='reflect')
        sf.write(str(dst_folder / f"{prefix}_{existing}.wav"),
                 y.astype(np.float32), SR)
        clips = 1
    else:
        step = WIN_LEN // 2
        for start in range(0, len(y) - WIN_LEN + 1, step):
            if clips >= max_clips:
                break
            seg = y[start: start + WIN_LEN]
            if np.sqrt(np.mean(seg ** 2)) < 5e-5:
                continue
            sf.write(str(dst_folder / f"{prefix}_{existing + clips}.wav"),
                     seg.astype(np.float32), SR)
            clips += 1

    return clips


def count(folder: Path) -> int:
    return len(list(folder.glob("*.wav")))


def copy_audio_to(src_dir: Path, dst_folder: Path,
                  prefix: str, needed: int) -> int:
    """src_dir dagi barcha audio fayllarni dst_folder ga ko'chiradi."""
    if needed <= 0:
        return 0

    all_files = (list(src_dir.rglob("*.wav")) +
                 list(src_dir.rglob("*.mp3")) +
                 list(src_dir.rglob("*.ogg")) +
                 list(src_dir.rglob("*.flac")))
    random.shuffle(all_files)
    print(f"  Topilgan: {len(all_files)} ta audio fayl")
    print(f"  Kerak:    {needed} ta klip")

    added = 0
    pbar  = tqdm(all_files, desc=f"  -> {dst_folder.name}", unit="fayl",
                 bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for f in pbar:
        if added >= needed:
            break
        n = process_audio(str(f), dst_folder, prefix)
        added += n
        pbar.set_postfix({"qo'shildi": added, "kerak": needed})

    return added


# ------------------------------------------------------------------
# 1. SNORING DATASET
# ------------------------------------------------------------------

def load_snoring():
    tmp = TMP / "snoring"
    ok  = try_slugs(SNORING_SLUGS, tmp, "SNORING DATASET (xurrak + xurrak emas)")
    if not ok:
        return

    # Papka strukturasini avtomatik aniqlash
    found_any = False
    for sub in sorted(tmp.rglob("*")):
        if not sub.is_dir():
            continue
        name = sub.name.lower()

        if name in ("1", "snoring", "snore", "snores"):
            target, pfx = OUT_1, "snore_kg"
        elif name in ("0", "notsnoring", "not_snoring", "nosnore",
                      "non_snoring", "nonsnoring", "normal", "noise",
                      "no_snore"):
            target, pfx = OUT_0, "nonsnore_kg"
        else:
            continue

        need = max(0, (TARGET_1 if target == OUT_1 else TARGET_0) - count(target))
        if need == 0:
            print(f"  {target} to'ldi, {name} o'tkazildi")
            continue

        print(f"\n  {name}/ -> {target}/")
        n = copy_audio_to(sub, target, pfx, need)
        print(f"  [OK] +{n} ta -> {target} (jami: {count(target)})")
        found_any = True

    if not found_any:
        print("  [!!] Snoring dataset ichida 0/ yoki 1/ papkalar topilmadi")
        print("  Barcha fayllar ko'rsatilmoqda:")
        for f in list(tmp.rglob("*"))[:20]:
            print(f"    {f}")


# ------------------------------------------------------------------
# 2. SPEECH DATA (nutq — xurrak emas deb o'rganishi uchun)
# ------------------------------------------------------------------

def load_speech():
    if count(OUT_0) >= TARGET_0:
        print("\n  dataset/0 to'ldi, speech kerak emas.")
        return

    tmp = TMP / "speech"
    ok  = try_slugs(SPEECH_SLUGS, tmp,
                    "SPEECH DATA (nutq — non-snoring o'rgatish uchun)")
    if not ok:
        return

    need  = max(0, TARGET_0 - count(OUT_0))
    added = copy_audio_to(tmp, OUT_0, "speech", need)
    print(f"  [OK] Speech: +{added} -> dataset/0 (jami: {count(OUT_0)})")


# ------------------------------------------------------------------
# 3. ENVIRONMENTAL NOISE (shovqin — xurrak emas deb o'rganishi uchun)
# ------------------------------------------------------------------

def load_noise():
    if count(OUT_0) >= TARGET_0:
        print("\n  dataset/0 to'ldi, noise kerak emas.")
        return

    tmp = TMP / "noise"
    ok  = try_slugs(NOISE_SLUGS, tmp,
                    "NOISE DATA (shovqin — non-snoring o'rgatish uchun)")
    if not ok:
        return

    need  = max(0, TARGET_0 - count(OUT_0))
    added = copy_audio_to(tmp, OUT_0, "noise", need)
    print(f"  [OK] Noise: +{added} -> dataset/0 (jami: {count(OUT_0)})")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    random.seed(42)

    print()
    print("=" * 55)
    print("  SNORE DETECTOR — DATASET YUKLOVCHI")
    print("=" * 55)
    print()
    print("  Maqsad: Model faqat XURRAKNI tanishi uchun")
    print("  dataset/0 ga: nutq, shovqin, boshqa ovozlar")
    print("  dataset/1 ga: faqat xurrak ovozlari")
    print()
    print(f"  Mavjud holat:")
    print(f"    dataset/0 (non-snoring): {count(OUT_0):>5} ta")
    print(f"    dataset/1 (snoring):     {count(OUT_1):>5} ta")

    OUT_0.mkdir(parents=True, exist_ok=True)
    OUT_1.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    if not check_kaggle():
        sys.exit(1)

    # 1. Snoring dataset
    print()
    load_snoring()

    # 2. Speech (nutq)
    print()
    load_speech()

    # 3. Noise (shovqin)
    print()
    load_noise()

    # Tozalash
    if TMP.exists():
        shutil.rmtree(TMP)

    # Yakuniy hisobot
    c0 = count(OUT_0)
    c1 = count(OUT_1)
    print()
    print("=" * 55)
    print("  YAKUNIY HOLAT")
    print("=" * 55)
    print(f"  dataset/0 (non-snoring): {c0:>5} ta")
    print(f"  dataset/1 (snoring):     {c1:>5} ta")
    print(f"  Jami:                    {c0+c1:>5} ta")
    print()

    if c0 < 800 or c1 < 800:
        print("  [!!] OGOHLANTIRISH: Biror klassda 800 dan kam fayl!")
        print("       Ko'proq data kerak — model yaxshi o'qimaydi.")
    elif abs(c0 - c1) > min(c0, c1) * 0.3:
        print("  [!!] OGOHLANTIRISH: Klasslar muvozanatsiz!")
        print(f"       dataset/0: {c0}  vs  dataset/1: {c1}")
        print("       Kamroq klassga data qo'shish tavsiya etiladi.")
    else:
        print("  [OK] Dataset muvozanatli va yetarli!")
        print()
        print("  Endi o'qitish:")
        print("    python train_and_export.py")

    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
