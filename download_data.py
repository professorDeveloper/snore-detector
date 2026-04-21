#!/usr/bin/env python3
"""
Dataset tayyorlash — Kaggle'dan audio yuklab olish
=======================================================
Nima qiladi:
  dataset/0/  ->  nutq (speech) + boshqa ovozlar  [NON-SNORING]
  dataset/1/  ->  xurrak ovozlar                   [SNORING]

Ishlatish:
  1. Kaggle API token olish:  https://www.kaggle.com/settings -> API -> Create Token
  2. Tokenni qo'yish:  ~/.kaggle/kaggle.json  (Mac/Linux)
                       C:\\Users\\<name>\\.kaggle\\kaggle.json  (Windows)
  3. pip install kaggle librosa soundfile tqdm
  4. python download_data.py

Yuklanadigan datasetlar (Kaggle):
  - google-speech-commands  ->  dataset/0  (nutq, so'zlar)
  - snoring-detection       ->  dataset/0 va dataset/1  (to'ldirish uchun)
"""

import os
import sys
import shutil
import zipfile
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
OUT_0        = Path("dataset/0")   # Non-snoring
OUT_1        = Path("dataset/1")   # Snoring
SAMPLE_RATE  = 16000
DURATION     = 1.0
WIN_LEN      = int(SAMPLE_RATE * DURATION)

TARGET_0     = 2000   # dataset/0 ga nechta fayl kerak
TARGET_1     = 2000   # dataset/1 ga nechta fayl kerak

TMP_DIR      = Path("_tmp_download")

# Kaggle dataset sluglari (bir nechta variant — biri ishlamasa keyingisi sinab ko'riladi)
DATASETS_SNORING = [
    "emilianogalimberti/snoring-dataset",
    "tareqtaha/snoring-dataset",
    "datasets/snoring-detection",
]
DATASETS_NONSPEECH = [
    "mmoreaux/environmental-sound-classification",   # ESC-50
    "chrisfilo/urbansound8k",
    "soumendrakumar/urban-sound-classification",
]


# ------------------------------------------------------------------
# YORDAMCHI FUNKSIYALAR
# ------------------------------------------------------------------

def check_kaggle():
    """kaggle CLI mavjudligini tekshiradi."""
    try:
        result = subprocess.run(["kaggle", "--version"],
                                capture_output=True, text=True)
        print(f"  [OK] Kaggle CLI: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("\n  [!!] kaggle topilmadi. O'rnatish:")
        print("       pip install kaggle")
        print("       Keyin:  https://www.kaggle.com/settings -> API -> Create New Token")
        return False


def kaggle_download(slug: str, out_dir: Path) -> bool:
    """Kaggle dataset yuklab oladi, zip ni ochadi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Yuklanmoqda: {slug}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug,
         "-p", str(out_dir), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err = (result.stderr.strip() or result.stdout.strip())[:300]
        print(f"  [!!] Xato ({slug}):")
        print(f"       {err if err else 'noma`lum xato'}")
        return False
    print(f"  [OK] Yuklandi -> {out_dir}")
    return True


def kaggle_download_first(slugs: list, out_dir: Path) -> bool:
    """Sluglar ro'yxatidan birinchi muvaffaqiyatli yuklanganini oladi."""
    for slug in slugs:
        if kaggle_download(slug, out_dir):
            return True
    print(f"  [!!] Hech qaysi slug ishlamadi: {slugs}")
    return False


def process_audio(src_path: str, dst_folder: Path,
                  prefix: str, max_clips: int = None) -> int:
    """
    Audio faylni 16kHz mono ga o'tkazadi, 1s parchalarga bo'ladi.
    Qaytaradi: qo'shilgan kliplar soni.
    """
    dst_folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(dst_folder.glob("*.wav")))

    try:
        y, _ = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return 0

    # Ovoz bor-yo'qligini tekshirish (juda jim fayllarni o'tkazib yuborish)
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-4:
        return 0

    clips_added = 0

    if len(y) < WIN_LEN:
        # Qisqa — pad qilib 1 klip
        y = np.pad(y, (0, WIN_LEN - len(y)), mode='reflect')
        out_path = dst_folder / f"{prefix}_{existing + clips_added}.wav"
        sf.write(str(out_path), y.astype(np.float32), SAMPLE_RATE)
        clips_added += 1
    else:
        # Uzun — 1s oynalarga bo'lish
        step = WIN_LEN // 2  # 50% overlap
        for start in range(0, len(y) - WIN_LEN + 1, step):
            if max_clips and clips_added >= max_clips:
                break
            seg = y[start: start + WIN_LEN]
            seg_rms = np.sqrt(np.mean(seg ** 2))
            if seg_rms < 1e-4:
                continue
            out_path = dst_folder / f"{prefix}_{existing + clips_added}.wav"
            sf.write(str(out_path), seg.astype(np.float32), SAMPLE_RATE)
            clips_added += 1

    return clips_added


def count(folder: Path) -> int:
    return len(list(folder.glob("*.wav")))


# ------------------------------------------------------------------
# DATASET 1: SPEECH COMMANDS (dataset/0 uchun nutq)
# ------------------------------------------------------------------

def load_nonspeech_sounds():
    """ESC-50 / UrbanSound — turli ovozlar (non-snoring uchun)."""
    print("\n" + "=" * 55)
    print("  NON-SNORING OVOZLAR yuklanmoqda...")
    print("=" * 55)

    tmp = TMP_DIR / "nonspeech"
    ok  = kaggle_download_first(DATASETS_NONSPEECH, tmp)
    if not ok:
        print("  [!!] Non-snoring data yuklanmadi, o'tkaziladi")
        return

    all_audio = (list(tmp.rglob("*.wav")) +
                 list(tmp.rglob("*.ogg")) +
                 list(tmp.rglob("*.mp3")))
    random.shuffle(all_audio)
    print(f"  Topilgan: {len(all_audio)} ta fayl")

    needed = max(0, TARGET_0 - count(OUT_0))
    print(f"  Kerak: {needed} ta klip dataset/0 ga")

    added = 0
    pbar  = tqdm(all_audio, desc="  Ovoz->dataset/0")
    for f in pbar:
        if added >= needed:
            break
        n = process_audio(str(f), OUT_0, "env", max_clips=2)
        added += n
        pbar.set_postfix(added=added)

    print(f"  [OK] +{added} ta klip -> dataset/0 (jami: {count(OUT_0)})")


# ------------------------------------------------------------------
# DATASET 2: SNORING DATASET (ikki klass uchun)
# ------------------------------------------------------------------

def load_snoring_dataset():
    """Snoring dataset -> dataset/0 va dataset/1 ga qo'shadi."""
    print("\n" + "=" * 55)
    print("  SNORING DATASET yuklanmoqda...")
    print("=" * 55)

    tmp = TMP_DIR / "snoring"
    ok  = kaggle_download_first(DATASETS_SNORING, tmp)
    if not ok:
        print("  [!!] Snoring dataset yuklanmadi, o'tkaziladi")
        return

    # Papka strukturasini topish
    # Ko'p datasetlarda: 0/ va 1/ yoki Snoring/ va NotSnoring/ bo'ladi
    for subfolder in tmp.rglob("*"):
        if not subfolder.is_dir():
            continue

        name = subfolder.name.lower()
        wavs = list(subfolder.glob("*.wav"))
        if not wavs:
            wavs = list(subfolder.glob("*.ogg")) + list(subfolder.glob("*.mp3"))
        if not wavs:
            continue

        # Qaysi klass ekanini aniqlash
        if name in ("1", "snoring", "snore", "xurrak"):
            target = OUT_1
            prefix = "snore_kaggle"
        elif name in ("0", "notsnoring", "not_snoring", "nosnore",
                      "non_snoring", "nonsnoring", "normal", "noise"):
            target = OUT_0
            prefix = "nonsnore_kaggle"
        else:
            continue

        needed = max(0, TARGET_1 - count(target)) if target == OUT_1 \
            else max(0, TARGET_0 - count(target))
        if needed == 0:
            continue

        random.shuffle(wavs)
        added = 0
        for wav in tqdm(wavs, desc=f"  {subfolder.name} -> {target.name}"):
            if added >= needed:
                break
            n = process_audio(str(wav), target, prefix, max_clips=2)
            added += n

        print(f"  [OK] {subfolder.name}: +{added} -> {target} (jami: {count(target)})")


# ------------------------------------------------------------------
# DATASET 3: URBANSOUND (qo'shimcha non-snoring)
# ------------------------------------------------------------------

def load_urbansound():
    """Qo'shimcha non-snoring (agar hali ham kam bo'lsa)."""
    if count(OUT_0) >= TARGET_0:
        print("\n  dataset/0 to'ldi, qo'shimcha kerak emas.")
        return

    print("\n" + "=" * 55)
    print("  QO'SHIMCHA NON-SNORING yuklanmoqda...")
    print("=" * 55)

    tmp = TMP_DIR / "urban2"
    slugs = [s for s in DATASETS_NONSPEECH if (TMP_DIR / "nonspeech").exists() is False or s != DATASETS_NONSPEECH[0]]
    ok  = kaggle_download_first(slugs, tmp)
    if not ok:
        print("  [!!] Qo'shimcha data yuklanmadi, o'tkaziladi")
        return

    all_wavs = list(tmp.rglob("*.wav"))
    random.shuffle(all_wavs)
    print(f"  Topilgan: {len(all_wavs)} ta fayl")

    needed = max(0, TARGET_0 - count(OUT_0))
    added  = 0
    pbar   = tqdm(all_wavs, desc="  Urban->dataset/0")
    for wav in pbar:
        if added >= needed:
            break
        n = process_audio(str(wav), OUT_0, "urban", max_clips=1)
        added += n
        pbar.set_postfix(added=added)

    print(f"  [OK] Urban: +{added} -> dataset/0 (jami: {count(OUT_0)})")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    random.seed(42)

    print("\n" + "=" * 55)
    print("  DATASET TAYYORLOVCHI SKRIPT")
    print("  Mac va Windows da ishlaydi")
    print("=" * 55)

    # Papkalarni yaratish
    OUT_0.mkdir(parents=True, exist_ok=True)
    OUT_1.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Mavjud holat:")
    print(f"    dataset/0 (non-snoring): {count(OUT_0)} ta")
    print(f"    dataset/1 (snoring):     {count(OUT_1)} ta")

    # kaggle tekshirish
    if not check_kaggle():
        sys.exit(1)

    # 1. Snoring dataset (ikkala klass uchun)
    load_snoring_dataset()

    # 2. Non-snoring ovozlar (ESC-50 / UrbanSound)
    if count(OUT_0) < TARGET_0:
        load_nonspeech_sounds()

    # 3. Hali ham kam bo'lsa — yana boshqa variant
    if count(OUT_0) < TARGET_0:
        load_urbansound()

    # Tmp papkani o'chirish
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
        print("\n  Vaqtinchalik fayllar o'chirildi.")

    # Yakuniy hisobot
    c0 = count(OUT_0)
    c1 = count(OUT_1)
    print("\n" + "=" * 55)
    print("  TAYYOR!")
    print(f"  dataset/0 (non-snoring): {c0} ta")
    print(f"  dataset/1 (snoring):     {c1} ta")
    print(f"  Jami:                    {c0 + c1} ta")

    if c0 < 500 or c1 < 500:
        print("\n  [!!] Ogohlantirish: bir klassda 500 dan kam fayl.")
        print("       Ko'proq data qo'shish tavsiya etiladi.")
    else:
        print("\n  Endi o'qitish uchun:")
        print("    python train_and_export.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
