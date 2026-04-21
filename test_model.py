"""
Modelni audio fayl bilan tekshirish
Ishlatish:
    python test_model.py audio.wav
    python test_model.py  (dataset dan avtomatik oladi)

Uzun audiolar (>1s) uchun sliding window ishlatiladi:
  - Har 0.5s da 1s oyna siljiydi
  - Xurrak topilgan segmentlar ko'rsatiladi
"""

import sys
import os
import numpy as np
import librosa
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

SAMPLE_RATE  = 16000
DURATION     = 1.0
STEP         = 0.5   # sliding window qadam (soniya)
N_MELS       = 128
N_FFT        = 1024
HOP_LENGTH   = 128
IMG_SIZE     = 224


def make_model():
    model = models.efficientnet_v2_s(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),    nn.Sigmoid()
    )
    return model


def extract_segment(y_seg):
    """1 soniyalik numpy massivdan feature chiqaradi."""
    target = int(SAMPLE_RATE * DURATION)
    if len(y_seg) < target:
        y_seg = np.pad(y_seg, (0, target - len(y_seg)), mode='reflect')
    else:
        y_seg = y_seg[:target]

    # Ovoz balandligini normalize qilish — past/baland yozilgan audiodan qat'i nazar
    rms = np.sqrt(np.mean(y_seg ** 2))
    if rms > 1e-6:
        y_seg = y_seg / (rms + 1e-9) * 0.08

    mel    = librosa.feature.melspectrogram(y=y_seg, sr=SAMPLE_RATE,
                n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    d1     = librosa.feature.delta(mel_db, order=1)
    d2     = librosa.feature.delta(mel_db, order=2)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    ch0 = cv2.resize(norm(mel_db), (IMG_SIZE, IMG_SIZE))
    ch1 = cv2.resize(norm(d1),     (IMG_SIZE, IMG_SIZE))
    ch2 = cv2.resize(norm(d2),     (IMG_SIZE, IMG_SIZE))
    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)


def predict(model, device, file_path):
    """
    Qisqa audio (<=1s): bitta natija.
    Uzun audio (>1s): sliding window bilan har segmentni tekshiradi.
    """
    y_full, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    total_dur = len(y_full) / SAMPLE_RATE
    name      = os.path.basename(file_path)

    win_len  = int(SAMPLE_RATE * DURATION)
    step_len = int(SAMPLE_RATE * STEP)

    # Qisqa audio - bitta oyna
    if total_dur <= DURATION + 0.05:
        feat   = extract_segment(y_full)
        tensor = torch.FloatTensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(tensor).item()
        label = "XURRAK   " if score > 0.5 else "XURRAK EMAS"
        bar   = int(score * 30)
        print(f"  {name:<30} [{('#'*bar).ljust(30)}] {score:.3f}  =>  {label}")
        return score

    # Uzun audio - sliding window
    print(f"\n  {name}  ({total_dur:.1f}s)  -  sliding window ({DURATION}s oyna, {STEP}s qadam)")
    print(f"  {'Vaqt':<12} {'Natija':<45} Score")
    print("  " + "-"*70)

    scores = []
    starts = np.arange(0, len(y_full) - win_len + 1, step_len)
    if len(starts) == 0:
        starts = [0]

    for start in starts:
        seg    = y_full[start: start + win_len]
        feat   = extract_segment(seg)
        tensor = torch.FloatTensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(tensor).item()

        t_start = start / SAMPLE_RATE
        t_end   = t_start + DURATION
        is_seg  = score > 0.5
        bar     = int(score * 30)
        marker  = " <<<" if is_seg else ""
        seg_lbl = "XURRAK   " if is_seg else "xurrak emas"
        print(f"  {t_start:4.1f}s-{t_end:4.1f}s   [{('#'*bar).ljust(30)}] {score:.3f}  {seg_lbl}{marker}")
        scores.append(score)

    n          = len(scores)
    avg_score  = float(np.mean(scores))
    max_score  = float(max(scores))
    snore_segs = sum(1 for s in scores if s > 0.5)
    ratio      = snore_segs / n  # xurrak segmentlar ulushi

    # Uzluksiz ketma-ket xurrak segmentlar (0.5 threshold)
    max_consec = 0
    cur_run    = 0
    for s in scores:
        if s > 0.5:
            cur_run   += 1
            max_consec = max(max_consec, cur_run)
        else:
            cur_run = 0

    # -- Hukm mantiq ----------------------------------------------
    # Xurrak odamdan odamga farq qiladi - ba'zisi past, uzluksiz emas.
    # Shuning uchun faqat ketma-ket emas, NISBAT ham mezon.
    #
    # XURRAK BOR, agar:
    #   A) segmentlarning >= 35% xurrak (tarqoq xurrak ham shu yerga kiradi)
    #   B) YOKI kamida 2 ta ketma-ket segment (uzluksiz xurrak)
    # -------------------------------------------------------------
    rule_a   = ratio >= 0.50          # 35% -> 50%: nutqni xurrak demaslik uchun
    rule_b   = max_consec >= 3        # 2   -> 3:   tasodifiy match larni kamaytirish
    is_snore = rule_a or rule_b

    # Ishonch darajasi
    if ratio >= 0.65:
        confidence = "YUQORI"
    elif ratio >= 0.50:
        confidence = "O'RTA"
    elif max_consec >= 3:
        confidence = "O'RTA"
    else:
        confidence = "PAST"

    print()
    print(f"  {'-'*60}")
    print(f"  Jami segment:      {n} ta")
    print(f"  Xurrak segment:    {snore_segs}/{n}  ({ratio*100:.0f}%)")
    print(f"  Uzluksiz max:      {max_consec} ta segment")
    print(f"  O'rtacha score:    {avg_score:.3f}   |   Max: {max_score:.3f}")
    print(f"  {'-'*60}")

    if is_snore:
        print(f"  [OK]  XURRAK BOR  [{confidence} ishonch]")
        if rule_b and not rule_a:
            print(f"      (ketma-ket {max_consec} ta segment)")
        elif rule_a:
            print(f"      ({ratio*100:.0f}% segmentda xurrak belgilari)")
    else:
        print(f"  [!!]  XURRAK YO'Q  [{confidence} ishonch]")
        if snore_segs > 0:
            print(f"      ({snore_segs} ta segment biroz yuqori, lekin yetarli emas)")

    print(f"  {'-'*60}")
    return avg_score


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    print("  Model yuklanmoqda...")

    model = make_model()
    model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=False))
    model.eval().to(device)

    print("  Tayyor!\n")

    # Agar argument berilsa - shu faylni tekshir
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            if os.path.exists(f):
                predict(model, device, f)
            else:
                print(f"  Topilmadi: {f}")

    # Aks holda dataset dan namuna oladi
    else:
        print("  Fayl                           Natija                          Score   Label")
        print("  " + "-"*80)
        print("  [dataset/1 dan 5 ta XURRAK namunasi]")
        folder1 = "dataset/1"
        files1  = [f for f in os.listdir(folder1) if f.endswith('.wav')][:5]
        for f in files1:
            predict(model, device, os.path.join(folder1, f))

        print()
        print("  [dataset/0 dan 5 ta XURRAK EMAS namunasi]")
        folder0 = "dataset/0"
        files0  = [f for f in os.listdir(folder0) if f.endswith('.wav')][:5]
        for f in files0:
            predict(model, device, os.path.join(folder0, f))

    print()


if __name__ == "__main__":
    main()
