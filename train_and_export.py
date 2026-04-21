"""
================================================================
  SNORE DETECTOR - RTX 4060 OPTIMIZED (PyTorch + CUDA)
  EfficientNetV2-S + SpecAugment + Mixup + 5-Fold CV
  Target: 97%+ accuracy
================================================================
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# ---------------------------------------------
#  KONFIGURATSIYA
# ---------------------------------------------
DATASET_PATH    = "dataset"
SAMPLE_RATE     = 16000
DURATION        = 1.0
N_MELS          = 128
N_FFT           = 1024
HOP_LENGTH      = 128
IMG_SIZE        = 224
BATCH_SIZE      = 64        # RTX 4060 8GB VRAM uchun optimal
EPOCHS_FROZEN   = 25
EPOCHS_FINETUNE = 20
N_FOLDS         = 5
MODEL_OUTPUT    = "best_model.pt"
ONNX_OUTPUT     = "snore_model.onnx"


# ---------------------------------------------
#  GPU SOZLASH
# ---------------------------------------------
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  [OK] GPU: {name}")
        print(f"  [OK] VRAM: {vram:.1f} GB")
        print(f"  [OK] Mixed Precision (AMP) yoqildi - 2x tez")
        torch.backends.cudnn.benchmark = True
        return device
    print("  [?]  GPU topilmadi - CPU ishlatiladi")
    return torch.device('cpu')


# ---------------------------------------------
#  FEATURE EXTRACTION - 3 kanal (C, H, W)
#  Kanal 0: Mel-Spectrogram
#  Kanal 1: Delta
#  Kanal 2: Delta-Delta
# ---------------------------------------------
def wav_to_features(y):
    """Numpy waveform dan 3-kanal feature chiqaradi."""
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='reflect')
    else:
        y = y[:target_len]

    # RMS normalization — past/baland yozilgan ovozdan qat'i nazar bir xil
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-6:
        y = y / (rms + 1e-9) * 0.08

    mel    = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    delta1 = librosa.feature.delta(mel_db, order=1)
    delta2 = librosa.feature.delta(mel_db, order=2)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    ch0 = cv2.resize(norm(mel_db), (IMG_SIZE, IMG_SIZE))
    ch1 = cv2.resize(norm(delta1),  (IMG_SIZE, IMG_SIZE))
    ch2 = cv2.resize(norm(delta2),  (IMG_SIZE, IMG_SIZE))
    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)


def extract_features(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        return wav_to_features(y)
    except Exception:
        return None


# ---------------------------------------------
#  AUDIO AUGMENTATION (waveform darajasida)
#  Shovqinli muhitlarda ishlashi uchun
# ---------------------------------------------
def add_noise(y, snr_db=None):
    """Tasodifiy SNR bilan oq shovqin qo'shadi (10-30 dB)."""
    if snr_db is None:
        snr_db = np.random.uniform(10, 30)
    signal_power = np.mean(y ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def pitch_shift(y, sr=16000):
    """Tovush balandligini biroz o'zgartiradi (+/-2 ton)."""
    steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


def time_stretch(y):
    """Tezlikni biroz o'zgartiradi (0.85-1.15x)."""
    rate = np.random.uniform(0.85, 1.15)
    y2   = librosa.effects.time_stretch(y, rate=rate)
    target = int(SAMPLE_RATE * DURATION)
    if len(y2) < target:
        y2 = np.pad(y2, (0, target - len(y2)), mode='reflect')
    return y2[:target].astype(np.float32)


def audio_augment(y):
    """Waveform ga tasodifiy augmentatsiya qo'llaydi."""
    if np.random.random() < 0.6:
        y = add_noise(y)
    if np.random.random() < 0.3:
        y = pitch_shift(y)
    if np.random.random() < 0.3:
        y = time_stretch(y)
    # Volume o'zgartirish
    if np.random.random() < 0.5:
        gain = np.random.uniform(0.6, 1.4)
        y    = (y * gain).clip(-1, 1)
    return y


# ---------------------------------------------
#  SPEC AUGMENT
# ---------------------------------------------
def spec_augment(img, freq_mask=20, time_mask=20, num_masks=2):
    aug = img.copy()  # (C, H, W)
    _, h, w = aug.shape
    for _ in range(num_masks):
        f  = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, max(1, h - f))
        aug[:, f0:f0+f, :] = 0
        t  = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, max(1, w - t))
        aug[:, :, t0:t0+t] = 0
    return aug


# ---------------------------------------------
#  MIXUP
# ---------------------------------------------
def mixup_batch(X_b, y_b, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(X_b))
    return lam * X_b + (1-lam) * X_b[idx], lam * y_b + (1-lam) * y_b[idx]


# ---------------------------------------------
#  DATASET
# ---------------------------------------------
class SnoreDataset(Dataset):
    def __init__(self, X, y, raw_waves=None, augment=False):
        self.X         = X
        self.y         = y.astype(np.float32)
        self.raw_waves = raw_waves  # waveform larni saqlash (audio augment uchun)
        self.augment   = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Audio darajasida augmentatsiya (shovqin, pitch, stretch)
        if self.augment and self.raw_waves is not None and np.random.random() > 0.3:
            y_aug = audio_augment(self.raw_waves[idx].copy())
            x     = wav_to_features(y_aug)
        else:
            x = self.X[idx].copy()

        # Spectrogram darajasida augmentatsiya
        if self.augment and np.random.random() > 0.4:
            x = spec_augment(x)
        return torch.FloatTensor(x), torch.FloatTensor([self.y[idx]])


# ---------------------------------------------
#  FOCAL LOSS
# ---------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred  = pred.float().clamp(1e-7, 1-1e-7)
        target = target.float()
        bce   = -target * torch.log(pred) - (1-target) * torch.log(1-pred)
        p_t   = target * pred + (1-target) * (1-pred)
        return torch.mean(self.alpha * torch.pow(1-p_t, self.gamma) * bce)


# ---------------------------------------------
#  MODEL - EfficientNetV2-S
# ---------------------------------------------
def build_model():
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model


# ---------------------------------------------
#  TRAIN / VALIDATE
# ---------------------------------------------
def train_epoch(model, loader, optimizer, criterion, scaler, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == 'cuda'

    for X_b, y_b in loader:
        X_np = X_b.numpy()
        y_np = y_b.numpy()

        if use_mixup and np.random.random() > 0.5:
            X_np, y_np = mixup_batch(X_np, y_np)

        X_b = torch.FloatTensor(X_np).to(device, non_blocking=True)
        y_b = torch.FloatTensor(y_np).to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(X_b)
            loss = criterion(pred, y_b)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        predicted   = (pred.detach() > 0.5).float()
        correct    += (predicted == y_b).sum().item()
        total      += y_b.numel()

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(X_b)
                loss = criterion(pred, y_b)

            total_loss += loss.item()
            predicted   = (pred > 0.5).float()
            correct    += (predicted == y_b).sum().item()
            total      += y_b.numel()

    return total_loss / len(loader), correct / total


# ---------------------------------------------
#  DATASET YUKLASH
# ---------------------------------------------
def load_dataset():
    print("\n" + "="*55)
    print("  DATASET YUKLANMOQDA...")
    print("="*55)
    X, y, waves = [], [], []
    for label in [0, 1]:
        folder = os.path.join(DATASET_PATH, str(label))
        if not os.path.exists(folder):
            print(f"\n  [!!] XATO: {folder} topilmadi!")
            print(f"  dataset/0/  va  dataset/1/  bo'lishi kerak")
            exit(1)
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        name  = "XURRAK" if label == 1 else "XURRAK EMAS"
        print(f"\n  {name}: {len(files)} ta fayl")
        for fname in tqdm(files, desc=f"  {name}"):
            path = os.path.join(folder, fname)
            try:
                wave, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
            except Exception:
                continue
            feat = wav_to_features(wave)
            if feat is not None:
                X.append(feat)
                y.append(label)
                target = int(SAMPLE_RATE * DURATION)
                if len(wave) < target:
                    wave = np.pad(wave, (0, target - len(wave)), mode='reflect')
                waves.append(wave[:target].astype(np.float32))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\n  [OK] Jami: {len(X)}  |  Xurrak: {sum(y==1)}  |  Xurrak emas: {sum(y==0)}")
    return X, y, waves


# ---------------------------------------------
#  5-FOLD CROSS VALIDATION
# ---------------------------------------------
def run_cv(X, y, waves, device):
    print("\n" + "="*55)
    print(f"  {N_FOLDS}-FOLD CROSS VALIDATION")
    print("="*55)

    waves_arr   = np.array(waves, dtype=np.float32)
    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    best_acc    = 0
    best_model  = None
    scaler      = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    criterion   = FocalLoss()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'-'*55}\n  FOLD {fold}/{N_FOLDS}\n{'-'*55}")

        X_tr, X_val   = X[tr_idx], X[val_idx]
        y_tr, y_val   = y[tr_idx], y[val_idx]
        w_tr          = waves_arr[tr_idx]

        train_loader = DataLoader(
            SnoreDataset(X_tr, y_tr, raw_waves=w_tr, augment=True),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            SnoreDataset(X_val, y_val),
            batch_size=BATCH_SIZE, num_workers=0, pin_memory=True
        )

        model = build_model().to(device)

        # -- Bosqich 1: Frozen - faqat classifier o'qitiladi --
        optimizer   = optim.Adam(model.classifier.parameters(), lr=1e-3)
        scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3, min_lr=1e-7
        )
        best_state  = None
        best_val_acc = 0
        patience_cnt = 0

        print(f"  Bosqich 1: Frozen ({EPOCHS_FROZEN} epoch max)")
        for epoch in range(EPOCHS_FROZEN):
            tr_loss, tr_acc = train_epoch(
                model, train_loader, optimizer, criterion, scaler, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(f"  Ep {epoch+1:2d}/{EPOCHS_FROZEN} | "
                  f"loss {tr_loss:.4f} | acc {tr_acc*100:.1f}% | "
                  f"val_acc {val_acc*100:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 6:
                    print("  Early stopping")
                    break

        model.load_state_dict(best_state)

        # -- Bosqich 2: Fine-tune - oxirgi 50 qatlamni ochish --
        for param in model.features.parameters():
            param.requires_grad = True
        feat_params = list(model.features.parameters())
        for param in feat_params[:-50]:
            param.requires_grad = False

        optimizer    = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6
        )
        scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3, min_lr=1e-8
        )
        best_val_acc = 0
        patience_cnt = 0

        print(f"  Bosqich 2: Fine-tune ({EPOCHS_FINETUNE} epoch max)")
        for epoch in range(EPOCHS_FINETUNE):
            tr_loss, tr_acc = train_epoch(
                model, train_loader, optimizer, criterion, scaler, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(f"  Ep {epoch+1:2d}/{EPOCHS_FINETUNE} | "
                  f"loss {tr_loss:.4f} | acc {tr_acc*100:.1f}% | "
                  f"val_acc {val_acc*100:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
                torch.save(model.state_dict(), f'fold_{fold}_best.pt')
            else:
                patience_cnt += 1
                if patience_cnt >= 5:
                    print("  Early stopping")
                    break

        model.load_state_dict(best_state)

        # Fold accuracy
        model.eval()
        preds = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                out = model(X_b.to(device)).cpu().numpy().flatten()
                preds.extend((out > 0.5).astype(int))

        acc = float(np.mean(np.array(preds) == y_val))
        fold_scores.append(acc)
        print(f"\n  Fold {fold} Accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc   = acc
            best_model = model
            torch.save(model.state_dict(), MODEL_OUTPUT)
            print(f"  [OK] Yangi eng yaxshi model saqlandi!")

        if model is not best_model:
            del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"  CROSS VALIDATION NATIJALARI")
    for i, s in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {s*100:.2f}%")
    print(f"  {'-'*35}")
    print(f"  O'rtacha:   {np.mean(fold_scores)*100:.2f}% +/- {np.std(fold_scores)*100:.2f}%")
    print(f"  Eng yaxshi: {max(fold_scores)*100:.2f}%")

    return best_model, fold_scores


# ---------------------------------------------
#  FINAL TEST + GRAFIKLAR
# ---------------------------------------------
def final_evaluation(model, X_test, y_test, fold_scores, device):
    print("\n" + "="*55)
    print("  FINAL TEST NATIJALARI")
    print("="*55)

    model.eval()
    test_loader = DataLoader(SnoreDataset(X_test, y_test), batch_size=BATCH_SIZE)
    preds = []

    with torch.no_grad():
        for X_b, _ in test_loader:
            out = model(X_b.to(device)).cpu().numpy().flatten()
            preds.extend((out > 0.5).astype(int))

    preds = np.array(preds)
    print("\n" + classification_report(
        y_test, preds,
        target_names=['Xurrak emas (0)', 'Xurrak (1)'],
        digits=4
    ))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Xurrak emas', 'Xurrak'],
                yticklabels=['Xurrak emas', 'Xurrak'])
    axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Haqiqiy')
    axes[0].set_xlabel('Bashorat')

    colors = ['#2196F3' if s == max(fold_scores) else '#90CAF9' for s in fold_scores]
    axes[1].bar(range(1, N_FOLDS+1), [s*100 for s in fold_scores],
                color=colors, edgecolor='white', linewidth=1.5)
    axes[1].axhline(y=np.mean(fold_scores)*100, color='red', linestyle='--',
                    label=f"O'rtacha: {np.mean(fold_scores)*100:.2f}%")
    axes[1].set_title('5-Fold Cross Validation', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim([80, 101])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Snore Detection - EfficientNetV2-S + SpecAugment',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  results.png saqlandi")


# ---------------------------------------------
#  ONNX EXPORT (Android - ONNX Runtime)
# ---------------------------------------------
def export_onnx(model, device):
    print("\n" + "="*55)
    print("  ONNX EXPORT (Android uchun)...")
    print("="*55)

    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    torch.onnx.export(
        model, dummy, ONNX_OUTPUT,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )

    size_mb = os.path.getsize(ONNX_OUTPUT) / (1024 * 1024)
    print(f"\n  [OK] {ONNX_OUTPUT} - {size_mb:.2f} MB")
    print("  Android'da ONNX Runtime bilan ishlatish mumkin")

    # Test inference
    dummy_np = dummy.cpu().numpy()
    print(f"  Input shape: {dummy_np.shape}")
    with torch.no_grad():
        result = model(dummy.to(device)).cpu().item()
    label = 'Xurrak' if result > 0.5 else 'Xurrak emas'
    print(f"  Test inference: {result:.4f} -> {label}")


# ---------------------------------------------
#  MAIN
# ---------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SNORE DETECTOR - RTX 4060 OPTIMIZED")
    print("  EfficientNetV2-S + SpecAugment + Mixup + 5-Fold CV")
    print("  Backend: PyTorch + CUDA")
    print("="*55)

    device = setup_device()

    X, y, waves = load_dataset()

    X_trainval, X_test, y_trainval, y_test, w_trainval, _ = train_test_split(
        X, y, waves, test_size=0.15, random_state=42, stratify=y
    )
    print(f"\n  Train+Val: {len(X_trainval)}  |  Final Test: {len(X_test)}")

    best_model, fold_scores = run_cv(X_trainval, y_trainval, w_trainval, device)

    final_evaluation(best_model, X_test, y_test, fold_scores, device)

    export_onnx(best_model, device)

    print("\n" + "="*55)
    print("  [OK] HAMMASI TAYYOR!")
    print(f"  -> {MODEL_OUTPUT}     - PyTorch modeli")
    print(f"  -> {ONNX_OUTPUT}  - Android (ONNX Runtime)")
    print("="*55 + "\n")
