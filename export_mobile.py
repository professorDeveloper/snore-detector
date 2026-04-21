"""
Android uchun PyTorch Mobile export
Kirish:  FloatTensor [N]  — 1–10 soniya audio (16kHz, mono)
Chiqish: FloatTensor [3]  — [avg_score, max_consec_ratio, is_snore]
           is_snore = 1.0 -> XURRAK BOR
           is_snore = 0.0 -> XURRAK YO'Q

Android tarafida:
    float[] out = module.forward(IValue.from(tensor)).toTensor().getDataAsFloatArray();
    boolean isSnore = out[2] > 0.5f;
"""

import os
from typing import List

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as AF
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile

SAMPLE_RATE  = 16000
DURATION     = 1.0
STEP         = 0.5
WIN_LEN      = int(SAMPLE_RATE * DURATION)              # 16000
STEP_LEN     = int(SAMPLE_RATE * STEP)                  # 8000
MAX_SECONDS  = 10
MAX_SAMPLES  = SAMPLE_RATE * MAX_SECONDS                # 160000
MAX_WINDOWS  = (MAX_SAMPLES - WIN_LEN) // STEP_LEN + 1  # 19
SEG_THRESH   = 0.5   # segment xurrak chegarasi
MIN_RATIO    = 0.50  # kamida 50% segment xurrak bo'lsa (nutqni kamaytirish uchun)
MIN_CONSEC   = 3     # YOKI kamida 3 ta ketma-ket segment


def make_backbone() -> nn.Module:
    model = models.efficientnet_v2_s(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),    nn.Sigmoid()
    )
    return model


class SnoreDetector(nn.Module):
    """
    Kirish:  waveform FloatTensor [N]  (1-10s, 16kHz, mono)
    Chiqish: FloatTensor [3]
               [0] avg_score       -- o'rtacha ishonch (0-1)
               [1] max_consec_norm -- uzluksiz max / max_win (0-1)
               [2] is_snore        -- 1.0=XURRAK BOR, 0.0=YO'Q
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

        # TorchScript uchun barcha konstantalar self.* bo'lishi kerak
        self.img_size:    int   = 224
        self.win_len:     int   = WIN_LEN
        self.step_len:    int   = STEP_LEN
        self.max_samples: int   = MAX_SAMPLES
        self.max_win:     int   = MAX_WINDOWS
        self.seg_thresh:  float = SEG_THRESH
        self.min_ratio:   float = MIN_RATIO
        self.min_consec:  float = float(MIN_CONSEC)

        self.mel_transform   = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024,
            hop_length=128, n_mels=128, f_max=8000.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mn = x.min()
        mx = x.max()
        return (x - mn) / (mx - mn + 1e-8)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.interpolate(
            x, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        )
        return x.squeeze(0).squeeze(0)

    def _segment_score(self, seg: torch.Tensor) -> torch.Tensor:
        if seg.shape[0] < self.win_len:
            pad = self.win_len - seg.shape[0]
            seg = torch.nn.functional.pad(seg, (0, pad))
        else:
            seg = seg[:self.win_len]

        # RMS normalization — train bilan bir xil preprocessing
        rms = torch.sqrt(torch.mean(seg ** 2))
        if rms > 1e-6:
            seg = seg / (rms + 1e-9) * 0.08

        mel    = self.mel_transform(seg)
        mel_db = self.amplitude_to_db(mel)
        mel_3d = mel_db.unsqueeze(0)
        d1     = AF.compute_deltas(mel_3d).squeeze(0)
        d2     = AF.compute_deltas(d1.unsqueeze(0)).squeeze(0)

        ch0 = self._resize(self._norm(mel_db))
        ch1 = self._resize(self._norm(d1))
        ch2 = self._resize(self._norm(d2))

        img = torch.stack([ch0, ch1, ch2], dim=0).unsqueeze(0)
        return self.backbone(img).squeeze()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        max_s = self.max_samples
        if waveform.shape[0] > max_s:
            waveform = waveform[:max_s]

        n_samples = waveform.shape[0]
        win_len   = self.win_len
        step_len  = self.step_len
        max_win   = self.max_win

        scores: List[torch.Tensor] = []
        start: int = 0
        for _ in range(max_win):
            if start + win_len > n_samples:
                break
            seg   = waveform[start: start + win_len]
            score = self._segment_score(seg)
            scores.append(score)
            start = start + step_len

        if len(scores) == 0:
            score = self._segment_score(waveform)
            scores.append(score)

        score_tensor = torch.stack(scores)
        avg_score    = score_tensor.mean()
        n_win        = float(len(scores))

        # Uzluksiz ketma-ket xurrak segmentlar soni
        max_consec = torch.zeros(1)
        cur_run    = torch.zeros(1)
        thresh     = self.seg_thresh
        for i in range(len(scores)):
            if scores[i].item() > thresh:
                cur_run = cur_run + 1.0
                if cur_run[0].item() > max_consec[0].item():
                    max_consec = cur_run.clone()
            else:
                cur_run = torch.zeros(1)

        snore_count = torch.zeros(1)
        for i in range(len(scores)):
            if scores[i].item() > self.seg_thresh:
                snore_count = snore_count + 1.0
        ratio = snore_count[0].item() / n_win

        # XURRAK BOR: >= 35% segment YOKI ketma-ket >= 2 segment
        rule_a = ratio >= self.min_ratio
        rule_b = max_consec[0].item() >= self.min_consec
        is_snore = torch.zeros(1)
        if rule_a or rule_b:
            is_snore = torch.ones(1)

        max_consec_norm = max_consec / n_win

        return torch.stack([
            avg_score.reshape(1),
            max_consec_norm.reshape(1),
            is_snore.reshape(1)
        ]).squeeze(-1)


def main() -> None:
    device = torch.device('cpu')

    print("  Weights yuklanmoqda...")
    backbone = make_backbone()
    state    = torch.load('best_model.pt', map_location=device, weights_only=False)
    backbone.load_state_dict(state)
    backbone.eval()

    print("  SnoreDetector qurilmoqda...")
    full_model = SnoreDetector(backbone)
    full_model.eval()

    print("  TorchScript...")
    scripted = torch.jit.script(full_model)

    print("\n  Test natijalar:")
    tests = [
        ("1s  tovush (past)", torch.randn(16000) * 0.05),
        ("3s  tovush",        torch.randn(48000) * 0.1),
        ("10s tovush",        torch.randn(160000) * 0.1),
    ]
    for name, audio in tests:
        with torch.no_grad():
            out = scripted(audio)
        verdict = "XURRAK BOR" if out[2].item() > 0.5 else "XURRAK YO'Q"
        print(f"    {name}: avg={out[0].item():.3f}  consec={out[1].item():.2f}  => {verdict}")

    print("\n  Mobile optimize...")
    optimized = optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter('snore_mobile.ptl')

    size_mb = os.path.getsize('snore_mobile.ptl') / 1024 ** 2
    print(f"\n  OK: snore_mobile.ptl -- {size_mb:.1f} MB")
    print()
    print("  Android (Java/Kotlin):")
    print("  --------------------------------------------------")
    print("  Module module = LiteModuleLoader.load(assetFilePath(\"snore_mobile.ptl\"));")
    print("  Tensor input  = Tensor.fromBlob(audioFloats, new long[]{audioFloats.length});")
    print("  float[] out   = module.forward(IValue.from(input)).toTensor().getDataAsFloatArray();")
    print("  boolean snore = out[2] > 0.5f;  // true = XURRAK BOR")
    print("  --------------------------------------------------")


if __name__ == '__main__':
    main()
